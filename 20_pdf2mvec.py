import time
import math
from pathlib import Path
import torch
from PIL import Image
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
import json
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    pass


def convert_pdf_to_images(src_file_path: Path, img_path: Path) -> List[Path]:
    try:
        tgt_file_path_prefix = src_file_path.stem
        images = convert_from_path(str(src_file_path))
        if not images:
            raise ProcessingError(f"No images extracted from {src_file_path}")

        digits = int(math.log10(len(images))) + 1
        img_paths = []

        for i, img in enumerate(images, start=1):
            img_file_path = (
                img_path / f"{tgt_file_path_prefix}_{i:0{digits}d}_of_{len(images)}.png"
            )
            img.save(img_file_path, "PNG")
            img_paths.append(img_file_path)

        return img_paths
    except Exception as e:
        raise ProcessingError(f"Failed to convert PDF to images: {str(e)}")


def process_images_to_vectors(
    img_paths: List[Path],
    model: ColPali,
    processor: ColPaliProcessor,
    batch_size: int = 8,
) -> torch.Tensor:
    image_embeddings = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        try:
            # Load and process images in batches
            batch_images = []
            for img_path in batch_paths:
                with Image.open(img_path) as img:
                    batch_images.append(img.convert("RGB"))

            processed_batch = processor.process_images(batch_images).to(model.device)
            logger.info(
                f"Processing images {i+1} to {min(i+batch_size, len(img_paths))}..."
            )

            with torch.no_grad():
                start_time = time.time()
                batch_embeddings = model(**processed_batch)
                image_embeddings.append(batch_embeddings)
                logger.info(f"Batch processing time: {time.time() - start_time:.2f}s")

        except Exception as e:
            raise ProcessingError(f"Failed to process images {batch_paths}: {str(e)}")

    return torch.cat(image_embeddings)


def save_embeddings_to_json(
    src_file_path: Path,
    img_paths: List[Path],
    image_embeddings: torch.Tensor,
    output_dir: Path,
) -> None:
    try:
        image_embeddings_dict = {
            img_path.stem: img_emb.cpu().to(torch.float32).numpy().tolist()
            for img_path, img_emb in zip(img_paths, image_embeddings)
        }

        output_path = output_dir / f"{src_file_path.stem}_embeddings.json"

        with open(output_path, "w") as f:
            json.dump(image_embeddings_dict, f)

        logger.info(f"Saved embeddings to {output_path}")
    except Exception as e:
        raise ProcessingError(f"Failed to save embeddings: {str(e)}")


def cleanup_images(img_paths: List[Path]) -> None:
    for img_path in img_paths:
        try:
            img_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temporary image {img_path}: {str(e)}")


def main():
    try:
        src_path = Path("./data/src")
        img_path = Path("./data/img")
        output_dir = Path("./data/embeddings")

        # Create directories
        for path in [src_path, img_path, output_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Load model
        model_name = "vidore/colpali-v1.3"
        try:
            model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="mps",  # If CUDA is available, use "cuda"
            ).eval()
            processor = ColPaliProcessor.from_pretrained(model_name)
        except Exception as e:
            raise ProcessingError(f"Failed to load model: {str(e)}")

        # Process PDFs
        src_file_paths = list(src_path.glob("*.pdf"))
        if not src_file_paths:
            logger.warning("No PDF files found in source directory")
            return

        for src_file_path in src_file_paths:
            output_path = output_dir / f"{src_file_path.stem}_embeddings.json"

            # Skip if output file already exists
            if output_path.exists():
                logger.info(f"Skipping {src_file_path} - embeddings already exist")
                continue

            try:
                logger.info(f"Processing {src_file_path}...")
                img_paths = convert_pdf_to_images(src_file_path, img_path)
                image_embeddings = process_images_to_vectors(
                    img_paths, model, processor
                )
                save_embeddings_to_json(
                    src_file_path, img_paths, image_embeddings, output_dir
                )
                cleanup_images(img_paths)
            except ProcessingError as e:
                logger.error(f"Failed to process {src_file_path}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
