from pathlib import Path
import torch
import h5py
import numpy as np
import logging
from helpers import ProcessingError, convert_pdf_to_images, process_images_to_vectors, get_model_and_processor, SRC_DIR, IMG_DIR, EMBEDDING_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_embeddings_to_hdf5(
    src_file_path: Path,
    img_paths: list[Path],
    image_embeddings: torch.Tensor,
    output_dir: Path,
) -> None:
    try:
        hdf5_path = output_dir / f"{src_file_path.stem}_embeddings.h5"

        with h5py.File(hdf5_path, 'w') as hdf5_file:
            hdf5_file.create_dataset('embeddings', data=image_embeddings.cpu().to(torch.float32).numpy())
            hdf5_file.create_dataset('filepaths', data=[str(p) for p in img_paths])
        logger.info(f"Embeddings saved to {hdf5_path}")
    except Exception as e:
        raise ProcessingError(f"Failed to save embeddings: {str(e)}")


def save_embeddings_to_npz(
    src_file_path: Path,
    img_paths: list[Path],
    image_embeddings: torch.Tensor,
    output_dir: Path,
) -> None:
    try:
        npz_path = output_dir / f"{src_file_path.stem}_embeddings.npz"

        np.savez_compressed(
            npz_path,
            embeddings=image_embeddings.cpu().to(torch.float32).numpy(),
            filepaths=[str(p) for p in img_paths]
        )
        logger.info(f"Embeddings saved to {npz_path}")
    except Exception as e:
        raise ProcessingError(f"Failed to save embeddings: {str(e)}")


def cleanup_images(img_paths: list[Path]) -> None:
    for img_path in img_paths:
        try:
            img_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temporary image {img_path}: {str(e)}")


def main():
    try:
        # Create directories
        for path in [SRC_DIR, IMG_DIR, EMBEDDING_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        # Load model
        try:
            model, processor = get_model_and_processor()
        except Exception as e:
            raise ProcessingError(f"Failed to load model: {str(e)}")

        # Process PDFs
        src_file_paths = list(SRC_DIR.glob("*.pdf"))
        if not src_file_paths:
            logger.warning("No PDF files found in source directory")
            return

        for src_file_path in src_file_paths:
            output_path = EMBEDDING_DIR / f"{src_file_path.stem}_embeddings.npz"

            # Skip if output file already exists
            if output_path.exists():
                logger.info(f"Skipping {src_file_path} - embeddings already exist")
                continue

            try:
                logger.info(f"Processing {src_file_path}...")
                img_paths = convert_pdf_to_images(src_file_path, IMG_DIR)
                image_embeddings = process_images_to_vectors(
                    img_paths, model, processor
                )
                save_embeddings_to_npz(
                    src_file_path, img_paths, image_embeddings, EMBEDDING_DIR
                )
                # cleanup_images(img_paths)
            except ProcessingError as e:
                logger.error(f"Failed to process {src_file_path}: {str(e)}")
                continue


    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
