import time
import math
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
import logging
from typing import Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SRC_DIR = Path("./data/src")
IMG_DIR = Path("./data/img")
OUT_DIR = Path("./outputs")
EMBEDDING_DIR = Path("./data/embeddings")

WEAVIATE_COLLECTION_NAME = "pdf_embeddings"

# MODEL_NAME = "vidore/colpali-v1.3"

# def get_model_and_processor():
#     model = ColPali.from_pretrained(
#         MODEL_NAME,
#         torch_dtype=torch.bfloat16,
#         device_map="mps",  # If CUDA is available, use "cuda"
#     ).eval()
#     processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
#     return model, processor


class ProcessingError(Exception):
    pass


def convert_pdf_to_images(src_file_path: Path, img_path: Path) -> list[Path]:
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


# def process_images_to_vectors(
#     img_paths: list[Path],
#     model: ColPali,
#     processor: ColPaliProcessor,
#     batch_size: int = 8,
# ) -> torch.Tensor:
#     image_embeddings = []

#     for i in range(0, len(img_paths), batch_size):
#         batch_paths = img_paths[i : i + batch_size]
#         try:
#             # Load and process images in batches
#             batch_images = []
#             for img_path in batch_paths:
#                 with Image.open(img_path) as img:
#                     batch_images.append(img.convert("RGB"))

#             processed_batch = processor.process_images(batch_images).to(model.device)
#             logger.info(
#                 f"Processing images {i+1} to {min(i+batch_size, len(img_paths))}..."
#             )

#             with torch.no_grad():
#                 start_time = time.time()
#                 batch_embeddings = model(**processed_batch)
#                 image_embeddings.append(batch_embeddings)
#                 logger.info(f"Batch processing time: {time.time() - start_time:.2f}s")

#         except Exception as e:
#             raise ProcessingError(f"Failed to process images {batch_paths}: {str(e)}")

#     return torch.cat(image_embeddings)


# def text_to_colpali(
#     texts: list[str],
#     model: Optional[ColPali] = None,
#     processor: Optional[ColPaliProcessor] = None,
# ) -> torch.Tensor:
#     if not model or not processor:
#         model, processor = get_model_and_processor()

#     try:
#         # Process the text using the processor
#         processed_text = processor.process_queries(texts).to(model.device)

#         # Get embedding
#         with torch.no_grad():
#             embedding = model(**processed_text)

#         return embedding

#     except Exception as e:
#         raise ProcessingError(f"Failed to process text: {str(e)}")


def render_svg_file(svg_file_path, width=None, height=None):
    import base64

    with open(svg_file_path, "r") as f:
        svg = f.read()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add width and height attributes to the img tag
    style_attributes = ""
    if width:
        style_attributes += f'width="{width}" '
    if height:
        style_attributes += f'height="{height}" '

    html = f'<img src="data:image/svg+xml;base64,{b64}" {style_attributes}/>'
    return html
