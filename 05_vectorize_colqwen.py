# Example adapted from https://huggingface.co/docs/transformers/en/model_doc/colpali
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pathlib import Path
import time


model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    # device_map="cuda:0",  # or "mps" if on Apple Silicon
    device_map="mps",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,  # or "eager" if "mps"
).eval()
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

img_path = Path("./data/img")
img_paths = Path.glob(img_path, "*.png")
img_paths = sorted(img_paths)

print(img_paths)

images = [
    Image.open(img_path).convert("RGB") for img_path in img_paths
]

queries = [
    "In block-max WAND, how is the information organised into an index, and why does that speed up retrieval?",
    "What early termination algorithms are used in inverted index search, like early stopping, skipping within lists and omitting lists, or partial scoring?",
    "What are the main differences between block-max WAND and regular WAND algorithm? How does block-max WAND organise the information and speed up retrieval?",
]

# Process the inputs
start_time = time.time()
batch_images = processor.process_images(images).to(model.device)
print(f"Time to process images: {time.time() - start_time}")

start_time = time.time()
batch_queries = processor.process_queries(queries).to(model.device)
print(f"Time to process queries: {time.time() - start_time}")

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

# Score the queries against the images
scores = processor.score_multi_vector(query_embeddings, image_embeddings)

print(scores)
