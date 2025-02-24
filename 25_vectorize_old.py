# Example adapted from https://huggingface.co/docs/transformers/en/model_doc/colpali
import time
from pathlib import Path
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
import numpy as np

model_name = "vidore/colpali-v1.3"

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps",
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

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
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    start_time = time.time()
    image_embeddings = model(**batch_images)
    print(f"Time to process images: {time.time() - start_time}")
    start_time = time.time()
    query_embeddings = model(**batch_queries)
    print(f"Time to process queries: {time.time() - start_time}")

# Score the queries against the images
scores = processor.score_multi_vector(query_embeddings, image_embeddings)

print(scores)
