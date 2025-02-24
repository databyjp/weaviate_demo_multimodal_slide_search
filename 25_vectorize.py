# Example adapted from https://huggingface.co/docs/transformers/en/model_doc/colpali
import time
from pathlib import Path
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
import json

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

img_paths = img_paths[:8] # Limit to 8 images for demonstration purposes

images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

# Process inputs in batches of 8 images at a time
batch_size = 8
image_embeddings = []

for i in range(0, len(images), batch_size):
    batch_images = processor.process_images(images[i : i + batch_size]).to(model.device)
    print(f"Processing images {i} to {i + batch_size}...")
    with torch.no_grad():
        start_time = time.time()
        image_embeddings.append(model(**batch_images))
        print(f"Time to process images: {time.time() - start_time}")

    image_embeddings = torch.cat(image_embeddings)

# Convert the image embeddings to a dictionary with lists
image_embeddings_dict = {
    img_path.stem: img_emb.cpu().to(torch.float32).numpy().tolist() for img_path, img_emb in zip(img_paths, image_embeddings)
}

# Save the image embeddings to a JSON file
with open("image_embeddings.json", "w") as f:
    json.dump(image_embeddings_dict, f)
