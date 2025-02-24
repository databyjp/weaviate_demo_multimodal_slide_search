# Using pdf2image (https://github.com/Belval/pdf2image)
from pathlib import Path
from pdf2image import convert_from_path
import math

src_path = Path("./data/src")
img_path = Path("./data/img")

src_path.mkdir(exist_ok=True)
img_path.mkdir(exist_ok=True)

src_file_paths = list(src_path.glob("*.pdf"))
for src_file_path in src_file_paths:
    tgt_file_path_prefix = src_file_path.stem

    print(f"Converting {src_file_path} to images...")
    images = convert_from_path(src_file_path)
    digits = int(math.log10(len(images))) + 1

    print(f"Saving {len(images)} images...")
    for i, img in enumerate(images, start=1):
        img.save(img_path / f"{tgt_file_path_prefix}_{i:0{digits}d}_of_{len(images)}.png", "PNG")
