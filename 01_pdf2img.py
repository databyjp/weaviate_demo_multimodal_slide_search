# Using pdf2image (https://github.com/Belval/pdf2image)
from pathlib import Path
from pdf2image import convert_from_path
import math

src_path = Path("./data/src")
img_path = Path("./data/img")

src_path.mkdir(exist_ok=True)
img_path.mkdir(exist_ok=True)

images = convert_from_path(src_path / "bmw.pdf")
digits = int(math.log10(len(images))) + 1

for i, img in enumerate(images, start=1):
    img.save(img_path / f"bmw_{i:0{digits}d}_of_{len(images)}.png", "PNG")
