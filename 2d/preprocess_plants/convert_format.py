from pathlib import Path
from PIL import Image
from tqdm import tqdm
from tqdm import tqdm

root = Path("/data/scavone/plants_3d2cut/val/raw")  # dataset root
quality = 95  # 90–95 is usually safe

for img_path in tqdm(list(root.rglob("*"))):

    if img_path.suffix.lower() not in (".jpg", ".jpeg"):
        continue

    img = Image.open(img_path).convert("RGB")

    new_path = img_path.with_suffix(".jpg")

    # Skip if already .jpg with correct name
    if img_path.suffix.lower() == ".jpg":
        continue

    img.save(new_path, format="JPEG", quality=quality, subsampling=0)

    # remove old .jpeg file
    img_path.unlink()
