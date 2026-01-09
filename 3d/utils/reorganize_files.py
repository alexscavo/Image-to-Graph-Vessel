#!/usr/bin/env python3

from pathlib import Path
import shutil
from tqdm import tqdm

# =========================
# USER CONFIG
# =========================

ROOT_DIR = Path("/data/scavone/20cities/patches_mie")
SPLITS = ["train", "test", "val"]
MOVE_FILES = True   # True = move, False = copy
DRY_RUN = False     # True = preview only

# =========================
# LOGIC
# =========================

def is_segmentation(p: Path) -> bool:
    return p.suffix.lower() == ".png" and p.name.endswith("_gt.png")


def process_split(split_dir: Path):
    raw_dir = split_dir / "raw"
    seg_dir = split_dir / "seg"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw/: {raw_dir}")

    if not DRY_RUN:
        seg_dir.mkdir(exist_ok=True)

    moved = skipped = ignored = 0

    files = list(raw_dir.iterdir())
    for f in tqdm(files, desc=f"Processing {split_dir.name}", unit="file"):
        if not f.is_file():
            continue

        if not is_segmentation(f):
            ignored += 1
            continue

        dest = seg_dir / f.name
        if dest.exists():
            skipped += 1
            continue

        if DRY_RUN:
            moved += 1
            continue

        if MOVE_FILES:
            f.replace(dest)
        else:
            shutil.copy2(f, dest)

        moved += 1

    return moved, skipped, ignored


def main():
    for split in SPLITS:
        split_dir = ROOT_DIR / split
        moved, skipped, ignored = process_split(split_dir)

        action = "WOULD MOVE" if DRY_RUN else ("MOVED" if MOVE_FILES else "COPIED")
        print(f"[{split}] {action}: {moved} | skipped: {skipped} | ignored: {ignored}")


if __name__ == "__main__":
    main()
