from argparse import ArgumentParser
from pathlib import Path
import csv
import shutil

# Hardcoded supported image extensions (in priority order)
IMAGE_EXTS = (".jpg", ".jpeg")
ANNOTATION_SUFFIX = "_annotation.json"


def read_splits(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV or missing header: {csv_path}")

        # Expected: sample_id, split
        if "sample_id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(
                f"Expected columns sample_id,split in {csv_path}, got: {reader.fieldnames}"
            )

        for r in reader:
            sid = (r.get("sample_id") or "").strip()
            sp = (r.get("split") or "").strip()
            if sid:
                rows.append((sid, sp))

    if not rows:
        raise RuntimeError(f"No rows read from {csv_path}")
    return rows


def find_image_path(sample_id: str, src_dir: Path) -> Path:
    for ext in IMAGE_EXTS:
        candidate = src_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    # Helpful diagnostics: list what exists with that prefix
    matches = sorted([p.name for p in src_dir.glob(f"{sample_id}.*") if p.is_file()])
    raise FileNotFoundError(
        f"Missing image for {sample_id} in {src_dir} (tried {IMAGE_EXTS}). "
        f"Found with same prefix: {matches}"
    )


def copy_pair(sample_id: str, src_dir: Path, dst_dir: Path) -> None:
    ann_src = src_dir / f"{sample_id}{ANNOTATION_SUFFIX}"
    if not ann_src.exists():
        matches = sorted([p.name for p in src_dir.glob(f"{sample_id}*") if p.is_file()])
        raise FileNotFoundError(
            f"Missing annotation for {sample_id}: {ann_src}. Found with same prefix: {matches}"
        )

    img_src = find_image_path(sample_id, src_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, dst_dir / img_src.name)
    shutil.copy2(ann_src, dst_dir / ann_src.name)


def main(args):
    root = Path(args.root).resolve()
    trainval_dir = root / "01-TrainAndValidationSet"
    test_dir = root / "test"

    splits_csv = Path(args.splits_csv).resolve() if args.splits_csv else (root / "splits.csv")
    out_dir = Path(args.out).resolve() if args.out else (root / "data")

    if not splits_csv.exists():
        raise FileNotFoundError(f"splits.csv not found: {splits_csv}")
    if not trainval_dir.exists():
        raise FileNotFoundError(f"Train/Val dir not found: {trainval_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    rows = read_splits(splits_csv)

    for sample_id, split in rows:
        src = test_dir if split == "test" else trainval_dir
        copy_pair(sample_id, src, out_dir)

    print(f"Copied {len(rows)} samples (image + annotation) into: {out_dir}")
    print(f"Image extensions supported: {IMAGE_EXTS}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--root", required=True, help="Root folder, e.g. plants_3d2cut/3D2cut_Single_Guyot")
    p.add_argument("--splits_csv", default=None, help="Path to splits.csv (default: <root>/splits.csv)")
    p.add_argument("--out", default=None, help="Output folder (default: <root>/data)")
    args = p.parse_args([
        '--root', '/data/scavone/plants_3d2cut/3D2cut_Single_Guyot',
        '--splits_csv', '/data/scavone/plants_3d2cut/3D2cut_Single_Guyot/splits.csv',
        '--out', '/data/scavone/plants_3d2cut/data', 
    ])
    main(args)
