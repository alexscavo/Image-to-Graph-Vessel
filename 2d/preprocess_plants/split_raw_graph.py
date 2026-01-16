from argparse import ArgumentParser
from pathlib import Path
import shutil

# Hardcoded supported image extensions
IMAGE_EXTS = (".jpg", ".jpeg")
ANNOTATION_SUFFIX = "_annotation.json"


def is_image_file(p: Path) -> bool:
    lower = p.name.lower()
    return any(lower.endswith(ext) for ext in IMAGE_EXTS)


def is_annotation_file(p: Path) -> bool:
    return p.name.endswith(ANNOTATION_SUFFIX)


def main(args):
    in_dir = Path(args.input).resolve()
    out_dir = Path(args.out).resolve() if args.out else in_dir

    raw_dir = out_dir / "raw"
    graphs_dir = out_dir / "graphs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    moved = 0
    copied = 0

    for p in in_dir.iterdir():
        if not p.is_file():
            continue

        if is_image_file(p):
            dest = raw_dir / p.name
        elif is_annotation_file(p):
            dest = graphs_dir / p.name
        else:
            continue

        if args.copy:
            shutil.copy2(p, dest)
            copied += 1
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            p.replace(dest)
            moved += 1

    action = "Copied" if args.copy else "Moved"
    count = copied if args.copy else moved
    print(f"{action} {count} files into:")
    print(f"  {raw_dir}")
    print(f"  {graphs_dir}")
    print(f"Image extensions supported: {IMAGE_EXTS}")
    print(f"Annotation suffix: {ANNOTATION_SUFFIX}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--input", required=True, help="Folder containing mixed images and *_annotation.json files")
    p.add_argument("--out", default=None, help="Output base folder (default: same as --input)")
    p.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = p.parse_args([
        '--input', '/data/scavone/plants_3d2cut/data',
        '--out', '/data/scavone/plants_3d2cut/new_data', 
    ])
    main(args)
