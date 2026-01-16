from argparse import ArgumentParser
from pathlib import Path
import csv
import random
from typing import Dict, List, Tuple, Set

DEFAULT_RATIOS = (0.8, 0.2)  # train/val among TrainAndValidationSet only
RANDOM_SEED = 42


def _sample_id_from_file(p: Path, annotation_suffix: str) -> str | None:
    """
    Returns sample_id for:
      - *.jpeg                -> stem
      - *{annotation_suffix}  -> name without suffix
    Otherwise None.
    """
    name = p.name
    lower = name.lower()
    if lower.endswith(".jpeg") or lower.endswith(".jpg"):
        return p.stem

    if name.endswith(annotation_suffix):
        return name[: -len(annotation_suffix)]
    return None


def list_sample_ids(folder: Path, annotation_suffix: str) -> Set[str]:
    """Collect sample IDs from both jpeg + annotation files (union)."""
    ids: Set[str] = set()
    if not folder.exists():
        return ids
    for p in folder.iterdir():
        if not p.is_file():
            continue
        sid = _sample_id_from_file(p, annotation_suffix)
        if sid:
            ids.add(sid)
    return ids


def validate_pairs(folder: Path, sample_ids: Set[str], annotation_suffix: str, image_exts: tuple[str, ...]) -> List[str]:
    """Return list of sample_ids that are missing either an image or annotation."""
    missing = []
    for sid in sorted(sample_ids):
        ann = folder / f"{sid}{annotation_suffix}"
        has_ann = ann.exists()
        has_img = any((folder / f"{sid}{ext}").exists() for ext in image_exts)
        if not has_img or not has_ann:
            missing.append(sid)
    return missing


def split_train_val(sample_ids: List[str], ratios: Tuple[float, float]) -> Dict[str, str]:
    rnd = random.Random(RANDOM_SEED)
    shuffled = sample_ids[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_train = min(n_train, n)
    n_val = n - n_train

    assignment: Dict[str, str] = {}
    for sid in shuffled[:n_train]:
        assignment[sid] = "train"
    for sid in shuffled[n_train:]:
        assignment[sid] = "val"
    return assignment


def write_splits_csv(output_path: Path, assignment: Dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "split"])
        for sid, split in sorted(assignment.items()):
            writer.writerow([sid, split])


def main(args):
    root = Path(args.root).resolve()
    train_folder = root / "01-TrainAndValidationSet"
    test_folder = root / "test"

    if not train_folder.exists():
        raise FileNotFoundError(f"Train/Val folder not found: {train_folder}")
    if not test_folder.exists():
        raise FileNotFoundError(f"Test folder not found: {test_folder}")

    annotation_suffix = args.annotation_suffix

    test_ids = list_sample_ids(test_folder, annotation_suffix)
    trainval_ids = list_sample_ids(train_folder, annotation_suffix)

    if not test_ids:
        raise RuntimeError(f"No samples found under: {test_folder}")
    if not trainval_ids:
        raise RuntimeError(f"No samples found under: {train_folder}")

    # Optional validation (recommended)
    if args.validate_pairs:
        image_exts = tuple(args.image_exts.split(","))
        missing_test = validate_pairs(test_folder, test_ids, annotation_suffix, image_exts) 
        missing_trainval = validate_pairs(train_folder, trainval_ids, annotation_suffix, image_exts)    

        if missing_test or missing_trainval:
            msg = []
            if missing_test:
                msg.append('-'*50)
                msg.append(f"Missing test files: {len(missing_test)}")
                msg.append(f"Test missing pairs for: {missing_test[:10]}{'...' if len(missing_test) > 10 else ''}")
            if missing_trainval:
                msg.append('-'*50)
                msg.append(f"Missing train/val files: {len(missing_trainval)}")
                msg.append(f"Train/Val missing pairs for: {missing_trainval[:10]}{'...' if len(missing_trainval) > 10 else ''}")
            raise RuntimeError("Pair validation failed:\n" + "\n".join(msg))

    # Ensure test is forced test and excluded from train/val pool
    pool_ids = sorted(list(trainval_ids - test_ids))
    if not pool_ids:
        raise RuntimeError("After removing test_ids from train/val pool, nothing remains to split.")

    train_val_assignment = split_train_val(pool_ids, (args.train_ratio, 1.0 - args.train_ratio))

    assignment: Dict[str, str] = {}
    assignment.update({sid: "test" for sid in test_ids})
    assignment.update(train_val_assignment)

    output = Path(args.output).resolve() if args.output else (root / "splits.csv")
    write_splits_csv(output, assignment)
    print(f"Saved splits to: {output}")
    print(f"Counts: train={sum(v=='train' for v in assignment.values())}, "
          f"val={sum(v=='val' for v in assignment.values())}, "
          f"test={sum(v=='test' for v in assignment.values())}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--root", required=True, help="Root folder, e.g. plants_3d2cut/3D2cut_Single_Guyot")
    p.add_argument("--output", default=None, help="Where to write splits.csv (default: <root>/splits.csv)")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio within TrainAndValidationSet (rest is val)")
    p.add_argument("--annotation_suffix", default="_annotation.json", help="Annotation filename suffix")
    p.add_argument("--validate_pairs", action="store_true", help="Fail if any sample is missing jpeg or annotation")
    p.add_argument("--image_exts", default=".jpg,.jpeg", help="Comma-separated image extensions to accept")

    args = p.parse_args([
        '--root', '/data/scavone/plants_3d2cut/3D2cut_Single_Guyot',
        '--validate_pairs'
    ])
    main(args)
