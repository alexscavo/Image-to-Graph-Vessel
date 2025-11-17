from argparse import ArgumentParser
import math
from pathlib import Path
import argparse
import csv
import os
import random
from typing import Dict, List, Tuple, Optional


DEFAULT_PATIENT_RATIOS = (0.38, 0.095, 0.525)  # train/val/test (paper-like patch proportions)
RANDOM_SEED = 42


def list_samples(root: Path, base_fn) -> List[str]:
    samples = []
    for entry in sorted(root.iterdir()):
        samples.append(base_fn(entry))  # <<-- apply _base() here
    return samples

def simple_patient_ratio_split(samples: List[str],
                               ratios: Tuple[float, float, float]) -> Dict[str, str]:
    """
    Split by patient counts according to given ratios.
    """
    rnd = random.Random(RANDOM_SEED)
    shuffled = samples[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # Ensure all samples are assigned (avoid rounding gaps)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = max(0, n - n_train - n_val)

    assignment: Dict[str, str] = {}
    for pid in shuffled[:n_train]:
        assignment[pid] = 'train'
    for pid in shuffled[n_train:n_train + n_val]:
        assignment[pid] = 'val'
    for pid in shuffled[n_train + n_val:]:
        assignment[pid] = 'test'
    return assignment

def write_splits_csv(output_path: Path, assignment: Dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'split'])
        for pid, split in sorted(assignment.items()):
            writer.writerow([pid, split])



def main(args):
    
    def _base(p: Path) -> str:
        s = p.stem
        if s.endswith("graph"):
            splitted = s.split("_")
            s = f"{splitted[0]}_{splitted[1]}"
        if s.startswith("G_"):
            s = s[2:]
        return s
    
    root = Path(args.root)
    
    output = Path(args.output).resolve() if args.output else (root / "splits.csv")
    
    root = root / "vtp"
    root = root.resolve()
    
    if not root.exists():
        raise FileNotFoundError(f"--root not found: {root}")

    samples = list_samples(root, _base)
    num_samples = len(samples)
    if not samples:
        raise RuntimeError(f"No patient folders found under {root}")

    if args.ratio:
        # Targets must be provided
        if args.train_patches_num is None or args.val_patches_num is None or args.test_patches_num is None:
            raise ValueError("--ratio requires --train_patches_num and --test_patches_num")
        
        num_patches_train = args.train_patches_num
        num_patches_val = args.val_patches_num
        num_patches_test = args.test_patches_num
        
        total_num_patches = num_patches_train + num_patches_val + num_patches_test
        
        if total_num_patches <= 0:
            raise ValueError("Sum of patch targets must be > 0")
        
        ratio_train = num_patches_train / total_num_patches
        ratio_val = num_patches_val / total_num_patches
        
        n_train = math.floor(num_samples * ratio_train)
        n_val = math.floor(num_samples * ratio_val)
        n_test = num_samples - (n_train + n_val)
        
        assert n_train + n_val + n_test == num_samples
        
        rnd = random.Random(RANDOM_SEED)
        shuffled = samples[:]
        rnd.shuffle(shuffled)
        
        train_samples = set(shuffled[:n_train])
        val_samples   = set(shuffled[n_train:n_train + n_val])
        test_samples  = set(shuffled[n_train + n_val:])
        
        assignment = {}
        for pid in train_samples:
            assignment[pid] = "train"
        for pid in val_samples:
            assignment[pid] = "val"
        for pid in test_samples:
            assignment[pid] = "test"
        
    else:
        # Patient-count split using default paper-like proportions (train/val/test).
        assignment = simple_patient_ratio_split(samples, DEFAULT_PATIENT_RATIOS)

    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    total = len(assignment)
    
    print("\nSplit statistics:")
    print(f"  Train: {counts['train']} samples ({counts['train'] / total:.2%})")
    print(f"  Val:   {counts['val']} samples ({counts['val'] / total:.2%})")
    print(f"  Test:  {counts['test']} samples ({counts['test'] / total:.2%})\n")

    write_splits_csv(output, assignment)
    
    print(f"Finished saving in {output}")
    
    




if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--root', default=None, required=True, help='path of the root folder containing the data to convert')
    parser.add_argument('--ratio', default=False, action='store_true', help='If you want to compute the splits based on the number of patches for train/val/test')
    parser.add_argument('--train_patches_num', default=None, type=float, help='Number of training patches')
    parser.add_argument('--val_patches_num', default=None, type=float, help='Number of validation patches')
    parser.add_argument('--test_patches_num', default=None, type=float, help='Number of testing patches')
    parser.add_argument('--output', default=None, help='output path of the file that will contain the splits')
    
    
    args = parser.parse_args(['--root', "/data/scavone/20cities",
                              '--ratio',
                              '--train_patches_num', '99200',
                              '--val_patches_num', '24800',
                              '--test_patches_num', '25000',
                             ])
    
    main(args)