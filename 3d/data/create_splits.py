from argparse import ArgumentParser
import math
from pathlib import Path
import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Tuple, Optional


# DEFAULT_PATIENT_RATIOS = (0.38, 0.095, 0.525)  # train/val/test (paper-like patch proportions)
RANDOM_SEED = 42


def list_patients(root: Path) -> List[str]:
    """List patient IDs as immediate subdirectories under root."""
    patients = []
    for entry in sorted(root.iterdir()):
        
        pid = entry.name.split(".")[0]     # remove everything after first dot
        patients.append(pid)
    
    return patients

def simple_patient_ratio_split(patients: List[str],
                               ratios: Tuple[float, float, float]) -> Dict[str, str]:
    """
    Split by patient counts according to given ratios.
    """
    rnd = random.Random(RANDOM_SEED)
    shuffled = patients[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # Ensure all patients are assigned (avoid rounding gaps)
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
    
    root = Path(args.root)
    
    output = Path(args.output).resolve() if args.output else (root / "splits.csv")
    
    root = root / "raw"
    root = root.resolve()
    
    if not root.exists():
        raise FileNotFoundError(f"--root not found: {root}")

    patients = list_patients(root)
    num_patients = len(patients)
    if not patients:
        raise RuntimeError(f"No patient folders found under {root}")

    if args.ratio:
        # Targets must be provided
        if args.train_patches_num is None or args.val_patches_num is None or args.test_patches_num is None:
            raise ValueError("--ratio requires --train_patches_num,  and --test_patches_num")
        
        num_patches_train = args.train_patches_num
        num_patches_val = args.val_patches_num
        num_patches_test = args.test_patches_num
        
        total_num_patches = num_patches_train + num_patches_val + num_patches_test
        
        if total_num_patches <= 0:
            raise ValueError("Sum of patch targets must be > 0")
        
        ratio_train = num_patches_train / total_num_patches
        ratio_val = num_patches_val / total_num_patches
        
        n_train = math.floor(num_patients * ratio_train)
        n_val = math.floor(num_patients * ratio_val)
        n_test = num_patients - (n_train + n_val)
        
        print('Number of images per set:')
        print(f'→ Train set: {n_train}')
        print(f'→ Val set: {n_val}')
        print(f'→ Test set: {n_test}')
        
        assert n_train + n_val + n_test == num_patients
        
        rnd = random.Random(RANDOM_SEED)
        shuffled = patients[:]
        rnd.shuffle(shuffled)
        
        train_patients = set(shuffled[:n_train])
        val_patients   = set(shuffled[n_train:n_train + n_val])
        test_patients  = set(shuffled[n_train + n_val:])
        
        assignment = {}
        for pid in train_patients:
            assignment[pid] = "train"
        for pid in val_patients:
            assignment[pid] = "val"
        for pid in test_patients:
            assignment[pid] = "test"
    
    else:
        print("Please specify a ratio!")
        sys.exit()


    write_splits_csv(output, assignment)
    
    print(f"Finished saving in {output}")
    
    




if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--root', default=None, required=True, help='path of the root folder containing the data to convert')
    parser.add_argument('--ratio', default=False, required=True, action='store_true', help='If you want to compute the splits based on the number of patches for train/val/test')
    parser.add_argument('--train_patches_num', default=None, type=float, help='Number of training patches')
    parser.add_argument('--val_patches_num', default=None, type=float, help='Number of validation patches')
    parser.add_argument('--test_patches_num', default=None, type=float, help='Number of testing patches')
    parser.add_argument('--output', default=None, help='output path of the file that will contain the splits')
    
    
    args = parser.parse_args(['--root', "/data/scavone/syntheticMRI",
                              '--ratio',
                              '--train_patches_num', '4000',
                              '--val_patches_num', '1000',
                              '--test_patches_num', '5000',
                             ])
    
    main(args)