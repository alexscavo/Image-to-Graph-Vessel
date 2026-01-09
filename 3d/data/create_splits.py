from argparse import ArgumentParser
import math
from pathlib import Path
import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Tuple, Optional


RANDOM_SEED = 42


def list_patients(root: Path) -> List[str]:
    """List patient IDs as immediate subdirectories under root."""
    patients = []
    for entry in sorted(root.iterdir()):
        pid = entry.name.split(".")[0]     # remove everything after first dot
        patients.append(pid)
    return patients


def write_splits_csv(output_path: Path, assignment: Dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'split'])
        for pid, split in sorted(assignment.items()):
            writer.writerow([pid, split])


def compute_volume_split(
    num_volumes: int,
    target_train_patches: int,
    target_val_patches: int,
    target_test_patches: int,
    max_patches_per_volume: int
) -> Tuple[int, int, int]:
    """
    Compute how many volumes to allocate to each split given:
    - Total number of volumes
    - Target patches per split
    - Maximum patches extractable per volume
    
    Returns: (n_train_volumes, n_val_volumes, n_test_volumes)
    """
    
    total_target_patches = target_train_patches + target_val_patches + target_test_patches
    
    # Calculate minimum volumes needed per split
    min_train_vols = math.ceil(target_train_patches / max_patches_per_volume)
    min_val_vols = math.ceil(target_val_patches / max_patches_per_volume)
    min_test_vols = math.ceil(target_test_patches / max_patches_per_volume)
    
    min_total_vols = min_train_vols + min_val_vols + min_test_vols
    
    print("\n=== VOLUME SPLIT COMPUTATION ===")
    print(f"Total volumes available: {num_volumes}")
    print(f"Max patches per volume: {max_patches_per_volume}")
    print(f"\nTarget patches:")
    print(f"  Train: {target_train_patches}")
    print(f"  Val:   {target_val_patches}")
    print(f"  Test:  {target_test_patches}")
    print(f"  Total: {total_target_patches}")
    
    print(f"\nMinimum volumes needed:")
    print(f"  Train: {min_train_vols} volumes × {max_patches_per_volume} = {min_train_vols * max_patches_per_volume} patches")
    print(f"  Val:   {min_val_vols} volumes × {max_patches_per_volume} = {min_val_vols * max_patches_per_volume} patches")
    print(f"  Test:  {min_test_vols} volumes × {max_patches_per_volume} = {min_test_vols * max_patches_per_volume} patches")
    print(f"  Total: {min_total_vols} volumes")
    
    # Check feasibility
    if min_total_vols > num_volumes:
        print(f"\n⚠️  WARNING: Not enough volumes!")
        print(f"   Need {min_total_vols} volumes but only have {num_volumes}")
        print(f"   Some splits will not reach their target.")
        
        # Allocate proportionally
        total_min = min_train_vols + min_val_vols + min_test_vols
        n_train = max(1, int(round(num_volumes * min_train_vols / total_min)))
        n_val = max(1, int(round(num_volumes * min_val_vols / total_min)))
        n_test = max(1, num_volumes - n_train - n_val)
        
    else:
        # We have enough volumes - distribute optimally
        spare_volumes = num_volumes - min_total_vols
        
        print(f"\n✓ Enough volumes! {spare_volumes} spare volumes to distribute.")
        
        # Distribute spare volumes proportionally to patch targets
        # This ensures each split gets closer to its ideal patches-per-volume ratio
        if spare_volumes > 0:
            # Calculate ideal patches per volume for each split
            ideal_ppv_train = target_train_patches / (min_train_vols + 0.001)
            ideal_ppv_val = target_val_patches / (min_val_vols + 0.001)
            ideal_ppv_test = target_test_patches / (min_test_vols + 0.001)
            
            # Splits that need more patches per volume get priority
            needs = {
                'train': max(0, ideal_ppv_train - max_patches_per_volume),
                'val': max(0, ideal_ppv_val - max_patches_per_volume),
                'test': max(0, ideal_ppv_test - max_patches_per_volume)
            }
            
            total_need = sum(needs.values())
            
            if total_need > 0:
                # Distribute proportionally to need
                extra_train = int(round(spare_volumes * needs['train'] / total_need))
                extra_val = int(round(spare_volumes * needs['val'] / total_need))
                extra_test = spare_volumes - extra_train - extra_val
            else:
                # No specific need, distribute by target patch proportions
                total_patches = target_train_patches + target_val_patches + target_test_patches
                extra_train = int(round(spare_volumes * target_train_patches / total_patches))
                extra_val = int(round(spare_volumes * target_val_patches / total_patches))
                extra_test = spare_volumes - extra_train - extra_val
            
            n_train = min_train_vols + extra_train
            n_val = min_val_vols + extra_val
            n_test = min_test_vols + extra_test
        else:
            n_train = min_train_vols
            n_val = min_val_vols
            n_test = min_test_vols
    
    # Final adjustment to ensure we use exactly num_volumes
    total_allocated = n_train + n_val + n_test
    if total_allocated != num_volumes:
        diff = num_volumes - total_allocated
        # Add difference to largest split
        if n_test >= n_train and n_test >= n_val:
            n_test += diff
        elif n_train >= n_val:
            n_train += diff
        else:
            n_val += diff
    
    # Sanity check
    assert n_train + n_val + n_test == num_volumes, f"Volume allocation error: {n_train}+{n_val}+{n_test} != {num_volumes}"
    assert n_train > 0 and n_val > 0 and n_test > 0, "Each split must have at least 1 volume"
    
    print(f"\n=== FINAL ALLOCATION ===")
    print(f"Train: {n_train} volumes → {n_train * max_patches_per_volume} max patches (target: {target_train_patches})")
    print(f"Val:   {n_val} volumes → {n_val * max_patches_per_volume} max patches (target: {target_val_patches})")
    print(f"Test:  {n_test} volumes → {n_test * max_patches_per_volume} max patches (target: {target_test_patches})")
    print(f"Total: {n_train + n_val + n_test} volumes")
    
    # Calculate actual patches per volume needed
    actual_ppv_train = target_train_patches / n_train if n_train > 0 else 0
    actual_ppv_val = target_val_patches / n_val if n_val > 0 else 0
    actual_ppv_test = target_test_patches / n_test if n_test > 0 else 0
    
    print(f"\n=== PATCHES PER VOLUME NEEDED ===")
    print(f"Train: {actual_ppv_train:.1f} patches/volume (limit: {max_patches_per_volume})")
    print(f"Val:   {actual_ppv_val:.1f} patches/volume (limit: {max_patches_per_volume})")
    print(f"Test:  {actual_ppv_test:.1f} patches/volume (limit: {max_patches_per_volume})")
    
    # Check if any split will fall short
    warnings = []
    if actual_ppv_train > max_patches_per_volume:
        shortage = target_train_patches - (n_train * max_patches_per_volume)
        warnings.append(f"  Train: SHORT by ~{shortage} patches")
    if actual_ppv_val > max_patches_per_volume:
        shortage = target_val_patches - (n_val * max_patches_per_volume)
        warnings.append(f"  Val: SHORT by ~{shortage} patches")
    if actual_ppv_test > max_patches_per_volume:
        shortage = target_test_patches - (n_test * max_patches_per_volume)
        warnings.append(f"  Test: SHORT by ~{shortage} patches")
    
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for w in warnings:
            print(w)
        print(f"\nTo fix: increase --max_patches_per_volume or reduce target patches")
    else:
        print(f"\n✓ All splits can reach their targets!")
    
    print("=" * 50)
    
    return n_train, n_val, n_test


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

    print(f"\nFound {num_patients} volumes in {root}")

    if args.ratio:
        # Targets must be provided
        if args.train_patches_num is None or args.val_patches_num is None or args.test_patches_num is None:
            raise ValueError("--ratio requires --train_patches_num, --val_patches_num, and --test_patches_num")
        
        if args.max_patches_per_volume is None:
            raise ValueError("--ratio requires --max_patches_per_volume")
        
        num_patches_train = args.train_patches_num
        num_patches_val = args.val_patches_num
        num_patches_test = args.test_patches_num
        max_patches_per_volume = args.max_patches_per_volume
        
        # Compute optimal volume split
        n_train, n_val, n_test = compute_volume_split(
            num_volumes=num_patients,
            target_train_patches=num_patches_train,
            target_val_patches=num_patches_val,
            target_test_patches=num_patches_test,
            max_patches_per_volume=max_patches_per_volume
        )
        
        # Shuffle and assign
        rnd = random.Random(RANDOM_SEED)
        shuffled = patients[:]
        rnd.shuffle(shuffled)
        
        train_patients = set(shuffled[:n_train])
        val_patients = set(shuffled[n_train:n_train + n_val])
        test_patients = set(shuffled[n_train + n_val:])
        
        assignment = {}
        for pid in train_patients:
            assignment[pid] = "train"
        for pid in val_patients:
            assignment[pid] = "val"
        for pid in test_patients:
            assignment[pid] = "test"
    
    else:
        print("ERROR: Please specify --ratio!")
        sys.exit(1)

    write_splits_csv(output, assignment)
    
    print(f"\n✓ Splits saved to: {output}")
    print(f"\nSummary:")
    print(f"  Train: {len(train_patients)} volumes")
    print(f"  Val:   {len(val_patients)} volumes")
    print(f"  Test:  {len(test_patients)} volumes")
    print(f"  Total: {len(assignment)} volumes")


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Create train/val/test splits based on target patch counts")
    parser.add_argument('--root', required=True, 
                        help='Path of the root folder containing raw/ subfolder with volumes')
    parser.add_argument('--ratio', action='store_true', required=True,
                        help='Compute splits based on target patch numbers')
    parser.add_argument('--train_patches_num', type=int, default=None,
                        help='Target number of training patches')
    parser.add_argument('--val_patches_num', type=int, default=None,
                        help='Target number of validation patches')
    parser.add_argument('--test_patches_num', type=int, default=None,
                        help='Target number of testing patches')
    parser.add_argument('--max_patches_per_volume', type=int, default=None,
                        help='Maximum patches that can be extracted from one volume (REQUIRED)')
    parser.add_argument('--output', default=None,
                        help='Output path for splits.csv (default: <root>/splits.csv)')
    
    # Example usage for your case:
    # 136 volumes, targeting 4k/1k/5k patches with max 150 patches/volume
    args = parser.parse_args([
        '--root', '/data/scavone/syntheticMRI',
        '--ratio',
        '--train_patches_num', '4000',
        '--val_patches_num', '1000',
        '--test_patches_num', '5000',
        '--max_patches_per_volume', '74',  # NEW: Required parameter
    ])
    
    main(args)