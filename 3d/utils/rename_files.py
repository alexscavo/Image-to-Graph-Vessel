#!/usr/bin/env python3
"""
Create a new dataset folder "patches_new" from an existing patches folder, renaming files to:

  raw/: sample_XXXXXX_data.png
  seg/: sample_XXXXXX_seg.png
  vtp/: sample_XXXXXX_graph.vtp

Where XXXXXX is a 6-digit running index PER SPLIT (train/test/val...), starting at 000000.

Behavior:
- Reads input structure: <INPUT_ROOT>/<split>/{raw,seg,vtp}/...
- For each sample, retrieves its trio:
    raw: region_<r>_<id>_sat(.png or no ext)
    seg: region_<r>_<id>_gt(.png or no ext)
    vtp: region_<r>_<id>_gt_graph.vtp
- Matches by (region, id) so correspondence is preserved.
- Writes/copys into <OUTPUT_ROOT> with the exact same structure.
- Does NOT overwrite existing output files; it finds the next free index if needed.
- By default COPIES files (recommended). You can switch to MOVE.

No CLI args; edit config below.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil

# =========================
# USER CONFIG
# =========================

INPUT_ROOT = Path("/data/scavone/20cities/patches_mie")   # contains train/, test/, val/...
OUTPUT_ROOT = Path("/data/scavone/20cities/patches_mie")           # will be created next to where you run this
SPLITS = ["train", "test", "val"]

COPY_FILES = False   # True=copy, False=move (move will remove originals!)
DRY_RUN = False     # True=print actions only

# If a trio is incomplete, what to do?
REQUIRE_TRIO = True     # True=only export samples with raw+seg+vtp all present
                         # False=export whatever exists (still consistent naming for that sample)

# How to decide export order (deterministic)
SORT_BY = "region_then_id"  # "region_then_id" or "id_then_region"

# Safety cap for searching next free sample index
MAX_INDEX_SEARCH = 10_000_000

# =========================
# PATTERNS
# =========================

# Images can be .png or extensionless
RAW_RE = re.compile(r"^region_(?P<region>\d+)_(?P<pid>\d{6})_sat(?P<ext>\.png)?$", re.IGNORECASE)
SEG_RE = re.compile(r"^region_(?P<region>\d+)_(?P<pid>\d{6})_gt(?P<ext>\.png)?$", re.IGNORECASE)
VTP_RE = re.compile(r"^region_(?P<region>\d+)_(?P<pid>\d{6})_gt_graph\.vtp$", re.IGNORECASE)

# Detect already-exported filenames in output (if you re-run)
OUT_RAW_RE = re.compile(r"^sample_(\d{6})_data\.png$", re.IGNORECASE)
OUT_SEG_RE = re.compile(r"^sample_(\d{6})_seg\.png$", re.IGNORECASE)
OUT_VTP_RE = re.compile(r"^sample_(\d{6})_graph\.vtp$", re.IGNORECASE)


def ensure_dir(p: Path) -> None:
    if not DRY_RUN:
        p.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path) -> None:
    if DRY_RUN:
        print(f"WOULD WRITE: {src} -> {dst}")
        return
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)


def scan_split(split_in: Path) -> dict[tuple[int, int], dict[str, Path]]:
    """
    Returns mapping (region, pid) -> {'raw': Path?, 'seg': Path?, 'vtp': Path?}
    """
    samples: dict[tuple[int, int], dict[str, Path]] = {}

    raw_dir = split_in / "raw"
    seg_dir = split_in / "seg"
    vtp_dir = split_in / "vtp"

    if raw_dir.exists():
        for f in raw_dir.iterdir():
            if not f.is_file():
                continue
            m = RAW_RE.match(f.name)
            if not m:
                continue
            region = int(m.group("region"))
            pid = int(m.group("pid"))
            samples.setdefault((region, pid), {})["raw"] = f

    if seg_dir.exists():
        for f in seg_dir.iterdir():
            if not f.is_file():
                continue
            m = SEG_RE.match(f.name)
            if not m:
                continue
            region = int(m.group("region"))
            pid = int(m.group("pid"))
            samples.setdefault((region, pid), {})["seg"] = f

    if vtp_dir.exists():
        for f in vtp_dir.iterdir():
            if not f.is_file():
                continue
            m = VTP_RE.match(f.name)
            if not m:
                continue
            region = int(m.group("region"))
            pid = int(m.group("pid"))
            samples.setdefault((region, pid), {})["vtp"] = f

    return samples


def used_output_indices(split_out: Path) -> set[int]:
    """Find sample indices already present in the output split (supports re-runs)."""
    used: set[int] = set()
    for sub, rx in (("raw", OUT_RAW_RE), ("seg", OUT_SEG_RE), ("vtp", OUT_VTP_RE)):
        d = split_out / sub
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            m = rx.match(f.name)
            if m:
                used.add(int(m.group(1)))
    return used


def index_is_free(split_out: Path, idx: int) -> bool:
    s = f"{idx:06d}"
    return (
        not (split_out / "raw" / f"sample_{s}_data.png").exists()
        and not (split_out / "seg" / f"sample_{s}_seg.png").exists()
        and not (split_out / "vtp" / f"sample_{s}_graph.vtp").exists()
    )


def next_free_index(split_out: Path, start: int, used: set[int]) -> int:
    idx = start
    for _ in range(MAX_INDEX_SEARCH):
        if idx not in used and index_is_free(split_out, idx):
            return idx
        idx += 1
    raise RuntimeError(f"Could not find a free output index starting at {start} in {split_out}")


def sort_keys(keys: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if SORT_BY == "region_then_id":
        return sorted(keys, key=lambda t: (t[0], t[1]))
    if SORT_BY == "id_then_region":
        return sorted(keys, key=lambda t: (t[1], t[0]))
    raise ValueError(f"Unknown SORT_BY={SORT_BY}")


def export_split(split: str) -> None:
    split_in = INPUT_ROOT / split
    split_out = OUTPUT_ROOT / split

    raw_out = split_out / "raw"
    seg_out = split_out / "seg"
    vtp_out = split_out / "vtp"

    if not split_in.exists():
        raise FileNotFoundError(f"Missing split input folder: {split_in}")

    ensure_dir(raw_out)
    ensure_dir(seg_out)
    ensure_dir(vtp_out)

    samples = scan_split(split_in)
    keys = sort_keys(list(samples.keys()))

    used = used_output_indices(split_out)
    idx_cursor = 0

    exported_groups = 0
    exported_files = 0
    skipped_incomplete = 0

    for key in keys:
        parts = samples[key]
        has_raw = "raw" in parts
        has_seg = "seg" in parts
        has_vtp = "vtp" in parts

        if REQUIRE_TRIO and not (has_raw and has_seg and has_vtp):
            skipped_incomplete += 1
            continue

        # pick an output index (bump if already used)
        out_idx = next_free_index(split_out, idx_cursor, used)
        used.add(out_idx)
        idx_cursor = out_idx + 1

        s = f"{out_idx:06d}"

        # Always write in the expected output names (images forced to .png)
        if has_raw:
            dst = raw_out / f"sample_{s}_data.png"
            copy_or_move(parts["raw"], dst)
            exported_files += 1
        if has_seg:
            dst = seg_out / f"sample_{s}_seg.png"
            copy_or_move(parts["seg"], dst)
            exported_files += 1
        if has_vtp:
            dst = vtp_out / f"sample_{s}_graph.vtp"
            copy_or_move(parts["vtp"], dst)
            exported_files += 1

        exported_groups += 1

    print(
        f"[{split}] exported_groups={exported_groups} exported_files={exported_files} "
        f"skipped_incomplete={skipped_incomplete} (REQUIRE_TRIO={REQUIRE_TRIO})"
    )


def main() -> None:
    if not DRY_RUN:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        export_split(split)


if __name__ == "__main__":
    main()
