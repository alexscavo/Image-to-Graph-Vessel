#!/usr/bin/env python3
import os, argparse, pickle, csv, re, random, json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------
# Graph I/O
# ---------------------------

def save_patch_graph_vtp(nodes_p, edges_p, S, out_path):
    """
    Save the patch graph as a .vtp polyline file (similar to generate_sat_data.py).
    nodes_p : (N,2) float32 local patch coordinates
    edges_p : (E,2) int node indices
    S       : patch size (used to normalize coords to [0,1])
    """
    import numpy as np
    try:
        import pyvista as pv
    except ImportError as e:
        raise RuntimeError(
            "pyvista is required when using --graph_format vtp or both. "
            "Install it with `pip install pyvista`."
        ) from e

    nodes_p = np.asarray(nodes_p, dtype=np.float32)
    E = np.asarray(edges_p, dtype=np.int64)

    if nodes_p.size == 0:
        # empty polydata
        mesh = pv.PolyData(np.zeros((0, 3), dtype=np.float32))
        mesh.save(str(out_path))
        return

    # 2D -> 3D points, normalized by patch size like in generate_sat_data.py
    pts = np.zeros((nodes_p.shape[0], 3), dtype=np.float32)
    pts[:, 0] = nodes_p[:, 0] / float(S)
    pts[:, 1] = nodes_p[:, 1] / float(S)
    # z = 0

    if E.size:
        E = E.reshape(-1, 2)
        # prepend "2" to each edge to indicate a 2-point line
        lines = np.concatenate(
            (2 * np.ones((E.shape[0], 1), dtype=np.int32), E.astype(np.int32)),
            axis=1
        )
    else:
        lines = np.zeros((0, 3), dtype=np.int32)

    mesh = pv.PolyData(pts)
    if lines.size:
        mesh.lines = lines.flatten()
    mesh.save(str(out_path))

def save_graph(
    nodes,
    edges,
    out_path: Path,
    fmt: str,
    S: int = 128,
):
    if fmt == "pickle":
        with open(out_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump({"nodes": nodes, "edges": edges}, f)

    elif fmt == "json":
        data = {
            "nodes": [{"x": float(x), "y": float(y)} for x, y in nodes],
            "edges": [[int(i), int(j)] for i, j in edges],
        }
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)

    elif fmt == "vtp":
        save_patch_graph_vtp(nodes, edges, S, out_path.with_suffix(".vtp"))

    else:
        raise ValueError(f"Unknown graph format: {fmt}")


def load_graph_from_json(file_path: Path):
    """
    Returns:
      nodes_xy: (N,2) float32, pixel coords (x,y)
      edges:    (E,2) int64, indices into nodes_xy (not FeatureID)
      feat_ids: (N,) int64, original FeatureID per row in nodes_xy
      types:    list[str], FeatureType per node (optional but often useful)
    """
    
    # open the annotation file
    with open(file_path, 'r') as f:
        ann = json.load(f)
        
    # get the plant image annotations and features
    plant_img = ann["VineImage"][0]
    features = plant_img["VineFeature"][0]
    
    features_ids = [int(feat["FeatureID"]) for feat in features]
    
    # nodes in ID space
    nodes_by_id = {}
    for feat in features:
        fid = int(feat["FeatureID"])
        nodes_by_id[fid] = tuple(feat["FeatureCoordinates"])
        
    # edges in ID space
    edges_by_id = []
    for feat in features:
        pid = feat.get("ParentID", None)    # ParentID
        if pid is None:
            continue
        
        pid = int(pid)
        cid = int(feat["FeatureID"])  # child ID
    
        edges_by_id.append((pid, cid))

    # convert node ids into indeces to have dense indeces between 0 - N-1
    feat_ids = sorted(nodes_by_id.keys())
    id_to_idx = {fid: idx for idx, fid in enumerate(feat_ids)}
    
    nodes_xy = np.array([nodes_by_id[fid] for fid in feat_ids], dtype=np.float32)
    edges_ix = np.array([[id_to_idx[pid], id_to_idx[cid]] for pid, cid in edges_by_id], dtype=np.int64)
    
    return nodes_xy, edges_ix
# ---------------------------
# Graph utils (crop & rescale)
# ---------------------------

def rescale_graph(nodes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if nodes.size == 0:
        return nodes.astype(np.float32).reshape(0, 2)
    out = nodes.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out

import numpy as np

def clip_segment_to_rect(p0, p1, x0, y0, x1, y1):
    """
    Clip segment p0->p1 to axis-aligned rectangle.
    Returns (c0, c1) or None if no intersection.
    """
    x_min, x_max = x0, x1
    y_min, y_max = y0, y1

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    t0, t1 = 0.0, 1.0

    def clip(p, q, t0, t1):
        if p == 0:
            if q < 0:
                return None
            return t0, t1
        r = q / p
        if p < 0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
        return t0, t1

    res = clip(-dx, p0[0] - x_min, t0, t1)
    if res is None: return None
    t0, t1 = res

    res = clip(dx, x_max - p0[0], t0, t1)
    if res is None: return None
    t0, t1 = res

    res = clip(-dy, p0[1] - y_min, t0, t1)
    if res is None: return None
    t0, t1 = res

    res = clip(dy, y_max - p0[1], t0, t1)
    if res is None: return None
    t0, t1 = res

    c0 = p0 + t0 * np.array([dx, dy])
    c1 = p0 + t1 * np.array([dx, dy])
    return c0, c1


def crop_graph_to_patch(nodes_xy, edges_idx, x0, y0, S):
    x1, y1 = x0 + S, y0 + S

    # keep original inside nodes
    inside = (
        (nodes_xy[:, 0] >= x0) & (nodes_xy[:, 0] < x1) &
        (nodes_xy[:, 1] >= y0) & (nodes_xy[:, 1] < y1)
    )

    keep_nodes = np.where(inside)[0]
    nodes_local = nodes_xy[keep_nodes].copy()
    nodes_local[:, 0] -= x0
    nodes_local[:, 1] -= y0

    old2new = -np.ones(len(nodes_xy), dtype=np.int64)
    old2new[keep_nodes] = np.arange(len(keep_nodes), dtype=np.int64)

    # map endpoints to new indices (-1 if outside)
    e0 = old2new[edges_idx[:, 0]]
    e1 = old2new[edges_idx[:, 1]]

    # start with edges fully inside
    mask_inside = (e0 >= 0) & (e1 >= 0)
    edges_local = (
        np.stack([e0[mask_inside], e1[mask_inside]], axis=1)
        if np.any(mask_inside)
        else np.zeros((0, 2), dtype=np.int64)
    )

    # boundary/intersection nodes
    inter_pts = []
    inter_edges = []
    pt2idx = {}

    def add_pt(pt):
        # dedupe within patch
        key = (round(pt[0] - x0, 3), round(pt[1] - y0, 3))
        if key in pt2idx:
            return pt2idx[key]
        idx = len(nodes_local) + len(inter_pts)
        pt2idx[key] = idx
        inter_pts.append([pt[0] - x0, pt[1] - y0])
        return idx

    # CASE 2: exactly one endpoint inside (one in, one out)
    partial = (e0 >= 0) ^ (e1 >= 0)
    for k in np.where(partial)[0]:
        u, v = edges_idx[k]
        clipped = clip_segment_to_rect(nodes_xy[u], nodes_xy[v], x0, y0, x1, y1)
        if clipped is None:
            continue
        c0, c1 = clipped

        if e0[k] >= 0:
            inside_idx = e0[k]
            boundary_pt = c1
        else:
            inside_idx = e1[k]
            boundary_pt = c0

        boundary_idx = add_pt(boundary_pt)
        if inside_idx != boundary_idx:
            inter_edges.append([inside_idx, boundary_idx])

    # CASE 1: both endpoints outside but edge crosses patch
    both_outside = (e0 < 0) & (e1 < 0)
    for k in np.where(both_outside)[0]:
        u, v = edges_idx[k]
        clipped = clip_segment_to_rect(nodes_xy[u], nodes_xy[v], x0, y0, x1, y1)
        if clipped is None:
            continue
        c0, c1 = clipped
        i0 = add_pt(c0)
        i1 = add_pt(c1)
        if i0 != i1:
            inter_edges.append([i0, i1])

    # merge appended points/edges
    if inter_pts:
        nodes_local = np.vstack([nodes_local, np.asarray(inter_pts, dtype=np.float32)])
        if inter_edges:
            inter_edges = np.asarray(inter_edges, dtype=np.int64)
            edges_local = (
                np.vstack([edges_local, inter_edges])
                if edges_local.size
                else inter_edges
            )

    return nodes_local.astype(np.float32), edges_local.astype(np.int64)

# ---------------------------
# Resize helpers
# ---------------------------

def _pil_resample():
    try:
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    except AttributeError:
        return Image.BILINEAR, Image.NEAREST

def resize_pair(image: np.ndarray, seg: np.ndarray, out_w: int, out_h: int):
    bilinear, nearest = _pil_resample()
    img = Image.fromarray(image)
    sg  = Image.fromarray(seg)
    img_r = np.array(img.resize((out_w, out_h), resample=bilinear))
    seg_r = np.array(sg.resize((out_w, out_h), resample=nearest))
    return img_r, seg_r

# ---------------------------
# Split CSV
# ---------------------------

def _normalize_split(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("train", "tr"):
        return "train"
    if s in ("val", "valid", "validation"):
        return "val"
    if s in ("test", "te"):
        return "test"
    # default/fallback
    return "train"

def read_split_csv(csv_path: Path):
    split_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("split CSV missing headers")
        id_col = "id" if "id" in reader.fieldnames else ("sample_id" if "sample_id" in reader.fieldnames else None)
        if id_col is None or "split" not in reader.fieldnames:
            raise ValueError("split CSV must have headers: id(or sample_id), split")
        for row in reader:
            rid = Path(row[id_col]).stem
            split_map[rid] = _normalize_split(row["split"])
    return split_map

# ---------------------------
# Main
# ---------------------------

def get_split_limit(args, split: str):
    if split == "train":
        return args.num_train
    if split == "val":
        return args.num_val
    if split == "test":
        return args.num_test
    return None

def main(args):
    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    raw_dir = root / "raw"
    seg_dir = root / "seg"
    vtp_dir = root /"graphs"

    split_map = read_split_csv(Path(args.split))

    # find regions via *.jpg in raw/
    img_files_all = sorted(raw_dir.glob("*.jpg"))
    if not img_files_all:
        raise RuntimeError(f"No .jpg images found in {raw_dir}")

    # --- buckets from split csv (skip if not listed) ---
    buckets = {"train": [], "val": [], "test": []}
    for img_path in img_files_all:
        base = img_path.stem  # you must implement _base() (see below)
        sp = split_map.get(base, None)
        if sp is None:
            # if not in CSV, skip cleanly
            print(f"[skip] {img_path.name} not found in splits.csv")
            continue
        buckets[sp].append(img_path)

    # --- patch params ---
    ps = int(args.patch_size)
    overlap = float(args.overlap)
    stride = int(round(ps * (1.0 - overlap)))
    stride = max(1, min(ps, stride))

    # --- create output dirs per split ---
    out_roots = {}
    for sp in ("train", "val", "test"):
        root_sp = Path(args.out_root) / sp
        out_img = root_sp / "raw"
        out_seg = root_sp / "seg"
        out_graph = root_sp / "graphs"
        for d in (out_img, out_seg, out_graph):
            d.mkdir(parents=True, exist_ok=True)
        out_roots[sp] = (out_img, out_seg, out_graph)


    # resizing
    resize_w = args.resize_width
    resize_h = args.resize_height
    if (resize_w is None) != (resize_h is None):
        raise ValueError("Use both --resize_width and --resize_height together (or neither).")

    written_total = 0
    for sp in ("train", "val", "test"):
        files = buckets[sp]
        if not files:
            continue

        split_limit = get_split_limit(args, sp)  # None => unlimited
        written_sp = 0
        
        out_img, out_seg, out_graph = out_roots[sp]

        for sat_path in tqdm(files, desc=f"Regions ({sp})"):
            if split_limit is not None and written_sp >= split_limit:
                break

            stem = sat_path.stem  # "Set00_IMG_3283"

            seg_path = seg_dir / f"{stem}_seg.png"
            if not seg_path.exists():
                print(f"[warn] missing seg: {seg_path}")
                continue

            graph_path = vtp_dir / f"{stem}_annotation.json"
            if not graph_path.exists():
                print(f"[warn] missing graph json: {graph_path}")
                continue


            img = np.array(Image.open(sat_path).convert("RGB"))
            seg = np.array(Image.open(seg_path))
            if seg.ndim == 3:
                seg = seg[..., 0]
            seg = (seg > 0).astype(np.uint8)

            nodes, edges = load_graph_from_json(graph_path)

            # Decide resize target
            H, W = img.shape[0], img.shape[1]
            if resize_w is not None and resize_h is not None:
                out_w, out_h = int(resize_w), int(resize_h)
            else:
                f = float(args.downscale_factor)
                if f is None or f <= 1.0:
                    out_w, out_h = W, H
                else:
                    out_w = int(round(W / f))
                    out_h = int(round(H / f))
                    out_w = max(ps, out_w)
                    out_h = max(ps, out_h)

            if out_w != W or out_h != H:
                sx = out_w / float(W)
                sy = out_h / float(H)
                img, seg = resize_pair(img, seg, out_w, out_h)
                nodes = rescale_graph(nodes, sx, sy)

            H, W = img.shape[0], img.shape[1]
            if H < ps or W < ps:
                print(f"[warn] resized image smaller than patch_size ({W}x{H} < {ps}); skipping {stem}")
                continue

            per_image_limit = args.patches_per_image  # None => unlimited
            saved_here = 0

            pi = 0
            for y0 in range(0, H - ps + 1, stride):
                for x0 in range(0, W - ps + 1, stride):
                    if split_limit is not None and written_sp >= split_limit:
                        break
                    if per_image_limit is not None and saved_here >= per_image_limit:
                        break

                    img_p = img[y0:y0+ps, x0:x0+ps]
                    seg_p = seg[y0:y0+ps, x0:x0+ps]

                    nodes_p, edges_p = crop_graph_to_patch(nodes, edges, x0, y0, ps)

                    # Constraints: non-empty
                    if args.min_fg_pixels > 0 and int(np.count_nonzero(seg_p)) < int(args.min_fg_pixels):
                        continue
                    if nodes_p.shape[0] == 0 and edges_p.shape[0] == 0:
                        continue
                    if args.max_nodes is not None and nodes_p.shape[0] > int(args.max_nodes):
                        continue

                    patch_name = f"{stem}_{pi:06d}"

                    # image patch
                    Image.fromarray(img_p).save(out_img / f"{patch_name}.png")

                    # segmentation patch
                    Image.fromarray((seg_p.astype(np.uint8) * 255)).save(out_seg / f"{patch_name}_seg.png")

                    # Save graph in same legacy pickle dict format expected by downstream
                    save_graph(
                        nodes_p.astype(np.float32),
                        edges_p.astype(np.int32),
                        out_graph / f"{patch_name}_gt_graph",
                        args.graph_out_format,
                        args.patch_size,
                    )

                    pi += 1
                    saved_here += 1
                    written_sp += 1
                    written_total += 1

                if split_limit is not None and written_sp >= split_limit:
                    break
                if per_image_limit is not None and saved_here >= per_image_limit:
                    break

    print(f"[done] wrote {written_total} patches into {out_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to dataset root (expects raw/ and vtp/ subfolders)")
    ap.add_argument("--out_root", required=True, help="Output root for patches")
    ap.add_argument("--split", required=True, help="Path to splits.csv with headers id(or patient_id),split")

    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--overlap", type=float, default=0.35, help="Overlap fraction between consecutive patches (0..1)")
    ap.add_argument("--min_fg_pixels", type=int, default=50, help="Reject patch if segmentation foreground < this")
    ap.add_argument("--max_nodes", type=int, default=None, help="Skip patches with more than this many nodes (after cropping).")

    # NEW: resize-first
    ap.add_argument("--downscale_factor", type=float, default=6.0,
                    help="Downscale factor before patching (default 6: 4032x3024 -> 672x504). Set <=1 to disable.")
    ap.add_argument("--resize_width", type=int, default=None, help="Override resized width (requires --resize_height).")
    ap.add_argument("--resize_height", type=int, default=None, help="Override resized height (requires --resize_width).")

    # OPTIONAL caps (if omitted => extract as much as possible)
    ap.add_argument("--patches_per_image", type=int, default=None,
                    help="Optional cap per image. If not set, extracts all valid overlapping patches.")
    ap.add_argument("--num_train", type=int, default=None,
                    help="Optional cap for TRAIN split. If not set, extracts all valid patches for train.")
    ap.add_argument("--num_val", type=int, default=None,
                    help="Optional cap for VAL split. If not set, extracts all valid patches for val.")
    ap.add_argument("--num_test", type=int, default=None,
                    help="Optional cap for TEST split. If not set, extracts all valid patches for test.")
    ap.add_argument(
        "--graph_out_format",
        type=str,
        default="pickle",
        choices=["pickle", "vtp", "json"],
        help="Output format for patch graphs"
    )
    

    args = ap.parse_args(['--root', '/data/scavone/plants_3d2cut',
                          '--out_root', '/data/scavone/plants_3d2cut/patches',
                          '--split', '/data/scavone/plants_3d2cut/splits.csv',
                          '--overlap', '0.35',
                          '--downscale_factor', '6',
                          '--graph_out_format', 'vtp',
                        ])
    
    main(args)
