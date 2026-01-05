from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image
import argparse, csv
import torch
from tqdm import tqdm
import csv
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.csv_graph import csv_graph_loading, csv_graph_saving


# ---------------------------
# I/O helpers (keep OCTA-Synth CSV format)
# ---------------------------

_BRACKET_VEC = re.compile(r"\[([^\]]+)\]")
_SLIDE_CACHE = {}

def print_args(args):
    print("\n" + "=" * 60)
    print("⚙️  Configuration")
    print("=" * 60)
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        print(f"{k:>15}: {v}")
    print("=" * 60 + "\n")

def make_rng(base_seed, image_key=None):
    if base_seed is None or base_seed < 0:
        return np.random.default_rng()  # non-deterministic
    mix = hash((base_seed, image_key)) & 0xFFFFFFFF
    return np.random.default_rng(mix)

def pil_to_rgb(im: Image.Image):
    if im.mode != "RGB":
        return im.convert("RGB")
    return im

def clip_segment_to_rect(p0, p1, x0, y0, x1, y1):
    # Liang–Barsky. Returns (clipped_p0, clipped_p1) in global coords, or None if no intersection.
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    p = np.array([-dx, dx, -dy, dy], dtype=float)
    q = np.array([p0[0]-x0, x1-p0[0], p0[1]-y0, y1-p0[1]], dtype=float)

    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:  # parallel and outside
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u1: return None
                if t > u0: u0 = t
            else:
                if t < u0: return None
                if t < u1: u1 = t

    c0 = p0 + u0 * np.array([dx, dy])
    c1 = p0 + u1 * np.array([dx, dy])
    return c0, c1

def save_graph_csv_single(csv_path: Path, nodes_xy_px: np.ndarray, edges_idx: np.ndarray, canvas_size: int):
    """
    Save graph in OCTA-Synth CSV format: node1,node2,radius
    - node coords are *normalized* to [0,1] (x/canvas, y/canvas)
    - we don’t track radii through clipping; write 0.0 for radius to keep format
    """
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node1", "node2", "radius"])
        inv = 1.0 / float(canvas_size)
        for (i, j) in edges_idx.tolist():
            x1, y1 = nodes_xy_px[int(i)] * inv
            x2, y2 = nodes_xy_px[int(j)] * inv
            w.writerow([f"[{x1:.8f} {y1:.8f} 0.0]", f"[{x2:.8f} {y2:.8f} 0.0]", "0.0"])  # keep radius column

def crop_with_clipping(img_np, seg_np, nodes_xy, edges_idx, x0, y0, S):
    x1, y1 = x0 + S, y0 + S
    img_p = img_np[y0:y1, x0:x1, :]
    seg_p = seg_np[y0:y1, x0:x1]

    # inside nodes
    inside = (nodes_xy[:,0] >= x0) & (nodes_xy[:,0] < x1) & (nodes_xy[:,1] >= y0) & (nodes_xy[:,1] < y1)
    keep_nodes = np.where(inside)[0]
    nodes_local = nodes_xy[keep_nodes].copy()
    nodes_local[:,0] -= x0
    nodes_local[:,1] -= y0

    # map old->new for inside nodes
    old2new = -np.ones(nodes_xy.shape[0], dtype=np.int64)
    old2new[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)

    # collect edges fully inside
    e0 = old2new[edges_idx[:,0]]
    e1 = old2new[edges_idx[:,1]]
    mask_inside_edges = (e0 >= 0) & (e1 >= 0)
    edges_local = np.stack([e0[mask_inside_edges], e1[mask_inside_edges]], axis=1) if np.any(mask_inside_edges) else np.zeros((0,2), np.int64)

    # clip crossing edges (both endpoints outside but segment intersects)
    outside_u = (nodes_xy[:,0] < x0) | (nodes_xy[:,0] >= x1) | (nodes_xy[:,1] < y0) | (nodes_xy[:,1] >= y1)
    crossing = outside_u[edges_idx[:,0]] & outside_u[edges_idx[:,1]]  # both endpoints outside
    crossing_idx = np.where(crossing)[0]

    # store new intersection points, dedup by rounding to subpixel grid (here 1e-3 px)
    inter_pts = []
    inter_edges = []
    pt2idx = {}

    def add_pt(pt):
        key = (float(np.round(pt[0]-x0, 3)), float(np.round(pt[1]-y0, 3)))  # local coords key
        if key in pt2idx:
            return pt2idx[key]
        pt2idx[key] = len(nodes_local) + len(inter_pts)
        inter_pts.append([pt[0]-x0, pt[1]-y0])
        return pt2idx[key]

    for k in crossing_idx:
        u, v = edges_idx[k]
        p0 = nodes_xy[u]
        p1 = nodes_xy[v]
        clipped = clip_segment_to_rect(p0, p1, x0, y0, x1, y1)
        if clipped is None:
            continue
        c0, c1 = clipped
        i0 = add_pt(c0)
        i1 = add_pt(c1)
        if i0 != i1:
            inter_edges.append([i0, i1])
            
    # edge handling where one point is inside the patch and one is outside 
    
    partial = (e0 >= 0) ^ (e1 >= 0)  # XOR: exactly one is inside
    partial_idx = np.where(partial)[0]

    for k in partial_idx:
        u, v = edges_idx[k]
        p0 = nodes_xy[u]
        p1 = nodes_xy[v]
        
        # Clip the segment to the patch boundary
        clipped = clip_segment_to_rect(p0, p1, x0, y0, x1, y1)
        if clipped is None:
            continue
        
        c0, c1 = clipped
        
        # Determine which endpoint was inside
        if e0[k] >= 0:  # u is inside
            inside_idx = e0[k]
            boundary_pt = c1  # the other clipped point is on boundary
        else:  # v is inside
            inside_idx = e1[k]
            boundary_pt = c0
        
        # Add boundary intersection point
        boundary_idx = add_pt(boundary_pt)
        
        if inside_idx != boundary_idx:
            inter_edges.append([inside_idx, boundary_idx])

    if inter_pts:
        inter_pts = np.asarray(inter_pts, dtype=np.float32)
        nodes_local = np.vstack([nodes_local, inter_pts])
        if inter_edges:
            edges_local = np.vstack([edges_local, np.asarray(inter_edges, dtype=np.int64)]) if edges_local.size else np.asarray(inter_edges, dtype=np.int64)

    return img_p, seg_p, nodes_local.astype(np.float32), edges_local.astype(np.int64)
           
def _passes_two_point_length(nodes_p_xy, edges_p, S, frac=0.05, relative='diag'):
    """
    Return True if (1) not the 2-nodes-1-edge case, or (2) the single edge length
    >= frac * reference, where reference is S (side) or sqrt(2)*S (diag).
    """
    import numpy as np

    if nodes_p_xy is None or edges_p is None:
        return True

    if nodes_p_xy.shape[0] == 2 and edges_p.shape[0] == 1:
        i, j = int(edges_p[0, 0]), int(edges_p[0, 1])
        # be lenient if indices look odd
        if i not in (0, 1) or j not in (0, 1) or i == j:
            return True
        p = nodes_p_xy[i].astype(float)
        q = nodes_p_xy[j].astype(float)
        d = float(np.linalg.norm(p - q))
        ref = (S * (2 ** 0.5)) if relative == 'diag' else float(S)
        return d >= frac * ref

    return True    
    
def sweep_best_patch(img_np, seg_np, nodes_xy, edges_ix, S, stride=32, crop_fn=crop_with_clipping, min_two_point_edge_frac=0.05, two_point_len_relative='diag', require_graph=False,):
    H, W = seg_np.shape
    best = None   # (score, x0,y0, img_p, seg_p, nodes_p, edges_p)

    def score(nodes_p, edges_p, seg_p):
        # reject if require_graph and none present
        has_edge = int(edges_p.shape[0] > 0)
        has_node = int(nodes_p.shape[0] > 0)
        if require_graph and not (has_edge or has_node):
            return -1
        # NEW: reject if 2pts-1edge but edge too short
        if not _passes_two_point_length(
            nodes_p, edges_p, S,
            frac=float(min_two_point_edge_frac),
            relative=two_point_len_relative
        ):
            return -1
        fg = int(np.count_nonzero(seg_p))
        return has_edge*10_000 + has_node*1_000 + min(fg, 999)

    for y0 in range(0, H - S + 1, stride):
        for x0 in range(0, W - S + 1, stride):
            img_p, seg_p, nodes_p, edges_p = crop_fn(img_np, seg_np, nodes_xy, edges_ix, x0, y0, S)
            sc = score(nodes_p, edges_p, seg_p)
            if best is None or sc > best[0]:
                best = (sc, x0, y0, img_p, seg_p, nodes_p, edges_p)

    # If nothing scored > 0, do a final dense sweep just over the seg bbox (cheap if small)
    if best is None or best[0] == 0:
        ys, xs = (seg_np > 0).nonzero()
        if ys.size:
            y_min, y_max = max(0, ys.min()-S), min(H-S, ys.max())
            x_min, x_max = max(0, xs.min()-S), min(W-S, xs.max())
            for y0 in range(y_min, y_max+1):
                for x0 in range(x_min, x_max+1):
                    img_p, seg_p, nodes_p, edges_p = crop_fn(img_np, seg_np, nodes_xy, edges_ix, x0, y0, S)
                    sc = score(nodes_p, edges_p, seg_p)
                    if best is None or sc > best[0]:
                        best = (sc, x0, y0, img_p, seg_p, nodes_p, edges_p)

    if best is None:
        # truly no foreground, return the top-left tile as a last resort
        x0 = y0 = 0
        return x0, y0, crop_fn(img_np, seg_np, nodes_xy, edges_ix, x0, y0, S)

    _, x0, y0, img_p, seg_p, nodes_p, edges_p = best
    return x0, y0, img_p, seg_p, nodes_p, edges_p

def normalize_xy(nodes, size):
    """Return nodes with x,y divided by size; leave z as-is if present."""
    arr = nodes.detach().cpu().numpy() if torch.is_tensor(nodes) else np.asarray(nodes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("nodes must be (N,>=2)")
    out = arr.astype(np.float32).copy()
    out[:, :2] /= float(size)
    return out

def sample_patch(
    img_np, seg_np, nodes_xy, edges_ix, S,
    min_fg_pixels=0, require_graph=False, max_tries=2000,  # kept for API compat
    stem=None, rng=None,
    # NEW:
    overlap=0.5,            # fraction in [0, 0.99]; 0.5 → 50% like before
    random_start=False,     # if True, start from a random index in the positions list
    random_offset=False,    # if True, jitter the grid origin by a random offset < stride
    allow_empty=False,      # if True, allow empty patches when constraints fail
    empty_keep_prob=0.0,     # keep prob for empty patches if allow_empty=True (0..1)
    min_two_point_edge_frac=0.05,
    two_point_len_relative='diag'
):
    """
    Sliding-window patch extractor with:
      - tunable overlap,
      - optional random starting point or random grid offset,
      - controllable behavior for empty/no-graph patches.

    Returns: x0, y0, img_p, seg_p, nodes_p_xy, edges_p
    """
    import numpy as np

    H, W = seg_np.shape
    # compute stride from overlap; clamp to [1, S]
    stride = int(round(S * (1.0 - float(overlap))))
    stride = max(1, min(stride, S))

    # RNG
    _rng = rng if rng is not None else np.random.default_rng()

    def _crop_xy(img_np, seg_np, nodes_xy_in, edges_ix_in, x0_in, y0_in, S_in):
        # your crop_with_clipping uses (row,col); convert as you had
        if nodes_xy_in.size:
            nodes_rc = nodes_xy_in[:, [1, 0]]
        else:
            nodes_rc = nodes_xy_in

        img_p, seg_p, nodes_p_rc, edges_p = crop_with_clipping(
            img_np, seg_np, nodes_rc, edges_ix_in, x0_in, y0_in, S_in
        )

        if nodes_p_rc.size:
            nodes_p_xy = nodes_p_rc[:, [1, 0]]
        else:
            nodes_p_xy = nodes_p_rc
        return img_p, seg_p, nodes_p_xy, edges_p

    # ---- build (or reuse) the positions for this image ----
    key = stem if stem is not None else "__default__"
    state = _SLIDE_CACHE.get(key)

    if state is None:
        # optional random grid origin (offset smaller than stride)
        ox = _rng.integers(0, stride) if random_offset else 0
        oy = _rng.integers(0, stride) if random_offset else 0

        # make sure at least one valid position exists
        xs = list(range(ox, max(1, W - S + 1), stride)) if W >= S else [0]
        ys = list(range(oy, max(1, H - S + 1), stride)) if H >= S else [0]

        positions = [(x0, y0) for y0 in ys for x0 in xs]
        if not positions:
            positions = [(0, 0)]

        # choose a random starting index if requested; otherwise 0
        start_idx = int(_rng.integers(0, len(positions))) if random_start else 0

        state = {"positions": positions, "idx": start_idx}
        _SLIDE_CACHE[key] = state

    positions = state["positions"]
    start_idx = state["idx"]
    n = len(positions)

    # ---- scan positions until constraints are met ----
    for step in range(n):
        idx = (start_idx + step) % n
        x0, y0 = positions[idx]

        img_p, seg_p, nodes_p_xy, edges_p = _crop_xy(
            img_np, seg_np, nodes_xy, edges_ix, x0, y0, S
        )

        fg = int(np.count_nonzero(seg_p))
        ok_fg = (min_fg_pixels <= 0) or (fg >= min_fg_pixels)

        # graph present?
        has_nodes = (nodes_p_xy is not None and nodes_p_xy.shape[0] > 0)
        has_edges = (edges_p is not None and edges_p.shape[0] > 0)
        ok_presence = (not require_graph) or (has_nodes or has_edges)

        # NEW: 2-points-1-edge minimum length constraint
        ok_pair_len = _passes_two_point_length(
            nodes_p_xy, edges_p, S,
            frac=float(min_two_point_edge_frac),
            relative=two_point_len_relative
        )

        ok_graph = ok_presence and ok_pair_len

        if ok_fg and ok_graph:
            state["idx"] = (idx + 1) % n
            return x0, y0, img_p, seg_p, nodes_p_xy, edges_p

        # optionally allow empty/no-graph patches with some probability
        if allow_empty and _rng.random() < float(empty_keep_prob):
            state["idx"] = (idx + 1) % n
            return x0, y0, img_p, seg_p, nodes_p_xy, edges_p

    # ---- fallback: deterministic sweep for "best" patch when nothing matched ----
    x0, y0, img_p, seg_p, nodes_p_xy, edges_p = sweep_best_patch(
        img_np, seg_np, nodes_xy, edges_ix, S,
        stride=max(8, S // 4),
        crop_fn=_crop_xy,
        require_graph=require_graph,
    )
    state["idx"] = (state["idx"] + 1) % len(state["positions"])
    return x0, y0, img_p, seg_p, nodes_p_xy, edges_p

def split_reader(split):
    
    def _base(p: Path) -> str:
        s = p.stem
        if s.endswith("data"):
            s = s[:-4]
        if s.startswith("G_"):
            s = s[2:]
        return s

    def _normalize_split(name: str) -> str:
        name = (name or "").strip().lower()
        if name in {"val", "valid", "validation", "dev"}:
            return "val"
        if name in {"test", "testing"}:
            return "test"
        return "train"
    
    split_map = {}
    split_csv = Path(split)
    if not split_csv.exists():
        print(f"[error] split file not found: {split_csv}")
        return
    with open(split_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "patient_id" not in reader.fieldnames or "split" not in reader.fieldnames:
            print("[error] split CSV must have headers: patient_id,split")
            return
        for row in reader:
            pid = row["patient_id"].strip()
            # normalize to 'base' using same logic used for files
            base = _base(Path(pid))
            split_map[base] = _normalize_split(row["split"])

    return split_map

# ---------------------------
# Main preprocessing (OCTA-Synth → patches) using 20cities clipping/sampling
# ---------------------------

def main(args):
    
    print_args(args)
    
    def _base(p: Path) -> str:
        s = p.stem
        if s.endswith("data"):
            s = s[:-4]
        if s.startswith("G_"):
            s = s[2:]
        return s
    
    split_file = Path(args.split)
    
    split_map = split_reader(split_file)
    
    in_img = Path(args.root) / 'raw'
    in_seg = Path(args.root) / 'labels'
    in_graph = Path(args.root) / 'graphs'


    img_files_all = sorted(in_img.glob("*.png"))
    if not img_files_all:
        print(f"[error] no images found in {in_img}")
        return
    
    
    buckets = {"train": [], "val": [], "test": []}
    for img_path in img_files_all:
        base = _base(img_path)
        sp = split_map.get(base, None)
        if sp is None:
            # if not in CSV, skip cleanly
            print(f"[skip] {img_path.name} not found in splits.csv")
            continue
        buckets[sp].append(img_path)

    try:
        RESIZE_BILINEAR = Image.Resampling.BILINEAR
        RESIZE_NEAREST  = Image.Resampling.NEAREST
    except AttributeError:
        RESIZE_BILINEAR = Image.BILINEAR
        RESIZE_NEAREST  = Image.NEAREST
        
    # create output dirs per split
    out_roots = {}
    for sp in ("train", "val", "test"):
        root_sp = Path(args.out_root) / sp
        out_img = root_sp / "raw"
        out_seg = root_sp / "labels"
        out_csv = root_sp / "graphs"
        for d in (out_img, out_seg, out_csv):
            d.mkdir(parents=True, exist_ok=True)
        out_roots[sp] = (out_img, out_seg, out_csv)

    written = 0
    want_total = int(args.target_total) if args.target_total else 0

    # determine allocation per split
    # priority: explicit per-split numbers > target_total proportional > default (no cap)
    explicit = {
        'train': int(args.num_train) if args.num_train is not None else None,
        'val':   int(args.num_val)   if args.num_val   is not None else None,
        'test':  int(args.num_test)  if args.num_test  is not None else None,
    }

    counts = {sp: len(buckets[sp]) for sp in buckets}
    total_imgs = sum(counts.values())

    if any(v is not None for v in explicit.values()):
        # use provided per-split totals (missing ones default to 0)
        alloc = {sp: max(0, (explicit[sp] if explicit[sp] is not None else 0)) for sp in buckets}
        total_target = sum(alloc.values())
    elif want_total and total_imgs > 0:
        # distribute target_total across splits proportionally to number of images
        alloc = {sp: int(np.round(want_total * (counts[sp] / total_imgs))) for sp in buckets}
        # adjust rounding to hit exact total
        diff = want_total - sum(alloc.values())
        if diff != 0:
            order = sorted(buckets.keys(), key=lambda k: counts[k], reverse=True)
            for k in order:
                if diff == 0: break
                alloc[k] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
        total_target = want_total
    else:
        # no cap specified; process with default per-image quotas (5)
        alloc = {sp: 0 for sp in buckets}
        total_target = 0
     
    pbar = tqdm(total=total_target if total_target else None, desc="Patches", leave=True)

    for sp in ("train", "val", "test"):
        files = buckets[sp]
        if not files:
            continue

        # per-image quota for this split
        if want_total:
            want_total_sp = max(0, alloc.get(sp, 0))
            N = len(files)
            per_img_quota = max(1, int(np.ceil(want_total_sp / max(1, N)))) if want_total_sp else 5
        else:
            per_img_quota = 5   # debug value

        out_img, out_seg, out_csv = out_roots[sp]

        written_sp = 0
        for i, img_path in enumerate(tqdm(files, desc=f"Processing images ({sp})", leave=False)):
            if alloc.get(sp, 0) and written_sp >= alloc[sp]:
                break
            base = _base(img_path)
            seg_path   = in_seg / f"{base}.png"
            graph_path = in_graph / f"{base}"
            
            
            
            if not seg_path.exists() or not graph_path.exists():
                print(f"[skip] missing seg/graph for {base}")
                # print('-'*50)
                # print(img_path)
                # print(seg_path)
                # print(graph_path)
                # print('-'*50)
                continue

            # load & force to common canvas
            im  = pil_to_rgb(Image.open(img_path))
            seg = Image.open(seg_path).convert("L")
            if im.size != (args.canvas_size, args.canvas_size):
                print('ERROR! Canvas size different from image size!')
                sys.exit()
                # im = im.resize((args.canvas_size, args.canvas_size), resample=RESIZE_BILINEAR)
            if seg.size != (args.canvas_size, args.canvas_size):
                # seg = seg.resize((args.canvas_size, args.canvas_size), resample=RESIZE_NEAREST)
                print('ERROR! Canvas size different from segmentation size!')
                sys.exit()

            img_np = np.array(im, dtype=np.uint8)
            seg_np = np.array(seg, dtype=np.uint8)
            if seg_np.max() > 1:
                seg_np = (seg_np > 0).astype(np.uint8)

            # load graph from OCTA-Synth CSV (normalized → pixels)
            nodes_t, edges_t = csv_graph_loading(graph_path)

            # convert to numpy + scale normalized coords to pixels expected by downstream code
            nodes_xy = (nodes_t.cpu().numpy() * float(args.canvas_size)).astype(np.float32)
            edges_ix = edges_t.cpu().numpy().astype(np.int64)

            rng = make_rng(args.seed, image_key=base)

            for pi in range(per_img_quota):
                x0, y0, img_p, seg_p, nodes_p, edges_p = sample_patch(
                    img_np, seg_np, nodes_xy, edges_ix,
                    S=args.crop,
                    min_fg_pixels=0,
                    require_graph=True,
                    max_tries=2000,
                    stem=base,
                    rng=rng,
                    allow_empty=False,
                    overlap=args.overlap,
                    random_start=True,
                    random_offset=True,
                    min_two_point_edge_frac=0.05,
                    two_point_len_relative='diag'
                )
                 
                nodes_norm = normalize_xy(nodes_p, args.crop)   # normalize to patch size

                patch_name = f"{base}_{pi:03d}"
                Image.fromarray(img_p).save(out_img / f"{patch_name}.png")
                seg_to_save = (seg_p > 0).astype(np.uint8) * 255
                Image.fromarray(seg_to_save, mode="L").save(out_seg / f"{patch_name}.png")
                # save_graph_csv_single(out_csv / f"{patch_name}.csv", nodes_p, edges_p, canvas_size=args.crop)
                csv_graph_saving(out_csv / patch_name, nodes_norm, edges_p)
                

                written += 1
                written_sp += 1
                if pbar is not None:
                    pbar.update(1)

                # stop when split meets its allocation
                if alloc.get(sp, 0) and written_sp >= alloc[sp]:
                    break

                # stop globally when we hit the total target (explicit or proportional)
                if (sum(alloc.values()) > 0) and written >= sum(alloc.values()):
                    if pbar is not None:
                        pbar.close()
                    print(f"[done] wrote {written} patches (target was {sum(alloc.values())})")
                    return

    if pbar is not None:
        pbar.close()
    print(f"[done] wrote {written} patches (target was {args.target_total or written})")
    
    


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--canvas_size", type=int, default=2048)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--target_total", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overlap", type=float, default=0.3, required=True, help="Extension of the overlap between consecutive patches")
    ap.add_argument("--num_train", type=int, default=None,
                    help="Exact number of TRAIN patches to generate; overrides target_total for this split.")
    ap.add_argument("--num_val", type=int, default=None,
                    help="Exact number of VAL patches to generate; overrides target_total for this split.")
    ap.add_argument("--num_test", type=int, default=None,
                    help="Exact number of TEST patches to generate; overrides target_total for this split.")
    ap.add_argument("--split", default=None, required=True)
    
    args = ap.parse_args(['--root', 'C:/Users/Utente/Desktop/tesi/datasets/octa-synth',
                          '--out_root', 'C:/Users/Utente/Desktop/tesi/datasets/octa-synth/patches',
                          '--split', 'C:/Users/Utente/Desktop/tesi/datasets/octa-synth/splits.csv',
                          '--canvas_size', '1400',
                          '--overlap', '0.35',
                          '--num_train', '480',
                          '--num_val', '220', 
                          '--num_test', '2000']) 
    
    main(args)

