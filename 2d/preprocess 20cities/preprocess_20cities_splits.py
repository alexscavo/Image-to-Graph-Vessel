#!/usr/bin/env python3
import os, argparse, pickle, csv, re, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from textwrap import indent
import sys

# ---------------------------
# Graph I/O (20Cities pickle -> nodes, edges)
# ---------------------------

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


def load_graph_pickle_20cities(pkl_path):
    """
    Expects a pickled Python dict: {(x,y): [(x1,y1), (x2,y2), ...], ...}
    Returns:
        nodes_xy : (N,2) float32 pixel coords
        edges_ix : (E,2) int64 undirected edges (indices into nodes)
    """
    with open(pkl_path, "rb") as f:
        g = pickle.load(f)

    # keys and neighbor lists are (x,y) integer tuples in pixel coords
    # build a stable index for unique nodes
    node_to_idx = {}
    nodes = []
    for k in g.keys():
        if k not in node_to_idx:
            node_to_idx[k] = len(nodes)
            nodes.append(k)
        for nb in g[k]:
            if nb not in node_to_idx:
                node_to_idx[nb] = len(nodes)
                nodes.append(nb)

    nodes_xy = np.array(nodes, dtype=np.float32)
    # edges: (k -> nb) for all neighbors; undirected — dedup by sorting pairs
    edge_set = set()
    for k, nbs in g.items():
        u = node_to_idx[k]
        for nb in nbs:
            v = node_to_idx[nb]
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            edge_set.add((a, b))

    edges_ix = np.array(sorted(edge_set), dtype=np.int64)
    return nodes_xy, edges_ix

# ---------------------------
# Graph simplification (remove near-straight degree-2 nodes)
# ---------------------------

def angle_at_node(p_u, p_n, p_v, eps=1e-8):
    a = p_u - p_n
    b = p_v - p_n
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 180.0
    cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def simplify_degree2_nodes(nodes_xy, edges_idx, theta_straight=160.0, max_passes=10):
    N = nodes_xy.shape[0]
    adj = [set() for _ in range(N)]
    for u, v in edges_idx:
        if u == v: 
            continue
        adj[u].add(int(v)); adj[v].add(int(u))

    alive = np.ones(N, dtype=bool)
    changed = True
    passes = 0

    while changed and passes < max_passes:
        changed = False
        passes += 1
        for n in range(N):
            if not alive[n] or len(adj[n]) != 2:
                continue
            u, v = tuple(adj[n])
            if not (alive[u] and alive[v]): 
                continue
            ang = angle_at_node(nodes_xy[u], nodes_xy[n], nodes_xy[v])
            if ang >= theta_straight:  # near straight, drop n
                adj[u].discard(n); adj[v].discard(n)
                adj[u].add(v);     adj[v].add(u)
                adj[n].clear()
                alive[n] = False
                changed = True

    new_edges = set()
    for u in range(N):
        if not alive[u]: continue
        for v in adj[u]:
            if alive[v] and u < v:
                new_edges.add((u, v))
    new_edges = np.array(sorted(new_edges), dtype=np.int64)

    old2new = -np.ones(N, dtype=np.int64)
    keep_idx = np.where(alive)[0]
    old2new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    nodes_new = nodes_xy[keep_idx]
    if new_edges.size > 0:
        new_edges = np.stack([old2new[new_edges[:,0]], old2new[new_edges[:,1]]], axis=1)
    return nodes_new, new_edges, old2new

# ---------------------------
# Crop triplet (image, seg, graph)
# ---------------------------

def crop_triplet(img_np, seg_np, nodes_xy, edges_idx, x0, y0, patch, stem=None):
    x1, y1 = x0 + patch, y0 + patch
    img_p = img_np[y0:y1, x0:x1, :]
    seg_p = seg_np[y0:y1, x0:x1]

    in_box = (nodes_xy[:,0] >= x0) & (nodes_xy[:,0] < x1) & \
             (nodes_xy[:,1] >= y0) & (nodes_xy[:,1] < y1)
    keep_idx = np.where(in_box)[0]

    nodes_local = nodes_xy[keep_idx].copy()
    nodes_local[:,0] -= x0
    nodes_local[:,1] -= y0

    old2new = -np.ones(nodes_xy.shape[0], dtype=np.int64)
    old2new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    e0 = old2new[edges_idx[:,0]]
    e1 = old2new[edges_idx[:,1]]
    mask = (e0 >= 0) & (e1 >= 0)
    edges_local = np.stack([e0[mask], e1[mask]], axis=1) if np.any(mask) else np.zeros((0,2), np.int64)

    return img_p, seg_p, nodes_local, edges_local

def clip_segment_to_rect(p0, p1, x0, y0, x1, y1):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    p = np.array([-dx, dx, -dy, dy], dtype=float)
    q = np.array([p0[0]-x0, x1-p0[0], p0[1]-y0, y1-p0[1]], dtype=float)

    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
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

def save_patch_graph(nodes_p, edges_p, S, out_path):
    nodes_int = np.rint(nodes_p).astype(np.int64)
    nodes_int[:, 0] = np.clip(nodes_int[:, 0], 0, S - 1)
    nodes_int[:, 1] = np.clip(nodes_int[:, 1], 0, S - 1)

    coords_yx = [(int(y), int(x)) for (x, y) in nodes_int]

    adj = {c: set() for c in coords_yx}

    E = np.asarray(edges_p, dtype=np.int64)
    if E.size:
        E = E.reshape(-1, 2)
        for u, v in E:
            a = coords_yx[int(u)]
            b = coords_yx[int(v)]
            if a == b:
                continue
            adj[a].add(b)
            adj[b].add(a)

    graph_dict = { (int(y), int(x)): sorted(list(neigh))
                   for (y, x), neigh in adj.items() }

    with open(out_path, "wb") as f:
        pickle.dump(graph_dict, f, protocol=4)

def crop_with_clipping(img_np, seg_np, nodes_xy, edges_idx, x0, y0, S):
    x1, y1 = x0 + S, y0 + S
    img_p = img_np[y0:y1, x0:x1, :]
    seg_p = seg_np[y0:y1, x0:x1]

    inside = (nodes_xy[:,0] >= x0) & (nodes_xy[:,0] < x1) & (nodes_xy[:,1] >= y0) & (nodes_xy[:,1] < y1)
    keep_nodes = np.where(inside)[0]
    nodes_local = nodes_xy[keep_nodes].copy()
    nodes_local[:,0] -= x0
    nodes_local[:,1] -= y0

    old2new = -np.ones(nodes_xy.shape[0], dtype=np.int64)
    old2new[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)

    e0 = old2new[edges_idx[:,0]]
    e1 = old2new[edges_idx[:,1]]
    mask_inside_edges = (e0 >= 0) & (e1 >= 0)
    edges_local = np.stack([e0[mask_inside_edges], e1[mask_inside_edges]], axis=1) if np.any(mask_inside_edges) else np.zeros((0,2), np.int64)

    outside_u = (nodes_xy[:,0] < x0) | (nodes_xy[:,0] >= x1) | (nodes_xy[:,1] < y0) | (nodes_xy[:,1] >= y1)
    crossing = outside_u[edges_idx[:,0]] & outside_u[edges_idx[:,1]]
    crossing_idx = np.where(crossing)[0]

    inter_pts = []
    inter_edges = []
    pt2idx = {}

    def add_pt(pt):
        key = (float(np.round(pt[0]-x0, 3)), float(np.round(pt[1]-y0, 3)))
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

    e0 = old2new[edges_idx[:,0]]
    e1 = old2new[edges_idx[:,1]]
    partial = (e0 >= 0) ^ (e1 >= 0)
    partial_idx = np.where(partial)[0]
    for k in partial_idx:
        u, v = edges_idx[k]
        p0 = nodes_xy[u]
        p1 = nodes_xy[v]
        clipped = clip_segment_to_rect(p0, p1, x0, y0, x1, y1)
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

    if inter_pts:
        inter_pts = np.asarray(inter_pts, dtype=np.float32)
        nodes_local = np.vstack([nodes_local, inter_pts])
        if inter_edges:
            edges_local = np.vstack([edges_local, np.asarray(inter_edges, dtype=np.int64)]) if edges_local.size else np.asarray(inter_edges, dtype=np.int64)

    return img_p, seg_p, nodes_local.astype(np.float32), edges_local.astype(np.int64)

def sweep_best_patch(img_np, seg_np, nodes_xy, edges_ix, S, stride=32, crop_fn=crop_with_clipping):
    H, W = seg_np.shape
    best = None

    def score(nodes_p, edges_p, seg_p):
        has_edge = int(edges_p.shape[0] > 0)
        has_node = int(nodes_p.shape[0] > 0)
        fg = int(np.count_nonzero(seg_p))
        return has_edge*10_000 + has_node*1_000 + min(fg, 999)

    for y0 in range(0, H - S + 1, stride):
        for x0 in range(0, W - S + 1, stride):
            img_p, seg_p, nodes_p, edges_p = crop_fn(img_np, seg_np, nodes_xy, edges_ix, x0, y0, S)
            sc = score(nodes_p, edges_p, seg_p)
            if best is None or sc > best[0]:
                best = (sc, x0, y0, img_p, seg_p, nodes_p, edges_p)

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
        x0 = y0 = 0
        return x0, y0, crop_fn(img_np, seg_np, nodes_xy, edges_ix, x0, y0, S)

    _, x0, y0, img_p, seg_p, nodes_p, edges_p = best
    return x0, y0, img_p, seg_p, nodes_p, edges_p

            
def sample_patch(
    img_np, seg_np, nodes_xy, edges_ix, S,
    min_fg_pixels=0, require_graph=False, max_tries=2000,
    stem=None, rng=None,
    overlap=0.5,          # 0..0.99
    random_start=False,   # pick a random starting index in positions list (once per stem)
    random_offset=False,  # jitter grid origin by < stride (once per stem)
    allow_empty=False,    # allow empty patches if constraints fail
    empty_keep_prob=0.0   # prob to keep empty patch when allow_empty=True
):
    """
    Sliding-window (overlapping) patch extractor that starts at an initial
    position and then walks forward sequentially across calls (per `stem`).
    Returns: (x0, y0, img_p, seg_p, nodes_p, edges_p)
    """
    import numpy as np

    H, W = seg_np.shape
    # stride from overlap; clamp to [1, S]
    stride = int(round(S * (1.0 - float(overlap))))
    stride = max(1, min(stride, S))

    # RNG (only needed if random_* are used)
    _rng = rng if rng is not None else np.random.default_rng()

    # --- (optional) wrapper if your crop expects (row,col) rather than (x,y) ---
    # If your crop_with_clipping already expects XY, leave as-is.
    def _crop_xy(img_np, seg_np, nodes_in_xy, edges_in, x0, y0, S_):
        return crop_with_clipping(img_np, seg_np, nodes_in_xy, edges_in, x0, y0, S_)

    # ---------- build (or reuse) positions for this image ----------
    key = stem if stem is not None else "__default__"
    state = _SLIDE_CACHE.get(key)

    if state is None:
        # jitter the grid origin (pixel offset < stride)
        ox = int(_rng.integers(0, stride)) if random_offset else 0
        oy = int(_rng.integers(0, stride)) if random_offset else 0

        xs = list(range(ox, max(1, W - S + 1), stride)) if W >= S else [0]
        ys = list(range(oy, max(1, H - S + 1), stride)) if H >= S else [0]
        positions = [(x0, y0) for y0 in ys for x0 in xs]
        if not positions:
            positions = [(0, 0)]

        # choose a starting index once (per stem)
        start_idx = int(_rng.integers(0, len(positions))) if random_start else 0
        state = {"positions": positions, "idx": start_idx}
        _SLIDE_CACHE[key] = state

    positions = state["positions"]
    start_idx = state["idx"]
    n = len(positions)

    # ---------- sequential walk over positions ----------
    # Try up to `n` positions (full cycle) this call
    for step in range(n):
        idx = (start_idx + step) % n
        x0, y0 = positions[idx]

        img_p, seg_p, nodes_p, edges_p = _crop_xy(
            img_np, seg_np, nodes_xy, edges_ix, x0, y0, S
        )

        fg = int(np.count_nonzero(seg_p))
        ok_fg = (min_fg_pixels <= 0) or (fg >= min_fg_pixels)
        ok_graph = (not require_graph) or (
            (nodes_p is not None and nodes_p.shape[0] > 0) or
            (edges_p is not None and edges_p.shape[0] > 0)
        )

        if ok_fg and ok_graph:
            # advance index for next call (sequential behavior)
            state["idx"] = (idx + 1) % n
            return x0, y0, img_p, seg_p, nodes_p, edges_p

        if allow_empty and _rng.random() < float(empty_keep_prob):
            state["idx"] = (idx + 1) % n
            return x0, y0, img_p, seg_p, nodes_p, edges_p

    # ---------- fallback when no position satisfied constraints ----------
    x0, y0, img_p, seg_p, nodes_p, edges_p = sweep_best_patch(
        img_np, seg_np, nodes_xy, edges_ix, S,
        stride=max(8, S // 4),
        crop_fn=lambda a,b,c,d,x,y,S_: _crop_xy(a,b,c,d,x,y,S_)
    )
    state["idx"] = (state["idx"] + 1) % n
    return x0, y0, img_p, seg_p, nodes_p, edges_p


# ---------------------------
# Split helpers
# ---------------------------

def _normalize_split(name: str) -> str:
    name = (name or "").strip().lower()
    if name in {"val", "valid", "validation", "dev"}:
        return "val"
    if name in {"test", "testing"}:
        return "test"
    return "train"

# ---------------------------
# Main preprocessing with split-aware generation
# ---------------------------

def main(args):

    if args.total_patches is not None and args.patches_per_image is not None:
        raise ValueError("Use either --total_patches or --patches_per_image, not both.")
    
    
    print_args(args)
    

    random.seed(args.seed); np.random.seed(args.seed)

    raw_dir = Path(args.root) / "raw"
    vtp_dir = Path(args.root) / "vtp"

    # --- read split mapping (stem -> split) ---
    split_map = {}
    if args.split:
        split_csv = Path(args.split)
        if not split_csv.exists():
            print(f"[error] split file not found: {split_csv}")
            return
        with open(split_csv, newline="") as f:
            reader = csv.DictReader(f)
            if "id" not in reader.fieldnames and "patient_id" not in reader.fieldnames:
                print("[error] split CSV must have headers: id or patient_id, and split")
                return
            id_col = "id" if "id" in reader.fieldnames else "patient_id"
            if "split" not in reader.fieldnames:
                print("[error] split CSV must contain a 'split' column")
                return
            for row in reader:
                rid = Path(row[id_col]).stem
                split_map[rid] = _normalize_split(row["split"])

    # find regions via *_sat.png
    sats = sorted(raw_dir.glob("region_*_sat.png"))
    if not sats:
        print(f"[error] no satellite images like region_*_sat.png found in {raw_dir}")
        return

    # bucket regions by split; skip those not in CSV (to mirror other pipeline)
    buckets = {"train": [], "val": [], "test": []}
    for sat_path in sats:
        stem = sat_path.stem.replace("_sat", "")
        sp = split_map.get(stem, None) if split_map else "train"
        if sp is None:
            print(f"[skip] {sat_path.name} not found in splits.csv")
            continue
        buckets[sp].append(sat_path)

    counts = {sp: len(buckets[sp]) for sp in buckets}
    total_imgs = sum(counts.values())
    if total_imgs == 0:
        print("[error] no images matched the provided splits")
        return

    # --- determine allocation across splits ---
    explicit = {
        "train": args.num_train if args.num_train is not None else None,
        "val":   args.num_val   if args.num_val   is not None else None,
        "test":  args.num_test  if args.num_test  is not None else None,
    }

    if any(v is not None for v in explicit.values()):
        alloc = {sp: max(0, int(explicit[sp]) if explicit[sp] is not None else 0) for sp in buckets}
        grand_total = sum(alloc.values())
    elif args.total_patches is not None:
        alloc = {sp: int(np.round(args.total_patches * (counts[sp] / total_imgs))) for sp in buckets}
        diff = int(args.total_patches) - sum(alloc.values())
        order = sorted(buckets.keys(), key=lambda k: counts[k], reverse=True)
        for k in order:
            if diff == 0: break
            alloc[k] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
        grand_total = int(args.total_patches)
    else:
        alloc = {sp: 0 for sp in buckets}  # 0 means "no cap"
        grand_total = 0

    # --- create split-specific output roots ---
    out_roots = {}
    for sp in ("train", "val", "test"):
        root_sp = Path(args.out_root) / sp
        out_raw = root_sp / "raw"
        out_seg = root_sp / "seg"
        out_vtp = root_sp / "vtp"

        for d in (out_raw, out_seg, out_vtp):
            assert d.parent.name in {"train", "val", "test"}
            d.mkdir(parents=True, exist_ok=True)

        out_roots[sp] = (out_raw, out_seg, out_vtp)


    # Pillow resample compat
    try:
        RESIZE_BILINEAR = Image.Resampling.BILINEAR
        RESIZE_NEAREST  = Image.Resampling.NEAREST
    except AttributeError:
        RESIZE_BILINEAR = Image.BILINEAR
        RESIZE_NEAREST  = Image.NEAREST

    written = 0
    pbar = tqdm(total=grand_total if grand_total else None, desc="Patches", leave=True)

    global_idx = {"train": 0, "val": 0, "test": 0}

    for sp in ("train", "val", "test"):
        files = buckets[sp]
        if not files:
            continue

        # per-image quota for this split
        if grand_total:
            want_total_sp = max(0, alloc.get(sp, 0))
            N = len(files)
            per_img_quota = max(1, int(np.ceil(want_total_sp / max(1, N)))) if want_total_sp else (args.patches_per_image or 5)
        else:
            per_img_quota = args.patches_per_image or 5

        out_raw, out_seg, out_vtp = out_roots[sp]

        written_sp = 0
        for si, sat_path in enumerate(tqdm(files, desc=f"Processing regions ({sp})", leave=False)):
            if alloc.get(sp, 0) and written_sp >= alloc[sp]:
                break

            stem = sat_path.stem.replace("_sat", "")
            gt_path  = raw_dir / f"{stem}_gt.png"
            pkl_path = vtp_dir / f"{stem}_gt_graph.pickle"

            if not gt_path.exists() or not pkl_path.exists():
                print(f"[skip] missing gt/pickle for {stem}")
                continue

            img = Image.open(sat_path).convert("RGB")
            seg = Image.open(gt_path).convert("L")

            if seg.size != img.size:
                seg = seg.resize(img.size, resample=RESIZE_NEAREST)

            W, H = img.size
            if W < args.patch_size or H < args.patch_size:
                print(f"[skip] region too small for patch_size: {stem}")
                continue

            img_np = np.array(img, dtype=np.uint8)
            seg_np = (np.array(seg, dtype=np.uint8) > 0).astype(np.uint8)

            nodes_xy, edges_ix = load_graph_pickle_20cities(pkl_path)

            def hit_rate(nodes_px, segarr):
                H2, W2 = segarr.shape
                pts = np.round(nodes_px).astype(int)
                m = (pts[:,0] >= 0) & (pts[:,0] < W2) & (pts[:,1] >= 0) & (pts[:,1] < H2)
                pts = pts[m]
                if pts.size == 0: 
                    return 0.0
                return float(segarr[pts[:,1], pts[:,0]].mean())

            nodes_base = nodes_xy
            nodes_swap = nodes_xy[:, [1, 0]]
            hr_base = hit_rate(nodes_base, seg_np)
            hr_swap = hit_rate(nodes_swap, seg_np)
            nodes = nodes_swap if hr_swap > hr_base else nodes_base

            mins = nodes.min(axis=0)
            maxs = nodes.max(axis=0)
            if 0.0 <= mins.min() and maxs.max() <= 1.2:
                nodes[:, 0] *= W
                nodes[:, 1] *= H

            rng = make_rng(args.seed, image_key=sat_path.stem)

            for pi in tqdm(range(per_img_quota), desc=f"Patches ({si+1}/{len(files)})", leave=False, unit='patch'):
                x0, y0, img_p, seg_p, nodes_p, edges_p = sample_patch(
                    img_np, seg_np, nodes, edges_ix,
                    S=args.patch_size,
                    min_fg_pixels=args.min_fg_pixels,
                    require_graph=True,
                    max_tries=2000,
                    stem=stem,
                    rng=rng,
                    allow_empty=False, 
                    overlap=args.overlap,
                    random_start=True,
                    random_offset=True
                )

                if args.simplify_theta is not None and edges_p.shape[0] > 0:
                    nodes_p, edges_p, _ = simplify_degree2_nodes(
                        nodes_p, edges_p, theta_straight=float(args.simplify_theta)
                    )

                global_idx[sp] += 1
                idx = global_idx[sp]

                base = f"sample_{idx:06d}"

                # raw image
                Image.fromarray(img_p).save(
                    out_raw / f"{base}_data.png"
                )

                # segmentation
                seg_vis = (seg_p.astype(np.uint8) * 255)
                Image.fromarray(seg_vis, mode="L").save(
                    out_seg / f"{base}_seg.png"
                )

                # graph
                graph_base = out_vtp / f"{base}_graph"


                # Save according to chosen format
                if args.graph_format in ("pickle", "both"):
                    save_patch_graph(
                        nodes_p, edges_p, args.patch_size,
                        graph_base.with_suffix(".pickle")
                    )

                if args.graph_format in ("vtp", "both"):
                    save_patch_graph_vtp(
                        nodes_p, edges_p, args.patch_size,
                        graph_base.with_suffix(".vtp")
                    )



                written += 1
                written_sp += 1
                if pbar is not None:
                    pbar.update(1)

                if alloc.get(sp, 0) and written_sp >= alloc[sp]:
                    break

                if grand_total and written >= grand_total:
                    if pbar is not None:
                        pbar.close()
                    print(f"[done] wrote {written} patches (target was {grand_total})")
                    return

    if pbar is not None:
        pbar.close()
    print(f"[done] wrote {written} patches to {args.out_root}")

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path/to/20cities")
    ap.add_argument("--out_root", required=True, help="output root for patches")
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--patches_per_image", type=int, default=None, help="random patches per region (used if no totals provided)")
    ap.add_argument("--graph_format", choices=["pickle", "vtp", "both"], default="pickle", help="Graph file format to save: 'pickle' (20Cities dict), 'vtp' (PolyData), or 'both'.")
    ap.add_argument("--simplify_theta", type=float, default=160.0, help="remove degree-2 nodes with angle >= theta")
    ap.add_argument("--min_fg_pixels", type=int, default=50, help="reject random patch if seg foreground < this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overlap", type=float, default=0.3, required=True, help="Extension of the overlap between consecutive patches")
    # NEW: split-aware args and allocation
    ap.add_argument("--split", default=None, required=True,
                    help="Path to splits.csv with headers id(or patient_id),split. Split in {train,val/test}.")
    ap.add_argument("--total_patches", type=int, default=None,
                    help="Global target; distributed across splits and images proportionally.")
    ap.add_argument("--num_train", type=int, default=None,
                    help="Exact number of TRAIN patches to generate; overrides total_patches for this split.")
    ap.add_argument("--num_val", type=int, default=None,
                    help="Exact number of VAL patches to generate; overrides total_patches for this split.")
    ap.add_argument("--num_test", type=int, default=None,
                    help="Exact number of TEST patches to generate; overrides total_patches for this split.")

    args = ap.parse_args(['--root', 'C:/Users/Utente/Desktop/tesi/datasets/20cities',
                          '--out_root', 'C:/Users/Utente/Desktop/tesi/datasets/20cities/patches',
                          '--split', 'C:/Users/Utente/Desktop/tesi/datasets/20cities/splits.csv',
                          '--graph_format', 'both',
                          '--overlap', '0.35',
                          '--num_train', '99200',
                          '--num_val', '24800',
                          '--num_test', '25000'])
    
    main(args)
