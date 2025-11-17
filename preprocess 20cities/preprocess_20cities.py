#!/usr/bin/env python3
import os, argparse, pickle, csv, re, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import sys


# ---------------------------
# Graph I/O (20Cities pickle -> nodes, edges)
# ---------------------------

def make_rng(base_seed, image_key=None):
    if base_seed is None or base_seed < 0:
        return np.random.default_rng()  # non-deterministic
    mix = hash((base_seed, image_key)) & 0xFFFFFFFF
    return np.random.default_rng(mix)

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
    """
    img_np: (H,W,3) uint8 image
    seg_np: (H,W)   uint8 labels (0/1 or 0..K)
    nodes_xy: (N,2) float32 node coords in pixels (same canvas as image)
    edges_idx: (E,2) int64 edges (indices into nodes_xy)
    """
    x1, y1 = x0 + patch, y0 + patch

    # crop image + labels
    img_p = img_np[y0:y1, x0:x1, :]
    seg_p = seg_np[y0:y1, x0:x1]

    
    # keep only nodes inside the crop (inclusive on left/top, exclusive on right/bottom)
    in_box = (nodes_xy[:,0] >= x0) & (nodes_xy[:,0] < x1) & \
             (nodes_xy[:,1] >= y0) & (nodes_xy[:,1] < y1)
    keep_idx = np.where(in_box)[0]

    # translate nodes into the patch frame
    nodes_local = nodes_xy[keep_idx].copy()
    nodes_local[:,0] -= x0
    nodes_local[:,1] -= y0

    # remap edges so they point to the kept nodes only
    old2new = -np.ones(nodes_xy.shape[0], dtype=np.int64)
    old2new[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)
    e0 = old2new[edges_idx[:,0]]
    e1 = old2new[edges_idx[:,1]]
    mask = (e0 >= 0) & (e1 >= 0)
    edges_local = np.stack([e0[mask], e1[mask]], axis=1) if np.any(mask) else np.zeros((0,2), np.int64)

    return img_p, seg_p, nodes_local, edges_local


            
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

def save_patch_graph(nodes_p, edges_p, S, out_path):
    """
    Save graph in *dataset-compliant* format:
      - adjacency dict keyed by integer (y, x) tuples
      - patch-local pixel coordinates (0..S-1)
      - undirected, no self-loops, no duplicates
    """

    # round to integer pixel coords and clip to patch bounds
    nodes_int = np.rint(nodes_p).astype(np.int64)   # nodes_p is (x, y) in patch coords
    nodes_int[:, 0] = np.clip(nodes_int[:, 0], 0, S - 1)  # x
    nodes_int[:, 1] = np.clip(nodes_int[:, 1], 0, S - 1)  # y

    # dataset uses (y, x) ordering (row, col)
    coords_yx = [(int(y), int(x)) for (x, y) in nodes_int]

    # adjacency with sets (avoid duplicates); include isolated nodes
    adj = {c: set() for c in coords_yx}

    E = np.asarray(edges_p, dtype=np.int64)
    if E.size:
        E = E.reshape(-1, 2)
        for u, v in E:
            a = coords_yx[int(u)]
            b = coords_yx[int(v)]
            if a == b:        # skip self-loops from rounding collapse
                continue
            adj[a].add(b)
            adj[b].add(a)

    # convert sets -> sorted lists (stable & compact), keys are plain Python ints
    graph_dict = { (int(y), int(x)): sorted(list(neigh))
                   for (y, x), neigh in adj.items() }

    # NOTE: if you later want FULL-IMAGE coords, add (y0, x0) offsets here before building adj.
    with open(out_path, "wb") as f:
        pickle.dump(graph_dict, f, protocol=4)
            
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
        
def sweep_best_patch(img_np, seg_np, nodes_xy, edges_ix, S, stride=32, crop_fn=crop_with_clipping):
    H, W = seg_np.shape
    best = None   # (score, x0,y0, img_p, seg_p, nodes_p, edges_p)

    def score(nodes_p, edges_p, seg_p):
        # priority: any edge > any node > foreground pixels
        has_edge = int(edges_p.shape[0] > 0)
        has_node = int(nodes_p.shape[0] > 0)
        fg = int(np.count_nonzero(seg_p))
        # lexicographic via weighted sum
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
        
def sample_patch(img_np, seg_np, nodes_xy, edges_ix, S, min_fg_pixels=0,
                 require_graph=False, max_tries=2000, stem=None, rng=None):
    
    H, W = seg_np.shape
    tries = 0
    
    assert rng is not None
    
    while True:
        
        x0 = int(rng.integers(0, W - S + 1))
        y0 = int(rng.integers(0, H - S + 1))
            

        img_p, seg_p, nodes_p, edges_p = crop_with_clipping(
            img_np, seg_np, nodes_xy, edges_ix, x0, y0, S
        )

        # robust foreground count (works for binary or multi-class labels)
        # if your background is 0, this is safe:
        fg = int(np.count_nonzero(seg_p))

        ok_fg = (min_fg_pixels <= 0) or (fg >= min_fg_pixels)
        ok_graph = (not require_graph) or (nodes_p.shape[0] > 0 or edges_p.shape[0] > 0)

        if ok_fg and ok_graph:
            # print(f'sample {stem}: tries: {tries}')
            return x0, y0, img_p, seg_p, nodes_p, edges_p

        tries += 1
        
        
        
        if tries >= max_tries:
            # deterministic sweep fallback: guarantees a patch with a node/edge if one exists
            # tip: if you have `stem` or `image_key`, log it here
            # print(f"[sample] random failed after {tries} tries → sweeping…")
            return sweep_best_patch(
                img_np, seg_np, nodes_xy, edges_ix, S,
                stride=max(8, S//4),         # e.g., 32 when S=128
                crop_fn=crop_with_clipping   # ensure clipping is used
            )

# ---------------------------
# Main preprocessing
# ---------------------------

def main(args):
    
    if args.total_patches is not None and args.patches_per_image is not None:
        raise ValueError("Use either --total_patches or --patches_per_image, not both.")

    random.seed(args.seed); np.random.seed(args.seed)
    
    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    raw_dir = Path(args.root) / "raw"
    vtp_dir = Path(args.root) / "vtp"
    out_vtp = Path(args.out_root) / "vtp"
    output_seg_and_img = Path(args.out_root) / "raw"
    for d in (output_seg_and_img, out_vtp):
        d.mkdir(parents=True, exist_ok=True)

    # find regions via *_sat.png
    sats = sorted(raw_dir.glob("region_*_sat.png"))
    n_imgs = len(sats)
    
    if args.total_patches is not None:
        base = args.total_patches // n_imgs
        extra = args.total_patches % n_imgs      # e.g., 20 when 99,200 over 180
        # deterministic order: shuffle if you want fairness across runs
        per_img = [base + 1]*extra + [base]*(n_imgs - extra)
    else:
        per_img = [args.patches_per_image]*n_imgs

    if not sats:
        print(f"[error] no satellite images like region_*_sat.png found in {raw_dir}")
        return

    # Pillow resample compat
    try:
        RESIZE_BILINEAR = Image.Resampling.BILINEAR
        RESIZE_NEAREST  = Image.Resampling.NEAREST
    except AttributeError:
        RESIZE_BILINEAR = Image.BILINEAR
        RESIZE_NEAREST  = Image.NEAREST

    written = 0
    for si, sat_path in enumerate(tqdm(sats, desc="Regions", position=0)):
        stem = sat_path.stem.replace("_sat", "")
        gt_path  = raw_dir / f"{stem}_gt.png"
        pkl_path = vtp_dir / f"{stem}_gt_graph.pickle"

        if not gt_path.exists() or not pkl_path.exists():
            print(f"[skip] missing gt/pickle for {stem}")
            continue

        # load images
        img = Image.open(sat_path).convert("RGB")
        seg = Image.open(gt_path).convert("L")  # binary or multi-class mask

        # ensure same size
        if seg.size != img.size:
            seg = seg.resize(img.size, resample=RESIZE_NEAREST)

        W, H = img.size
        img_np = np.array(img, dtype=np.uint8)
        seg_np = (np.array(seg, dtype=np.uint8) > 0).astype(np.uint8)  # binarize to {0,1}

        # load graph
        nodes_xy, edges_ix = load_graph_pickle_20cities(pkl_path)    
        
        def hit_rate(nodes_px, seg):
            H, W = seg.shape
            pts = np.round(nodes_px).astype(int)
            m = (pts[:,0] >= 0) & (pts[:,0] < W) & (pts[:,1] >= 0) & (pts[:,1] < H)
            pts = pts[m]
            if pts.size == 0: 
                return 0.0
            return float(seg[pts[:,1], pts[:,0]].mean())  # seg is 0/1
        
        nodes_base = nodes_xy
        nodes_swap = nodes_xy[:, [1, 0]]  # <-- swap (y,x) -> (x,y)

        hr_base = hit_rate(nodes_base, seg_np)
        hr_swap = hit_rate(nodes_swap, seg_np)
        # print(f"hit-rate base={hr_base:.3f}, swap={hr_swap:.3f}")

        nodes_xy = nodes_swap if hr_swap > hr_base else nodes_base
        
        mins = nodes_xy.min(axis=0)
        maxs = nodes_xy.max(axis=0)
        nodes = nodes_xy.copy().astype(np.float32)
        
        if 0.0 <= mins.min() and maxs.max() <= 1.2:
            print("[fix] nodes look normalized; scaling by (W,H)")
            nodes[:, 0] *= W
            nodes[:, 1] *= H

        # 3) Detect swapped axes by range mismatch
        if maxs[0] > W*1.05 and maxs[1] <= W*1.05 and maxs[0] <= H*1.05:
            print("[fix] nodes look like (y,x); swapping columns")
            nodes = nodes[:, [1, 0]]

        # 4) Detect need to flip Y (origin at bottom)
        if nodes[:,1].min() >= 0 and nodes[:,1].max() > H*0.9 and nodes[:,1].mean() > H/2:
            pass  # ranges alone are not decisive; use the hit-test below

        rng = make_rng(args.seed, image_key=sat_path.stem)
        
        # generate random patches
        for pi in tqdm(range(per_img[si]), desc=f"Patches ({si+1}/{len(sats)})", position=1, leave=False, unit="patch"):
            
            if W < args.patch_size or H < args.patch_size:
                continue
            
            # rng = make_rng(args.seed)
            
            x0, y0, img_p, seg_p, nodes_p, edges_p = sample_patch(
                img_np, seg_np, nodes_xy, edges_ix,
                S=args.patch_size,
                min_fg_pixels=args.min_fg_pixels,   # same meaning as before
                require_graph=True,                 # set False if you just want foreground
                max_tries=2000,
                stem=stem,
                rng=rng
            )

            # optional: simplify graph if present
            if args.simplify_theta is not None and edges_p.shape[0] > 0:
                nodes_p, edges_p, _ = simplify_degree2_nodes(
                    nodes_p, edges_p, theta_straight=float(args.simplify_theta)
                )

            # save
            patch_name = f"{stem}_{pi:06d}"

            Image.fromarray(img_p).save(output_seg_and_img / f"{patch_name}_sat.png")  

            # labels: store training mask (0/1) as 0..255 for visualization/train loaders
            seg_vis = (seg_p.astype(np.uint8) * 255) if seg_p.dtype != np.uint8 else (seg_p * 255)
            Image.fromarray(seg_vis, mode="L").save(output_seg_and_img / f"{patch_name}_gt.png")

            # aggiungere salvataggio grafo 
            save_patch_graph(nodes_p, edges_p, args.patch_size, out_vtp / f"{patch_name}_gt_graph.pickle")

            written += 1

    print(f"[done] wrote {written} patches to {args.out_root}")
    

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path/to/20cities")
    ap.add_argument("--out_root", required=True, help="output root for patches")
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--patches_per_image", type=int, default=None, help="random patches per region")
    ap.add_argument("--simplify_theta", type=float, default=160.0, help="remove degree-2 nodes with angle >= theta")
    ap.add_argument("--min_fg_pixels", type=int, default=50, help="reject random patch if seg foreground < this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total_patches", type=int, default=None, help="if set, distribute this total across images")
    
    args = ap.parse_args(['--root', '/data/scavone/20cities',
                          '--out_root', '/data/scavone/20cities/patches/test',
                          '--total_patches', '25000'])
    
    main(args)


