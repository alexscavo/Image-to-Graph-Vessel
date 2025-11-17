
"""
viz_nodes_edges.py — Reusable node/edge visualizer (PIL-based)
(Fix) Correct pruning stats:
  - nodes_before_prune = unique snapped coords encountered (pre-prune view)
  - nodes_after_prune  = actual node count used
  - merged_nodes       = nodes_before_prune - nodes_after_prune (never negative)
"""
from __future__ import annotations

import os, math, csv, argparse, json, colorsys
from typing import Tuple, Optional, Dict, List
import numpy as np
from PIL import Image, ImageDraw


# -----------------------------
# CSV loader (normalized coords)
# -----------------------------
def _parse_xyz(txt: str) -> Tuple[float, float]:
    arr = np.fromstring(txt.strip().strip('[]'), sep=' ', dtype=np.float32)
    if arr.size < 2:
        raise ValueError(f"Bad node format: {txt!r}")
    return float(arr[0]), float(arr[1])  # ignore z


class _SpatialHash:
    """Simple grid-based spatial hash for proximity queries in 2D."""
    def __init__(self, cell_size: float):
        self.s = max(1e-6, float(cell_size))
        self.grid: Dict[Tuple[int,int], List[int]] = {}

    def _key(self, x: float, y: float) -> Tuple[int,int]:
        return int(x // self.s), int(y // self.s)

    def insert(self, x: float, y: float, idx: int):
        k = self._key(x, y)
        self.grid.setdefault(k, []).append(idx)

    def nearby(self, x: float, y: float) -> List[int]:
        gx, gy = self._key(x, y)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                lst = self.grid.get((gx+dx, gy+dy))
                if lst:
                    out.extend(lst)
        return out


def load_graph_csv_edges(csv_path: str, size: Tuple[int, int], *,
                         norm_coords: bool = True,
                         swap_xy: bool = False,
                         flip_y: bool = False,
                         min_width_px: int = 1,
                         max_width_px: int = 6,
                         radius_scale: float = 1.0,
                         dedup_to_pixel: bool = False,
                         min_node_sep_px: float = 0.0):
    """
    Load CSV rows: node1,node2,radius  where node is \"[x y z]\".

    Parameters:
      dedup_to_pixel: if True, snap coordinates to integer pixels when indexing
      min_node_sep_px: if > 0, nodes closer than this distance (Euclidean, in pixels)
                       will be MERGED into an existing node (greedy, stream order).

    Returns:
      nodes_xy: (N,2) float32 in pixel coords (x,y)
      edges_ix: (M,2) int32 indices into nodes_xy
      edge_w:   (M,) float32 per-edge width in pixels (or None)
      stats:    dict with 'nodes_before_prune', 'nodes_after_prune', 'merged_nodes'
    """
    W, H = size
    coord2idx = {}
    coords = []
    edges = []
    widths = []

    # Track unique pre-prune snapped coordinates (independent of prune decisions)
    preprune_snap_keys = set()

    # For pruning-by-distance we need spatial hash + mapping
    use_prune = (min_node_sep_px is not None) and (float(min_node_sep_px) > 0.0)
    shash = _SpatialHash(min_node_sep_px if use_prune else 1.0)

    def snap(x, y):
        return (round(x), round(y)) if dedup_to_pixel else (round(x, 4), round(y, 4))

    def upsert_node(x: float, y: float):
        # Record pre-prune (snapped) identity for correct "before" count
        q_pre = snap(x, y)
        preprune_snap_keys.add(q_pre)

        # Optional distance-based merge first
        if use_prune and len(coords) > 0:
            cand_idx = shash.nearby(x, y)
            if cand_idx:
                xy = np.asarray(coords, dtype=np.float32)
                cx = xy[cand_idx, 0]; cy = xy[cand_idx, 1]
                dx = cx - x; dy = cy - y
                d2 = dx*dx + dy*dy
                j = int(np.argmin(d2))
                if float(d2[j]) <= float(min_node_sep_px)**2:
                    return cand_idx[j]

        # Snap-based de-dup after pruning (keeps behavior compatible)
        q = snap(x, y)
        u = coord2idx.get(q)
        if u is None:
            u = len(coords)
            coord2idx[q] = u
            coords.append((float(q[0]), float(q[1])))
            if use_prune:
                shash.insert(x, y, u)
        return u

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # If header doesn't look like a header, rewind
        if header and not any("node" in h.lower() for h in header):
            f.seek(0)
            reader = csv.reader(f)

        for row in reader:
            if not row or len(row) < 2:
                continue
            n1_txt, n2_txt = row[0], row[1]
            r_txt = row[2] if len(row) >= 3 else ""

            x1, y1 = _parse_xyz(n1_txt)
            x2, y2 = _parse_xyz(n2_txt)
            if swap_xy:
                x1, y1 = y1, x1
                x2, y2 = y2, x2

            if norm_coords:
                x1 *= W; y1 *= H
                x2 *= W; y2 *= H

            if flip_y:
                y1 = H - y1
                y2 = H - y2

            u = upsert_node(x1, y1)
            v = upsert_node(x2, y2)

            if u == v:
                continue  # skip zero-length edges after merge

            edges.append((u, v))

            # width from CSV radius if present, scaled & clamped
            if r_txt.strip():
                try:
                    w = float(r_txt) * radius_scale * max(W, H)
                    w = int(max(min_width_px, min(max_width_px, round(w))))
                except Exception:
                    w = min_width_px
            else:
                w = min_width_px
            widths.append(w)

    nodes_xy = np.asarray(coords, dtype=np.float32)
    edges_ix = np.asarray(edges, dtype=np.int32) if edges else np.zeros((0, 2), np.int32)
    edge_w = np.asarray(widths, dtype=np.float32) if widths else None

    nodes_before = int(len(preprune_snap_keys))
    nodes_after = int(len(nodes_xy))
    merged = max(0, nodes_before - nodes_after)

    stats = {
        "nodes_before_prune": nodes_before,
        "nodes_after_prune": nodes_after,
        "merged_nodes": merged,
        "min_node_sep_px": float(min_node_sep_px),
        "dedup_to_pixel": bool(dedup_to_pixel),
    }
    return nodes_xy, edges_ix, edge_w, stats


# -----------------------------
# Coloring helpers (unchanged)
# -----------------------------
def _distinct_hues(n: int, *, seed: Optional[int] = None) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    phi = (1 + 5 ** 0.5) / 2  # golden ratio
    step = 1.0 / phi
    base = 0.0 if seed is None else (hash(seed) % 1000) / 1000.0
    hues = (base + step * np.arange(n)) % 1.0
    return hues.astype(np.float32)


def _hsv_to_rgb_uint8(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def make_node_colors(nodes_xy: np.ndarray,
                     edges_ix: np.ndarray,
                     mode: str = "solid",
                     solid_rgb=(255, 255, 0),
                     alpha: int = 220,
                     seed: Optional[int] = None) -> np.ndarray:
    N = len(nodes_xy)
    if N == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    A = max(0, min(255, int(alpha)))
    out = np.empty((N, 4), dtype=np.uint8)

    if mode == "solid":
        out[:, 0:3] = np.array(solid_rgb, dtype=np.uint8)
        out[:, 3] = A
        return out

    if mode == "distinct":
        hues = _distinct_hues(N, seed=seed)
        for i, h in enumerate(hues):
            r, g, b = _hsv_to_rgb_uint8(h, 0.8, 0.95)
            out[i, 0:3] = (r, g, b)
            out[i, 3] = A
        return out

    if mode == "degree":
        deg = np.zeros((N,), dtype=np.int32)
        if len(edges_ix) > 0:
            np.add.at(deg, edges_ix[:, 0], 1)
            np.add.at(deg, edges_ix[:, 1], 1)
        dmin, dmax = int(deg.min()), int(deg.max()) if N > 0 else (0, 1)
        span = max(1, dmax - dmin)
        for i, d in enumerate(deg):
            t = (d - dmin) / span
            h = (2/3) * (1 - t)  # 0.666.. -> 0
            r, g, b = _hsv_to_rgb_uint8(h, 0.9, 1.0)
            out[i, 0:3] = (r, g, b)
            out[i, 3] = A
        return out

    out[:, 0:3] = np.array(solid_rgb, dtype=np.uint8)
    out[:, 3] = A
    return out


# -----------------------------
# Drawing (unchanged)
# -----------------------------
def draw_graph_layer(size: Tuple[int, int], nodes_xy, edges_ix, *,
                     edge_color=(0, 255, 0, 180),
                     node_color=(255, 255, 0, 220),
                     default_edge_w=2,
                     node_r=2,
                     edge_w_per_edge=None,
                     draw_nodes=True,
                     node_colors_rgba: Optional[np.ndarray] = None,
                     max_edges_to_draw=50000,
                     max_nodes_to_draw=50000) -> Image.Image:
    W, H = size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    E = len(edges_ix)
    N = len(nodes_xy)
    e_step = max(1, math.ceil(E / max(1, max_edges_to_draw)))
    n_step = max(1, math.ceil(N / max(1, max_nodes_to_draw)))

    # Draw edges
    for k, (u, v) in enumerate(edges_ix[::e_step]):
        x1, y1 = nodes_xy[u]
        x2, y2 = nodes_xy[v]
        w = default_edge_w
        if edge_w_per_edge is not None:
            idx = k * e_step
            if idx < len(edge_w_per_edge) and edge_w_per_edge[idx] > 0:
                w = int(edge_w_per_edge[idx])
        draw.line(((x1, y1), (x2, y2)), fill=edge_color, width=max(1, w))

    # Draw nodes
    if draw_nodes:
        r = max(1, int(node_r))
        use_per_node = node_colors_rgba is not None and len(node_colors_rgba) == N
        for i, (x, y) in enumerate(nodes_xy[::n_step]):
            if use_per_node:
                idx = i * n_step
                fill = tuple(int(v) for v in node_colors_rgba[idx])
            else:
                fill = node_color
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=None)

    return layer


# -----------------------------
# Simple overlay helper (unchanged)
# -----------------------------
def overlay_on_background(bg_img: Image.Image, graph_layer: Image.Image, out_path: str) -> None:
    bg_rgba = bg_img.convert("RGBA")
    composite = Image.alpha_composite(bg_rgba, graph_layer)
    composite.save(out_path)


# -----------------------------
# CLI (minimal edits to include corrected stats)
# -----------------------------
def _parse_size(s: str) -> Tuple[int, int]:
    if 'x' not in s:
        raise argparse.ArgumentTypeError("Size must be like 1024x768")
    w, h = s.split('x')
    return int(w), int(h)


def main():
    parser = argparse.ArgumentParser(description="Visualize CSV graph as RGBA layer, with optional background overlay.")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV with node1,node2[,radius] in \"[x y z]\" form')
    parser.add_argument('--bg', type=str, default='', help='Optional background image to overlay on')
    parser.add_argument('--size', type=_parse_size, default=None, help='Canvas size WxH if no background (e.g., 1024x768)')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--basename', type=str, default='viz')

    # Graph parsing flags
    parser.add_argument('--norm-coords', type=int, default=1, help='Treat CSV coords as normalized [0,1] (1/0)')
    parser.add_argument('--swap-xy', type=int, default=0, help='Swap x<->y from CSV (1/0)')
    parser.add_argument('--flip-y', type=int, default=0, help='Flip y from top-left to bottom-left origin (1/0)')

    # Width/radius & dedup/prune
    parser.add_argument('--min-width', type=int, default=1)
    parser.add_argument('--max-width', type=int, default=4)
    parser.add_argument('--radius-scale', type=float, default=1.0)
    parser.add_argument('--dedup-to-pixel', type=int, default=0)
    parser.add_argument('--min-node-sep', type=float, default=0.0, help='Merge nodes closer than this many pixels (greedy). 0=disabled')

    # Drawing
    parser.add_argument('--edge-a', type=int, default=180, help='Edge alpha 0-255')
    parser.add_argument('--edge-rgb', type=str, default='0,255,0', help='Edge RGB like \"0,255,0\"')
    parser.add_argument('--node-a', type=int, default=220, help='Node alpha 0-255')
    parser.add_argument('--node-rgb', type=str, default='255,255,0', help='Node RGB like \"255,255,0\"')
    parser.add_argument('--draw-nodes', type=int, default=1)
    parser.add_argument('--node-r', type=int, default=3)
    parser.add_argument('--default-edge-w', type=int, default=2)

    # Node coloring
    parser.add_argument('--node-coloring', type=str, choices=['solid','distinct','degree'], default='solid',
                        help='How to color nodes: solid (single color), distinct (unique hue per node), degree (by node degree)')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed to rotate hues when node-coloring=distinct')

    name = '20251028_230010_000'

    args = parser.parse_args([
        '--csv', f'/data/scavone/octa-synth-packed/patches/train/csv/{name}.csv',
        '--bg', f'/data/scavone/octa-synth-packed/patches/train/raw/{name}.png',
        '--out', '/data/scavone/prove octasynth/nuove',
        '--draw-nodes', '1', '--norm-coords', '1', '--swap-xy', '0', '--flip-y', '0',
        '--min-width', '1', '--max-width', '4', '--radius-scale', '1.0', '--node-r', '2',
        # '--node_coloring', 'distinct',
        '--min-node-sep', '6.0 '
    ])

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    if args.bg:
        bg = Image.open(args.bg).convert("RGB")
        size = bg.size
    else:
        if args.size is None:
            raise SystemExit("When --bg is not provided, you must pass --size WxH")
        size = args.size

    def _rgb_tuple(s):
        parts = [int(x) for x in s.split(',')]
        if len(parts) != 3:
            raise ValueError("--edge-rgb/--node-rgb must be R,G,B")
        return tuple(parts)

    edge_color = (*_rgb_tuple(args.edge_rgb), max(0, min(255, args.edge_a)))
    solid_node_rgb = _rgb_tuple(args.node_rgb)

    # Load
    nodes_xy, edges_ix, edge_w, stats = load_graph_csv_edges(
        args.csv, size=size,
        norm_coords=bool(args.norm_coords),
        swap_xy=bool(args.swap_xy),
        flip_y=bool(args.flip_y),
        min_width_px=args.min_width,
        max_width_px=args.max_width,
        radius_scale=args.radius_scale,
        dedup_to_pixel=bool(args.dedup_to_pixel),
        min_node_sep_px=float(args.min_node_sep),
    )

    # Node colors
    node_colors_rgba = None
    coloring_mode = args.node_coloring if hasattr(args, 'node_coloring') else 'solid'
    if coloring_mode != 'solid':
        node_colors_rgba = make_node_colors(
            nodes_xy, edges_ix, mode=coloring_mode,
            solid_rgb=solid_node_rgb, alpha=args.node_a, seed=args.seed
        )

    # Draw layer
    layer = draw_graph_layer(
        size, nodes_xy, edges_ix,
        edge_color=edge_color,
        node_color=(*solid_node_rgb, max(0, min(255, args.node_a))),
        default_edge_w=args.default_edge_w,
        node_r=args.node_r,
        edge_w_per_edge=edge_w,
        draw_nodes=bool(args.draw_nodes),
        node_colors_rgba=node_colors_rgba,
    )

    # Always save the transparent layer
    layer_path = os.path.join(out_dir, f"{args.basename}_graph_layer.png")
    layer.save(layer_path)

    # Optional composite with background
    composite_path = None
    if args.bg:
        composite_path = os.path.join(out_dir, f"{args.basename}_on_bg.png")
        overlay_on_background(Image.open(args.bg), layer, composite_path)

    # Report & persist counts and pruning stats
    N = int(len(nodes_xy))
    E = int(len(edges_ix))
    report = {
        "nodes": N, "edges": E,
        "nodes_before_prune": stats.get("nodes_before_prune", N),
        "nodes_after_prune": stats.get("nodes_after_prune", N),
        "merged_nodes": stats.get("merged_nodes", 0),
        "min_node_sep_px": stats.get("min_node_sep_px", 0.0),
        "dedup_to_pixel": stats.get("dedup_to_pixel", False),
        "coloring": coloring_mode,
        "layer_path": layer_path, "composite_path": composite_path
    }
    counts_path = os.path.join(out_dir, f"{args.basename}_counts.json")
    with open(counts_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved:")
    print(" -", layer_path)
    if composite_path:
        print(" -", composite_path)
    print(f"Nodes: {N} (before prune: {report['nodes_before_prune']}, merged: {report['merged_nodes']})")
    print(f"Edges: {E}")
    print("Node coloring:", report["coloring"])
    print("Counts JSON:", counts_path)


if __name__ == "__main__":
    main()
