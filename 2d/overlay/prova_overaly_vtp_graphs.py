from PIL import Image, ImageDraw
import numpy as np
import os
import pyvista as pv

"""
Visualizer for satellite + GT + VTP graph files.

This is a VTP-based version of `prova_overaly_pickle_graphs.py`.
It loads a graph stored as a VTK PolyData (.vtp), extracts:
  - nodes_xy: (N, 2) array of (x, y) pixel coordinates
  - edges_ix: (M, 2) array of indices into nodes_xy
and overlays the graph on top of satellite and GT images.

You only need to edit the three paths below:
  sat_path, gt_path, graph_path
"""

# ---------- paths (EDIT THESE) ----------
sat_path   = '/data/scavone/20cities/patches/train/raw/sample_000044_data.png'
gt_path    = '/data/scavone/20cities/patches/train/seg/sample_000044_seg.png'
graph_path = '/data/scavone/20cities/patches/train/vtp/sample_000044_graph.vtp'

out_dir    = "/data/scavone/prova_vtp_overlay"
os.makedirs(out_dir, exist_ok=True)

# ---------- overlay params ----------
mask_color = (255, 0, 0)         # red overlay for GT
mask_opacity = 0.6               # 0..1 opacity of GT on satellite
satellite_opacity = 0.5          # 0..1 opacity of satellite on GT
edge_color_rgba = (0, 255, 0, 220)      # graph edges in green
node_color_rgba = (255, 255, 0, 220)    # graph nodes in yellow
edge_width = 2
node_radius = 2


# ---------- helpers ----------
def load_graph_vtp(vtp_path, img_size=None):
    """
    Load graph from a .vtp PolyData file.

    If img_size is given, assumes points are in [0,1] normalized coords
    and rescales them to pixel coords.

    Args:
        vtp_path: path to .vtp file
        img_size: (W, H) of the image/patch, e.g. (128, 128)

    Returns:
        nodes_xy: (N, 2) float32 array of node positions in image pixel coords (x, y)
        edges_ix: (M, 2) int32 array of edges as indices into nodes_xy
    """
    mesh = pv.read(vtp_path)

    # Extract node positions (normalized [0,1] from your patch generator)
    pts = np.asarray(mesh.points, dtype=np.float32)
    nodes_xy = pts[:, :2]

    # If image size is given, rescale [0,1] → [0,W]x[0,H]
    if img_size is not None:
        W, H = img_size
        nodes_xy[:, 0] *= W
        nodes_xy[:, 1] *= H

    # Extract edges from VTK lines
    if mesh.lines.size == 0:
        edges_ix = np.zeros((0, 2), dtype=np.int32)
        return nodes_xy, edges_ix

    # lines are flattened: [npts, i0, i1, npts, i2, i3, ...]
    lines = mesh.lines.reshape(-1, 3)  # (num_lines, 3) where first col is npts (=2)
    if not np.all(lines[:, 0] == 2):
        raise ValueError("Expected only 2-point line cells in VTP mesh.lines.")

    edges_ix = lines[:, 1:].astype(np.int32)

    return nodes_xy, edges_ix


def align_nodes_to_seg(nodes_xy, seg_np, verbose=True):
    """
    Choose between (x,y) and (y,x) by maximizing overlap with positive GT pixels.
    seg_np must be 0/1 (will binarize if 0/255).
    """
    nodes_xy = np.asarray(nodes_xy, dtype=np.float32)
    H, W = seg_np.shape
    s = seg_np
    if s.max() > 1:
        s = (s > 0).astype(np.uint8)

    def hit_rate(nodes):
        pts = np.rint(nodes).astype(int)
        m = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        pts = pts[m]
        if pts.size == 0:
            return 0.0
        return float(s[pts[:, 1], pts[:, 0]].mean())

    base = nodes_xy
    swap = nodes_xy[:, [1, 0]]
    hr_base, hr_swap = hit_rate(base), hit_rate(swap)
    if verbose:
        print(f"hit-rate base={hr_base:.3f}, swap={hr_swap:.3f}")
        best = swap if hr_swap > hr_base else base
        mins, maxs = best.min(0), best.max(0)
        print("nodes bounds (min,max):", mins, maxs)
    return swap if hr_swap > hr_base else base


def draw_graph_layer(size, nodes_xy, edges_ix,
                     edge_color=edge_color_rgba, node_color=node_color_rgba,
                     edge_w=edge_width, node_r=node_radius):
    """
    Returns an RGBA image with the graph rendered (edges then nodes).
    """
    W, H = size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    # sub-sample if huge for speed
    e_step = 1
    n_step = 1
    if len(edges_ix) > 20000:
        e_step = max(1, len(edges_ix) // 20000)
    if len(nodes_xy) > 10000:
        n_step = max(1, len(nodes_xy) // 10000)

    # draw edges
    for (u, v) in edges_ix[::e_step]:
        x1, y1 = nodes_xy[u]
        x2, y2 = nodes_xy[v]
        draw.line(((x1, y1), (x2, y2)), fill=edge_color, width=edge_w)

    # draw nodes
    r = node_r
    for (x, y) in nodes_xy[::n_step]:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=node_color, width=1)

    return layer


def main():
    # ---------- load imagery ----------
    sat = Image.open(sat_path).convert("RGB")
    gt  = Image.open(gt_path)

    # ensure same size (resize GT to SAT if needed)
    if gt.size != sat.size:
        gt = gt.resize(sat.size, Image.NEAREST)

    # GT to single-channel 0..255
    if gt.mode != "L":
        gt = gt.convert("L")
    mask_np = np.array(gt)
    if mask_np.max() == 1:
        mask_np = (mask_np * 255).astype(np.uint8)
    gt = Image.fromarray(mask_np, mode="L")

    # ---------- base overlays (no graph) ----------
    # A) GT over Satellite
    alpha_from_mask = (mask_np.astype(np.float32) * mask_opacity).clip(0, 255).astype(np.uint8)
    overlay = Image.new("RGBA", sat.size, (*mask_color, 0))
    overlay.putalpha(Image.fromarray(alpha_from_mask, mode="L"))
    sat_rgba = sat.convert("RGBA")
    sat_with_mask = Image.alpha_composite(sat_rgba, overlay)

    # B) Satellite over GT
    gt_rgb  = gt.convert("RGB")
    gt_rgba = gt_rgb.convert("RGBA")
    sat_rgba2 = sat.convert("RGBA")
    sat_rgba2.putalpha(int(round(255 * satellite_opacity)))
    sat_on_gt = Image.alpha_composite(gt_rgba, sat_rgba2)

    # ---------- load & align graph from VTP ----------
    nodes_xy, edges_ix = load_graph_vtp(graph_path, img_size=sat.size)
    print('Loaded graph:')
    print('  nodes shape:', nodes_xy.shape)
    print('  edges shape:', edges_ix.shape)
    nodes_xy = align_nodes_to_seg(nodes_xy, (mask_np > 0).astype(np.uint8), verbose=True)

    # ---------- draw graph on top ----------
    graph_layer = draw_graph_layer(sat.size, nodes_xy, edges_ix)

    # sat + gt + graph
    sat_gt_graph = Image.alpha_composite(sat_with_mask, graph_layer)
    sat_gt_graph.save(os.path.join(out_dir, "overlay_gt_on_sat_graph.png"))

    # gt + semi-sat + graph
    gt_sat_graph = Image.alpha_composite(sat_on_gt, graph_layer)
    gt_sat_graph.save(os.path.join(out_dir, "overlay_sat_on_gt_graph.png"))

    # (optional) also save graph on plain sat / plain gt for inspection
    sat_graph_only = Image.alpha_composite(sat.convert("RGBA"), graph_layer)
    sat_graph_only.save(os.path.join(out_dir, "graph_on_sat.png"))

    gt_graph_only = Image.alpha_composite(gt.convert("RGBA"), graph_layer)
    gt_graph_only.save(os.path.join(out_dir, "graph_on_gt.png"))

    print("Saved overlays to:", out_dir)


if __name__ == '__main__':
    main()
