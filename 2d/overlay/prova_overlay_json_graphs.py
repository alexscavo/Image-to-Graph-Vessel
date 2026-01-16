import sys
from PIL import Image, ImageDraw
import sys, os

from networkx import edges
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import load_graph_from_json
import numpy as np
import pickle, os, csv, math

# ---------- paths ----------
raw_path   = '/data/scavone/plants_3d2cut/train/raw/Set00_IMG_3283.jpg'
label_path = '/data/scavone/plants_3d2cut/train/seg/Set00_IMG_3283_seg.png'
graph_path = '/data/scavone/plants_3d2cut/train/graphs/Set00_IMG_3283_annotation.json'  
out_dir    = "/data/scavone/overlay_prova/plants"
 


os.makedirs(out_dir, exist_ok=True)

# ---------- overlay params ----------
mask_color = (255, 0, 0)          # red overlay for GT
mask_opacity = 0.6
satellite_opacity = 0.5
edge_color_rgba = (0, 255, 0, 220)      # edges in green
node_color_rgba = (255, 255, 0, 220)    # nodes in yellow
edge_width = 10
node_radius = 15                          # bigger so dots pop over lines

# ---------- helpers ----------
def draw_graph_layer(size, nodes, edges,
                     edge_color=(0, 255, 0, 180),
                     node_color=(255, 255, 0, 180),
                     default_edge_w=2,
                     node_r=3,                    # <- larger default
                     edge_w_per_edge=None,
                     draw_nodes=True,             # <- ON: draw yellow dots
                     max_edges_to_draw=5000,
                     max_nodes_to_draw=5000):
    W, H = size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    E = len(edges)
    N = len(nodes)

    # cap for sanity
    e_step = max(1, math.ceil(E / max(1, max_edges_to_draw)))
    n_step = max(1, math.ceil(N / max(1, max_nodes_to_draw)))

    # 1) draw edges (green lines)
    for k, (u, v) in enumerate(edges[::e_step]):
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        w = default_edge_w
        if edge_w_per_edge is not None:
            idx = k * e_step
            if idx < len(edge_w_per_edge) and edge_w_per_edge[idx] > 0:
                w = int(edge_w_per_edge[idx])
        draw.line(((x1, y1), (x2, y2)), fill=edge_color, width=max(1, w))

    # 2) draw nodes (filled yellow dots) *on top* of edges
    if draw_nodes:
        r = max(1, int(node_r))
        for (x, y) in nodes[::n_step]:
            bbox = (x - r, y - r, x + r, y + r)
            draw.ellipse(bbox, fill=node_color, outline=None)  # <- filled

    return layer

def main():
    # ---------- load imagery ----------
    sat = Image.open(raw_path).convert("RGB")
    gt  = Image.open(label_path)

    if gt.size != sat.size:
        gt = gt.resize(sat.size, Image.NEAREST)

    if gt.mode != "L":
        gt = gt.convert("L")
    mask_np = np.array(gt)
    if mask_np.max() == 1:
        mask_np = (mask_np * 255).astype(np.uint8)
    gt = Image.fromarray(mask_np, mode="L")

    # ---------- base overlays ----------
    alpha_from_mask = (mask_np.astype(np.float32) * mask_opacity).clip(0, 255).astype(np.uint8)
    overlay = Image.new("RGBA", sat.size, (*mask_color, 0))
    overlay.putalpha(Image.fromarray(alpha_from_mask, mode="L"))
    sat_rgba = sat.convert("RGBA")
    sat_with_mask = Image.alpha_composite(sat_rgba, overlay)

    gt_rgb  = gt.convert("RGB")
    gt_rgba = gt_rgb.convert("RGBA")
    sat_rgba2 = sat.convert("RGBA")
    sat_rgba2.putalpha(int(round(255 * satellite_opacity)))
    sat_on_gt = Image.alpha_composite(gt_rgba, sat_rgba2)

    # ---------- load graph (CSV) ----------    
    nodes, edges = load_graph_from_json(graph_path)   # load the graph from the 
    
    nodes_np = np.asarray(nodes, dtype=np.float32)
    edges_np = np.asarray(edges, dtype=np.int32)    
    
    edge_w = None

    # ---------- draw graph on top (lines + dots) ----------
    graph_layer = draw_graph_layer(
        sat.size, nodes_np, edges_np,
        edge_color=edge_color_rgba,
        node_color=node_color_rgba,
        default_edge_w=edge_width,
        node_r=node_radius,
        edge_w_per_edge=edge_w,
        draw_nodes=True,             # <— ON: show nodes as dots
        max_edges_to_draw=8000,
        max_nodes_to_draw=2000
    )

    # Compose and save
    sat_gt_graph = Image.alpha_composite(sat_with_mask, graph_layer)
    os.makedirs(out_dir, exist_ok=True)
    sat_gt_graph.save(os.path.join(out_dir, "overlay_gt_on_sat_graph.png"))

    gt_sat_graph = Image.alpha_composite(gt.convert("RGBA"), graph_layer)
    gt_sat_graph.save(os.path.join(out_dir, "overlay_sat_on_gt_graph.png"))

    sat_graph_only = Image.alpha_composite(sat.convert("RGBA"), graph_layer)
    sat_graph_only.save(os.path.join(out_dir, "graph_on_sat.png"))

    gt_graph_only = Image.alpha_composite(gt.convert("RGBA"), graph_layer)
    gt_graph_only.save(os.path.join(out_dir, "graph_on_gt.png"))

    print("Saved overlays to:", out_dir)

if __name__ == '__main__':
    main()
