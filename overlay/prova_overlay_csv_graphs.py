import sys
from PIL import Image, ImageDraw
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.csv_graph import csv_graph_loading
import numpy as np
import pickle, os, csv, math

# ---------- paths ----------
# raw_path   = '/data/scavone/octa-synth-packed/patches/train/raw/20251028_243530_005.png'
# label_path = '/data/scavone/octa-synth-packed/patches/train/labels/20251028_243530_005.png'
# graph_path = '/data/scavone/octa-synth-packed/patches/train/graphs/20251028_243530_005'  
# out_dir    = "/data/scavone/overlay_prova/octasynth_strana"

# raw_path   = '/data/scavone/octa500/raw/10003.bmp'
# label_path = '/data/scavone/octa500/labels/10003.bmp'
# graph_path = '/data/scavone/octa500/graphs/10003'  
# out_dir    = "/data/scavone/overlay_prova/octa500_prova"

raw_path   = '/data/scavone/octa500/patches/train/raw/10002_001.png'
label_path = '/data/scavone/octa500/patches/train/labels/10002_001.png'
graph_path = '/data/scavone/octa500/patches/train/graphs/10002_001'  
out_dir    = "/data/scavone/overlay_prova/octa500_prova_patch"

os.makedirs(out_dir, exist_ok=True)

# ---------- overlay params ----------
mask_color = (255, 0, 0)          # red overlay for GT
mask_opacity = 0.6
satellite_opacity = 0.5
edge_color_rgba = (0, 255, 0, 220)      # edges in green
node_color_rgba = (255, 255, 0, 220)    # nodes in yellow
edge_width = 4
node_radius = 3                          # bigger so dots pop over lines

# ---------- helpers ----------
def load_graph_pickle_generic(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        nodes = np.array(obj[0], dtype=np.float32)
        edges = np.array(obj[1], dtype=np.int32)
        return nodes, edges, None
    if isinstance(obj, dict) and "nodes" in obj and "edges" in obj:
        nodes = np.array(obj["nodes"], dtype=np.float32)
        edges = np.array(obj["edges"], dtype=np.int32)
        return nodes, edges, None
    if isinstance(obj, dict):
        keys = list(obj.keys())
        try:
            nodes = np.array(keys, dtype=np.float32)
        except Exception:
            nodes = np.array([tuple(k) for k in keys], dtype=np.float32)
        pos2idx = {tuple(p): i for i, p in enumerate(nodes)}
        edges = []
        for u_pos, nbrs in obj.items():
            u = pos2idx.get(tuple(u_pos))
            if u is None: continue
            for v_pos in nbrs:
                v = pos2idx.get(tuple(v_pos))
                if v is None: continue
                if u < v:
                    edges.append((u, v))
        edges = np.array(edges, dtype=np.int32) if edges else np.zeros((0,2), np.int32)
        return nodes, edges, None
    raise ValueError(f"Unrecognized graph pickle format at {pkl_path!r}")


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
    
    nodes[:, 0] *= W                      # x
    nodes[:, 1] *= H                      # y

    E = len(edges)
    N = len(nodes)

    # cap for sanity
    e_step = max(1, math.ceil(E / max(1, max_edges_to_draw)))
    n_step = max(1, math.ceil(N / max(1, max_nodes_to_draw)))

    # 1) draw edges (green lines)
    for k, (u, v) in enumerate(edges[::e_step]):
        x1, y1, z1 = nodes[u]
        x2, y2, z2 = nodes[v]
        w = default_edge_w
        if edge_w_per_edge is not None:
            idx = k * e_step
            if idx < len(edge_w_per_edge) and edge_w_per_edge[idx] > 0:
                w = int(edge_w_per_edge[idx])
        draw.line(((x1, y1), (x2, y2)), fill=edge_color, width=max(1, w))

    # 2) draw nodes (filled yellow dots) *on top* of edges
    if draw_nodes:
        r = max(1, int(node_r))
        for (x, y, z) in nodes[::n_step]:
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
    nodes, edges = csv_graph_loading(path=graph_path)   # load the graph from the 
    
    nodes_np = np.asarray(nodes.detach().cpu(), dtype=np.float32)
    edges_np = np.asarray(edges.detach().cpu(), dtype=np.int32)
    
    # nodes_np[:, [0, 1]] = nodes_np[:, [1, 0]]   # SWAP ONLY FOR VISUALIZATION FOR OCTA-SYNTH!!!!
    
    print('nodes:')
    print(nodes[:15])
    print('-'*50)
    print('edges:')
    print(edges[:15])
    
    
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
