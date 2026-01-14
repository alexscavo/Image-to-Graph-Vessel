"""
3D Debug Visualizer that uses the *loaded segmentation tensor* (no file paths).

This version assumes:
  - You already have `segs` in your training loop (B, C, D, H, W) or (B, D, H, W).
  - Your GT and Predicted nodes are in the SAME voxel coordinate system as `segs`.
  - You don't need real-world (mm) coordinates; visualization is in voxel space.

It will:
  - Take one sample from the batch (by `batch_index`).
  - Extract the corresponding segmentation volume.
  - Run marching cubes on that volume to create a mesh.
  - Overlay:
        Left  : segmentation mesh + GT graph
        Right : segmentation mesh + Pred graph
  - Save an interactive Plotly HTML file.

Typical usage in your 3D trainer:

    from vis_debug_3d_from_seg import DebugVisualizer3D

    self.viz3d = DebugVisualizer3D(
        out_dir=out_dir,
        prob=config.display_prob,
        max_per_epoch=8,
        show_seg=True,
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda eng: self.viz3d.start_epoch())

    # inside _iteration:
    infer = relation_infer(...)
    pred_nodes_list = infer["nodes"]
    pred_edges_list = infer["edges"]

    if hasattr(self, "viz3d"):
        self.viz3d.maybe_save(
            segs=segs,                      # tensor (B, C, D, H, W) or (B, D, H, W)
            gt_nodes_list=nodes,            # list length B; each (Ni,3)
            gt_edges_list=edges,            # list length B; each (Ei,2)
            pred_nodes_list=pred_nodes_list,
            pred_edges_list=pred_edges_list,
            epoch=epoch,
            step=iteration,
            batch_index=0,
            tag="train",
        )

"""

import os
import random

import numpy as np
from skimage import measure

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline


# -----------------------------------------------------------------------------
# Mesh from segmentation tensor
# -----------------------------------------------------------------------------

def seg_tensor_to_mesh(seg_volume: np.ndarray, level: float = 0.5):
    """
    Convert a 3D segmentation volume to a surface mesh using marching cubes.

    seg_volume: 3D numpy array, e.g. (D, H, W)
                Expected to be binary or probability-like.

    Returns:
        verts: (N, 3) vertices in voxel coordinates
        faces: (F, 3) faces as indices into verts
    """
    # Make sure it's 3D
    if seg_volume.ndim != 3:
        raise ValueError(f"seg_volume must be 3D, got shape {seg_volume.shape}")

    # marching_cubes expects float32
    vol = (seg_volume > 0).astype(np.float32)   # binarize too

    # If it's not binary, threshold implicitly via `level`
    verts, faces, normals, values = measure.marching_cubes(vol, level=level)
    return verts, faces


# -----------------------------------------------------------------------------
# Plotly helpers: build side-by-side figure in voxel space
# -----------------------------------------------------------------------------

def make_side_by_side_figure_from_nodes(
    D, H, W,
    seg_mesh,
    img_volume,
    gt_nodes: np.ndarray,
    gt_edges: np.ndarray,
    pred_nodes: np.ndarray,
    pred_edges: np.ndarray,
    title: str = "",
    show_seg: bool = True,
):
    """
    Create a Plotly Figure with two 3D scenes:
      - Left  scene: segmentation mesh + GT graph
      - Right scene: segmentation mesh + Pred graph

    seg_mesh:
      None OR (verts, faces), where verts is (Nv,3), faces is (F,3).
    *_nodes:
      (N,3) arrays in voxel coordinates.
    *_edges:
      (E,2) integer arrays of node indices.
    """
    def add_image_volume(row, col):
        if img_volume is None:
            return
        
        v = img_volume.astype(np.float32)
        v_min, v_max = float(v.min()), float(v.max())
        thresh = v_min + 0.05 * (v_max - v_min)
        
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]

        fig.add_trace(
            go.Volume(
                x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
                value=v.flatten(),
                isomin=thresh,       # ignore low / background values
                isomax=v_max,
                opacity=0.08,
                surface_count=6,
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                name="image",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        
    def get_legend_ref(col):
        return "legend" if col == 1 else "legend2"
    
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("GT graph", "Predicted graph"),
    )

    def add_seg_mesh(row, col):
        if seg_mesh is None or not show_seg:
            return
        verts, faces = seg_mesh
        x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]  # optional reordering
        # x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]  # swapped
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color="lightgray",
                opacity=0.20,
                name="segmentation",
                visible=True,
                showlegend=True,
                hoverinfo="skip",
                legend=get_legend_ref(col)
            ),
            row=row,
            col=col,
        )

    def add_graph(nodes, edges, color, label, row, col):
        if nodes is None or nodes.size == 0:
            return

        # nodes: assume (N,3) as (z, y, x) or (x, y, z); here we treat them as (z,y,x)
        xs = nodes[:, 2]
        ys = nodes[:, 1]
        zs = nodes[:, 0]

        # scatter nodes
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(size=3, color=color),
                name=f"nodes: {label}",
                hoverinfo="none",
                legend = get_legend_ref(col)
            ),
            row=row,
            col=col,
        )

        # scatter edges
        if edges is not None and edges.size > 0:
            edges = np.asarray(edges, dtype=np.int64)
            edge_x, edge_y, edge_z = [], [], []
            for e0, e1 in edges:
                x0, y0, z0 = xs[e0], ys[e0], zs[e0]
                x1, y1, z1 = xs[e1], ys[e1], zs[e1]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(width=2, color=color),
                    name=f"edges: {label}",
                    hoverinfo="skip",
                    legend = get_legend_ref(col)
                ),
                row=row,
                col=col,
            )

    # Left: seg + GT
    # add_image_volume(row=1, col=1)      # <--- NEW
    add_seg_mesh(row=1, col=1)
    add_graph(gt_nodes, gt_edges, "red", "GT", row=1, col=1)

    # Right: seg + Pred
    add_seg_mesh(row=1, col=2)
    add_graph(pred_nodes, pred_edges, "blue", "Pred", row=1, col=2)
    
    

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(range=[0, W-1]),
            yaxis=dict(range=[0, H-1]),
            zaxis=dict(range=[0, D-1]),
        ),
        scene2=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(range=[0, W-1]),
            yaxis=dict(range=[0, H-1]),
            zaxis=dict(range=[0, D-1]),
        ),
        legend=dict(
            x=0.01, y=0.99, 
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.5)"
        ),
        # Legend 2: Positioned Top-Middle (for Pred)
        legend2=dict(
            x=0.51, y=0.99, 
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.5)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


# -----------------------------------------------------------------------------
# High-level: seg tensor + nodes/edges -> HTML
# -----------------------------------------------------------------------------

def visualize_sample_3d_from_seg_and_graphs(
    seg_volume: np.ndarray,
    img_volume: np.ndarray,
    gt_nodes: np.ndarray,
    gt_edges: np.ndarray,
    pred_nodes: np.ndarray,
    pred_edges: np.ndarray,
    out_html: str,
    show_seg: bool = True,
    level: float = 0.5,
):
    """
    Given a single 3D segmentation volume and GT/Pred graphs, save a
    browser-viewable HTML file with side-by-side 3D scenes.

    seg_volume: (D, H, W) numpy array (voxel space).
    gt_nodes, pred_nodes: (N,3) arrays in same voxel space.
    gt_edges, pred_edges: (E,2) arrays of node indices.
    """
    D, H, W = seg_volume.shape
    
    seg_mesh = None
    try:
        verts, faces = seg_tensor_to_mesh(seg_volume, level=level)
        seg_mesh = (verts, faces)
    except Exception as e:
        print(f"[WARNING] Failed to create seg mesh from tensor: {e}")

    # ---- NEW: rescale normalized coords to voxel indices ----
    D, H, W = seg_volume.shape

    def norm_to_vox(nodes):
        # nodes ~ [0,1]; map to 0..(size-1)
        n = nodes.copy()
        # assume n[:,0], n[:,1], n[:,2] correspond to (z,y,x) or similar;
        # since D,H,W are equal or similar, exact order won't visually break much.
        n_vox = np.zeros_like(n)
        n_vox[:, 0] = n[:, 0] * (D - 1)
        n_vox[:, 1] = n[:, 1] * (H - 1)
        n_vox[:, 2] = n[:, 2] * (W - 1)
        return n_vox

    gt_nodes_vox   = norm_to_vox(gt_nodes)
    pred_nodes_vox = norm_to_vox(pred_nodes)
    # ---------------------------------------------



    fig = make_side_by_side_figure_from_nodes(
        D, H, W,
        seg_mesh,
        img_volume,
        gt_nodes_vox,
        gt_edges,
        pred_nodes_vox,
        pred_edges,
        title="GT vs Pred graphs + seg (voxel space)",
        show_seg=show_seg,
    )

    out_dir = os.path.dirname(out_html)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plot_offline(fig, filename=str(out_html), auto_open=False)
    print(f"Saved interactive HTML to {out_html}")


# -----------------------------------------------------------------------------
# Trainer-friendly wrapper (DebugVisualizer3D)
# -----------------------------------------------------------------------------

class DebugVisualizer3D:
    """
    3D visualizer for the training loop, producing an HTML file with
    two 3D panels: (seg + GT graph) vs (seg + Pred graph) in voxel space.

    Usage (inside your trainer):

        self.viz3d = DebugVisualizer3D(
            out_dir=out_dir,
            prob=config.display_prob,
            max_per_epoch=8,
            show_seg=True,
        )
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda eng: self.viz3d.start_epoch())```

        ...

        if hasattr(self, "viz3d"):
            self.viz3d.maybe_save(
                segs=segs,
                gt_nodes_list=nodes,
                gt_edges_list=edges,
                pred_nodes_list=pred_nodes_list,
                pred_edges_list=pred_edges_list,
                epoch=epoch,
                step=iteration,
                batch_index=0,
                tag="train",
            )
    """

    def __init__(
        self,
        out_dir,
        prob=0.001,
        max_per_epoch=8,
        show_seg=True,
        html_subdir="html_debug",
        level=0.5,
    ):
        self.out_dir = out_dir
        self.prob = float(prob)
        self.max_per_epoch = int(max_per_epoch)
        self.show_seg = bool(show_seg)
        self.html_subdir = html_subdir
        self.level = float(level)

        self._emitted_in_epoch = 0

        # base debug directory
        os.makedirs(self.out_dir, exist_ok=True)
        # subdir for HTMLs
        self.html_dir = os.path.join(self.out_dir, self.html_subdir)
        os.makedirs(self.html_dir, exist_ok=True)

    def start_epoch(self):
        """Call at EPOCH_STARTED."""
        self._emitted_in_epoch = 0

    def maybe_save(
        self,
        segs,
        images,
        gt_nodes_list,
        gt_edges_list,
        pred_nodes_list,
        pred_edges_list,
        epoch,
        step,
        batch_index=0,
        tag="train",
        pred_seg=None,
    ):
        """
        segs:  torch.Tensor or np.ndarray with shape (B, C, D, H, W) or (B, D, H, W).
        *_nodes_list: list length B, each (Ni,3) tensor/array in voxel coords.
        *_edges_list: list length B, each (Ei,2) tensor/array of node indices.
        """
        # budget per epoch
        if self._emitted_in_epoch >= self.max_per_epoch:
            return
        # probabilistic trigger
        if random.random() >= self.prob:
            return

        # Select batch index
        try:
            import torch
            is_torch = isinstance(segs, torch.Tensor)
        except ImportError:
            is_torch = False

        if is_torch:
            if batch_index >= segs.size(0):
                print(f"[viz3d] batch_index {batch_index} out of range for segs (B={segs.size(0)})")
                return
            seg_sample = segs[batch_index].detach().cpu().numpy()
        else:
            segs_np = np.asarray(segs)
            if batch_index >= segs_np.shape[0]:
                print(f"[viz3d] batch_index {batch_index} out of range for segs (B={segs_np.shape[0]})")
                return
            seg_sample = segs_np[batch_index]

        # Drop channel dimension if present
        if seg_sample.ndim == 4:
            # assume (C, D, H, W) and take first channel
            seg_sample = seg_sample[0]
        elif seg_sample.ndim != 3:
            print(f"[viz3d] Unexpected seg_sample shape {seg_sample.shape}, expected (D,H,W) or (C,D,H,W)")
            return
        
        # --- image sample ---
        img_sample = None
        try:
            import torch
            is_torch_img = isinstance(images, torch.Tensor)
        except ImportError:
            is_torch_img = False

        if is_torch_img:
            if batch_index >= images.size(0):
                print(f"[viz3d] batch_index {batch_index} out of range for images (B={images.size(0)})")
                return
            # important: .detach().cpu().numpy()
            img_sample = images[batch_index].detach().cpu().numpy()
        else:
            images_np = np.asarray(images)
            if batch_index >= images_np.shape[0]:
                print(f"[viz3d] batch_index {batch_index} out of range for images (B={images_np.shape[0]})")
                return
            img_sample = images_np[batch_index]

        # drop channel dimension if present (C, D, H, W)
        if img_sample.ndim == 4:
            img_sample = img_sample[0]
        elif img_sample.ndim != 3:
            print(f"[viz3d] Unexpected img_sample shape {img_sample.shape}, expected (D,H,W) or (C,D,H,W)")
            img_sample = None




        # Extract per-sample GT / Pred
        try:
            gt_nodes = gt_nodes_list[batch_index]
            gt_edges = gt_edges_list[batch_index]
            pred_nodes = pred_nodes_list[batch_index]
            pred_edges = pred_edges_list[batch_index]
        except Exception as e:
            print(f"[viz3d] Failed to index nodes/edges for batch_index={batch_index}: {e}")
            return

        # Convert tensors to numpy
        def to_numpy(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
            except ImportError:
                pass
            return np.asarray(x)

        gt_nodes_np = to_numpy(gt_nodes)
        gt_edges_np = to_numpy(gt_edges)
        pred_nodes_np = to_numpy(pred_nodes)
        pred_edges_np = to_numpy(pred_edges)

        # Build a unique output HTML filename
        fname = f"{tag}_e{int(epoch):03d}_it{int(step):06d}_b{int(batch_index)}.html"
        out_html = os.path.join(self.html_dir, fname)

        try:
            visualize_sample_3d_from_seg_and_graphs(
                seg_volume=seg_sample,
                img_volume=img_sample,
                gt_nodes=gt_nodes_np,
                gt_edges=gt_edges_np,
                pred_nodes=pred_nodes_np,
                pred_edges=pred_edges_np,
                out_html=out_html,
                show_seg=self.show_seg,
                level=self.level,
            )
            print(f"[viz3d] Saved HTML debug view to {out_html}")
            self._emitted_in_epoch += 1
        except Exception as e:
            print(f"[viz3d] Failed to create 3D visualization from seg tensor: {e}")
