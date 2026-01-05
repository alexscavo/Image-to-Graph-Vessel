# training/vis_debug.py
import os, random
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from metrics.metric_smd import get_point_cloud, SinkhornDistance

# ---------------- viz-only coordinate options ----------------
# Leave your dataset/model coordinates untouched.
# Apply these ONLY when turning points into pixels for display.
SWAP_XY_FOR_VIZ = True    # <- set True to match your overlay behavior
FLIP_Y_FOR_VIZ  = False   # <- set True if your Y is "up" and you want image coords "down"

# ---------- utilities (aligned with your working script) ----------
def denorm_img(t_img, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    """t_img: (C,H,W) torch tensor in normalized space -> numpy (H,W,C) in [0,1]."""
    img = t_img.detach().cpu()
    if img.shape[0] == 3:
        mean = torch.tensor(mean)[:, None, None]
        std  = torch.tensor(std)[:, None, None]
        img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        cmap = None
    else:
        # grayscale
        img = img[0].clamp(0, 1).numpy()
        cmap = "gray"
    return img, cmap

def _apply_viz_conventions(arr, H, W):
    """
    arr: np.float32 (N,2) in *pixel* units.
    Applies SWAP_XY_FOR_VIZ and FLIP_Y_FOR_VIZ (viz-only).
    """
    if arr.size == 0:
        return arr
    out = arr.copy()
    if SWAP_XY_FOR_VIZ:
        out[:, [0, 1]] = out[:, [1, 0]]
    if FLIP_Y_FOR_VIZ:
        out[:, 1] = H - out[:, 1]
    return out

def to_pixel_space(nodes, H, W):
    """
    nodes: (...,2) torch/np with (x,y) in either normalized [0,1] or pixels.
    Heuristic: if max <= 1.5 -> normalized; else assume already pixels.
    Returns np.float32 shape (N,2) in pixels, with viz-only conventions applied.
    """
    if nodes is None:
        return np.zeros((0,2), dtype=np.float32)
    arr = nodes.detach().cpu().numpy() if isinstance(nodes, torch.Tensor) else np.asarray(nodes)
    if arr.size == 0:
        return arr.reshape(0,2).astype(np.float32)

    arr = arr.astype(np.float32)
    mx = float(np.nanmax(arr))
    if mx <= 1.5:  # normalized coords
        arr[:, 0] *= W
        arr[:, 1] *= H

    # apply viz-only swaps/flips
    arr = _apply_viz_conventions(arr, H, W)
    return arr

def as_int_edges(edges):
    """edges: (E,2) -> numpy int64, gracefully handle empty/None."""
    if edges is None:
        return np.zeros((0,2), dtype=np.int64)
    E = edges.detach().cpu().numpy() if isinstance(edges, torch.Tensor) else np.asarray(edges)
    if E.size == 0:
        return E.reshape(0,2).astype(np.int64)
    return E.astype(np.int64)

'''def visualize_graph_batch(
    samples,
    n=6,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    point_size=10,
    linewidth=1.0,
    figsize_per_row=(6, 6),
    alpha_img=1.0,
):
    """
    Layout per sample (2 rows, 4 columns):
      Row 1: [ Raw Image | GT Graph Overlay | Pred Graph Overlay | Empty/Extra ]
      Row 2: [ GT Segmentation | Pred Segmentation | SMD Plot | Empty ]
    """

    imgs = samples["images"]
    B = len(imgs) if isinstance(imgs, (list, tuple)) else imgs.shape[0]
    n = min(n, B)

    # 2 rows per sample, 3 columns fixed
    rows = n * 2
    cols = 3
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_row[0] * cols, figsize_per_row[1] * rows),
    )

    if rows == 1:
        axs = axs[None, :]  # ensure [row, col] indexing

    # Sinkhorn for SMD
    sinkhorn = SinkhornDistance(eps=1e-7, max_iter=100, reduction="none")

    for i in range(n):
        row0 = 2 * i       # first row (image + graphs)
        row1 = 2 * i + 1   # second row (SMD plots)

        # --------------------------------------------------
        # 1) IMAGE + GRAPHS (pixel space, as before)
        # --------------------------------------------------
        im_t = imgs[i] if isinstance(imgs, (list, tuple)) else imgs[i]
        vis, cmap = denorm_img(im_t, mean=mean, std=std)
        H, W = vis.shape[:2]

        # GT / Pred in pixel space for overlay
        gt_nodes_pix = to_pixel_space(samples.get("nodes", [None])[i], H, W)
        gt_edges     = as_int_edges(samples.get("edges", [None])[i])
        pr_nodes_pix = to_pixel_space(samples.get("pred_nodes", [None])[i], H, W)
        pr_edges     = as_int_edges(samples.get("pred_edges", [None])[i])

        # Panel (row0, col0): image
        ax0 = axs[row0, 0]
        ax0.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        ax0.set_title(f"Image {i} (H={H}, W={W})")
        ax0.axis("off")
        

        # Panel (row0, col1): GT graph
        ax1 = axs[row0, 1]
        ax1.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        if gt_nodes_pix is not None and len(gt_nodes_pix) > 0:
            ax1.scatter(gt_nodes_pix[:, 0], gt_nodes_pix[:, 1], s=point_size)
            if gt_edges is not None and len(gt_edges) > 0:
                for (u, v) in gt_edges:
                    x0, y0 = gt_nodes_pix[u]
                    x1p, y1p = gt_nodes_pix[v]
                    ax1.plot([x0, x1p], [y0, y1p], linewidth=linewidth)
        ax1.set_title(f"GT: {len(gt_nodes_pix)} nodes, {len(gt_edges)} edges")
        ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
        ax1.axis("off")

        # Panel (row0, col2): Pred graph
        ax2 = axs[row0, 2]
        ax2.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        if pr_nodes_pix is not None and len(pr_nodes_pix) > 0:
            ax2.scatter(pr_nodes_pix[:, 0], pr_nodes_pix[:, 1], s=point_size)
            if pr_edges is not None and len(pr_edges) > 0:
                for (u, v) in pr_edges:
                    x0, y0 = pr_nodes_pix[u]
                    x1p, y1p = pr_nodes_pix[v]
                    ax2.plot([x0, x1p], [y0, y1p], linewidth=linewidth)
        ax2.set_title(f"Pred: {len(pr_nodes_pix)} nodes, {len(pr_edges)} edges")
        ax2.set_xlim(0, W); ax2.set_ylim(H, 0)
        ax2.axis("off")

        # --------------------------------------------------
        # 2) SMD PLOTS (normalized + pixel space)
        # --------------------------------------------------
        gt_nodes_norm = samples.get("nodes", [None])[i]
        pr_nodes_norm = samples.get("pred_nodes", [None])[i]
        
        # first column of row1 unused
        axs[row1, 0].axis("off")
        axs[row1, 2].axis("off")

        if gt_nodes_norm is None or pr_nodes_norm is None:
            # nothing to show
            axs[row1, 0].axis("off")
            axs[row1, 1].axis("off")
            axs[row1, 2].axis("off")
            continue

        # --- ensure torch tensors for SMD
        gt_nodes_norm_t = torch.as_tensor(gt_nodes_norm, dtype=torch.float32)
        pr_nodes_norm_t = torch.as_tensor(pr_nodes_norm, dtype=torch.float32)

        # edges as torch for adj construction
        if gt_edges is not None and len(gt_edges) > 0:
            gt_edges_t = torch.as_tensor(gt_edges, dtype=torch.long)
        else:
            gt_edges_t = torch.zeros((0, 2), dtype=torch.long)

        if pr_edges is not None and len(pr_edges) > 0:
            pr_edges_t = torch.as_tensor(pr_edges, dtype=torch.long)
        else:
            pr_edges_t = torch.zeros((0, 2), dtype=torch.long)

        # ---------- A) SMD on normalized coordinates ----------
        A_norm = torch.zeros((gt_nodes_norm_t.shape[0], gt_nodes_norm_t.shape[0]), dtype=torch.float32)
        if len(gt_edges_t) > 0:
            A_norm[gt_edges_t[:, 0], gt_edges_t[:, 1]] = 1.0

        pred_A_norm = torch.zeros((pr_nodes_norm_t.shape[0], pr_nodes_norm_t.shape[0]), dtype=torch.float32)
        if len(pr_edges_t) > 0:
            pred_A_norm[pr_edges_t[:, 0], pr_edges_t[:, 1]] = 1.0

        y_pc_norm      = get_point_cloud(A_norm.T,      gt_nodes_norm_t, n_points=100)
        output_pc_norm = get_point_cloud(pred_A_norm.T, pr_nodes_norm_t, n_points=100)

        smd_norm, _, _ = sinkhorn(y_pc_norm, output_pc_norm)

        ax_smd_norm = axs[row1, 1]
        ax_smd_norm.scatter(y_pc_norm[:, 0].cpu(),      y_pc_norm[:, 1].cpu(),
                            s=5, label="GT (norm)")
        ax_smd_norm.scatter(output_pc_norm[:, 0].cpu(), output_pc_norm[:, 1].cpu(),
                            s=5, label="Pred (norm)")
        
        # DEBUG
        num_edges = int(A_norm.sum().item())


        ax_smd_norm.set_title(f"SMD (norm) = {smd_norm.item():.4f}: {len(gt_nodes_norm_t)} nodes, {num_edges} edges")
        ax_smd_norm.set_xlim(0, 1)
        ax_smd_norm.set_ylim(0, 1)
        ax_smd_norm.legend(fontsize=7)
        ax_smd_norm.axis("off")



    plt.tight_layout()
    return fig

'''

def visualize_graph_batch(
    samples,
    n=1,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    point_size=10,
    linewidth=1.0,
    figsize_per_row=(6, 6),
    alpha_img=1.0,
):
    """
    Layout per sample:
      Row 1: [ Image | GT graph | Pred graph | Empty ]
      Row 2: [ GT Seg | Pred Seg | SMD Plot | Empty ]
    """
    imgs = samples["images"]
    B = len(imgs) if isinstance(imgs, (list, tuple)) else imgs.shape[0]
    n = min(n, B)

    rows = n * 2
    cols = 4  # Expanded for side-by-side masks
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_row[0] * cols, figsize_per_row[1] * rows),
    )

    if rows == 1: axs = axs[None, :]

    sinkhorn = SinkhornDistance(eps=1e-7, max_iter=100, reduction="none")

    for i in range(n):
        row0 = 2 * i       
        row1 = 2 * i + 1   

        im_t = imgs[i]
        vis, cmap = denorm_img(im_t, mean=mean, std=std)
        H, W = vis.shape[:2]

        gt_nodes_pix = to_pixel_space(samples.get("nodes", [None])[i], H, W)
        gt_edges     = as_int_edges(samples.get("edges", [None])[i])
        pr_nodes_pix = to_pixel_space(samples.get("pred_nodes", [None])[i], H, W)
        pr_edges     = as_int_edges(samples.get("pred_edges", [None])[i])
        
        gt_seg = samples.get("seg", [None])[i]
        pr_seg = samples.get("pred_seg", [None])[i]

        # --- ROW 1: Raw Image + Graphs ---
        axs[row0, 0].imshow(vis, cmap=cmap)
        axs[row0, 0].set_title("Raw Image")
        axs[row0, 0].axis("off")

        for ax, nodes, edges, title, color in zip(
            [axs[row0, 1], axs[row0, 2]], 
            [gt_nodes_pix, pr_nodes_pix], 
            [gt_edges, pr_edges], 
            ["GT Graph", "Pred Graph"], ['r', 'b']
        ):
            ax.imshow(vis, alpha=alpha_img, cmap=cmap)
            if nodes is not None and len(nodes) > 0:
                ax.scatter(nodes[:, 0], nodes[:, 1], s=point_size, c=color)
                for (u, v) in edges:
                    ax.plot([nodes[u, 0], nodes[v, 0]], [nodes[u, 1], nodes[v, 1]], c=color, lw=linewidth)
            ax.set_title(title)
            ax.axis("off")
        axs[row0, 3].axis("off")

        # --- ROW 2: Segmentation + SMD ---
        # 1. GT Segmentation
        if gt_seg is not None:
            # Squeeze to handle [1, H, W] or [H, W]
            axs[row1, 0].imshow(gt_seg.detach().cpu().squeeze().numpy(), cmap='gray')
            axs[row1, 0].set_title("GT Segmentation")
        axs[row1, 0].axis("off")

        # 2. Predicted Segmentation (Roads)
        if pr_seg is not None:
            # Apply Sigmoid + Threshold as these are raw logits
            prob = torch.sigmoid(pr_seg).detach().cpu().squeeze().numpy()
            axs[row1, 1].imshow(prob > 0.5, cmap='gray')
            axs[row1, 1].set_title("Pred Segmentation")
        axs[row1, 1].axis("off")

        # 3. SMD Plot (Existing logic)
        gt_nodes_norm = samples.get("nodes", [None])[i]
        pr_nodes_norm = samples.get("pred_nodes", [None])[i]
        
        # first column of row1 unused
        axs[row1, 0].axis("off")
        axs[row1, 2].axis("off")

        if gt_nodes_norm is None or pr_nodes_norm is None:
            # nothing to show
            axs[row1, 0].axis("off")
            axs[row1, 1].axis("off")
            axs[row1, 2].axis("off")
            continue

        # --- ensure torch tensors for SMD
        gt_nodes_norm_t = torch.as_tensor(gt_nodes_norm, dtype=torch.float32)
        pr_nodes_norm_t = torch.as_tensor(pr_nodes_norm, dtype=torch.float32)

        # edges as torch for adj construction
        if gt_edges is not None and len(gt_edges) > 0:
            gt_edges_t = torch.as_tensor(gt_edges, dtype=torch.long)
        else:
            gt_edges_t = torch.zeros((0, 2), dtype=torch.long)

        if pr_edges is not None and len(pr_edges) > 0:
            pr_edges_t = torch.as_tensor(pr_edges, dtype=torch.long)
        else:
            pr_edges_t = torch.zeros((0, 2), dtype=torch.long)

        # ---------- A) SMD on normalized coordinates ----------
        A_norm = torch.zeros((gt_nodes_norm_t.shape[0], gt_nodes_norm_t.shape[0]), dtype=torch.float32)
        if len(gt_edges_t) > 0:
            A_norm[gt_edges_t[:, 0], gt_edges_t[:, 1]] = 1.0

        pred_A_norm = torch.zeros((pr_nodes_norm_t.shape[0], pr_nodes_norm_t.shape[0]), dtype=torch.float32)
        if len(pr_edges_t) > 0:
            pred_A_norm[pr_edges_t[:, 0], pr_edges_t[:, 1]] = 1.0

        y_pc_norm      = get_point_cloud(A_norm.T,      gt_nodes_norm_t, n_points=100)
        output_pc_norm = get_point_cloud(pred_A_norm.T, pr_nodes_norm_t, n_points=100)

        smd_norm, _, _ = sinkhorn(y_pc_norm, output_pc_norm)

        ax_smd_norm = axs[row1, 2]
        ax_smd_norm.scatter(y_pc_norm[:, 0].cpu(),      y_pc_norm[:, 1].cpu(),
                            s=5, label="GT (norm)")
        ax_smd_norm.scatter(output_pc_norm[:, 0].cpu(), output_pc_norm[:, 1].cpu(),
                            s=5, label="Pred (norm)")
        
        # DEBUG
        num_edges = int(A_norm.sum().item())


        ax_smd_norm.set_title(f"SMD (norm) = {smd_norm.item():.4f}: {len(gt_nodes_norm_t)} nodes, {num_edges} edges")
        ax_smd_norm.set_xlim(0, 1)
        ax_smd_norm.set_ylim(0, 1)
        ax_smd_norm.legend(fontsize=7)
        ax_smd_norm.axis("off")
        axs[row1, 2].axis("off")
        axs[row1, 3].axis("off")

    plt.tight_layout()
    return fig

# ---------- low-probability saver for training ----------
class DebugVisualizer:
    """
    Call `maybe_save(images, nodes, edges, pred_nodes, pred_edges, epoch, step, batch_index=0, tag="train")`
    from your training loop. Saves to out_dir with a small probability and per-epoch cap.
    """
    def __init__(self, out_dir, prob=0.001, max_per_epoch=8,
                 denorm_mean=(0.485,0.456,0.406), denorm_std=(0.229,0.224,0.225)):
        self.out_dir = out_dir
        self.prob = float(prob)
        self.max_per_epoch = int(max_per_epoch)
        self._emitted_in_epoch = 0
        self.denorm_mean = denorm_mean
        self.denorm_std = denorm_std
        os.makedirs(out_dir, exist_ok=True)

    def start_epoch(self):
        self._emitted_in_epoch = 0

    def maybe_save(self, images_bchw, gt_nodes_list, gt_edges_list, 
               pred_nodes_list, pred_edges_list, epoch, step, 
               batch_index=0, tag="train", gt_seg=None, pred_seg=None):
        
        if self._emitted_in_epoch >= self.max_per_epoch or random.random() >= self.prob:
            return

        sample = {
            "images":     [images_bchw[batch_index].detach().cpu()],
            "nodes":      [gt_nodes_list[batch_index]],
            "edges":      [gt_edges_list[batch_index]],
            "pred_nodes": [pred_nodes_list[batch_index]],
            "pred_edges": [pred_edges_list[batch_index]],
            "seg":        [gt_seg[batch_index] if gt_seg is not None else None],
            "pred_seg":   [pred_seg[batch_index] if pred_seg is not None else None],
        }

        fig = visualize_graph_batch(sample, n=1, mean=self.denorm_mean, std=self.denorm_std)

        fname = f"{tag}_e{int(epoch):03d}_it{int(step):06d}_b{int(batch_index)}.png"
        fpath = os.path.join(self.out_dir, fname)
        fig.savefig(fpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        self._emitted_in_epoch += 1
        print(f"[viz] Saved {fpath}")
