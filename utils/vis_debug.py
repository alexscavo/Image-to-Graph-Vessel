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

# ---------- core viz (image | GT | Pred) ----------
# def visualize_graph_batch(
#         samples,
#         n=6,
#         mean=(0.485,0.456,0.406),
#         std=(0.229,0.224,0.225),
#         point_size=10,
#         linewidth=1.0,
#         figsize_per_row=(6,6),
#         alpha_img=1.0,
#         show_smd=False,
#         smd_eps=1e-7,
#         smd_max_iter=100,
#         smd_n_points=100,
#     ):
#     """
#     samples must contain:
#       images:     list/tuple/Tensor of (C,H,W)
#       nodes:      list of (Ni,2) GT points (x,y) in px or [0,1]
#       edges:      list of (Ei,2) GT indices
#       pred_nodes: list of (Mi,2) predicted points in px or [0,1]
#       pred_edges: list of (Pi,2) predicted indices
#     """
#     imgs = samples["images"]
#     n = min(n, len(imgs))
#     rows = n
#     cols = 3 + (1 if show_smd else 0)
#     fig, axs = plt.subplots(rows, cols, figsize=(figsize_per_row[0]*cols, figsize_per_row[1]*rows))
#     if rows == 1:
#         axs = np.expand_dims(axs, 0)
        
#     sinkhorn = None
#     if show_smd:
#         sinkhorn = SinkhornDistance(eps=smd_eps, max_iter=smd_max_iter, reduction='none')

#     for i in range(n):
#         # --- image
#         im_t = imgs[i] if isinstance(imgs, (list, tuple)) else imgs[i]
#         vis, cmap = denorm_img(im_t)
#         H, W = vis.shape[:2]

#         # --- GT / Pred (converted to pixel space + viz conventions)
#         gt_nodes = to_pixel_space(samples.get("nodes", [None])[i], H, W)
#         gt_edges = as_int_edges(samples.get("edges", [None])[i])
#         pr_nodes = to_pixel_space(samples.get("pred_nodes", [None])[i], H, W)
#         pr_edges = as_int_edges(samples.get("pred_edges", [None])[i])
        
#         # Keep original (normalized) graph tensors for SMD, not pixel-converted
#         gt_nodes_norm = samples.get("nodes", [None])[i]
#         gt_edges_t    = samples.get("edges", [None])[i]
#         pr_nodes_norm = samples.get("pred_nodes", [None])[i]
#         pr_edges_t    = samples.get("pred_edges", [None])[i]

        # # Panel 0: Image
        # ax0 = axs[i, 0]
        # ax0.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        # ax0.set_title(f"Image {i}  (H={H}, W={W})")
        # ax0.axis("off")

        # # Panel 1: GT overlay
        # ax1 = axs[i, 1]
        # ax1.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        # if gt_nodes.size:
        #     ax1.scatter(gt_nodes[:, 0], gt_nodes[:, 1], s=point_size)
        #     for e0, e1 in gt_edges:
        #         if e0 < gt_nodes.shape[0] and e1 < gt_nodes.shape[0]:
        #             x0, y0 = gt_nodes[e0]; x1, y1 = gt_nodes[e1]
        #             ax1.plot([x0, x1], [y0, y1], linewidth=linewidth)
        # ax1.set_title(f"GT: {len(gt_nodes)} nodes, {len(gt_edges)} edges")
        # ax1.set_xlim(0, W); ax1.set_ylim(H, 0)  # image coords (y down)
        # ax1.axis("off")

        # # Panel 2: Pred overlay
        # ax2 = axs[i, 2]
        # ax2.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        # if pr_nodes.size:
        #     ax2.scatter(pr_nodes[:, 0], pr_nodes[:, 1], s=point_size)
        #     for e0, e1 in pr_edges:
        #         if e0 < pr_nodes.shape[0] and e1 < pr_nodes.shape[0]:
        #             x0, y0 = pr_nodes[e0]; x1, y1 = pr_nodes[e1]
        #             ax2.plot([x0, x1], [y0, y1], linewidth=linewidth)
        # ax2.set_title(f"Pred: {len(pr_nodes)} nodes, {len(pr_edges)} edges")
        # ax2.set_xlim(0, W); ax2.set_ylim(H, 0)
        # ax2.axis("off")
        
#         # Optional 4th panel: SMD point clouds
#         if show_smd and gt_nodes_norm is not None and pr_nodes_norm is not None:
#             # Build adjacency matrices for SMD (like in StreetMoverDistance)
#             gt_nodes_norm = gt_nodes_norm.detach()
#             pr_nodes_norm = pr_nodes_norm.detach()

#             A = torch.zeros(
#                 (gt_nodes_norm.shape[0], gt_nodes_norm.shape[0]),
#                 device=gt_nodes_norm.device
#             )
#             if gt_edges_t is not None and gt_edges_t.numel() > 0:
#                 A[gt_edges_t[:, 0], gt_edges_t[:, 1]] = 1.0

#             pred_A = torch.zeros(
#                 (pr_nodes_norm.shape[0], pr_nodes_norm.shape[0]),
#                 device=pr_nodes_norm.device
#             )
#             if pr_edges_t is not None and len(pr_edges_t) > 0:
#                 pred_A[pr_edges_t[:, 0], pr_edges_t[:, 1]] = 1.0

#             # Same logic as compute_meanSMD: build point clouds
#             y_pc      = get_point_cloud(A.T,      gt_nodes_norm, smd_n_points)
#             output_pc = get_point_cloud(pred_A.T, pr_nodes_norm, smd_n_points)

#             sink_dist, P, C = sinkhorn(y_pc, output_pc)

#             # Plot SMD point clouds in normalized [0,1] coordinates
#             ax_smd = axs[i, 3]  # 4th column
#             ax_smd.scatter(y_pc[:, 0].cpu(),      y_pc[:, 1].cpu(),      s=5, label="GT PC")
#             ax_smd.scatter(output_pc[:, 0].cpu(), output_pc[:, 1].cpu(), s=5, label="Pred PC")
#             ax_smd.set_title(f"SMD={sink_dist.item():.4f}")
#             ax_smd.set_xlim(0, 1)
#             ax_smd.set_ylim(0, 1)
#             ax_smd.legend(loc="best", fontsize=8)
#             ax_smd.axis("off")

#     plt.tight_layout()
#     return fig



def visualize_graph_batch(
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
    Layout per sample:
      Row 1: [ Image | GT graph | Pred graph ]
      Row 2: [ SMD (normalized) | SMD (pixel space) | empty ]
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

    def maybe_save(self,
                   images_bchw,           # (B,C,H,W) torch.Tensor
                   gt_nodes_list,         # list length B, each (Ni,2) tensor
                   gt_edges_list,         # list length B, each (Ei,2) tensor
                   pred_nodes_list,       # list length B, each (Mi,2) tensor/np
                   pred_edges_list,       # list length B, each (Pi,2) tensor/np
                   epoch, step, batch_index=0, tag="train"):
        if self._emitted_in_epoch >= self.max_per_epoch:
            return
        if random.random() >= self.prob:
            return

        # Build a one-sample dict compatible with visualize_graph_batch
        sample = {
            "images":     [images_bchw[batch_index].detach().cpu()],
            "nodes":      [gt_nodes_list[batch_index]],
            "edges":      [gt_edges_list[batch_index]],
            "pred_nodes": [pred_nodes_list[batch_index]],
            "pred_edges": [pred_edges_list[batch_index] if pred_edges_list is not None else np.zeros((0,2), dtype=np.int64)],
        }

        fig = visualize_graph_batch(sample, n=1,
                                    mean=self.denorm_mean, std=self.denorm_std,
                                    point_size=10, linewidth=1.0, figsize_per_row=(6,6), alpha_img=1.0)

        fname = f"{tag}_e{int(epoch):03d}_it{int(step):06d}_b{int(batch_index)}.png"
        fpath = os.path.join(self.out_dir, fname)
        fig.savefig(fpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        self._emitted_in_epoch += 1
        print(f"[viz] Saved {fpath}")
