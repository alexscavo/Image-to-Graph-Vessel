# training/vis_debug.py
import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def to_pixel_space(nodes, H, W, swap_xy=True):
    """nodes: (N,2[+...]) normalized torch/np -> returns np.float32 (N,2) in pixel coords."""
    if nodes is None:
        return np.zeros((0, 2), dtype=np.float32)

    # Convert safely to numpy
    if hasattr(nodes, "detach"):
        arr = nodes.detach().cpu().numpy()
    else:
        arr = np.asarray(nodes)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    # keep x,y only, copy to avoid mutating caller
    arr = arr[:, :2].astype(np.float32, copy=True)

    # swap x↔y for visualization if your dataset needs it
    if swap_xy:
        arr[:, [0, 1]] = arr[:, [1, 0]]

    # scale normalized coords to pixels
    arr[:, 0] *= W  # x
    arr[:, 1] *= H  # y
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
def visualize_graph_batch(samples, n=6, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                          point_size=10, linewidth=1.0, figsize_per_row=(6,6), alpha_img=1.0):
    """
    samples must contain:
      images:     list/tuple/Tensor of (C,H,W)
      nodes:      list of (Ni,2) GT points (x,y) in px or [0,1]
      edges:      list of (Ei,2) GT indices
      pred_nodes: list of (Mi,2) predicted points in px or [0,1]
      pred_edges: list of (Pi,2) predicted indices
    """
    imgs = samples["images"]
    n = min(n, len(imgs))
    rows, cols = n, 3
    fig, axs = plt.subplots(rows, cols, figsize=(figsize_per_row[0]*cols, figsize_per_row[1]*rows))
    if rows == 1:
        axs = np.expand_dims(axs, 0)

    for i in range(n):
        # --- image
        im_t = imgs[i] if isinstance(imgs, (list, tuple)) else imgs[i]
        vis, cmap = denorm_img(im_t)
        H, W = vis.shape[:2]

        # --- GT / Pred
        gt_nodes = to_pixel_space(samples.get("nodes", [None])[i], H, W, swap_xy=True)
        gt_edges = as_int_edges(samples.get("edges", [None])[i])
        pr_nodes = to_pixel_space(samples.get("pred_nodes", [None])[i], H, W, swap_xy=True)
        pr_edges = as_int_edges(samples.get("pred_edges", [None])[i])
        
        # Panel 0: Image
        ax0 = axs[i, 0]
        ax0.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        ax0.set_title(f"Image {i}  (H={H}, W={W})")
        ax0.axis("off")

        # Panel 1: GT overlay
        ax1 = axs[i, 1]
        ax1.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        if gt_nodes.size:
            ax1.scatter(gt_nodes[:, 0], gt_nodes[:, 1], s=point_size)
            for e0, e1 in gt_edges:
                if e0 < gt_nodes.shape[0] and e1 < gt_nodes.shape[0]:
                    x0, y0 = gt_nodes[e0]; x1, y1 = gt_nodes[e1]
                    ax1.plot([x0, x1], [y0, y1], linewidth=linewidth)
        ax1.set_title(f"GT: {len(gt_nodes)} nodes, {len(gt_edges)} edges")
        ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
        ax1.axis("off")

        # Panel 2: Pred overlay
        ax2 = axs[i, 2]
        ax2.imshow(vis, origin="upper", interpolation="nearest", alpha=alpha_img, cmap=cmap)
        if pr_nodes.size:
            ax2.scatter(pr_nodes[:, 0], pr_nodes[:, 1], s=point_size)
            for e0, e1 in pr_edges:
                if e0 < pr_nodes.shape[0] and e1 < pr_nodes.shape[0]:
                    x0, y0 = pr_nodes[e0]; x1, y1 = pr_nodes[e1]
                    ax2.plot([x0, x1], [y0, y1], linewidth=linewidth)
        ax2.set_title(f"Pred: {len(pr_nodes)} nodes, {len(pr_edges)} edges")
        ax2.set_xlim(0, W); ax2.set_ylim(H, 0)
        ax2.axis("off")

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
