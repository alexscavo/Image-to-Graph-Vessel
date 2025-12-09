import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from torchvision.transforms import Compose, Normalize

def draw_graph_3d(nodes, edges, ax):
    xs = nodes[:,0]
    ys = nodes[:,1]
    zs = nodes[:,2]
    ax.scatter(xs, ys, zs)

    # Add all edges
    for edge in edges:
        ax.plot([xs[edge[0]],xs[edge[1]]],[ys[edge[0]],ys[edge[1]]], [zs[edge[0]],zs[edge[1]]], color="green")
        
        
        
# def create_sample_visual_3d(samples, number_samples=10, max_points=5000, quantile=0.97):
#     """
#     Visualize 3D image volumes (as fed to the model) together with GT and predicted graphs.

#     Left  column: 3D point cloud of high-intensity voxels from samples["images"].
#     Middle column: GT graph (nodes/edges).
#     Right column:  Predicted graph (nodes/edges).

#     Args:
#         samples: dict with keys "images", "nodes", "edges", "pred_nodes", "pred_rels".
#                  images[i] is expected to be (D,H,W) or (1,D,H,W) torch tensor.
#         number_samples: how many samples to visualize.
#         max_points: max number of voxels to plot per sample (for speed).
#         quantile: keep voxels above this intensity quantile (0–1).
#     """
#     px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
#     fig, axs = plt.subplots(
#         number_samples,
#         3,
#         figsize=(1000 * px, number_samples * 300 * px),
#         subplot_kw={'projection': '3d'}
#     )

#     # If number_samples == 1, axs will be 1D; normalize to 2D [rows, cols]
#     if number_samples == 1:
#         axs = np.array([axs])

#     for i in range(number_samples):
#         if i >= len(samples["images"]):
#             break
#         try:
#             # ----- LEFT: 3D image volume -----
#             vol = samples["images"][i].squeeze().cpu().numpy()  # (D,H,W)
#             if vol.ndim != 3:
#                 raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

#             D, H, W = vol.shape

#             # Normalize to [0,1] for robust thresholding
#             vmin, vmax = vol.min(), vol.max()
#             if vmax > vmin:
#                 vol_norm = (vol - vmin) / (vmax - vmin)
#             else:
#                 vol_norm = np.zeros_like(vol)

#             # Keep only the brightest voxels (quantile threshold)
#             thr = np.quantile(vol_norm, quantile)
#             zz, yy, xx = np.where(vol_norm >= thr)

#             coords = np.stack([xx, yy, zz], axis=1)
#             if coords.shape[0] > max_points:
#                 idx = np.random.choice(coords.shape[0], max_points, replace=False)
#                 coords = coords[idx]

#             ax0 = axs[i, 0]
#             if coords.size > 0:
#                 ax0.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=1)
#             ax0.set_xlim(0, W)
#             ax0.set_ylim(0, H)
#             ax0.set_zlim(0, D)
#             ax0.set_xlabel("x")
#             ax0.set_ylabel("y")
#             ax0.set_zlabel("z")

#             # ----- MIDDLE: GT graph -----
#             plt.sca(axs[i, 1])
#             draw_graph_3d(
#                 samples["nodes"][i].cpu().numpy(),
#                 samples["edges"][i].cpu().numpy(),
#                 axs[i, 1]
#             )
#             axs[i, 1].set_xlim(0, 1)
#             axs[i, 1].set_ylim(0, 1)
#             axs[i, 1].set_zlim(0, 1)

#             # ----- RIGHT: predicted graph -----
#             plt.sca(axs[i, 2])
#             draw_graph_3d(
#                 samples["pred_nodes"][i],
#                 samples["pred_rels"][i],
#                 axs[i, 2]
#             )
#             axs[i, 2].set_xlim(0, 1)
#             axs[i, 2].set_ylim(0, 1)
#             axs[i, 2].set_zlim(0, 1)

#         except Exception as e:
#             print(f"visualization error occurred for sample {i}, skipping... ({e})")

#     fig.canvas.draw()
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return np.transpose(data, (2, 0, 1))


def create_sample_visual_3d(samples, number_samples=10):

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(number_samples, 3, figsize=(1000 * px, number_samples * 300 * px), subplot_kw={'projection':'3d'})

    for i in range(number_samples):
        if i >= len(samples["segs"]):
            break
        try:
            verts, faces, normals, values = measure.marching_cubes(samples["segs"][i].squeeze().cpu().numpy(), 0)

            mesh = Poly3DCollection(verts[faces])
            mesh.set_edgecolor('k')
            axs[i, 0].add_collection3d(mesh)

            axs[i, 0].set_xlim(0, 64)
            axs[i, 0].set_ylim(0, 64)
            axs[i, 0].set_zlim(0, 64)

            plt.sca(axs[i, 1])
            draw_graph_3d(samples["nodes"][i].cpu().numpy(), samples["edges"][i].cpu().numpy(), axs[i, 1])
            plt.sca(axs[i, 2])
            draw_graph_3d(samples['pred_nodes'][i], samples['pred_rels'][i], axs[i, 2])

            axs[i, 1].set_xlim(0, 1)
            axs[i, 1].set_ylim(0, 1)
            axs[i, 1].set_zlim(0, 1)
            axs[i, 2].set_xlim(0, 1)
            axs[i, 2].set_ylim(0, 1)
            axs[i, 2].set_zlim(0, 1)
        except:
            print("visualiazion error occured, skipping sample...")

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return np.transpose(data, (2, 0, 1))

def draw_graph_2d(nodes, edges, ax):
    xs = nodes[:, 0]
    ys = nodes[:, 1]
    ax.scatter(ys, xs, c="red")

    # Add all edges
    for edge in edges:
        ax.plot([ys[edge[0]], ys[edge[1]]], [xs[edge[0]], xs[edge[1]]], color="red")


def create_sample_visual_2d(samples, number_samples=10):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(number_samples, 2, figsize=(1000 * px, number_samples * 300 * px))

    if len(samples["segs"].shape) >= 5:
        indices = np.round(np.array(samples["z_pos"]) * (samples["segs"].shape[-1] - 1)).astype(int)

    for i in range(number_samples):
        if i >= len(samples["segs"]):
            break
        if len(samples["segs"].shape) >= 5:
            sample = samples["segs"][i, 0, :, :, indices[i]]
        else:
            sample = samples["images"][i, 0]
        axs[i, 0].imshow(sample.clone().cpu().detach())
        axs[i, 1].imshow(sample.clone().cpu().detach())

        plt.sca(axs[i, 0])
        draw_graph_2d(samples["nodes"][i].clone().cpu().detach() * sample.shape[0], samples["edges"][i].clone().cpu().detach(), axs[i, 0])
        plt.sca(axs[i, 1])
        draw_graph_2d(samples["pred_nodes"][i] * sample.shape[0], samples["pred_rels"][i], axs[i, 1])

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return np.transpose(data, (2, 0, 1))


inv_norm = Compose([
    Normalize(
        mean=[0., 0., 0.],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    Normalize(
        mean=[-0.485, -0.456, -0.406],
        std=[1., 1., 1.]
    ),
])

def create_gradcam_overlay_2d(samples):
    """
    Build a single image: raw slice + Grad-CAM overlay for sample 0.
    Returns [tensor] where tensor has shape [1, 3, H, W] for TensorBoardImageHandler.

    This matches TensorBoardImageHandler's expectation:
        show_outputs = output_transform(...)[index] -> [N, C, H, W]
    """

    # ---- if there is no Grad-CAM for this iteration, skip logging ----
    if "gradcam" not in samples or samples["gradcam"] is None:
        # output_transform must still be indexable; returning [None] means
        # show_outputs = None and the handler will just skip plotting.
        return [None]

    # ------------------------------------------------------------------
    # 1) Choose base volume: raw image if available, otherwise segs
    # ------------------------------------------------------------------
    if "image" in samples:
        vol = samples["image"]      # expected [B, 1, H, W, D]
    else:
        vol = samples["segs"]       # fallback

    cam = samples["gradcam"]

    # Move tensors to numpy
    if isinstance(vol, torch.Tensor):
        vol_np = vol.detach().cpu().numpy()
    else:
        vol_np = np.asarray(vol)

    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
    cam = np.asarray(cam)

    # ------------------------------------------------------------------
    # 2) Prepare Grad-CAM array to be 3D (D, H, W) or (H, W, D)
    # ------------------------------------------------------------------
    while cam.ndim > 3 and cam.shape[0] == 1:
        cam = cam[0]

    if cam.ndim != 3:
        # Unexpected shape, skip
        return [None]

    # ------------------------------------------------------------------
    # 3) Choose slice index (same logic as your 2D visualizer)
    # ------------------------------------------------------------------
    if vol_np.ndim < 5:
        # no depth dimension, can't overlay a slice
        return [None]

    # vol_np: [B, 1, H, W, D]
    vol0 = vol_np[0]                  # [1, H, W, D]
    depth = vol0.shape[-1]

    z_pos_arr = np.array(samples["z_pos"])
    indices = np.round(z_pos_arr * (depth - 1)).astype(int)

    if np.ndim(indices) == 0:
        idx0 = int(indices)
    else:
        idx0 = int(indices[0])

    idx0 = max(0, min(depth - 1, idx0))  # clamp to valid range

    # Base (raw) slice: [H, W]
    base = vol0[0, :, :, idx0]

    # ------------------------------------------------------------------
    # 4) Corresponding Grad-CAM slice
    # ------------------------------------------------------------------
    if cam.shape[0] == depth:
        # cam is [D, H, W]
        cam_slice = cam[idx0]
    elif cam.shape[-1] == depth:
        # cam is [H, W, D]
        cam_slice = cam[:, :, idx0]
    else:
        # fallback: treat first dim as depth
        cam_slice = cam[idx0]

    # ------------------------------------------------------------------
    # 5) Normalize both to [0, 1]
    # ------------------------------------------------------------------
    base_min, base_max = base.min(), base.max()
    if base_max > base_min:
        base = (base - base_min) / (base_max - base_min)
    else:
        base = np.zeros_like(base, dtype=np.float32)

    hm_min, hm_max = cam_slice.min(), cam_slice.max()
    if hm_max > hm_min:
        cam_slice = (cam_slice - hm_min) / (hm_max - hm_min)
    else:
        cam_slice = np.zeros_like(cam_slice, dtype=np.float32)

    # ------------------------------------------------------------------
    # 6) Create overlay figure (grayscale base + colored Grad-CAM)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))

    # raw volume slice in grayscale
    ax.imshow(base, cmap="gray")

    # Grad-CAM in color, e.g. "jet" for typical red/yellow heatmap
    ax.imshow(cam_slice, cmap="jet", alpha=0.4)

    ax.axis("off")
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # data: [H, W, 3] -> [1, 3, H, W]
    chw = torch.from_numpy(data).permute(2, 0, 1).float() / 255.0
    nchw = chw.unsqueeze(0)

    # IMPORTANT: wrap in a list so handler's [index] picks this tensor
    return [nchw]