import numpy as np
import open3d as o3d
from skimage import measure
import torch

def render_open3d_image(geometries, width=640, height=480):
    """
    Render a list of Open3D geometries to an offscreen RGB image (H, W, 3).
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)

    vis.get_render_option().background_color = np.array([0, 0, 0])

    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    img = np.asarray(img)  # H, W, 3 (float [0,1])
    img = (img * 255).astype(np.uint8)
    return img

def seg_to_mesh_open3d(seg_volume, level=0.5):
    """
    seg_volume: (D, H, W) numpy array
    """
    vol = (seg_volume > 0).astype(np.float32)
    verts, faces, normals, values = measure.marching_cubes(vol, level=level)

    # Open3D expects (x, y, z); current verts are (z, y, x)
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    xyz = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=-1)  # (N, 3)
    mesh.vertices = o3d.utility.Vector3dVector(xyz)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    return mesh

def nodes_to_pcd(nodes_vox, color):
    """
    nodes_vox: (N, 3) in voxel coords (z,y,x)
    color: [r,g,b] in [0,1]
    """
    import open3d as o3d
    if nodes_vox is None or nodes_vox.size == 0:
        return None

    xyz = np.stack([nodes_vox[:, 2], nodes_vox[:, 1], nodes_vox[:, 0]], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    colors = np.tile(np.array(color, dtype=np.float32)[None, :], (xyz.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def norm_nodes_to_vox(nodes, D, H, W):
    """
    Your nodes are in [0,1]^3; convert to voxel indices like in DebugVisualizer3D.
    """
    n = nodes.copy()
    out = np.zeros_like(n)
    out[:, 0] = n[:, 0] * (D - 1)
    out[:, 1] = n[:, 1] * (H - 1)
    out[:, 2] = n[:, 2] * (W - 1)
    return out


class Open3DTensorboardLogger:
    def __init__(
        self,
        writer,
        prob=1.0,
        max_per_epoch=1,
        level=0.5,
        width=640,
        height=480,
        tag_prefix="open3d",
    ):
        """
        prob: probability per call to log (use <1.0 to subsample)
        max_per_epoch: hard cap of images per epoch
        """
        self.writer = writer
        self.prob = float(prob)
        self.max_per_epoch = int(max_per_epoch)
        self.level = float(level)
        self.width = width
        self.height = height
        self.tag_prefix = tag_prefix
        self._emitted_in_epoch = 0

    def start_epoch(self):
        self._emitted_in_epoch = 0

    def maybe_log(
        self,
        segs,
        gt_nodes_list,
        pred_nodes_list,
        epoch,
        step,
        batch_index=0,
        tag="train",
    ):
        import random
        if self._emitted_in_epoch >= self.max_per_epoch:
            return
        if random.random() >= self.prob:
            return

        # ----- extract one sample from the batch -----
        if isinstance(segs, torch.Tensor):
            if batch_index >= segs.size(0):
                return
            seg_sample = segs[batch_index].detach().cpu().numpy()
        else:
            segs_np = np.asarray(segs)
            if batch_index >= segs_np.shape[0]:
                return
            seg_sample = segs_np[batch_index]

        # (C,D,H,W) → (D,H,W)
        if seg_sample.ndim == 4:
            seg_sample = seg_sample[0]
        if seg_sample.ndim != 3:
            return

        D, H, W = seg_sample.shape

        gt_nodes = gt_nodes_list[batch_index].detach().cpu().numpy()
        pred_nodes = pred_nodes_list[batch_index]
        if isinstance(pred_nodes, torch.Tensor):
            pred_nodes = pred_nodes.detach().cpu().numpy()

        gt_nodes_vox = norm_nodes_to_vox(gt_nodes, D, H, W)
        pred_nodes_vox = norm_nodes_to_vox(pred_nodes, D, H, W)

        # ----- build Open3D geometries -----
        geoms = []
        try:
            mesh = seg_to_mesh_open3d(seg_sample, level=self.level)
            geoms.append(mesh)
        except Exception as e:
            print(f"[open3d_tb] Failed to create mesh: {e}")

        gt_pcd = nodes_to_pcd(gt_nodes_vox, [1.0, 0.0, 0.0])   # red
        pred_pcd = nodes_to_pcd(pred_nodes_vox, [0.0, 0.0, 1.0]) # blue
        if gt_pcd is not None:
            geoms.append(gt_pcd)
        if pred_pcd is not None:
            geoms.append(pred_pcd)

        if not geoms:
            return

        # ----- render & send to TensorBoard -----
        img = render_open3d_image(geoms, width=self.width, height=self.height)  # H,W,3
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # C,H,W

        global_step = step  # or epoch, or epoch*epoch_len+local_step
        tb_tag = f"{self.tag_prefix}/{tag}_e{epoch:03d}"
        self.writer.add_image(tb_tag, img, global_step=global_step, dataformats="CHW")

        self._emitted_in_epoch += 1