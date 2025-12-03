import os
import glob
from pathlib import Path

import numpy as np
import pyvista as pv
import nibabel as nib
from skimage import measure

import plotly.graph_objects as go
from plotly.offline import plot as plot_offline


def get_paths_from_raw(raw_sample_path: str):
    """
    Expect structure like:
        .../patches/train/raw/sample_000001_0_data.nii.gz

    Returns:
        seg_path:  .../patches/train/seg/sample_000001_0_seg.nii.gz
        vtp_dir:   .../patches/train/vtp/
        base_id:   sample_000001_0
    """
    raw_path = Path(raw_sample_path).resolve()
    root_dir = raw_path.parent.parent      # parent of 'raw'

    name = raw_path.name
    if not name.endswith("_data.nii.gz"):
        raise ValueError(
            f"Unexpected raw filename format: {name}. "
            "Expected something like 'sample_000001_0_data.nii.gz'."
        )
    base_id = name.replace("_data.nii.gz", "")  # 'sample_000001_0'

    seg_dir = root_dir / "seg"
    vtp_dir = root_dir / "vtp"
    seg_path = seg_dir / f"{base_id}_seg.nii.gz"

    return seg_path, vtp_dir, base_id


def load_seg_mesh_and_affine(seg_path: Path, level: float = 0.5):
    """
    Load a segmentation NIfTI and extract a surface mesh using marching cubes.

    Returns:
        verts_world: (N,3) vertices in world (mm) coordinates
        faces:       (M,3) face indices
        affine:      (4,4) affine matrix used
    """
    print(f"Loading segmentation from {seg_path}")
    img = nib.load(str(seg_path))
    data = img.get_fdata()
    affine = img.affine

    # marching_cubes expects a 3D volume, returns verts in voxel index coords
    verts, faces, normals, values = measure.marching_cubes(data, level=level)

    # verts are (z, y, x) voxel indices; treat them as (i,j,k) and apply affine
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_world = (affine @ verts_h.T).T[:, :3]

    print(f"Segmentation mesh: {verts_world.shape[0]} vertices, {faces.shape[0]} faces")
    return verts_world, faces, affine


def apply_affine(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 affine to an (N,3) array of points.
    Assumes points are in voxel index space consistent with the seg.
    """
    pts_h = np.c_[points, np.ones(len(points))]
    pts_world = (affine @ pts_h.T).T[:, :3]
    return pts_world


def load_graph_data_for_sample(vtp_dir: Path, base_id: str, affine: np.ndarray = None, patch_size=(64, 64, 64)):
    """
    Load all .vtp graph files corresponding to base_id from vtp_dir.

    Returns a list of (points, lines, color, label, node_ids).
    """
    pattern = str(vtp_dir / f"{base_id}_graph.vtp")
    vtps = sorted(glob.glob(pattern))
    if not vtps:
        raise FileNotFoundError(f"No .vtp files found with pattern {pattern}")

    graphs = []
    offset_step = 0.0  # keep at 0.0 if you want graphs aligned with seg

    for idx, vtp_path in enumerate(vtps):
        mesh = pv.read(vtp_path)
        points = mesh.points.copy()
        
        points_voxel = points * np.array(patch_size)

        # optional: apply affine if points are in voxel space and you want world coords
        if affine is not None:
            points_world = apply_affine(points_voxel, affine)
        else:
            points_world = points_voxel

        # small shift if you want to separate multiple graphs visually
        if offset_step != 0.0:
            points_world = points_world + np.array([idx * offset_step, 0.0, 0.0])

        lines_raw = mesh.lines.reshape(-1, 3)
        lines = lines_raw[:, 1:]

        # try to get node IDs from point_data, fall back to indices
        if "id" in mesh.point_data:
            node_ids = np.array(mesh.point_data["id"])
        elif "node_id" in mesh.point_data:
            node_ids = np.array(mesh.point_data["node_id"])
        else:
            node_ids = np.arange(points_world.shape[0])

        fname = os.path.basename(vtp_path).lower()
        if "ref" in fname or "gt" in fname:
            color = "red"
        elif "pred" in fname:
            color = "blue"
        else:
            color = "green"

        graphs.append((points_world, lines, color, os.path.basename(vtp_path), node_ids))
        print(f"Loaded graph from {vtp_path}")

    return graphs


def make_plotly_figure(graphs, seg_mesh=None, title: str = "", show_seg: bool = True):
    """
    Create a Plotly Figure with interactive 3D:
      - optional segmentation mesh (semi-transparent, hideable)
      - graph nodes + edges with node IDs on hover
    """
    fig = go.Figure()

    # 1) Add segmentation mesh (toggle via legend; can start hidden)
    if seg_mesh is not None:
        verts_world, faces = seg_mesh
        x, y, z = verts_world[:, 0], verts_world[:, 1], verts_world[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                color="lightgray",
                opacity=0.20,
                name="segmentation",
                visible=True if show_seg else "legendonly",
                showlegend=True,
                hoverinfo="skip"
            )
        )

    # 2) Add graphs with node-id hover
    for points, lines, color, label, node_ids in graphs:
        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        # Prepare hover text for nodes
        node_text = [
            f"node: {nid}<br>"
            f"x: {x:.3f}<br>"
            f"y: {y:.3f}<br>"
            f"z: {z:.3f}"
            for nid, x, y, z in zip(node_ids, xs, ys, zs)
        ]

        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.9),
                text=node_text,
                hoverinfo="text",
                name=f"nodes",
            )
        )

        # edges
        edge_x, edge_y, edge_z = [], [], []
        for e0, e1 in lines:
            edge_x.extend([xs[e0], xs[e1], None])
            edge_y.extend([ys[e0], ys[e1], None])
            edge_z.extend([zs[e0], zs[e1], None])

        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(width=2, color=color),
                name=f"edges: {label}",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig



def visualize_sample_from_raw_browser(args, show_seg: bool = True):
    seg_path, vtp_dir, base_id = get_paths_from_raw(args.raw_path)

    out_html = args.out_html
    patch_size = args.patch_size

    print(f"Raw path: {args.raw_path}")
    print(f"Seg path (expected): {seg_path}")
    print(f"VTP directory: {vtp_dir}")
    print(f"Base id: {base_id}")

    seg_mesh = None
    affine = None
    if seg_path.exists() and show_seg:
        try:
            verts_world, faces, affine = load_seg_mesh_and_affine(seg_path)
            seg_mesh = (verts_world, faces)
        except Exception as e:
            print(f"[WARNING] Failed to create seg mesh: {e}")
    elif not seg_path.exists():
        print(f"[WARNING] Segmentation file not found: {seg_path}")

    if not vtp_dir.exists():
        raise FileNotFoundError(f"VTP directory does not exist: {vtp_dir}")

    graphs = load_graph_data_for_sample(vtp_dir, base_id, affine=affine, patch_size=patch_size)

    if out_html is None:
        out_html = Path(vtp_dir).parent / f"{base_id}_graphs.html"

    fig = make_plotly_figure(
        graphs,
        seg_mesh=seg_mesh,
        title=f"Graphs{' + seg' if show_seg else ''} for {base_id}",
        show_seg=show_seg,
    )

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    plot_offline(fig, filename=str(out_html), auto_open=False)
    print(f"Saved interactive HTML to {out_html}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Browser-based visualization of graph .vtp files + segmentation (Plotly HTML), starting from a raw patch path."
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        required=True,
        help="Path to raw patch, e.g. /path/to/patches/train/raw/sample_000001_0_data.nii.gz",
    )
    parser.add_argument(
        "--out_html",
        type=str,
        default=None,
        help="Output HTML path (optional). Default: <train>/<base_id>_graphs.html",
    )
    parser.add_argument(
        "--patch_size",
        type=float,
        default="64",
        help="Patch shape, report only 1 dimension. If the volume is 64x64x64, use 64",
    )
    args = parser.parse_args([
        "--raw_path", "/data/scavone/syntheticMRI/patches/train/raw/sample_000003_1_data.nii.gz"
    ])

    visualize_sample_from_raw_browser(args)
