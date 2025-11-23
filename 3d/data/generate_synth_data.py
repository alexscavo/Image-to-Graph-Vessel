from pathlib import Path
import sys
from tqdm import tqdm
from medpy.io import load, save
import pyvista
import numpy as np
import os
from patch_generator import PatchGraphGenerator
import scipy
import nibabel as nib
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import split_reader_nifti

patch_size = [64, 64, 64]
pad = [2, 2, 2]


# ---------- helpers for image + coordinates ----------

def load_image_and_affine(path):
    """
    Load NIfTI with nibabel to get both data and affine.
    """
    img = nib.load(path)  
    data = img.get_fdata()
    affine = img.affine        # voxel -> world
    return data, affine


def voxel_to_world(vox_xyz, affine):
    """
    vox_xyz: (N, 3) voxel coordinates (i, j, k)
    affine:  4x4 voxel->world matrix
    returns: (N, 3) world coordinates
    """
    vox_xyz = np.asarray(vox_xyz)
    ones = np.ones((vox_xyz.shape[0], 1))
    vox_h = np.concatenate([vox_xyz, ones], axis=1)      # (N, 4)
    world_h = vox_h @ affine.T                           # (N, 4)
    return world_h[:, :3]


def world_to_voxel(world_xyz, affine):
    """
    world_xyz: (N, 3) world coordinates
    affine:    4x4 voxel->world matrix
    returns:   (N, 3) voxel coordinates (float)
    """
    world_xyz = np.asarray(world_xyz)
    ones = np.ones((world_xyz.shape[0], 1))
    world_h = np.concatenate([world_xyz, ones], axis=1)  # (N, 4)
    affine_inv = np.linalg.inv(affine)                   # world->voxel
    vox_h = world_h @ affine_inv.T                       # (N, 4)
    return vox_h[:, :3]


# ---------- saving ----------

def save_input(save_path, idx, patch, patch_seg, patch_coord, patch_edge, num_saved):
    """
    Save patch image, seg, and graph (VTP).
    """
    save(
        patch,
        os.path.join(save_path, "raw", f"sample_{idx:06d}_{num_saved}_data.nii.gz"),
    )
    save(
        patch_seg,
        os.path.join(save_path, "seg", f"sample_{idx:06d}_{num_saved}_seg.nii.gz"),
    )

    patch_edge = np.concatenate(
        (np.int32(2 * np.ones((patch_edge.shape[0], 1))), patch_edge), axis=1
    )
    mesh = pyvista.PolyData(patch_coord)
    mesh.lines = patch_edge.flatten()
    mesh.save(
        os.path.join(save_path, "vtp", f"sample_{idx:06d}_{num_saved}_graph.vtp")
    )


# ---------- patch extraction with sliding window ----------

def patch_extract(save_path, image, seg, gen, affine, image_id,
                  max_patches=None,
                  stride=None,
                  overlap = 0.35,
                  device=None):

    """
    Extract patches with a sliding window (from the corner) and
    bring graph node coordinates into patch-local voxel space.
    """

    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    # effective patch size without padding
    eff_h = p_h - 2 * pad_h
    eff_w = p_w - 2 * pad_w
    eff_d = p_d - 2 * pad_d

    h, w, d = image.shape
    
    if stride is None:
        eff_sizes = np.array([eff_h, eff_w, eff_d], dtype=float)
        strides = np.rint(eff_sizes * (1.0 - float(overlap))).astype(int)
        strides = np.maximum(1, np.minimum(strides, eff_sizes.astype(int)))
        s_h, s_w, s_d = strides
    else:
        s_h, s_w, s_d = stride

    # start from the corner (0, 0, 0) in voxel space
    start_h_min = 0
    start_w_min = 0
    start_d_min = 0

    # last starting indices that still fit a full effective patch
    start_h_max = h - eff_h
    start_w_max = w - eff_w
    start_d_max = d - eff_d

    num_saved = 0

    for x in range(start_h_min, start_h_max + 1, s_h):
        for y in range(start_w_min, start_w_max + 1, s_w):
            for z in range(start_d_min, start_d_max + 1, s_d):

                if max_patches is not None and num_saved >= max_patches:
                    return num_saved

                # voxel-space patch bounds (effective region, before padding)
                start = np.array([x, y, z])
                end = start + np.array([eff_h, eff_w, eff_d]) - 1

                # extract image & seg patch (voxel, then pad)
                patch = np.pad(
                    image[x:x + eff_h, y:y + eff_w, z:z + eff_d],
                    ((pad_h, pad_h), (pad_w, pad_w), (pad_d, pad_d)),
                )
                patch_seg = np.pad(
                    seg[x:x + eff_h, y:y + eff_w, z:z + eff_d],
                    ((pad_h, pad_h), (pad_w, pad_w), (pad_d, pad_d)),
                )

                # Skip if bad SNR or too much/too little foreground
                fg_pixels = patch[patch_seg > 0]
                bg_pixels = patch[patch_seg == 0]

                if fg_pixels.size == 0 or bg_pixels.size == 0:
                    continue

                avg_intensity_fg = np.mean(fg_pixels)
                avg_intensity_bg = np.mean(bg_pixels)
                std_intensity_bg = np.std(bg_pixels) + 1e-8  # avoid /0

                snr = (avg_intensity_fg - avg_intensity_bg) / std_intensity_bg
                fgr = fg_pixels.size / patch.size

                if snr < 1.2 or fgr > 0.5:
                    print(f"skipping: {image_id}")
                    continue

                # ----- GRAPH PART -----
                # convert voxel bounds -> world bounds for PatchGraphGenerator
                corners_vox = np.stack([start, end], axis=0)      # (2, 3)
                corners_world = voxel_to_world(corners_vox, affine)  # (2, 3)

                bounds_world = np.array([
                    [corners_world[0, 0], corners_world[1, 0]],
                    [corners_world[0, 1], corners_world[1, 1]],
                    [corners_world[0, 2], corners_world[1, 2]],
                ])
                
                bounds_world = np.sort(bounds_world, axis=1)    # order the axis so that the min value is in the first col and the max in the second col
                
                gen.create_patch_graph(bounds_world)
                
                nodes, edges = gen.get_last_patch()

                if patch_seg.sum() <= 10 or len(nodes) < 3:
                    continue

                nodes = np.array(nodes)

                # nodes are in world coords -> bring into voxel coords
                coords_world = nodes[:, :3]
                coords_vox = world_to_voxel(coords_world, affine)

                # convert voxel coords -> patch-local coords (0..~patch_size),
                # accounting for padding
                local_coords = coords_vox - start + np.array(
                    [pad_h, pad_w, pad_d]
                )

                nodes[:, :3] = local_coords

                save_input(
                    save_path,
                    image_id,
                    patch,
                    patch_seg,
                    nodes,
                    np.array(edges),
                    num_saved
                )
                num_saved += 1
                
    return num_saved
                
                
# def main(args):
    
#     root_dir = Path(args.root)  # parent of raw/, seg/, graphs/
#     overlap = float(args.overlap)

#     raw_files = sorted((root_dir / "raw").glob("*.nii.gz"))
#     seg_files = sorted((root_dir / "seg").glob("*.nii.gz"))
#     graph_dirs = sorted((root_dir / "graphs").iterdir())
    
#     # simple function to extract just the name
#     def id_from_nii(path: Path) -> str:
#         # "1.nii.gz" -> "1"
#         return path.stem.split(".")[0]
    
#     raw_map = {id_from_nii(f): f for f in raw_files}
#     seg_map = {id_from_nii(f): f for f in seg_files}
#     graph_map = {d.name: d for d in graph_dirs if d.is_dir()}

#     # ----- TRAIN -----
#     train_path = root_dir / "patches" / "train"
#     if not train_path.is_dir():
#         seg_path_train = train_path / "seg"
#         vtp_path_train = train_path / "vtp"
#         raw_path_train = train_path / "raw"
        
#         seg_path_train.mkdir(parents=True, exist_ok=True)
#         vtp_path_train.mkdir(parents=True, exist_ok=True)
#         raw_path_train.mkdir(parents=True, exist_ok=True)


#     print("Preparing Train Data")
#     for seg_file in tqdm(seg_files):

#         file_id = id_from_nii(seg_file)
#         # print("seg_file:", seg_file, "id:", file_id)
        
#         raw_file = raw_map.get(file_id)
#         graph_dir = graph_map.get(file_id)
        
#         if raw_file is None or graph_dir is None:
#             print(f"WARNING: missing raw or graph for id {file_id}, skipping.")
#             continue
        
#         nodes_file       = graph_dir / "nodes.csv"
#         edges_file       = graph_dir / "edges.csv"
#         centerline_file  = graph_dir / "graph.vvg"

#         # image with affine (nibabel)
#         image_data, affine = load_image_and_affine(raw_file)
#         image_data = np.int32(image_data)

#         # seg with medpy
#         seg_data, _ = load(seg_file)
#         seg_data = np.int8(seg_data)

#         threshold = (
#             scipy.stats.median_abs_deviation(
#                 image_data.flatten(), scale="normal"
#             ) * 4 + np.median(image_data.flatten())
#         )
#         image_data[image_data > threshold] = threshold
#         image_data = image_data / threshold

#         gen = PatchGraphGenerator(
#             str(nodes_file),
#             str(edges_file),
#             str(centerline_file),
#             patch_mode="centerline",
#         )

#         patch_extract(
#             train_path,
#             image_data,
#             seg_data,
#             gen,
#             affine,
#             int(file_id),
#             max_patches=3,
#             overlap=overlap
#         )

#     # ----- TEST -----
#     test_path = root_dir / "patches" / "test"

#     seg_path_test = test_path / "seg"
#     vtp_path_test = test_path / "vtp"
#     raw_path_test = test_path / "raw"

#     seg_path_test.mkdir(parents=True, exist_ok=True)
#     vtp_path_test.mkdir(parents=True, exist_ok=True)
#     raw_path_test.mkdir(parents=True, exist_ok=True)

#     # split: first 21 segs -> train, rest -> test
#     seg_files_test = seg_files[21:]

#     print("Preparing Test Data")
#     for seg_file in tqdm(seg_files_test):
#         file_id = id_from_nii(seg_file)         # e.g. "23"

#         raw_file = raw_map.get(file_id)
#         graph_dir = graph_map.get(file_id)

#         if raw_file is None or graph_dir is None:
#             print(f"WARNING: missing raw or graph for id {file_id}, skipping.")
#             continue

#         nodes_file      = graph_dir / "nodes.csv"
#         edges_file      = graph_dir / "edges.csv"
#         centerline_file = graph_dir / "graph.vvg"

#         # image with affine (nibabel)
#         image_data, affine = load_image_and_affine(raw_file)
#         image_data = np.int32(image_data)

#         # seg with medpy
#         seg_data, _ = load(seg_file)
#         seg_data = np.int8(seg_data)

#         threshold = (
#             scipy.stats.median_abs_deviation(
#                 image_data.flatten(), scale="normal"
#             ) * 4
#             + np.median(image_data.flatten())
#         )
#         image_data[image_data > threshold] = threshold
#         image_data = image_data / threshold

#         gen = PatchGraphGenerator(
#             nodes_file,
#             edges_file,
#             centerline_file,
#             patch_mode="centerline",
#         )

#         # NOTE: assumes patch_extract signature includes file_name/file_id
#         patch_extract(
#             test_path,
#             image_data,
#             seg_data,
#             gen,
#             affine,
#             file_id,          # <-- replaces your old real_idx-based name
#             max_patches=3,
#             overlap=overlap,
#         )

def main(args):

    root_dir = Path(args.root)
    out_root = Path(args.out_root)
    overlap = float(args.overlap)

    # ---- collect raw/seg/graph files ----
    raw_files = sorted((root_dir / "raw").glob("*.nii.gz"))
    seg_files = sorted((root_dir / "seg").glob("*.nii.gz"))
    graph_dirs = sorted((root_dir / "graphs").iterdir())

    def id_from_nii(path: Path) -> str:
        # "1.nii.gz" -> "1"
        return path.stem.split(".")[0]

    raw_map  = {id_from_nii(f): f for f in raw_files}
    seg_map  = {id_from_nii(f): f for f in seg_files}
    graph_map = {d.name: d for d in graph_dirs if d.is_dir()}

    # ---- read split.csv → id -> {train,val,test} ----
    split_map = split_reader_nifti(Path(args.split))

    # bucket ids by split
    buckets = {"train": [], "val": [], "test": []}
    for seg_file in seg_files:
        fid = id_from_nii(seg_file)
        sp = split_map.get(fid, None)
        if sp is None:
            print(f"[skip] id {fid} not found in splits.csv")
            continue
        buckets[sp].append(fid)

    # ---- create output dirs per split ----
    out_roots = {}
    for sp in ("train", "val", "test"):
        root_sp = out_root / sp
        raw_sp  = root_sp / "raw"
        seg_sp  = root_sp / "seg"
        vtp_sp  = root_sp / "vtp"
        for d in (raw_sp, seg_sp, vtp_sp):
            d.mkdir(parents=True, exist_ok=True)
        out_roots[sp] = root_sp

    # desired total patches per split (may be None → uncapped)
    want = {
        "train": args.num_train,
        "val":   args.num_val,
        "test":  args.num_test,
    }

    # simple progress print
    print("Target patches:", want)
    total_target = sum(v for v in want.values() if v is not None)
    written_global = 0

    # ---- loop over splits ----
    for sp in ("train", "val", "test"):
        ids = buckets[sp]
        if not ids:
            print(f"[info] no {sp} ids in split.csv")
            continue

        root_sp = out_roots[sp]
        want_sp = want[sp] if want[sp] is not None else None

        # compute per-image quota for this split
        if want_sp is not None:
            n_img = len(ids)
            per_img_quota = max(1, int(np.ceil(float(want_sp) / max(1, n_img))))
        else:
            per_img_quota = 3   # fallback: same as old max_patches

        print(f"\nPreparing {sp} data (images={len(ids)}, per-image quota={per_img_quota})")
        written_sp = 0

        for fid in tqdm(ids, desc=f"{sp} volumes"):
            if want_sp is not None and written_sp >= want_sp:
                break

            raw_file  = raw_map.get(fid)
            seg_file  = seg_map.get(fid)
            graph_dir = graph_map.get(fid)

            if raw_file is None or seg_file is None or graph_dir is None:
                print(f"[WARNING] missing raw/seg/graph for id {fid}, skipping.")
                continue

            nodes_file      = graph_dir / "nodes.csv"
            edges_file      = graph_dir / "edges.csv"
            centerline_file = graph_dir / "graph.vvg"

            # image with affine (nibabel)
            image_data, affine = load_image_and_affine(raw_file)
            image_data = np.int32(image_data)

            # seg with medpy
            seg_data, _ = load(seg_file)
            seg_data = np.int8(seg_data)

            threshold = (
                scipy.stats.median_abs_deviation(
                    image_data.flatten(), scale="normal"
                ) * 4
                + np.median(image_data.flatten())
            )
            image_data[image_data > threshold] = threshold
            image_data = image_data / threshold

            gen = PatchGraphGenerator(
                str(nodes_file),
                str(edges_file),
                str(centerline_file),
                patch_mode="centerline",
            )

            # how many patches we are still allowed to write for this split
            if want_sp is not None:
                remaining = max(0, want_sp - written_sp)
                this_quota = min(per_img_quota, remaining)
            else:
                this_quota = per_img_quota

            if this_quota <= 0:
                break

            n_p = patch_extract(
                save_path=root_sp,
                image=image_data,
                seg=seg_data,
                gen=gen,
                affine=affine,
                image_id=int(fid),
                max_patches=this_quota,
                overlap=overlap,
            )

            written_sp += int(n_p)
            written_global += int(n_p)

            if want_sp is not None and written_sp >= want_sp:
                break

            if total_target and written_global >= total_target:
                print("\n[done] reached global target of patches.")
                return

    print(f"\n[done] wrote ~{written_global} patches in total.")

             


# ---------- main ----------

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--overlap", type=float, default=0.3, required=True, help="Overlap between consecutive patches (fraction)")
    ap.add_argument("--split", required=True, help="CSV file with columns: patient_id,split")
    ap.add_argument("--num_train", type=int, default=None, help="Target number of TRAIN patches (approx).")
    ap.add_argument("--num_val", type=int, default=None, help="Target number of VAL patches (approx).")
    ap.add_argument("--num_test", type=int, default=None, help="Target number of TEST patches (approx).")
    
    args = ap.parse_args(['--root', '/data/scavone/syntheticMRI',
                          '--out_root', '/data/scavone/syntheticMRI/patches',
                          '--split', '/data/scavone/syntheticMRI/splits.csv',
                          '--overlap', '0.35',
                          '--num_train', '4000', 
                          '--num_val', '1000',
                          '--num_test', '5000']) 
    
    
    main(args)
    
    
    
    
    

    
