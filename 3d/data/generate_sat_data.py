import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import os
import json
import csv
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d

patch_size = [128, 128, 1]
pad = [5, 5, 0]

total_min = 1
total_max = 0

# Global patch budgets and counters per split
patch_budget = {
    "train": None,
    "val": None,
    "test": None,
}

patch_count = {
    "train": 0,
    "val": 0,
    "test": 0,
}


def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min=-1, a_max=1))


def convert_graph(graph):
    node_list = []
    edge_list = []
    for n, v in graph.items():
        node_list.append(n)
    node_array = np.array(node_list)

    for ind, (n, v) in enumerate(graph.items()):
        for nei in v:
            idx = node_list.index(nei)
            edge_list.append(np.array((ind, idx)))
    edge_array = np.array(edge_list)
    return node_array, edge_array


vector_norm = 25.0


def neighbor_transpos(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (k[1], k[0])
        nv = []

        for _v in v:
            nv.append((_v[1], _v[0]))

        n_out[nk] = nv

    return n_out


def neighbor_to_integer(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))

        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []

        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))

            if new_n_k in nv:
                pass
            else:
                nv.append(new_n_k)

        n_out[nk] = nv

    return n_out


def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    """Save a single patch (image, seg, and graph) to disk."""

    global total_min
    global total_max
    imageio.imwrite(path + 'raw/sample_' + str(idx).zfill(6) + '_data.png', patch)
    imageio.imwrite(path + 'seg/sample_' + str(idx).zfill(6) + '_seg.png', patch_seg)

    patch_edge = np.concatenate(
        (np.int32(2 * np.ones((patch_edge.shape[0], 1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    cur_min = np.min(patch_coord[:, :2])
    cur_max = np.max(patch_coord[:, :2])
    if cur_min < total_min:
        total_min = cur_min
        print(total_min)
    if cur_max > total_max:
        total_max = cur_max
        print(total_max)
    mesh.lines = patch_edge.flatten()
    mesh.save(path + 'vtp/sample_' + str(idx).zfill(6) + '_graph.vtp')


def patch_extract(save_path, image, seg, mesh, split_name, max_patches_for_image=None, device=None, overlap=0.48):
    """Extract patches from an image and save them, respecting per-split budgets.

    Args:
        save_path (str): Base directory for this split (contains raw/ seg/ vtp/).
        image (np.ndarray): Input image (H, W, C).
        seg (np.ndarray): Segmentation mask (H, W).
        mesh (pyvista.PolyData): Full graph mesh.
        split_name (str): One of "train", "val", "test".
        max_patches_for_image (int, optional): Max patches to save from this image.
        device: Unused; kept for API compatibility.
    """
    global image_id, patch_budget, patch_count

    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad

    # effective (unpadded) patch size
    p_h = p_h - 2 * pad_h
    p_w = p_w - 2 * pad_w

    h, w, d = image.shape

    # --- tunable overlap: stride from overlap in [0, 1) ---
    # S is the effective patch size in each dimension
    S_h = p_h
    S_w = p_w

    # stride = S * (1 - overlap), clamped to [1, S]
    stride_h = int(round(S_h * (1.0 - float(overlap))))
    stride_w = int(round(S_w * (1.0 - float(overlap))))

    stride_h = max(1, min(stride_h, S_h))
    stride_w = max(1, min(stride_w, S_w))

    # starting coordinate offset used in your original code
    margin = 5

    start_y_list = list(range(margin, h - margin - p_h + 1, stride_h))
    start_x_list = list(range(margin, w - margin - p_w + 1, stride_w))

    # ensure last patch touches the end (like your linspace did)
    last_y = h - margin - p_h
    last_x = w - margin - p_w
    if start_y_list[-1] != last_y:
        start_y_list.append(last_y)
    if start_x_list[-1] != last_x:
        start_x_list.append(last_x)

    num_saved_from_image = 0

    for sy in start_y_list:
        for sx in start_x_list:
            # Global and per-image budget checks
            if patch_budget[split_name] is not None and patch_count[split_name] >= patch_budget[split_name]:
                return num_saved_from_image
            if max_patches_for_image is not None and num_saved_from_image >= max_patches_for_image:
                return num_saved_from_image

            start = np.array((sy, sx, 0))
            end = start + np.array(patch_size) - 1 - 2 * np.array(pad)

            patch = np.pad(
                image[start[0]:start[0] + p_h, start[1]:start[1] + p_w, :],
                ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
            )
            patch_list = [patch]

            patch_seg = np.pad(
                seg[start[0]:start[0] + p_h, start[1]:start[1] + p_w],
                ((pad_h, pad_h), (pad_w, pad_w))
            )
            seg_list = [patch_seg]

            input = torch.from_numpy(patch_seg).unsqueeze(0).unsqueeze(0).float() / 255.

            weights = torch.tensor([[1., 1., 1.],
                                    [1., 1., 1.],
                                    [1., 1., 1.]])
            weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1)

            x = conv2d(input, weight=weights, padding=1)
            for _ in range(3):
                x = conv2d(x, weight=weights, padding=1)

            aug_img = x[0][0]
            aug_img[aug_img > 1] = 1
            aug_img *= 255
            patch_list = [aug_img.byte()]

            bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]
            clipped_mesh = mesh.clip_box(bounds, invert=False)
            patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
            patch_edge = clipped_mesh.cells[np.sum(
                clipped_mesh.celltypes == 1) * 2:].reshape(-1, 3)

            patch_coord_ind = np.where(
                (np.prod(patch_coordinates >= start, 1) * np.prod(patch_coordinates <= end, 1)) > 0.0)
            patch_coordinates = patch_coordinates[patch_coord_ind[0], :]
            patch_edge = [tuple(l) for l in patch_edge[:, 1:] if l[0]
                        in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

            temp = np.array(patch_edge).flatten()
            temp = [np.where(patch_coord_ind[0] == ind_) for ind_ in temp]
            patch_edge = np.array(temp).reshape(-1, 2)

            if patch_coordinates.shape[0] < 2 or patch_edge.shape[0] < 1 or patch_coordinates.shape[0] > 80:
                continue

            patch_coordinates = (patch_coordinates - start + np.array(pad)) / np.array(patch_size)
            patch_coord_list = [patch_coordinates]
            patch_edge_list = [patch_edge]

            mod_patch_coord_list, mod_patch_edge_list = prune_patch(
                patch_coord_list, patch_edge_list)

            for patch, patch_seg, patch_coord, patch_edge in zip(
                    patch_list, seg_list, mod_patch_coord_list, mod_patch_edge_list):
                if patch_seg.sum() > 10:
                    # Re-check budgets just before saving
                    if patch_budget[split_name] is not None and patch_count[split_name] >= patch_budget[split_name]:
                        return num_saved_from_image

                    if max_patches_for_image is not None and num_saved_from_image >= max_patches_for_image:
                        return num_saved_from_image

                    save_input(save_path, image_id, patch, patch_seg, patch_coord, patch_edge)
                    image_id = image_id + 1
                    patch_count[split_name] += 1
                    num_saved_from_image += 1

    return num_saved_from_image


def prune_patch(patch_coord_list, patch_edge_list):
    """Prune patch graphs by removing nearly collinear degree-2 nodes."""
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):

        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:, 0], edge[:, 1]] = np.sum(
            (coord[edge[:, 0], :] - coord[edge[:, 1], :]) ** 2, 1)
        dist_adj[edge[:, 1], edge[:, 0]] = np.sum(
            (coord[edge[:, 0], :] - coord[edge[:, 1], :]) ** 2, 1)

        start = True
        node_mask = np.ones(coord.shape[0], dtype=bool)
        while start:
            degree = (dist_adj > 0).sum(1)
            deg_2 = list(np.where(degree == 2)[0])
            if len(deg_2) == 0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx, :] > 0)[0]

                p1 = coord[idx, :]
                p2 = coord[deg_2_neighbor[0], :]
                p3 = coord[deg_2_neighbor[1], :]
                l1 = p2 - p1
                l2 = p3 - p1
                node_angle = angle(l1, l2) * 180 / math.pi
                if node_angle > 160:
                    node_mask[idx] = False
                    dist_adj[deg_2_neighbor[0],
                             deg_2_neighbor[1]] = np.sum((p2 - p3) ** 2)
                    dist_adj[deg_2_neighbor[1],
                             deg_2_neighbor[0]] = np.sum((p2 - p3) ** 2)

                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    break
                elif n == len(deg_2) - 1:
                    start = False

        new_coord = coord[node_mask, :]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj) > 0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list


parser = ArgumentParser()
parser.add_argument('--source',
                    default=None,
                    required=True,
                    help='Path to source directory')
parser.add_argument('--target',
                    default=None,
                    required=True,
                    help='Path to target directory')
parser.add_argument('--source_number',
                    default=None,
                    type=int,
                    required=True,
                    help='Number of images in source directory')
parser.add_argument('--split',
                    default=0.8,
                    type=float,
                    help='(Unused if --split_csv is provided) Train/Test split.')
parser.add_argument('--city_names',
                    default=None,
                    required=False,
                    help='Path to json with city names that are prefixing the raw source images')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    required=False,
                    help='Random seed')
parser.add_argument('--split_csv',
                    default=None,
                    type=str,
                    required=True,
                    help='CSV file with columns id,split (train/val/test)')
parser.add_argument('--train_patches',
                    type=int,
                    required=True,
                    help='Target number of patches for training split')
parser.add_argument('--val_patches',
                    type=int,
                    required=True,
                    help='Target number of patches for validation split')
parser.add_argument('--test_patches',
                    type=int,
                    required=True,
                    help='Target number of patches for test split')
parser.add_argument('--overlap',
                    type=float,
                    default=0.5,
                    help='Fractional overlap between adjacent patches in [0, 1). 0.0 = no overlap, 0.5 = 50% overlap, 0.9 = very dense grid.')

image_id = 1


def generate_data(args):
    global image_id, patch_budget, patch_count

    root_dir = args.source
    target_dir = args.target
    amount_images = args.source_number
    overlap = args.overlap

    # If data has prefix with city names, a json with all names must be provided
    cities = []
    if args.city_names is not None:
        dataset_cfg_ = json.load(open(args.city_names, "r"))
        for item in dataset_cfg_:
            cities.append({"name": item["cityname"], "id": item["id"]})

    # Sets the seed for reproducibility
    random.seed(args.seed)

    # Initialize budgets and counters
    patch_budget["train"] = args.train_patches
    patch_budget["val"] = args.val_patches
    patch_budget["test"] = args.test_patches

    patch_count["train"] = 0
    patch_count["val"] = 0
    patch_count["test"] = 0

    indrange_train = []
    indrange_val = []
    indrange_test = []

    # Read split information from CSV
    with open(args.split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'patient_id' not in row or 'split' not in row:
                continue

            pid = row['patient_id'].strip()

            # Expect format like: region_0, region_10, region_100
            if pid.startswith("region_"):
                num_part = pid.replace("region_", "")
                # In case something like region_0_extra appears:
                num_part = num_part.split("_")[0]
                try:
                    idx = int(num_part)
                except ValueError:
                    continue
            else:
                continue

            # Range check
            if idx < 0 or idx >= amount_images:
                continue

            # Assign to split
            s = row['split'].strip().lower()
            if s in ('train', 'training'):
                indrange_train.append(idx)
            elif s in ('val', 'valid', 'validation'):
                indrange_val.append(idx)
            elif s in ('test', 'testing'):
                indrange_test.append(idx)


    # ------------------ TRAIN ------------------
    image_id = 1
    train_path = f"{target_dir}/train/"
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path + '/seg')
        os.makedirs(train_path + '/vtp')
        os.makedirs(train_path + '/raw')
    else:
        raise Exception("Train folder is non-empty")
    print('Preparing Train Data')

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_train:
        raw_files.append(f"{root_dir}/raw/region_{ind}_sat")
        seg_files.append(f"{root_dir}/raw/region_{ind}_gt.png")
        vtk_files.append(f"{root_dir}/vtp/region_{ind}_gt_graph.pickle")

    num_train_images = len(raw_files)

    for i in tqdm(range(num_train_images)):
        if patch_count["train"] >= patch_budget["train"]:
            break
        # print('Train image', i)
        try:
            sat_img = imageio.imread(raw_files[i] + ".png")
        except Exception:
            sat_img = imageio.imread(raw_files[i] + ".jpg")

        with open(vtk_files[i], 'rb') as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)

        gt_seg = imageio.imread(seg_files[i])
        patch_coord = np.concatenate(
            (node_array, np.int32(np.zeros((node_array.shape[0], 1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate(
            (np.int32(2 * np.ones((edge_array.shape[0], 1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()

        remaining_images = max(num_train_images - i, 1)
        remaining_patches = patch_budget["train"] - patch_count["train"]
        max_for_this_image = max(0, math.ceil(remaining_patches / remaining_images))

        patch_extract(train_path, sat_img, gt_seg, mesh,
                      split_name="train",
                      max_patches_for_image=max_for_this_image, overlap=overlap)

    # ------------------ VAL ------------------
    image_id = 1
    val_path = f"{target_dir}/val/"
    if not os.path.isdir(val_path):
        os.makedirs(val_path)
        os.makedirs(val_path + '/seg')
        os.makedirs(val_path + '/vtp')
        os.makedirs(val_path + '/raw')
    else:
        raise Exception("Val folder is non-empty")
    print('Preparing Val Data')

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_val:
        raw_files.append(f"{root_dir}/raw/region_{ind}_sat")
        seg_files.append(f"{root_dir}/raw/region_{ind}_gt.png")
        vtk_files.append(f"{root_dir}/vtp/region_{ind}_gt_graph.pickle")

    num_val_images = len(raw_files)

    for i in tqdm(range(num_val_images)):
        if patch_count["val"] >= patch_budget["val"]:
            break
        # print('Val image', i)
        try:
            sat_img = imageio.imread(raw_files[i] + ".png")
        except Exception:
            sat_img = imageio.imread(raw_files[i] + ".jpg")

        with open(vtk_files[i], 'rb') as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)

        gt_seg = imageio.imread(seg_files[i])
        patch_coord = np.concatenate(
            (node_array, np.int32(np.zeros((node_array.shape[0], 1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate(
            (np.int32(2 * np.ones((edge_array.shape[0], 1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()

        remaining_images = max(num_val_images - i, 1)
        remaining_patches = patch_budget["val"] - patch_count["val"]
        max_for_this_image = max(0, math.ceil(remaining_patches / remaining_images))

        patch_extract(val_path, sat_img, gt_seg, mesh,
                      split_name="val",
                      max_patches_for_image=max_for_this_image, overlap=overlap)

    # ------------------ TEST ------------------
    image_id = 1
    test_path = f"{target_dir}/test/"
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
        os.makedirs(test_path + '/seg')
        os.makedirs(test_path + '/vtp')
        os.makedirs(test_path + '/raw')
    else:
        raise Exception("Test folder is non-empty")

    print('Preparing Test Data')

    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_test:
        raw_files.append(f"{root_dir}/raw/region_{ind}_sat")
        seg_files.append(f"{root_dir}/raw/region_{ind}_gt.png")
        vtk_files.append(f"{root_dir}/vtp/region_{ind}_gt_graph.pickle")

    num_test_images = len(raw_files)

    for i in tqdm(range(num_test_images)):
        if patch_count["test"] >= patch_budget["test"]:
            break
        # print('Test image', i)
        try:
            sat_img = imageio.imread(raw_files[i] + ".png")
        except Exception:
            sat_img = imageio.imread(raw_files[i] + ".jpg")

        with open(vtk_files[i], 'rb') as f:
            graph = pickle.load(f)
        node_array, edge_array = convert_graph(graph)

        gt_seg = imageio.imread(seg_files[i])
        patch_coord = np.concatenate(
            (node_array, np.int32(np.zeros((node_array.shape[0], 1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate(
            (np.int32(2 * np.ones((edge_array.shape[0], 1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()

        remaining_images = max(num_test_images - i, 1)
        remaining_patches = patch_budget["test"] - patch_count["test"]
        max_for_this_image = max(0, math.ceil(remaining_patches / remaining_images))

        patch_extract(test_path, sat_img, gt_seg, mesh,
                      split_name="test",
                      max_patches_for_image=max_for_this_image, overlap=overlap)


if __name__ == "__main__":
    args = parser.parse_args([
        '--source', '/data/scavone/20cities',
        '--target', '/data/scavone/20cities/patches', 
        '--source_number', '180',
        '--split_csv', '/data/scavone/20cities/splits.csv',
        '--train_patches', '99200',
        '--val_patches', '24800',
        '--test_patches', '25000',
        '--overlap', '0.35'
    ])
    generate_data(args)
