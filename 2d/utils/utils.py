import math
import torch
import numpy as np
import pyvista
import json
from math import radians, sin, cos


def rotate_coordinates(points, angle, dataset_name=None):
    # points: (N, 2) in [0, 1], (x, y)
    rad = math.radians(angle)

    R = torch.tensor(
        [[math.cos(rad), -math.sin(rad)],
         [math.sin(rad),  math.cos(rad)]],
        dtype=points.dtype,
        device=points.device,
    )

    centered = points - 0.5        # rotate around center (0.5, 0.5)
    rotated = centered @ R.T       # NOTE: transpose here
    return rotated + 0.5


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def image_graph_collate(batch):
    images = torch.cat(
        [item_ for item in batch for item_ in item[0]], 0).contiguous()
    points = [item_ for item in batch for item_ in item[1]]
    edges = [item_ for item in batch for item_ in item[2]]
    return [images, points, edges]


def image_graph_collate_road_network(batch):
    images = torch.stack([item[0] for item in batch], 0).contiguous()
    seg = torch.stack([item[1] for item in batch], 0).contiguous()
    points = [item[2] for item in batch]
    edges = [item[3] for item in batch]
    domains = torch.tensor([item[4] for item in batch])
    return [images, seg, points, edges, domains]


def save_input(path, idx, patch, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """

    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)

    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'_sample_'+str(idx).zfill(3)+'_segmentation.stl')

    patch_edge = np.concatenate(
        (np.int32(2*np.ones((patch_edge.shape[0], 1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')


def save_output(path, idx, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    print('Num nodes:', patch_coord.shape[0],
          'Num edges:', patch_edge.shape[0])
    patch_edge = np.concatenate(
        (np.int32(2*np.ones((patch_edge.shape[0], 1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    if patch_edge.shape[0] > 0:
        mesh.lines = patch_edge.flatten()
    mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')


def Bresenham3D(p1, p2):
    """
    Function to compute direct connection in voxel space
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def upsample_edges(relation_pred, edge_labels, sample_ratio, acceptance_interval):
    """
    Upsamples the edges in a relation prediction tensor based on the given sample ratio and acceptance interval.

    Args:
        relation_pred (torch.Tensor): Tensor containing the relation predictions.
        edge_labels (torch.Tensor): Tensor containing the labels for each edge.
        sample_ratio (float): The desired ratio of positive edges to negative edges.
        acceptance_interval (float): The acceptable deviation from the desired sample ratio.

    Returns:
        torch.Tensor: Tensor containing the upsampled relation predictions.
        torch.Tensor: Tensor containing the upsampled edge labels.
    """
    target_pos_edges = relation_pred[edge_labels == 1]
    target_neg_edges = relation_pred[edge_labels == 0]

    actual_ratio = target_pos_edges.shape[0] / target_neg_edges.shape[0]

    if actual_ratio < sample_ratio - acceptance_interval:
        target_num = target_neg_edges.shape[0] * sample_ratio
        num_new_edges = int(target_num - target_pos_edges.shape[0])

        new_edges = target_pos_edges.repeat(int(num_new_edges / target_pos_edges.shape[0]), 1)
        new_edges = torch.cat((new_edges, target_pos_edges[:int(num_new_edges - new_edges.shape[0])]))

        new_labels = torch.ones(num_new_edges, dtype=torch.long, device=relation_pred.device)
    elif sample_ratio + acceptance_interval < actual_ratio:
        target_num = target_pos_edges.shape[0] * (1 / sample_ratio)
        num_new_edges = int(target_num - target_neg_edges.shape[0])

        new_edges = target_neg_edges.repeat(int(num_new_edges / target_neg_edges.shape[0]), 1)
        new_edges = torch.cat((new_edges, target_neg_edges[:int(num_new_edges - new_edges.shape[0])]))

        new_labels = torch.zeros(num_new_edges, dtype=torch.long, device=relation_pred.device)
    else:
        return relation_pred, edge_labels

    return torch.cat((relation_pred, new_edges)), torch.cat((edge_labels, new_labels))

def downsample_edges(relation_pred, edge_labels, sample_ratio, acceptance_interval):
    """
    Downsamples the edges based on the given sample ratio and acceptance interval.

    Args:
        relation_pred (torch.Tensor): The predicted relation values.
        edge_labels (torch.Tensor): The labels for the edges.
        sample_ratio (float): The desired ratio of positive to negative edges.
        acceptance_interval (float): The acceptable deviation from the sample ratio.

    Returns:
        tuple: A tuple containing the downsampled relation predictions and edge labels.
    """
    target_pos_edges = relation_pred[edge_labels == 1]
    target_neg_edges = relation_pred[edge_labels == 0]

    actual_ratio = target_pos_edges.shape[0] / target_neg_edges.shape[0]

    if actual_ratio < sample_ratio - acceptance_interval:
        # In this case, we have too many negative edges, so we need to remove some
        target_num = int(target_pos_edges.shape[0] * (1 / sample_ratio))

        target_neg_edges = target_neg_edges[:target_num]
    elif sample_ratio + acceptance_interval < actual_ratio:
        # In this case, we have too many positive edges, so we need to remove some
        target_num = int(target_neg_edges.shape[0] * sample_ratio)

        target_pos_edges = target_pos_edges[:target_num]
    else:
        return relation_pred, edge_labels
    
    return (
        torch.cat((target_pos_edges, target_neg_edges)), 
        torch.cat(
            (torch.ones(target_pos_edges.shape[0], dtype=torch.long, device=relation_pred.device), 
             torch.zeros(target_neg_edges.shape[0], dtype=torch.long, device=relation_pred.device)))
        )


def _to_edge_index(lines: torch.Tensor) -> torch.Tensor:
    """
    Convert various edge encodings to shape [E, 2] long tensor.
    - If already [E,2], keep.
    - If VTK lines format flattened or [E,3] with first col being '2', convert.
    """
    if lines.numel() == 0:
        return lines.view(0, 2).long()

    if lines.ndim == 2 and lines.shape[1] == 2:
        return lines.long()

    # VTK sometimes gives [E,3] = [2, u, v]
    if lines.ndim == 2 and lines.shape[1] == 3:
        return lines[:, 1:3].long()

    # Flattened VTK: [2, u, v, 2, u, v, ...]
    if lines.ndim == 1:
        flat = lines.long()
        flat = flat.view(-1, 3)
        return flat[:, 1:3].long()

    raise ValueError(f"Unsupported lines shape: {tuple(lines.shape)}")

def _bridge_severity_for_edges(num_nodes: int, edges_e2: torch.Tensor) -> torch.Tensor:
    """
    Compute severity per undirected edge in `edges_e2` aligned with input order.
    Severity definition: min(sideA, sideB)/component_size for bridge edges; 0 otherwise.
    Range [0, 0.5]. Works on forests too.
    """
    E = edges_e2.shape[0]
    if E == 0 or num_nodes == 0:
        return torch.zeros((E,), dtype=torch.float32)

    # Build adjacency list (undirected)
    adj = [[] for _ in range(num_nodes)]
    for (u, v) in edges_e2.tolist():
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)

    # Find connected component id + size for each node (so severity is per-component)
    comp_id = [-1] * num_nodes
    comp_sizes = []
    cid = 0
    for s in range(num_nodes):
        if comp_id[s] != -1:
            continue
        stack = [s]
        comp_id[s] = cid
        nodes = []
        while stack:
            x = stack.pop()
            nodes.append(x)
            for y in adj[x]:
                if comp_id[y] == -1:
                    comp_id[y] = cid
                    stack.append(y)
        comp_sizes.append(len(nodes))
        cid += 1

    # Tarjan bridge-finding with subtree sizes (per component root)
    tin = [-1] * num_nodes
    low = [-1] * num_nodes
    parent = [-1] * num_nodes
    sub = [0] * num_nodes
    timer = 0

    # Map undirected edge -> severity for lookup
    bridge_sev = {}

    def dfs(u: int):
        nonlocal timer
        tin[u] = low[u] = timer
        timer += 1
        sub[u] = 1
        for v in adj[u]:
            if v == parent[u]:
                continue
            if tin[v] != -1:
                low[u] = min(low[u], tin[v])
            else:
                parent[v] = u
                dfs(v)
                sub[u] += sub[v]
                low[u] = min(low[u], low[v])

                # Bridge condition
                if low[v] > tin[u]:
                    csize = comp_sizes[comp_id[u]]
                    a = sub[v]
                    b = csize - a
                    sev = float(min(a, b)) / float(csize) if csize > 0 else 0.0
                    key = (u, v) if u < v else (v, u)
                    bridge_sev[key] = sev

    for s in range(num_nodes):
        if tin[s] == -1:
            parent[s] = -1
            dfs(s)

    # Produce severity aligned with edge order
    out = torch.zeros((E,), dtype=torch.float32)
    for i, (u, v) in enumerate(edges_e2.tolist()):
        key = (u, v) if u < v else (v, u)
        out[i] = bridge_sev.get(key, 0.0)
    return out
