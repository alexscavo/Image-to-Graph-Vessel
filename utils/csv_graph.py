import sys
import numpy as np
import pandas as pd

from pathlib import Path

import torch


def csv_graph_loading(path):
    
    path = Path(path)   
    nodes_path = path / 'nodes.csv'
    edges_path = path / 'edges.csv'
    
    nodes_file = pd.read_csv(nodes_path, index_col="id")
    edges_file = pd.read_csv(edges_path, index_col="id")
    
    nodes = torch.tensor(nodes_file.to_numpy()[:, :3].astype(np.float32))
    edges = torch.tensor(edges_file.to_numpy()[:, :2].astype(int))
        
    return nodes, edges

def csv_graph_saving(path, nodes, edges):
    """
    Save graph to CSV files compatible with csv_graph_loading.
    Ensures nodes always have 3 coords (x, y, z), filling z=0 if missing.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # convert to numpy
    nodes_np = nodes.detach().cpu().numpy() if torch.is_tensor(nodes) else np.asarray(nodes)
    edges_np = edges.detach().cpu().numpy() if torch.is_tensor(edges) else np.asarray(edges)

    # ensure 2D with at least 2 columns
    if nodes_np.ndim != 2 or nodes_np.shape[1] < 2:
        raise ValueError("nodes must be a 2D array with at least x,y columns")

    # pad z=0 if missing
    if nodes_np.shape[1] == 2:
        z_col = np.zeros((nodes_np.shape[0], 1), dtype=np.float32)
        nodes_np = np.hstack([nodes_np, z_col])
    elif nodes_np.shape[1] > 3:
        nodes_np = nodes_np[:, :3]  # truncate extras

    # build dataframes
    nodes_df = pd.DataFrame(nodes_np.astype(np.float32), columns=["x", "y", "z"])
    nodes_df.index.name = "id"

    edges_df = pd.DataFrame(edges_np[:, :2].astype(np.int64), columns=["source", "target"])
    edges_df.index.name = "id"

    # save
    nodes_df.to_csv(path / "nodes.csv")
    edges_df.to_csv(path / "edges.csv")