"""
Analyze GT graph statistics (nodes/edges, positives/negatives, density, degree, components).

This script is modeled after `train_with_ema.py` but **does not train**.
It builds the same datasets/dataloaders and iterates through them to compute
graph-level stats.

Typical usage:
    python analyze_gt_graphs.py --config /path/to/config.yaml --split both --max-batches 200

Outputs:
    - JSON summary (aggregate stats)
    - CSV with per-sample stats (optional)

Notes:
    - "positive edges" = GT edges present in the graph
    - "negative edges" = all possible undirected node pairs that are NOT GT edges
      (i.e., C(n,2) - |E_undirected|)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functools import partial
from pathlib import Path
from argparse import ArgumentParser
import json
import math
import tqdm
import yaml
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.data import DataLoader

from utils.utils import image_graph_collate

# dataset builders (same as train_with_ema.py)
from data.dataset_mixed import build_mixed_data
from data.dataset_road_network import build_road_network_data
from data.dataset_plants import build_plants_network_data


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


@dataclass
class GraphStats:
    split: str
    sample_index: int
    domain: int

    n_nodes: int
    pos_edges: int
    neg_edges: int
    total_possible_edges: int
    density: float

    # degree stats (undirected)
    degree_min: int
    degree_max: int
    degree_mean: float
    degree_median: float

    # node-degree distribution helpers (over nodes, not graphs)
    node_degree_min: int
    node_degree_max: int
    node_degree_sum: int
    node_degree_sumsq: int
    node_deg0: int
    node_deg1: int
    node_deg2: int
    node_deg_ge3: int
    node_degree_hist_0_10: List[int]  # counts of degrees 0..10, last index=10

    # connectivity
    n_components: int
    largest_component_size: int
    n_bridges: int
    bridge_fraction: float
    bridge_incident_node_fraction: float
    bridge_severity_mean: float
    bridge_severity_max: float


def _edges_to_E2(edges: torch.Tensor) -> torch.Tensor:
    """
    Normalize various edge encodings to shape [E,2] int64.

    Supports:
      - [E,2] (already OK)
      - [E,3] VTK style: [2, u, v] per row -> drop first column
      - flat [2E] -> reshape
    """
    if edges is None:
        return torch.zeros((0, 2), dtype=torch.int64)

    if not torch.is_tensor(edges):
        edges = torch.as_tensor(edges)

    if edges.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.int64)

    edges = edges.to(dtype=torch.int64)

    if edges.ndim == 2 and edges.shape[1] == 2:
        return edges

    if edges.ndim == 2 and edges.shape[1] == 3:
        # VTK lines: [2, u, v]
        return edges[:, 1:3]

    if edges.ndim == 1 and edges.numel() % 2 == 0:
        return edges.view(-1, 2)

    raise ValueError(f"Unsupported edge tensor shape: {tuple(edges.shape)}")


def _unique_undirected_edges(edges: torch.Tensor) -> np.ndarray:
    """
    Convert edges [E,2] (directed or undirected, possibly duplicated)
    into unique undirected edges as sorted pairs.
    Returns numpy int64 array [E_u, 2].
    """
    edges = _edges_to_E2(edges)

    if edges.numel() == 0:
        return np.zeros((0, 2), dtype=np.int64)

    e = edges.detach().cpu().numpy().astype(np.int64)
    e = np.sort(e, axis=1)  # undirected canonical form

    # remove self-loops (just in case)
    e = e[e[:, 0] != e[:, 1]]
    if e.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # unique rows
    e = np.unique(e, axis=0)
    return e


def _compute_components(n_nodes: int, undirected_edges: np.ndarray) -> Tuple[int, int]:
    """
    Compute connected components for an undirected simple graph.
    Returns (n_components, largest_component_size).
    """
    if n_nodes <= 0:
        return 0, 0
    adj = [[] for _ in range(n_nodes)]
    for u, v in undirected_edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            adj[u].append(v)
            adj[v].append(u)
    seen = [False] * n_nodes
    comps = []
    for i in range(n_nodes):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        sz = 0
        while stack:
            x = stack.pop()
            sz += 1
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        comps.append(sz)
    n_components = len(comps)
    largest = max(comps) if comps else 0
    return n_components, largest


def _bridge_metrics(n_nodes: int, undirected_edges: np.ndarray):
    """
    Bridge metrics for an undirected simple graph.

    Returns:
      n_bridges: int
      bridge_inc_nodes: set[int]  (nodes incident to any bridge)
      sev_mean: float             (mean severity over bridges)
      sev_max: float              (max severity over bridges)

    Severity for a bridge edge is:
        min(subtree_size, component_size - subtree_size) / component_size
    where subtree_size refers to the DFS-child side of the bridge inside its connected component.
    """
    if n_nodes <= 1 or undirected_edges is None or undirected_edges.size == 0:
        return 0, set(), 0.0, 0.0

    # Build adjacency
    adj = [[] for _ in range(n_nodes)]
    for u, v in undirected_edges:
        u = int(u)
        v = int(v)
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adj[u].append(v)
            adj[v].append(u)

    tin = [-1] * n_nodes
    low = [0] * n_nodes
    sub_size = [0] * n_nodes
    comp_id = [-1] * n_nodes

    timer = 0
    bridges_child = []  # store (parent, child) where child is DFS child and edge is a bridge

    def dfs(v: int, p: int, cid: int):
        nonlocal timer
        comp_id[v] = cid
        tin[v] = timer
        low[v] = timer
        timer += 1
        sub_size[v] = 1

        for to in adj[v]:
            if to == p:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v, cid)
                sub_size[v] += sub_size[to]
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges_child.append((v, to))

    # Run DFS per connected component, assign comp_id
    cid = 0
    for v in range(n_nodes):
        if tin[v] == -1:
            dfs(v, -1, cid)
            cid += 1

    # Component sizes
    comp_sizes = [0] * cid
    for v in range(n_nodes):
        c = comp_id[v]
        if c >= 0:
            comp_sizes[c] += 1

    # Compute severity + incident nodes
    bridge_inc_nodes = set()
    severities = []

    for p, ch in bridges_child:
        bridge_inc_nodes.add(int(p))
        bridge_inc_nodes.add(int(ch))

        c = comp_id[p]
        comp_size = comp_sizes[c] if c >= 0 else n_nodes

        a = sub_size[ch]
        b = comp_size - a
        sev = (min(a, b) / comp_size) if comp_size > 0 else 0.0
        severities.append(float(sev))

    n_bridges = len(bridges_child)
    if severities:
        sev_mean = float(np.mean(severities))
        sev_max = float(np.max(severities))
    else:
        sev_mean = 0.0
        sev_max = 0.0

    return int(n_bridges), bridge_inc_nodes, sev_mean, sev_max


def compute_graph_stats(
    split: str,
    sample_index: int,
    domain: int,
    nodes: torch.Tensor,
    edges: torch.Tensor,
) -> GraphStats:
    n_nodes = int(nodes.shape[0]) if nodes is not None else 0
    undirected_edges = _unique_undirected_edges(edges)
    pos_edges = int(undirected_edges.shape[0])

    total_possible = int(n_nodes * (n_nodes - 1) // 2) if n_nodes >= 2 else 0
    neg_edges = int(max(total_possible - pos_edges, 0))

    density = float(pos_edges / total_possible) if total_possible > 0 else 0.0

    # degrees
    deg = np.zeros((n_nodes,), dtype=np.int64)
    for u, v in undirected_edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            deg[u] += 1
            deg[v] += 1

    if n_nodes > 0:
        degree_min = int(deg.min())
        degree_max = int(deg.max())
        degree_mean = float(deg.mean())
        degree_median = float(np.median(deg))
    else:
        degree_min = degree_max = 0
        degree_mean = degree_median = 0.0

    # node-level degree helpers (across nodes)
    if n_nodes > 0:
        node_degree_min = int(deg.min())
        node_degree_max = int(deg.max())
        node_degree_sum = int(deg.sum())
        node_degree_sumsq = int((deg.astype(np.int64) ** 2).sum())
        node_deg0 = int((deg == 0).sum())
        node_deg1 = int((deg == 1).sum())
        node_deg2 = int((deg == 2).sum())
        node_deg_ge3 = int((deg >= 3).sum())
        node_degree_hist_0_10 = [int((deg == k).sum()) for k in range(0, 11)]
    else:
        node_degree_min = 0
        node_degree_max = 0
        node_degree_sum = 0
        node_degree_sumsq = 0
        node_deg0 = 0
        node_deg1 = 0
        node_deg2 = 0
        node_deg_ge3 = 0
        node_degree_hist_0_10 = [0] * 11

    n_components, largest_component_size = _compute_components(n_nodes, undirected_edges)
    n_bridges, bridge_inc_nodes, sev_mean, sev_max = _bridge_metrics(n_nodes, undirected_edges)

    bridge_fraction = float(n_bridges / pos_edges) if pos_edges > 0 else 0.0
    bridge_incident_node_fraction = float(len(bridge_inc_nodes) / n_nodes) if n_nodes > 0 else 0.0
    bridge_severity_mean = float(sev_mean)
    bridge_severity_max = float(sev_max)

    return GraphStats(
        split=split,
        sample_index=sample_index,
        domain=int(domain),
        n_nodes=n_nodes,
        pos_edges=pos_edges,
        neg_edges=neg_edges,
        total_possible_edges=total_possible,
        density=density,
        degree_min=degree_min,
        degree_max=degree_max,
        degree_mean=degree_mean,
        degree_median=degree_median,
        node_degree_min=node_degree_min,
        node_degree_max=node_degree_max,
        node_degree_sum=node_degree_sum,
        node_degree_sumsq=node_degree_sumsq,
        node_deg0=node_deg0,
        node_deg1=node_deg1,
        node_deg2=node_deg2,
        node_deg_ge3=node_deg_ge3,
        node_degree_hist_0_10=node_degree_hist_0_10,
        n_components=n_components,
        largest_component_size=largest_component_size,
        n_bridges=n_bridges,
        bridge_fraction=bridge_fraction,
        bridge_incident_node_fraction=bridge_incident_node_fraction,
        bridge_severity_mean=bridge_severity_mean,
        bridge_severity_max=bridge_severity_max,
    )


def build_datasets_from_config(config: Any, debug: bool, continuous: bool):
    """
    Mirrors the dataset selection logic in train_with_ema.py.
    Returns (train_ds, val_ds, sampler, mixed_flag).
    """


    # NEW: plants dataset (edit these names to match your config.DATA.DATASET)
    
    train_ds, val_ds, sampler = build_plants_network_data(
        config,
        mode="split",
        max_samples=getattr(config.DATA, "NUM_SOURCE_SAMPLES", 0),
        domain_classification=-1,
        use_grayscale=getattr(config.DATA, "USE_GRAYSCALE", False),
        mixed=False,
        has_val=True,
    )
    mixed = False
    
    return train_ds, val_ds, sampler, mixed


def analyze_split(
    split_name: str,
    dataset,
    sampler,
    mixed: bool,
    batch_size: int,
    num_workers: int,
    pre2d: bool,
    max_batches: Optional[int],
    device: str,
) -> List[GraphStats]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda x: image_graph_collate(x, pre2d, gaussian_augment=mixed),
        pin_memory=False,
    )

    all_stats: List[GraphStats] = []
    sample_counter = 0

    dev = torch.device(device)

    for b_idx, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):
        if max_batches is not None and b_idx >= max_batches:
            break

        if not isinstance(batch, (list, tuple)):
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        # Support both:
        #   (images, segs, nodes, edges, z_pos, domains)   -> len==6
        #   (images, segs, nodes, edges, domains)         -> len==5  (plants)
        if len(batch) == 6:
            nodes_list = batch[2]
            edges_list = batch[3]
            domains = batch[5]
        elif len(batch) == 5:
            nodes_list = batch[2]
            edges_list = batch[3]
            domains = batch[4]
        else:
            raise ValueError(
                f"Unexpected batch length={len(batch)} for split={split_name}. "
                f"Expected 5 or 6 elements."
            )

        # Move domains to CPU quickly; nodes/edges can stay on CPU
        if torch.is_tensor(domains):
            domains_np = domains.detach().cpu().numpy().astype(np.int64)
        else:
            domains_np = np.array(domains, dtype=np.int64)

        # nodes/edges are lists (len=B)
        for i in range(len(nodes_list)):
            nodes = nodes_list[i]
            edges = edges_list[i]
            dom_i = int(domains_np[i]) if i < len(domains_np) else -1

            # ensure tensors on CPU
            if torch.is_tensor(nodes):
                nodes = nodes.to("cpu")
            else:
                nodes = torch.as_tensor(nodes, device="cpu")

            if torch.is_tensor(edges):
                edges = edges.to("cpu")
            else:
                edges = torch.as_tensor(edges, device="cpu")

            # normalize edge format to [E,2] so stats code is stable
            edges = _edges_to_E2(edges)

            st = compute_graph_stats(
                split=split_name,
                sample_index=sample_counter,
                domain=dom_i,
                nodes=nodes,
                edges=edges,
            )
            all_stats.append(st)
            sample_counter += 1

    return all_stats


def aggregate_stats(stats: List[GraphStats]) -> Dict[str, Any]:
    if not stats:
        return {"count": 0}

    def _agg(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    out: Dict[str, Any] = {"count": len(stats)}
    out["n_nodes"] = _agg([s.n_nodes for s in stats])
    out["pos_edges"] = _agg([s.pos_edges for s in stats])
    out["neg_edges"] = _agg([s.neg_edges for s in stats])
    out["density"] = _agg([s.density for s in stats])
    out["degree_mean"] = _agg([s.degree_mean for s in stats])
    out["n_components"] = _agg([s.n_components for s in stats])
    out["largest_component_size"] = _agg([s.largest_component_size for s in stats])
    out["n_bridges"] = _agg([s.n_bridges for s in stats])
    out["bridge_fraction"] = _agg([s.bridge_fraction for s in stats])
    out["bridge_incident_node_fraction"] = _agg([s.bridge_incident_node_fraction for s in stats])
    out["bridge_severity_mean"] = _agg([s.bridge_severity_mean for s in stats])
    out["bridge_severity_max"] = _agg([s.bridge_severity_max for s in stats])

    # node-level degree summary across ALL nodes in the split (not per-graph means)
    total_nodes = int(sum(s.n_nodes for s in stats))
    if total_nodes > 0:
        sum_deg = int(sum(s.node_degree_sum for s in stats))
        sumsq_deg = int(sum(s.node_degree_sumsq for s in stats))
        node_deg_mean = float(sum_deg / total_nodes)
        node_deg_var = float(max(sumsq_deg / total_nodes - node_deg_mean**2, 0.0))
        node_deg_std = float(math.sqrt(node_deg_var))
        nonempty = [s for s in stats if s.n_nodes > 0]
        node_deg_min = int(min(s.node_degree_min for s in nonempty)) if nonempty else 0
        node_deg_max = int(max(s.node_degree_max for s in nonempty)) if nonempty else 0
        deg0 = int(sum(s.node_deg0 for s in stats))
        deg1 = int(sum(s.node_deg1 for s in stats))
        deg2 = int(sum(s.node_deg2 for s in stats))
        degge3 = int(sum(s.node_deg_ge3 for s in stats))
        hist_0_10 = [0] * 11
        for s in stats:
            for k in range(11):
                hist_0_10[k] += int(s.node_degree_hist_0_10[k])
        out["node_degree"] = {
            "total_nodes": total_nodes,
            "mean": node_deg_mean,
            "std": node_deg_std,
            "min": float(node_deg_min),
            "max": float(node_deg_max),
            "p_zero": float(deg0 / total_nodes),
            "p_one": float(deg1 / total_nodes),
            "p_two": float(deg2 / total_nodes),
            "p_ge3": float(degge3 / total_nodes),
            "hist_0_10": hist_0_10,
        }
    else:
        out["node_degree"] = {
            "total_nodes": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p_zero": 0.0,
            "p_one": 0.0,
            "p_two": 0.0,
            "p_ge3": 0.0,
            "hist_0_10": [0] * 11,
        }

    # per-domain aggregates
    domains = sorted(set(int(s.domain) for s in stats))
    per_domain = {}
    for d in domains:
        sub = [s for s in stats if int(s.domain) == d]
        if not sub:
            continue
        per_domain[str(d)] = {
            "count": len(sub),
            "n_nodes": _agg([s.n_nodes for s in sub]),
            "pos_edges": _agg([s.pos_edges for s in sub]),
            "neg_edges": _agg([s.neg_edges for s in sub]),
            "density": _agg([s.density for s in sub]),
            "bridge_fraction": _agg([s.bridge_fraction for s in sub]),
            "bridge_incident_node_fraction": _agg([s.bridge_incident_node_fraction for s in sub]),
            "bridge_severity_mean": _agg([s.bridge_severity_mean for s in sub]),
            "bridge_severity_max": _agg([s.bridge_severity_max for s in sub]),
        }
    out["per_domain"] = per_domain

    # totals (sanity checks) — FIXED (previous version accidentally used `sub`)
    out["totals"] = {
        "nodes": int(sum(s.n_nodes for s in stats)),
        "pos_edges": int(sum(s.pos_edges for s in stats)),
        "neg_edges": int(sum(s.neg_edges for s in stats)),
        "total_possible_edges": int(sum(s.total_possible_edges for s in stats)),
        "bridge_fraction": _agg([s.bridge_fraction for s in stats]),
        "bridge_incident_node_fraction": _agg([s.bridge_incident_node_fraction for s in stats]),
        "bridge_severity_mean": _agg([s.bridge_severity_mean for s in stats]),
        "bridge_severity_max": _agg([s.bridge_severity_max for s in stats]),
    }
    return out


def main(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    train_ds, val_ds, sampler, mixed = build_datasets_from_config(config, debug=args.debug, continuous=args.continuous)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[GraphStats] = []

    if args.split in ("train", "both"):
        train_stats = analyze_split(
            split_name="train",
            dataset=train_ds,
            sampler=sampler,
            mixed=mixed,
            batch_size=int(config.DATA.BATCH_SIZE),
            num_workers=int(config.DATA.NUM_WORKERS),
            pre2d=args.pre2d,
            max_batches=args.max_batches,
            device=args.device,
        )
        all_stats.extend(train_stats)

        train_summary = aggregate_stats(train_stats)
        (out_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2))
        print(f"[train] analyzed samples: {len(train_stats)} -> {out_dir/'train_summary.json'}")

        if args.save_csv:
            import pandas as pd

            df = pd.DataFrame([asdict(s) for s in train_stats])
            df.to_csv(out_dir / "train_samples.csv", index=False)
            print(f"[train] per-sample CSV -> {out_dir/'train_samples.csv'}")

    if args.split in ("val", "both"):
        val_stats = analyze_split(
            split_name="val",
            dataset=val_ds,
            sampler=None,  # never sample val
            mixed=mixed,
            batch_size=int(config.DATA.BATCH_SIZE),
            num_workers=int(config.DATA.NUM_WORKERS),
            pre2d=args.pre2d,
            max_batches=args.max_batches,
            device=args.device,
        )
        all_stats.extend(val_stats)

        val_summary = aggregate_stats(val_stats)
        (out_dir / "val_summary.json").write_text(json.dumps(val_summary, indent=2))
        print(f"[val] analyzed samples: {len(val_stats)} -> {out_dir/'val_summary.json'}")

        if args.save_csv:
            import pandas as pd

            df = pd.DataFrame([asdict(s) for s in val_stats])
            df.to_csv(out_dir / "val_samples.csv", index=False)
            print(f"[val] per-sample CSV -> {out_dir/'val_samples.csv'}")

    # combined
    combined = aggregate_stats(all_stats)
    (out_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2))
    print(f"[combined] analyzed samples: {len(all_stats)} -> {out_dir/'combined_summary.json'}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--split",
        default="both",
        choices=["train", "val", "both"],
        help="Which split to analyze.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda (analysis is CPU-friendly).")
    parser.add_argument("--debug", action="store_true", help="Enable dataset debug mode.")
    parser.add_argument("--continuous", action="store_true", help="Match training continuous rotation option.")
    parser.add_argument("--pre2d", action="store_true", help="Match training pre2d option (passed to collate).")
    parser.add_argument("--max_batches", type=int, default=None, help="Stop after N batches per split.")
    parser.add_argument("--out_dir", default="./graph_stats_out", help="Output directory.")
    parser.add_argument("--save_csv", action="store_true", help="Save per-sample CSV.")

    args = parser.parse_args([
        '--config', '2d/configs/pretrain_plants_octa_synth.yaml', 
    ])
    main(args)
