import os, re
import numpy as np
import pandas as pd

num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_point(s):
    """Parse '[x y z]' or '[x, y, z]' → list of floats."""
    if pd.isna(s):
        return []
    return [float(v) for v in num_re.findall(str(s))]

def load_graph_from_csv(path, drop_z=True):
    """Return nodes (Nx2 or Nx3), edges (Mx2), radius (M,) if present."""
    df = pd.read_csv(path, dtype=str)  # keep strings as-is
    if not {"node1", "node2"}.issubset(df.columns):
        raise ValueError(f"Missing 'node1'/'node2' columns in {path}")

    pts = []         # all endpoints (with duplicates)
    radii = []

    for _, row in df.iterrows():
        p1 = parse_point(row["node1"])
        p2 = parse_point(row["node2"])
        if drop_z:
            p1 = p1[:2]
            p2 = p2[:2]
        if len(p1) < 2 or len(p2) < 2:
            raise ValueError(f"Bad point in {path}: {row.to_dict()}")
        pts.append(tuple(p1[:2] if drop_z else p1[:3]))
        pts.append(tuple(p2[:2] if drop_z else p2[:3]))
        if "radius" in df.columns:
            try:
                radii.append(float(row["radius"]))
            except Exception:
                radii.append(np.nan)

    # deduplicate points and map to indices
    uniq = {}
    for p in pts:
        if p not in uniq:
            uniq[p] = len(uniq)
    nodes = np.array(list(uniq.keys()), dtype=float)  # (N,2) or (N,3)

    # build edges as index pairs
    edges = []
    for i in range(0, len(pts), 2):
        u = uniq[pts[i]]
        v = uniq[pts[i+1]]
        edges.append((u, v))
    edges = np.array(edges, dtype=int)                # (M,2)
    radius = np.array(radii, dtype=float) if radii else None
    return nodes, edges, radius

def inspect_graph_folder(csv_dir):
    print(f"Scanning: {csv_dir}")
    bad = 0
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)
        try:
            nodes, edges, radius = load_graph_from_csv(path, drop_z=True)

            # sanity checks matching your loader expectations
            if nodes.ndim != 2 or nodes.shape[1] < 2:
                print(f"\n📄 {path}")
                print(f"  nodes shape: {nodes.shape}  | edges shape: {edges.shape}")
                print("  ❌ nodes must be 2D with ≥2 columns"); bad += 1; continue
            if edges.ndim != 2 or edges.shape[1] != 2:
                print(f"\n📄 {path}")
                print(f"  nodes shape: {nodes.shape}  | edges shape: {edges.shape}")
                print("  ❌ edges must be (M,2)"); bad += 1; continue
            if np.isnan(nodes).any() or np.isinf(nodes).any():
                print(f"\n📄 {path}")
                print(f"  nodes shape: {nodes.shape}  | edges shape: {edges.shape}")
                print("  ❌ nodes contain NaN/Inf"); bad += 1; continue
            if np.isnan(edges).any():
                print(f"\n📄 {path}")
                print(f"  nodes shape: {nodes.shape}  | edges shape: {edges.shape}")

                print("  ❌ edges contain NaN"); bad += 1; continue
            # print("  ✅ OK")
        except Exception as e:
            print(f"\n💥 {fname}: {e}")
            bad += 1
    print(f"\nDone. Problematic files: {bad}")


# Change this to your directory of CSV graph files
inspect_graph_folder("/data/scavone/octa-synth-packed_bigger_inverted/patches/train/csv")
