import os, csv, pickle, numpy as np

def validate_edges(arr, where):
    arr = np.asarray(arr)
    if arr.size == 0:
        if arr.shape != (0, 2):
            raise ValueError(f"{where}: empty edges must be shape (0,2), got {arr.shape}")
        return
    if arr.ndim == 1 and arr.size == 2:
        raise ValueError(f"{where}: single edge squeezed to (2,), should be (1,2)")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{where}: edges must be (E,2), got {arr.shape}")

def _parse_vec(s):
    # "[x y]" or "[x y z]"
    s = s.strip().lstrip('[').rstrip(']')
    v = np.fromstring(s, sep=' ', dtype=np.float32)
    if v.size not in (2, 3):
        raise ValueError(f"bad node format: '{s}'")
    return v

def _validate_octa_csv(path, where):
    nodes = []            # unique coords (drop z)
    idx_map = {}
    edges = []
    radii = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or not {"node1","node2"}.issubset(set(rdr.fieldnames)):
            raise ValueError(f"{where}: CSV missing headers node1/node2")
        for row in rdr:
            p = _parse_vec(row["node1"])[:2]
            q = _parse_vec(row["node2"])[:2]
            r = float(row.get("radius", 0.0))
            for pt in (p, q):
                if np.any(pt < -1e-6) or np.any(pt > 1+1e-6):
                    raise ValueError(f"{where}: normalized coord out of [0,1]: {pt}")
            tp, tq = tuple(p.tolist()), tuple(q.tolist())
            if tp not in idx_map:
                idx_map[tp] = len(nodes); nodes.append(tp)
            if tq not in idx_map:
                idx_map[tq] = len(nodes); nodes.append(tq)
            u, v = idx_map[tp], idx_map[tq]
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                edges.append((a, b))
                radii.append(r)

    edges = np.array(edges, dtype=np.int64).reshape(-1, 2)  # (E,2) even when E=0/1
    radii = np.array(radii, dtype=np.float32).reshape(-1)
    validate_edges(edges, where)
    if radii.shape[0] != edges.shape[0]:
        raise ValueError(f"{where}: radii len {radii.shape[0]} != edges len {edges.shape[0]}")

def scan_graphs(root, label):
    bad = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            try:
                if fn.endswith(".pickle"):
                    with open(p, "rb") as f:
                        obj = pickle.load(f)
                    if isinstance(obj, dict) and "edges" in obj:
                        validate_edges(obj["edges"], f"{label}:{p}")
                    elif isinstance(obj, dict):  # adjacency dict -> build edges then validate
                        nodes = np.array(list(obj.keys()), dtype=np.int64)
                        idx = {tuple(x): i for i, x in enumerate(nodes)}
                        es = set()
                        for u_xy, nbs in obj.items():
                            u = idx[tuple(u_xy)]
                            for v_xy in nbs:
                                v = idx[tuple(v_xy)]
                                if u != v:
                                    a, b = (u, v) if u < v else (v, u)
                                    es.add((a, b))
                        edges = np.array(sorted(es), dtype=np.int64).reshape(-1, 2)
                        validate_edges(edges, f"{label}:{p}")
                    else:
                        raise ValueError(f"{label}:{p} unexpected pickle top-level type {type(obj)}")
                elif fn.endswith(".csv"):
                    _validate_octa_csv(p, f"{label}:{p}")
                else:
                    continue
            except Exception as e:
                bad += 1
                print("[BAD]", e)
    print(f"[{label}] scan complete. bad files: {bad}")

# Run per dataset root (scan both .pickle and .csv files)
scan_graphs("/data/scavone/20cities/patches/vtp", "20cities")
scan_graphs("/data/scavone/octa-synth-packed/patches/vtp", "octa-synth")
