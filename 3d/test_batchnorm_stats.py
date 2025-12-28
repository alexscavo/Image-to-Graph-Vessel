#!/usr/bin/env python3
"""
BN diagnostics (updated):
1) Dump BN running stats (running_mean / running_var).
2) Quantify output differences between:
   - eval()
   - train() with dropout OFF, BN TRAIN (uses batch stats + updates running stats)
   - train() with dropout OFF, BN FROZEN (BN eval, uses running stats, no updates)
3) NEW: Compare BN running stats vs *actual batch stats* on the same batch (for a few BN layers).
4) NEW: Optional per-domain mode diffs (domain=0 vs domain=1 within the first batch).
5) Optional: BN calibration (forward-only), dump stats again, and compare.
6) Plots:
   - running_mean.mean histogram
   - log10(running_var.mean) histogram
   - scatter rm_mean vs rv_mean
   - top-K BN layers by rv_mean
   - mode-diff bars
   - if calibration ran: overlay before/after histograms

This script is meant to replace/extend your test_batchnorm_stats.py.
"""

import os
import json
import math
from argparse import ArgumentParser
from typing import Dict, Any, List, Tuple, Optional

import torch
import yaml
from tqdm import tqdm
from monai.data import DataLoader
from torch.nn.modules.batchnorm import _BatchNorm
import matplotlib.pyplot as plt

from data.dataset_vessel3d import build_vessel_data
from data.dataset_mixed import build_mixed_data
from models import build_model
from utils.utils import image_graph_collate


# --------------------------
# Small config helper
# --------------------------
class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


# --------------------------
# Mode control helpers
# --------------------------
def set_dropout_eval(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or "Dropout" in m.__class__.__name__:
            m.eval()

def set_bn_train_dropout_eval(model: torch.nn.Module):
    """BN in train mode (uses batch stats, updates running stats), dropout OFF."""
    model.train()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.train()
    set_dropout_eval(model)

def set_bn_eval_dropout_eval(model: torch.nn.Module):
    """Model in train() overall, but BN frozen in eval mode, dropout OFF."""
    model.train()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
    set_dropout_eval(model)


# --------------------------
# BN stats extraction (running stats)
# --------------------------
@torch.no_grad()
def collect_bn_stats(model: torch.nn.Module) -> List[Dict[str, Any]]:
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            rm = m.running_mean.detach().float().cpu() if m.running_mean is not None else None
            rv = m.running_var.detach().float().cpu() if m.running_var is not None else None

            def tensor_summary(x):
                if x is None:
                    return None
                return {
                    "shape": list(x.shape),
                    "mean": float(x.mean().item()),
                    "std": float(x.std(unbiased=False).item()),
                    "min": float(x.min().item()),
                    "max": float(x.max().item()),
                }

            nbt = None
            if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
                nbt = int(m.num_batches_tracked.detach().cpu().item())

            rows.append({
                "name": name,
                "type": m.__class__.__name__,
                "num_features": int(getattr(m, "num_features", -1)),
                "eps": float(getattr(m, "eps", 0.0)),
                "momentum": None if getattr(m, "momentum", None) is None else float(m.momentum),
                "affine": bool(getattr(m, "affine", False)),
                "track_running_stats": bool(getattr(m, "track_running_stats", False)),
                "num_batches_tracked": nbt,
                "running_mean": tensor_summary(rm),
                "running_var": tensor_summary(rv),
            })
    return rows


def write_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def bn_stats_diff(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare two BN snapshots (summary scalars only)."""
    b_map = {r["name"]: r for r in b}
    deltas = []
    for r in a:
        name = r["name"]
        if name not in b_map:
            continue
        r2 = b_map[name]

        def delta_field(k):
            x1 = r.get(k, None)
            x2 = r2.get(k, None)
            if x1 is None or x2 is None:
                return None
            out = {}
            for s in ["mean", "std", "min", "max"]:
                if x1.get(s) is None or x2.get(s) is None:
                    out[s] = None
                else:
                    out[s] = float(x2[s] - x1[s])
            return out

        deltas.append({
            "name": name,
            "running_mean_delta": delta_field("running_mean"),
            "running_var_delta": delta_field("running_var"),
            "num_batches_tracked_before": r.get("num_batches_tracked"),
            "num_batches_tracked_after": r2.get("num_batches_tracked"),
        })

    movers = []
    for d in deltas:
        dm = d["running_mean_delta"]
        if dm and dm.get("mean") is not None:
            movers.append((abs(dm["mean"]), d["name"]))
    movers.sort(reverse=True)
    top10 = movers[:10]

    return {"layer_deltas": deltas, "top10_by_abs_running_mean_mean_delta": top10}


# --------------------------
# Forward helpers + diffs
# --------------------------
def tensor_diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    d = (a.detach() - b.detach()).abs()
    return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())}

@torch.no_grad()
def forward_raw(net, x, z_pos, domains):
    h, out_raw, *_ = net(x, z_pos, domain_labels=domains)
    return h, out_raw


# --------------------------
# NEW: batch-statistics hooks for BN layers
# --------------------------
def list_bn_layer_names(model: torch.nn.Module) -> List[str]:
    names = []
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            names.append(name)
    return names

def choose_probe_layers(bn_names: List[str], strategy: str) -> List[str]:
    """
    strategy:
      - "first_middle_last": pick 3 layers (if possible)
      - "all": probe all BN layers (can be slower)
    """
    if not bn_names:
        return []
    if strategy == "all":
        return bn_names
    # default
    idxs = sorted(set([
        0,
        len(bn_names) // 2,
        len(bn_names) - 1
    ]))
    return [bn_names[i] for i in idxs]

@torch.no_grad()
def capture_bn_batch_vs_running(
    net: torch.nn.Module,
    x: torch.Tensor,
    z_pos,
    domains: torch.Tensor,
    probe_layer_names: List[str],
) -> Dict[str, Any]:
    """
    Runs ONE forward in BN-train mode (dropout OFF) and records:
      - batch mean/var (computed from BN input tensor)
      - running mean/var (from the module)
    for selected BN layers.
    """
    bn_batch_stats: Dict[str, Any] = {}
    handles = []

    # Map name->module for quick lookup
    name_to_mod = dict(net.named_modules())

    def make_hook(name: str):
        def fn(m: _BatchNorm, inp, out):
            # inp[0] is (N,C,...) pre-BN activation
            t = inp[0]
            # per-channel stats: mean/var over N and spatial dims
            dims = [0] + list(range(2, t.ndim))
            batch_mean = t.mean(dim=dims).detach().float().cpu()
            batch_var = t.var(dim=dims, unbiased=False).detach().float().cpu()

            bn_batch_stats[name] = {
                "batch_mean": {
                    "mean": float(batch_mean.mean().item()),
                    "std": float(batch_mean.std(unbiased=False).item()),
                    "min": float(batch_mean.min().item()),
                    "max": float(batch_mean.max().item()),
                },
                "batch_var": {
                    "mean": float(batch_var.mean().item()),
                    "std": float(batch_var.std(unbiased=False).item()),
                    "min": float(batch_var.min().item()),
                    "max": float(batch_var.max().item()),
                },
                "running_mean": {
                    "mean": float(m.running_mean.detach().float().cpu().mean().item()) if m.running_mean is not None else None,
                    "std": float(m.running_mean.detach().float().cpu().std(unbiased=False).item()) if m.running_mean is not None else None,
                    "min": float(m.running_mean.detach().float().cpu().min().item()) if m.running_mean is not None else None,
                    "max": float(m.running_mean.detach().float().cpu().max().item()) if m.running_mean is not None else None,
                },
                "running_var": {
                    "mean": float(m.running_var.detach().float().cpu().mean().item()) if m.running_var is not None else None,
                    "std": float(m.running_var.detach().float().cpu().std(unbiased=False).item()) if m.running_var is not None else None,
                    "min": float(m.running_var.detach().float().cpu().min().item()) if m.running_var is not None else None,
                    "max": float(m.running_var.detach().float().cpu().max().item()) if m.running_var is not None else None,
                },
                "num_batches_tracked": int(m.num_batches_tracked.detach().cpu().item()) if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None else None,
            }
        return fn

    for ln in probe_layer_names:
        m = name_to_mod.get(ln, None)
        if m is None or not isinstance(m, _BatchNorm):
            continue
        handles.append(m.register_forward_hook(make_hook(ln)))

    # Run forward once in BN-train mode so hooks see the true batch stats path
    set_bn_train_dropout_eval(net)
    net(x, z_pos, domain_labels=domains)

    # cleanup hooks
    for h in handles:
        h.remove()

    return bn_batch_stats


@torch.no_grad()
def exp_mode_diffs_on_first_batch(
    net,
    loader,
    device,
    use_seg: bool,
    do_per_domain: bool = False,
) -> Dict[str, Any]:
    """
    Compare outputs on ONE batch under 3 controlled settings:
      A) eval()
      B) train() with dropout OFF, BN TRAIN (batch stats)
      C) train() with dropout OFF, BN EVAL  (running stats)
    Optionally also compute the same diffs for domain==0 and domain==1 subsets.
    """
    batch = next(iter(loader))
    images, segs, nodes, edges, z_pos, domains = batch
    x = segs if use_seg else images
    x = x.to(device).float()
    domains = domains.to(device)

    # A) eval
    net.eval()
    _, out_e = forward_raw(net, x, z_pos, domains)

    # B) train, BN train, dropout off
    set_bn_train_dropout_eval(net)
    _, out_bt = forward_raw(net, x, z_pos, domains)

    # C) BN frozen, dropout off
    set_bn_eval_dropout_eval(net)
    _, out_bf = forward_raw(net, x, z_pos, domains)

    keys = []
    for k in ["pred_logits", "pred_nodes"]:
        if k in out_e and torch.is_tensor(out_e[k]):
            keys.append(k)

    diffs = {}
    for k in keys:
        diffs[k] = {
            "eval_vs_bnTrain": tensor_diff_stats(out_e[k], out_bt[k]),
            "eval_vs_bnFrozen": tensor_diff_stats(out_e[k], out_bf[k]),
            "bnTrain_vs_bnFrozen": tensor_diff_stats(out_bt[k], out_bf[k]),
        }

    out = {
        "compared_keys": keys,
        "diffs": diffs,
        "note": "If eval_vs_bnFrozen is small but eval_vs_bnTrain is large, BN (batch vs running stats) dominates the train/eval difference.",
    }

    if do_per_domain:
        per_dom = {}
        for dom_val in [0, 1]:
            idx = (domains == dom_val).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            # slice batch dimension only
            x_d = x.index_select(0, idx)
            z_d = z_pos  # if z_pos is per-sample tensor you may need to slice it too; keep as-is if constant/None
            d_d = domains.index_select(0, idx)

            net.eval()
            _, oe = forward_raw(net, x_d, z_d, d_d)
            set_bn_train_dropout_eval(net)
            _, obt = forward_raw(net, x_d, z_d, d_d)
            set_bn_eval_dropout_eval(net)
            _, obf = forward_raw(net, x_d, z_d, d_d)

            dd = {}
            for k in keys:
                if k in oe:
                    dd[k] = {
                        "eval_vs_bnTrain": tensor_diff_stats(oe[k], obt[k]),
                        "eval_vs_bnFrozen": tensor_diff_stats(oe[k], obf[k]),
                        "bnTrain_vs_bnFrozen": tensor_diff_stats(obt[k], obf[k]),
                    }
            per_dom[str(dom_val)] = {"batch_size": int(idx.numel()), "diffs": dd}

        out["per_domain"] = per_dom

    return out


# --------------------------
# BN calibration (forward-only)
# --------------------------
@torch.no_grad()
def bn_calibrate(model, calib_loader, device, use_seg: bool, num_batches: int = 200):
    set_bn_train_dropout_eval(model)  # BN updates happen in train mode
    n = 0
    for batch in tqdm(calib_loader, desc=f"BN calibration ({num_batches} batches)", leave=False):
        images, segs, nodes, edges, z_pos, domains = batch
        x = segs if use_seg else images
        x = x.to(device).float()
        domains = domains.to(device)
        model(x, z_pos, domain_labels=domains)  # forward only
        n += 1
        if n >= num_batches:
            break
    model.eval()


# --------------------------
# Plotting helpers
# --------------------------
def per_layer_vals(bn_rows, key: str) -> List[float]:
    out = []
    for r in bn_rows:
        d = r.get(key)
        if d and d.get("mean") is not None:
            out.append(float(d["mean"]))
    return out

def plot_bn_distributions(bn0, out_dir: str, bn1: Optional[List[Dict[str, Any]]] = None):
    os.makedirs(out_dir, exist_ok=True)

    rm0 = per_layer_vals(bn0, "running_mean")
    rv0 = per_layer_vals(bn0, "running_var")
    eps = 1e-12
    rv0_log = [math.log10(x + eps) for x in rv0]

    # running_mean histogram
    plt.figure()
    plt.hist(rm0, bins=30, alpha=0.7, label="loaded")
    if bn1 is not None:
        rm1 = per_layer_vals(bn1, "running_mean")
        plt.hist(rm1, bins=30, alpha=0.5, label="after_calib")
    plt.title("running_mean.mean across BN layers")
    plt.xlabel("running_mean.mean")
    plt.ylabel("#layers")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "running_mean_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # running_var histogram (log)
    plt.figure()
    plt.hist(rv0_log, bins=30, alpha=0.7, label="loaded")
    if bn1 is not None:
        rv1 = per_layer_vals(bn1, "running_var")
        rv1_log = [math.log10(x + eps) for x in rv1]
        plt.hist(rv1_log, bins=30, alpha=0.5, label="after_calib")
    plt.title("log10(running_var.mean) across BN layers")
    plt.xlabel("log10(running_var.mean)")
    plt.ylabel("#layers")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "running_var_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # scatter rm vs rv (log y)
    plt.figure()
    plt.scatter(rm0, rv0)
    plt.yscale("log")
    plt.title("BN layers: running_mean.mean vs running_var.mean (loaded)")
    plt.xlabel("running_mean.mean")
    plt.ylabel("running_var.mean")
    plt.savefig(os.path.join(out_dir, "rm_vs_rv_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # top-K by running_var.mean
    pairs = []
    for r in bn0:
        rv = r.get("running_var", {})
        if rv and rv.get("mean") is not None:
            pairs.append((float(rv["mean"]), r["name"]))
    pairs.sort(reverse=True)
    topk = pairs[:15]
    if topk:
        vals = [v for v, _ in topk]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(vals)), vals)
        plt.yscale("log")
        plt.title("Top 15 BN layers by running_var.mean (loaded)")
        plt.xlabel("Layer rank (sorted)")
        plt.ylabel("running_var.mean (log scale)")
        plt.xticks(range(len(vals)), [str(i + 1) for i in range(len(vals))])
        plt.savefig(os.path.join(out_dir, "top15_running_var.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_mode_diffs(mode_diffs: Dict[str, Any], out_dir: str, fname: str):
    diffs = mode_diffs.get("diffs", {})
    keys = list(diffs.keys())
    if not keys:
        return

    eval_bntrain = [diffs[k]["eval_vs_bnTrain"]["mean_abs"] for k in keys]
    eval_bnfrozen = [diffs[k]["eval_vs_bnFrozen"]["mean_abs"] for k in keys]
    bntrain_bnfrozen = [diffs[k]["bnTrain_vs_bnFrozen"]["mean_abs"] for k in keys]

    x = range(len(keys))
    plt.figure()
    plt.bar([i - 0.2 for i in x], eval_bntrain, width=0.2, label="eval vs BN-train")
    plt.bar([i for i in x], eval_bnfrozen, width=0.2, label="eval vs BN-frozen")
    plt.bar([i + 0.2 for i in x], bntrain_bnfrozen, width=0.2, label="BN-train vs BN-frozen")
    plt.title("Output difference (mean_abs) by mode")
    plt.xlabel("Output key")
    plt.ylabel("mean_abs")
    plt.xticks(list(x), keys)
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------
# Main
# --------------------------
def main(args):
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(cfg)

    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")

    # loaders
    if args.test_mode == "vessel":
        config.DATA.MIXED = False
        test_ds = build_vessel_data(config, mode="test", debug=False, max_samples=args.max_samples)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            collate_fn=lambda x: image_graph_collate(x, gaussian_augment=False),
            pin_memory=True,
        )

        calib_loader = None
        if args.bn_calibrate:
            calib_ds = build_vessel_data(config, mode=args.bn_calib_mode, debug=False, max_samples=args.bn_calib_max_samples)
            calib_loader = DataLoader(
                calib_ds,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                num_workers=config.DATA.NUM_WORKERS,
                collate_fn=lambda x: image_graph_collate(x, gaussian_augment=False),
                pin_memory=True,
            )

    elif args.test_mode == "mixed":
        config.DATA.MIXED = True
        train_ds, val_ds, sampler = build_mixed_data(config, mode="split", debug=False)

        test_ds = val_ds
        test_loader = DataLoader(
            test_ds,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            collate_fn=lambda x: image_graph_collate(x, gaussian_augment=False),
            pin_memory=True,
        )

        calib_loader = None
        if args.bn_calibrate:
            calib_loader = DataLoader(
                train_ds,
                batch_size=config.DATA.BATCH_SIZE,
                shuffle=False,
                sampler=sampler,
                num_workers=config.DATA.NUM_WORKERS,
                collate_fn=lambda x: image_graph_collate(x, gaussian_augment=False),
                pin_memory=True,
            )
    else:
        raise ValueError(f"Unknown test_mode: {args.test_mode}")

    # model
    if args.general_transformer:
        net = build_model(config, general_transformer=True, pretrain=False, output_dim=3).to(device)
    else:
        net = build_model(config).to(device)

    ckpt = torch.load(args.model, map_location=device)
    net.load_state_dict(ckpt["net"], strict=not args.no_strict_loading)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) BN running stats snapshot (loaded)
    bn0 = collect_bn_stats(net)
    write_json(os.path.join(args.out_dir, "bn_stats_loaded.json"), bn0)
    print(f"[OK] Wrote BN snapshot: {os.path.join(args.out_dir, 'bn_stats_loaded.json')}")
    print(f"[Info] Found {len(bn0)} BatchNorm layers.")

    # 2) Mode diffs on first batch (+ optional per-domain)
    if args.check_mode_diffs:
        diffs0 = exp_mode_diffs_on_first_batch(
            net, test_loader, device, args.seg, do_per_domain=args.per_domain_diffs
        )
        write_json(os.path.join(args.out_dir, "mode_diffs_before_calib.json"), diffs0)
        print(f"[OK] Wrote mode diffs: {os.path.join(args.out_dir, 'mode_diffs_before_calib.json')}")
        for k, v in diffs0["diffs"].items():
            print(f"  {k}: eval_vs_bnTrain={v['eval_vs_bnTrain']} | eval_vs_bnFrozen={v['eval_vs_bnFrozen']}")

        plot_mode_diffs(diffs0, args.out_dir, "mode_diffs_before_calib.png")

    # 3) NEW: BN batch vs running stats on the same first batch
    if args.check_bn_batch_vs_running:
        # grab the same first batch
        batch = next(iter(test_loader))
        images, segs, nodes, edges, z_pos, domains = batch
        x = segs if args.seg else images
        x = x.to(device).float()
        domains = domains.to(device)

        bn_names = list_bn_layer_names(net)
        probe_names = choose_probe_layers(bn_names, args.bn_probe_strategy)
        bn_bvr = capture_bn_batch_vs_running(net, x, z_pos, domains, probe_names)
        write_json(os.path.join(args.out_dir, "bn_batch_vs_running_first_batch.json"), bn_bvr)
        print(f"[OK] Wrote BN batch-vs-running stats: {os.path.join(args.out_dir, 'bn_batch_vs_running_first_batch.json')}")
        print("[Peek] Probed BN layers:", probe_names)
        for name, st in bn_bvr.items():
            print(f"  {name}: batch_mean.mean={st['batch_mean']['mean']:.4f} vs running_mean.mean={st['running_mean']['mean']:.4f} | "
                  f"batch_var.mean={st['batch_var']['mean']:.4f} vs running_var.mean={st['running_var']['mean']:.4f}")

    # 4) Optional BN calibration
    bn1 = None
    if args.bn_calibrate:
        print(f"[Action] BN calibration on split='{args.bn_calib_mode}' for {args.bn_calib_batches} batches ...")
        bn_calibrate(net, calib_loader, device, use_seg=args.seg, num_batches=args.bn_calib_batches)

        bn1 = collect_bn_stats(net)
        write_json(os.path.join(args.out_dir, "bn_stats_after_calib.json"), bn1)
        diff = bn_stats_diff(bn0, bn1)
        write_json(os.path.join(args.out_dir, "bn_stats_delta_after_calib.json"), diff)

        print(f"[OK] Wrote BN snapshot: {os.path.join(args.out_dir, 'bn_stats_after_calib.json')}")
        print(f"[OK] Wrote BN delta   : {os.path.join(args.out_dir, 'bn_stats_delta_after_calib.json')}")
        print("[Peek] Top-10 movers by |delta(running_mean.mean)|:", diff["top10_by_abs_running_mean_mean_delta"])

        if args.check_mode_diffs:
            diffs1 = exp_mode_diffs_on_first_batch(
                net, test_loader, device, args.seg, do_per_domain=args.per_domain_diffs
            )
            write_json(os.path.join(args.out_dir, "mode_diffs_after_calib.json"), diffs1)
            plot_mode_diffs(diffs1, args.out_dir, "mode_diffs_after_calib.png")
            print(f"[OK] Wrote mode diffs: {os.path.join(args.out_dir, 'mode_diffs_after_calib.json')}")

    # 5) Plots of BN distributions (loaded, and optionally after-calib overlay)
    plot_bn_distributions(bn0, args.out_dir, bn1=bn1)

    print("[Done]")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_samples", type=int, default=0)

    p.add_argument("--seg", action="store_true")
    p.add_argument("--general_transformer", action="store_true")
    p.add_argument("--no_strict_loading", action="store_true")

    p.add_argument("--check_mode_diffs", action="store_true",
                   help="Compare eval vs train(BN train) vs train(BN frozen) on the first batch.")
    p.add_argument("--per_domain_diffs", action="store_true",
                   help="Also compute mode diffs separately for domain==0 and domain==1 subsets (within the first batch).")

    # NEW
    p.add_argument("--check_bn_batch_vs_running", action="store_true",
                   help="Record batch mean/var vs running mean/var for a few BN layers on the first batch.")
    p.add_argument("--bn_probe_strategy", type=str, default="first_middle_last",
                   choices=["first_middle_last", "all"],
                   help="Which BN layers to probe for batch-vs-running stats.")

    p.add_argument("--bn_calibrate", action="store_true",
                   help="Run forward-only BN calibration to refresh running stats.")
    p.add_argument("--bn_calib_mode", type=str, default="train",
                   help="Which split to use for BN calibration.")
    p.add_argument("--bn_calib_batches", type=int, default=200)
    p.add_argument("--bn_calib_max_samples", type=int, default=0)

    p.add_argument("--test_mode", type=str, default="mixed", choices=["mixed", "vessel"])


    args = p.parse_args([
        '--config', '/home/scavone/cross-dim_i2g/3d/configs/mixed_synth_3D.yaml',
        '--model', '/data/scavone/cross-dim_i2g_3d/runs/pretraining_mixed_synth_1_20/models/checkpoint_epoch=50.pt',
        '--out_dir', '/data/scavone/cross-dim_i2g_3d/test_results',
        '--max_samples', '500',
        '--no_strict_loading',
        '--check_mode_diffs',
        '--bn_calibrate',
        '--bn_calib_mode', 'train',
        '--bn_calib_batches', '200',
        '--test_mode', 'mixed',
        '--per_domain_diffs',
    ])
    main(args)
