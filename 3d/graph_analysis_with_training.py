# ===================== STANDARD IMPORTS =====================
from functools import partial
import os
from pathlib import Path
import sys
import yaml
import json
import csv
import torch
import numpy as np
from argparse import ArgumentParser
from shutil import copyfile

from ignite.engine import Events
import ignite.distributed as igdist
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader
from PIL import Image

# ===================== PROJECT IMPORTS =====================
from training.inference import relation_infer
from training.evaluator import build_evaluator
from training.trainer import build_trainer
from training.losses import EDGE_SAMPLING_MODE, SetCriterion

from data.dataset_mixed import build_mixed_data
from data.dataset_road_network import build_road_network_data
from data.dataset_synth_octa_network import build_octa_network_data
from data.dataset_vessel3d import build_vessel_data

from models import build_model
from models.EMA_model import EMA_Model
from models.matcher import build_matcher

from utils.utils import image_graph_collate

# ===================== PIL BACKWARD COMPAT =====================
if not hasattr(Image, "ANTIALIAS"):
    from PIL import Image as _PILImage
    try:
        Image.NEAREST   = _PILImage.Resampling.NEAREST
        Image.BILINEAR  = _PILImage.Resampling.BILINEAR
        Image.BICUBIC   = _PILImage.Resampling.BICUBIC
        Image.LANCZOS   = _PILImage.Resampling.LANCZOS
        Image.ANTIALIAS = _PILImage.Resampling.LANCZOS
    except Exception:
        pass

# ===================== CONFIG UTILS =====================
class Obj:
    def __init__(self, d):
        self.__dict__.update(d)

def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=Obj)

# ===================== EDGE / GRAPH UTILS =====================
def _canon_undirected_edges_any(edges) -> np.ndarray:
    """Unique undirected edges (u<v), no self-loops."""
    if edges is None:
        return np.zeros((0, 2), dtype=np.int64)

    if torch.is_tensor(edges):
        if edges.numel() == 0:
            return np.zeros((0, 2), dtype=np.int64)
        e = edges.detach().cpu().numpy().astype(np.int64)
    else:
        e = np.asarray(edges, dtype=np.int64)

    if e.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int64)

    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    if e.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(e, axis=0)

def _edge_array_to_set(e_und: np.ndarray) -> set:
    return set((int(u), int(v)) for u, v in e_und)

def _degrees_from_edges(n_nodes: int, e_und: np.ndarray) -> np.ndarray:
    deg = np.zeros((n_nodes,), dtype=np.int64)
    for u, v in e_und:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            deg[u] += 1
            deg[v] += 1
    return deg

def _bridge_severity_edges(n_nodes: int, e_und: np.ndarray) -> dict:
    """Tarjan bridges with severity=min(split)/component_size."""
    if n_nodes <= 1 or e_und is None or e_und.size == 0:
        return {}

    adj = [[] for _ in range(n_nodes)]
    for u, v in e_und:
        u = int(u); v = int(v)
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adj[u].append(v)
            adj[v].append(u)

    tin = [-1] * n_nodes
    low = [0] * n_nodes
    sub = [0] * n_nodes
    comp_id = [-1] * n_nodes
    timer = 0
    bridges_child = []

    def dfs(v: int, p: int, cid: int):
        nonlocal timer
        comp_id[v] = cid
        tin[v] = timer
        low[v] = timer
        timer += 1
        sub[v] = 1
        for to in adj[v]:
            if to == p:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v, cid)
                sub[v] += sub[to]
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    bridges_child.append((v, to))

    cid = 0
    for v in range(n_nodes):
        if tin[v] == -1:
            dfs(v, -1, cid)
            cid += 1

    comp_sizes = [0] * cid
    for v in range(n_nodes):
        if comp_id[v] >= 0:
            comp_sizes[comp_id[v]] += 1

    out = {}
    for p, ch in bridges_child:
        c = comp_id[p]
        csz = comp_sizes[c] if c >= 0 else n_nodes
        a = sub[ch]
        b = csz - a
        sev = (min(a, b) / csz) if csz > 0 else 0.0
        u, v = (p, ch) if p < ch else (ch, p)
        out[(int(u), int(v))] = float(sev)
    return out

def _comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0

def _forward_network_robust(net, x, z_pos, alpha, domains):
    try:
        return net(x, z_pos, alpha, domain_labels=domains)
    except TypeError:
        pass
    try:
        return net(x, z_pos, alpha)
    except TypeError:
        pass
    try:
        return net(x, z_pos)
    except TypeError:
        pass
    return net(x)

def _safe_mean_std(vals):
    if len(vals) == 0:
        return (0.0, 0.0)
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())

def _compute_pred_gt_counts_for_loader(loader, net, device, config, seg_mode: bool, max_batches: int = 10):
    net.eval()
    obj_token = int(config.MODEL.DECODER.OBJ_TOKEN)
    rln_token = int(config.MODEL.DECODER.RLN_TOKEN)
    alpha = float(getattr(config.TRAIN, "ALPHA_COEFF", 0.0))
    severe_topk = int(getattr(config.TRAIN, "SEVERE_TOPK", 3))

    gt_nodes_l, gt_pos_l, gt_neg_l = [], [], []
    pr_nodes_l, pr_pos_l, pr_neg_l = [], [], []

    recall_all_l, recall_leaf_l, recall_hub_l, recall_severe_l = [], [], [], []
    gt_leaf_edges_l, gt_hub_edges_l, gt_severe_edges_l = [], [], []

    with torch.no_grad():
        for bi, batchdata in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break

            images, segs, nodes, edges, z_pos, domains = batchdata[:6]

            x = segs if seg_mode else images
            x = x.to(device, non_blocking=False).float()

            if torch.is_tensor(z_pos):
                z_pos = z_pos.to(device, non_blocking=False)
            elif isinstance(z_pos, (list, tuple)):
                z_pos = [zp.to(device, non_blocking=False) if torch.is_tensor(zp) else zp for zp in z_pos]

            domains_in = domains.to(device, non_blocking=False) if torch.is_tensor(domains) else domains

            out = _forward_network_robust(net, x, z_pos, alpha, domains_in)

            infer = relation_infer(
                out[0].detach() if isinstance(out, (list, tuple)) else None,
                out[1] if isinstance(out, (list, tuple)) else out,
                net,
                obj_token,
                rln_token,
                apply_nms=False,
            )
            pred_nodes_list = infer.get("pred_nodes", [])
            pred_edges_list = infer.get("pred_rels", [])

            batch_size = len(nodes)
            for i in range(batch_size):
                n_gt = int(nodes[i].shape[0]) if hasattr(nodes[i], "shape") else int(len(nodes[i]))

                gt_e_und = _canon_undirected_edges_any(edges[i])
                gt_set = _edge_array_to_set(gt_e_und)
                gt_pos = int(gt_e_und.shape[0])
                gt_neg = _comb2(n_gt) - gt_pos

                n_pred = int(len(pred_nodes_list[i])) if i < len(pred_nodes_list) else 0
                pred_e_und = _canon_undirected_edges_any(pred_edges_list[i]) if i < len(pred_edges_list) else np.zeros((0, 2), dtype=np.int64)
                pr_set = _edge_array_to_set(pred_e_und)
                pred_pos = int(pred_e_und.shape[0])
                pred_neg = _comb2(n_pred) - pred_pos

                gt_nodes_l.append(n_gt); gt_pos_l.append(gt_pos); gt_neg_l.append(gt_neg)
                pr_nodes_l.append(n_pred); pr_pos_l.append(pred_pos); pr_neg_l.append(pred_neg)

                recall_all = (len(gt_set & pr_set) / float(len(gt_set))) if gt_set else 0.0
                recall_all_l.append(float(recall_all))

                deg = _degrees_from_edges(n_gt, gt_e_und)
                leaf_set, hub_set = set(), set()
                for (u, v) in gt_set:
                    if deg[u] == 1 or deg[v] == 1:
                        leaf_set.add((u, v))
                    if deg[u] >= 3 or deg[v] >= 3:
                        hub_set.add((u, v))

                gt_leaf_edges_l.append(float(len(leaf_set)))
                gt_hub_edges_l.append(float(len(hub_set)))

                recall_leaf = (len(leaf_set & pr_set) / float(len(leaf_set))) if leaf_set else 0.0
                recall_hub = (len(hub_set & pr_set) / float(len(hub_set))) if hub_set else 0.0
                recall_leaf_l.append(float(recall_leaf))
                recall_hub_l.append(float(recall_hub))

                sev_map = _bridge_severity_edges(n_gt, gt_e_und)
                if sev_map:
                    severe_sorted = sorted(sev_map.items(), key=lambda kv: kv[1], reverse=True)
                    severe_edges = set([e for e, _ in severe_sorted[:max(1, severe_topk)]])
                else:
                    severe_edges = set()
                gt_severe_edges_l.append(float(len(severe_edges)))
                recall_severe = (len(severe_edges & pr_set) / float(len(severe_edges))) if severe_edges else 0.0
                recall_severe_l.append(float(recall_severe))

    samples = len(gt_nodes_l)

    gt_nodes_m, gt_nodes_s = _safe_mean_std(gt_nodes_l)
    gt_pos_m, gt_pos_s = _safe_mean_std(gt_pos_l)
    gt_neg_m, gt_neg_s = _safe_mean_std(gt_neg_l)

    pr_nodes_m, pr_nodes_s = _safe_mean_std(pr_nodes_l)
    pr_pos_m, pr_pos_s = _safe_mean_std(pr_pos_l)
    pr_neg_m, pr_neg_s = _safe_mean_std(pr_neg_l)

    r_all_m, r_all_s = _safe_mean_std(recall_all_l)
    r_l_m, r_l_s = _safe_mean_std(recall_leaf_l)
    r_h_m, r_h_s = _safe_mean_std(recall_hub_l)
    r_s_m, r_s_s = _safe_mean_std(recall_severe_l)

    gl_m, gl_s = _safe_mean_std(gt_leaf_edges_l)
    gh_m, gh_s = _safe_mean_std(gt_hub_edges_l)
    gs_m, gs_s = _safe_mean_std(gt_severe_edges_l)

    return {
        "samples": samples,
        "gt_nodes_mean": gt_nodes_m, "gt_nodes_std": gt_nodes_s,
        "gt_pos_edges_mean": gt_pos_m, "gt_pos_edges_std": gt_pos_s,
        "gt_neg_edges_mean": gt_neg_m, "gt_neg_edges_std": gt_neg_s,
        "pred_nodes_mean": pr_nodes_m, "pred_nodes_std": pr_nodes_s,
        "pred_pos_edges_mean": pr_pos_m, "pred_pos_edges_std": pr_pos_s,
        "pred_neg_edges_mean": pr_neg_m, "pred_neg_edges_std": pr_neg_s,
        "recall_all_mean": r_all_m, "recall_all_std": r_all_s,
        "recall_leaf_mean": r_l_m, "recall_leaf_std": r_l_s,
        "recall_hub_mean": r_h_m, "recall_hub_std": r_h_s,
        "recall_severe_mean": r_s_m, "recall_severe_std": r_s_s,
        "gt_leaf_edges_mean": gl_m, "gt_leaf_edges_std": gl_s,
        "gt_hub_edges_mean": gh_m, "gt_hub_edges_std": gh_s,
        "gt_severe_edges_mean": gs_m, "gt_severe_edges_std": gs_s,
    }

def match_name_keywords(n, name_keywords):
    return any(b in n for b in name_keywords)

# ===================== MAIN TRAINING =====================
def main(rank=0, args=None):
    with open(args.config) as f:
        config = dict2obj(yaml.load(f, Loader=yaml.FullLoader))

    config.log.exp_name = args.exp_name
    config.display_prob = args.display_prob

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", f"{args.exp_name}_{config.DATA.SEED}")
    os.makedirs(exp_path, exist_ok=True)
    try:
        copyfile(args.config, os.path.join(exp_path, "config.yaml"))
    except Exception:
        pass

    args.distributed = bool(args.parallel and igdist.get_world_size() > 1)
    device = torch.device(args.device)

    # ===================== DATA =====================
    if str(config.DATA.DATASET).startswith("mixed"):
        dataset_func = partial(build_mixed_data, upsample_target_domain=getattr(config.TRAIN, "UPSAMPLE_TARGET_DOMAIN", False))
        train_ds, val_ds, sampler = dataset_func(
            config, mode="split", debug=args.debug, rotate=True, continuous=args.continuous
        )
        config.DATA.MIXED = True
    elif config.DATA.DATASET == "road_dataset":
        train_ds, val_ds, sampler = build_road_network_data(
            config, mode="split", debug=args.debug, max_samples=config.DATA.NUM_SOURCE_SAMPLES,
            domain_classification=-1, gaussian_augment=True, rotate=True, continuous=args.continuous
        )
        config.DATA.MIXED = False
    elif config.DATA.DATASET == "synth_octa":
        train_ds, val_ds, sampler = build_octa_network_data(
            config, mode="split", debug=args.debug, max_samples=config.DATA.NUM_SOURCE_SAMPLES,
            domain_classification=-1, gaussian_augment=True, rotate=True, continuous=args.continuous
        )
        config.DATA.MIXED = False
    else:
        train_ds, val_ds, sampler = build_vessel_data(
            config, mode="split", debug=args.debug, max_samples=config.DATA.NUM_SOURCE_SAMPLES
        )
        config.DATA.MIXED = False

    loader = igdist.auto_dataloader if args.distributed else DataLoader

    train_loader = loader(
        train_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=not sampler, sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=lambda x: image_graph_collate(x, args.pre2d, gaussian_augment=config.DATA.MIXED),
        pin_memory=True
    )
    val_loader = loader(
        val_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=False, num_workers=config.DATA.NUM_WORKERS,
        collate_fn=lambda x: image_graph_collate(x, args.pre2d, gaussian_augment=config.DATA.MIXED),
        pin_memory=True
    )

    # ===================== MODEL =====================
    if getattr(args, "general_transformer", False):
        net = build_model(
            config, general_transformer=True, pretrain=args.pretrain_general,
            output_dim=2 if args.pretrain_general else 3
        )
    else:
        net = build_model(config, pre2d=args.pre2d)

    net = net.to(device)
    if args.distributed:
        net = igdist.auto_model(net)
        net_wo = net.module
    else:
        net_wo = net

    relation_embed = net_wo.relation_embed

    use_ema = bool(getattr(config.TRAIN, "USE_EMA", True))
    ema_relation = None
    if use_ema:
        ema_relation = EMA_Model(relation_embed, decay=float(getattr(config.TRAIN, "EMA_DECAY", 0.999))).to(device)

    matcher = build_matcher(config, dims=2 if args.pretrain_general else 3)

    if config.TRAIN.EDGE_SAMPLING_MODE == "none":
        edge_sampling_mode = EDGE_SAMPLING_MODE.NONE
    elif config.TRAIN.EDGE_SAMPLING_MODE == "up":
        edge_sampling_mode = EDGE_SAMPLING_MODE.UP
    elif config.TRAIN.EDGE_SAMPLING_MODE == "down":
        edge_sampling_mode = EDGE_SAMPLING_MODE.DOWN
    elif config.TRAIN.EDGE_SAMPLING_MODE == "random_up":
        edge_sampling_mode = EDGE_SAMPLING_MODE.RANDOM_UP
    else:
        raise ValueError("Invalid edge sampling mode")

    loss = SetCriterion(
        config, matcher, relation_embed,
        dims=2 if args.pretrain_general else 3,
        num_edge_samples=config.TRAIN.NUM_EDGE_SAMPLES,
        edge_sampling_mode=edge_sampling_mode,
        domain_class_weight=torch.tensor(config.TRAIN.DOMAIN_WEIGHTING, device=device),
        ema_relation_embed=ema_relation
    )
    val_loss = SetCriterion(
        config, matcher, relation_embed,
        dims=2 if args.pretrain_general else 3,
        edge_sampling_mode=EDGE_SAMPLING_MODE.NONE
    )

    param_dicts = [
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
            "lr": float(config.TRAIN.DOMAIN_LR),
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY),
        },
        {
            "params": [p for n, p in net.named_parameters() if not match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
            "lr": float(config.TRAIN.BASE_LR),
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY),
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=float(config.TRAIN.BASE_LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY))
    if args.distributed:
        optimizer = igdist.auto_optim(optimizer)

    # LR scheduler (polynomial warmup/decay)
    iter_per_epoch = len(train_loader)
    num_warmup_epoch = float(config.TRAIN.WARMUP_EPOCHS)
    warm_lr_init = float(config.TRAIN.WARMUP_LR)
    warm_lr_final = float(config.TRAIN.BASE_LR)
    num_warmup_iter = num_warmup_epoch * iter_per_epoch
    num_after_warmup_iter = config.TRAIN.EPOCHS * iter_per_epoch

    def lr_lambda_polynomial(it: int):
        if it < num_warmup_iter:
            lr_lamda0 = warm_lr_init / warm_lr_final
            return lr_lamda0 + (1 - lr_lamda0) * it / max(1, num_warmup_iter)
        return (1 - (it - num_warmup_iter) / max(1, num_after_warmup_iter)) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_polynomial)

    # Resume
    try:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            if args.sspt:
                checkpoint = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("momentum_encoder")}
            else:
                if args.ignore_backbone:
                    state_dict = net_wo.decoder.state_dict()
                    checkpoint["net"] = {k: v for k, v in checkpoint["net"].items() if k in state_dict}
                if args.load_only_decoder:
                    checkpoint["net"] = {k[8:]: v for k, v in checkpoint["net"].items() if k.startswith("decoder")}
                    net_wo.decoder.load_state_dict(checkpoint["net"], strict=not args.ignore_backbone and not args.no_strict_loading)
                else:
                    net_wo.load_state_dict(checkpoint["net"], strict=not args.ignore_backbone and not args.no_strict_loading)
                if args.restore_state:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"Checkpoint loaded from {args.resume}")
    except Exception as e:
        print("Error in loading checkpoint:", e)

    writer = SummaryWriter(log_dir=exp_path)

    evaluator = build_evaluator(
        val_loader, net, optimizer, scheduler, writer, config, device, val_loss,
        pre_2d=args.pre2d, pretrain_general=args.pretrain_general,
        gaussian_augment=config.DATA.MIXED, seg=args.seg
    )

    trainer = build_trainer(
        train_loader, net, loss, optimizer, scheduler, writer, evaluator, config, device,
        seg=args.seg, ema_relation=ema_relation
    )

    # ===================== CSV STATS (FULL) =====================
    stats_csv_primary = os.path.join(exp_path, "pred_edge_stats.csv")
    stats_csv_alias = os.path.join(exp_path, f"{str(args.exp_name).lower()}_stats.csv")
    stats_max_batches = int(getattr(config.TRAIN, "STATS_MAX_BATCHES", 10))

    stats_fields = [
        "epoch","split","samples",
        "gt_nodes_mean","gt_nodes_std",
        "gt_pos_edges_mean","gt_pos_edges_std",
        "gt_neg_edges_mean","gt_neg_edges_std",
        "pred_nodes_mean","pred_nodes_std",
        "pred_pos_edges_mean","pred_pos_edges_std",
        "pred_neg_edges_mean","pred_neg_edges_std",
        "recall_all_mean","recall_all_std",
        "recall_leaf_mean","recall_leaf_std",
        "recall_hub_mean","recall_hub_std",
        "recall_severe_mean","recall_severe_std",
        "gt_leaf_edges_mean","gt_leaf_edges_std",
        "gt_hub_edges_mean","gt_hub_edges_std",
        "gt_severe_edges_mean","gt_severe_edges_std",
    ]

    def _append_stats_row(path: str, row: dict):
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=stats_fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _log_pred_edge_stats(engine):
        epoch = int(engine.state.epoch)

        train_tot = _compute_pred_gt_counts_for_loader(
            train_loader, net, device, config, seg_mode=args.seg, max_batches=stats_max_batches
        )
        val_tot = _compute_pred_gt_counts_for_loader(
            val_loader, net, device, config, seg_mode=args.seg, max_batches=stats_max_batches
        )

        row_train = {"epoch": epoch, "split": "train", **train_tot}
        row_val = {"epoch": epoch, "split": "val", **val_tot}

        # write to both locations (so your existing hnm_proof_stats.csv habit keeps working)
        _append_stats_row(stats_csv_primary, row_train)
        _append_stats_row(stats_csv_primary, row_val)
        _append_stats_row(stats_csv_alias, row_train)
        _append_stats_row(stats_csv_alias, row_val)

        print(
            f"[PRED_STATS][epoch={epoch}] "
            f"train recall_all={train_tot['recall_all_mean']:.3f} leaf={train_tot['recall_leaf_mean']:.3f} "
            f"hub={train_tot['recall_hub_mean']:.3f} severe={train_tot['recall_severe_mean']:.3f} | "
            f"val recall_all={val_tot['recall_all_mean']:.3f} leaf={val_tot['recall_leaf_mean']:.3f} "
            f"hub={val_tot['recall_hub_mean']:.3f} severe={val_tot['recall_severe_mean']:.3f} "
            f"| csv={stats_csv_primary}"
        )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_pred_edge_stats)
    # ==========================================================

    if (not args.distributed) or (igdist.get_rank() == 0):
        ProgressBar().attach(trainer, output_transform=lambda x: {"loss": x["loss"]["total"]})

    trainer.run()


# ===================== ENTRY =====================
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--config',
                        default=None,
                        help='config file (.yml) containing the hyper-parameters for training. '
                            'If None, use the nnU-Net config. See /config for examples.')
    parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
    parser.add_argument('--restore_state', dest='restore_state', help='whether the state should be restored', action='store_true')
    parser.add_argument('--device', default='cuda',
                            help='device to use for training')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                            help='list of index where skip conn will be made')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--nproc_per_node", default=None, type=int)
    parser.add_argument('--debug', dest='debug', help='do fast debug', action='store_true')  #TODO: remove
    parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str,required=True)
    parser.add_argument('--pre2d', dest='pre2d', help='Pretraining the network using 2D-segmentation after backbone', action="store_true")
    parser.add_argument('--use_retina', dest='use_retina', help='Use synthetic retina data instead of street data for pretraining', action="store_true")
    parser.add_argument('--continuous', dest='continuous', help='Using continuous rotation for gaussian augment', action="store_true")
    parser.add_argument("--parallel", dest='parallel', help='Using distributed training', action="store_true")
    parser.add_argument("--ignore_backbone",  dest='ignore_backbone', help='', action="store_true")
    parser.add_argument('--seg', dest='seg', help='Using segmentation or raw images', action="store_true")
    parser.add_argument('--general_transformer', dest='general_transformer', help='Using a general transformer with an adapter', action="store_true")
    parser.add_argument('--pretrain_general', dest='pretrain_general', help='Pretraining the general transformer with 2D data', action="store_true")
    parser.add_argument('--load_only_decoder', dest='load_only_decoder', help='When resuming, only load state from decoder instead of whole model. Useful when doing multi-modal transfer learning', action="store_true")
    parser.add_argument('--no_strict_loading', default=False, action="store_true",
                        help="Whether the model was pretrained with domain adversarial. If true, the checkpoint will be loaded with strict=false")
    parser.add_argument('--sspt', default=False, action="store_true",
                        help="Whether the model was pretrained with self supervised pretraining. If true, the checkpoint will be loaded accordingly. Only combine with resume.")
    parser.add_argument('--display_prob', type=float, default=0.0018, help="Probability of plotting the overlay image with the graph")



    ########################################
    ###########  syntheticMRI  #############
    ########################################

    # --- PRE-TRAINING ---
    
    args = parser.parse_args([
        '--exp_name', 'HNS_proof',
        '--config', '/home/scavone/cross-dim_i2g/3d/configs/graph_analysis.yaml',
        '--continuous',
        '--display_prob', '0.0',
        # '--resume', '/data/scavone/cross-dim_i2g_3d/runs/pretraining_mixed_synth_3_20/models/checkpoint_key_metric=7.2568.pt',
        # '--restore_state',
    ])
    
    
    # --- FINETUNING ---
    
    # args = parser.parse_args([
    #     '--exp_name', 'finetuning_synth_1',
    #     '--config', '/home/scavone/cross-dim_i2g/3d/configs/synth_3D.yaml',
    #     # '--resume', '/data/scavone/cross-dim_i2g_3d/runs/pretraining_mixed_synth_1_20/models/checkpoint_epoch=50.pt',
    #     # '--restore_state',
    #     # '--no_strict_loading',
    #     '--continuous',
    #     '--display_prob', '0.002',
    # ])
    

    if args.parallel:
        with igdist.Parallel(backend='nccl', nproc_per_node=args.nproc_per_node) as parallel:
            parallel.run(main, args)
    else:
        main(args=args)
