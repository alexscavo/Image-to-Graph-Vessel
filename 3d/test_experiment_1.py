#!/usr/bin/env python3
"""
Experiment 1 (based on your run_batch_inference_eval.py): run ONE pass over the loader,
and for each batch make predictions twice:
  (A) with net.eval()
  (B) with net.train()   (still under torch.no_grad())

It saves two visualizations per batch ("eval" and "train") so you can see whether
.eval() materially changes outputs (BN / Dropout effects).

Notes:
- Using net.train() will also UPDATE BatchNorm running stats (even with no_grad).
  That is part of what "train mode" means. If you want "train behavior without
  updating BN stats", tell me and I’ll give a BN-frozen variant.
"""

import os
from pathlib import Path
import time
import torch
import yaml
import json
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from functools import partial
from monai.data import DataLoader

from data.dataset_vessel3d import build_vessel_data
from models import build_model
from training.inference import relation_infer
from utils.utils import image_graph_collate
from utils.vis_debug_3d import DebugVisualizer3D

from metrics.smd import compute_meanSMD, SinkhornDistance
from metrics.boxap import box_ap, iou_filter, get_unique_iou_thresholds, get_indices_of_iou_for_each_metric
from metrics.box_ops_np import box_iou_np
from metrics.coco import COCOMetric
from torch.nn.modules.batchnorm import _BatchNorm



class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def tensor_diff_stats(a: torch.Tensor, b: torch.Tensor):
    d = (a.detach() - b.detach()).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
    }

@torch.no_grad()
def forward_raw_and_infer(net, x, z_pos, domains, config):
    # Raw forward (tensors)
    h, out_raw, _, _, _, _ = net(x, z_pos, domain_labels=domains)

    # Postprocess (may turn tensors into lists)
    out_pp = relation_infer(
        h.detach(),
        out_raw,
        net,
        config.MODEL.DECODER.OBJ_TOKEN,
        config.MODEL.DECODER.RLN_TOKEN,
        apply_nms=False
    )
    return h, out_raw, out_pp

def set_bn_train_dropout_eval(model: torch.nn.Module):
    """
    Put model in train(), but:
      - BN layers in train() so they update running stats
      - Dropout layers in eval() so dropout is OFF (stability)
    """
    model.train()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.train()
        elif isinstance(m, torch.nn.Dropout) or "Dropout" in m.__class__.__name__:
            m.eval()

def set_bn_eval_dropout_eval(model: torch.nn.Module):
    """
    Put model in train(), but:
      - BN layers in eval() (freeze BN)
      - Dropout layers in eval() (dropout OFF)
    This matches your Experiment 2A setting (BN frozen).
    """
    model.train()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
        elif isinstance(m, torch.nn.Dropout) or "Dropout" in m.__class__.__name__:
            m.eval()

@torch.no_grad()
def bn_calibrate(model, calib_loader, device, use_seg: bool, num_batches: int = 200):
    """
    BN calibration: forward passes only to update BatchNorm running stats.
    No weights updated, no grads, no labels used.
    """
    set_bn_train_dropout_eval(model)

    n = 0
    for batch in tqdm(calib_loader, desc=f"BN calibration ({num_batches} batches)", leave=False):
        images, segs, nodes, edges, z_pos, domains = batch
        x = segs if use_seg else images
        x = x.to(device, non_blocking=False).float()
        domains = domains.to(device, non_blocking=False)

        # forward only to update BN running stats
        model(x, z_pos, domain_labels=domains)

        n += 1
        if n >= num_batches:
            break

    model.eval()

@torch.no_grad()
def exp1_diff_on_first_batch(model, loader, device, use_seg: bool, config):
    """
    Compute your Experiment-1 raw diffs on a single batch:
      eval()  vs  train() with dropout OFF (and BN behavior depends on caller).
    Returns (dlog, dnodes).
    """
    batch = next(iter(loader))
    images, segs, nodes, edges, z_pos, domains = batch
    x = segs if use_seg else images
    x = x.to(device, non_blocking=False).float()
    domains = domains.to(device, non_blocking=False)

    # eval mode
    model.eval()
    _, out_raw_e, _ = forward_raw_and_infer(model, x, z_pos, domains, config)

    # train mode with dropout OFF (BN behavior depends on current BN mode setting)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or "Dropout" in m.__class__.__name__:
            m.eval()
    _, out_raw_t, _ = forward_raw_and_infer(model, x, z_pos, domains, config)

    dlog = tensor_diff_stats(out_raw_e["pred_logits"], out_raw_t["pred_logits"])
    dnodes = tensor_diff_stats(out_raw_e["pred_nodes"], out_raw_t["pred_nodes"])
    return dlog, dnodes


def main(args):
    # Load config
    with open(args.config) as f:
        print("\n*** Config file")
        print(args.config)
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(cfg["log"]["exp_name"])
    config = dict2obj(cfg)

    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    config.display_prob = args.display_prob

    # IMPORTANT: ensure collate isn't injecting augmentation
    config.DATA.MIXED = False

    # Dataset / loader (keep consistent with your script, but shuffle=False for stable comparisons)
    dataset = build_vessel_data(config, mode="test", debug=False, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,  # <<< important for repeatability
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=lambda x: image_graph_collate(x, gaussian_augment=config.DATA.MIXED),
        pin_memory=True,
    )

    # Model
    if args.general_transformer:
        net = build_model(config, general_transformer=True, pretrain=False, output_dim=3).to(device)
    else:
        net = build_model(config).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    net.load_state_dict(checkpoint["net"], strict=not args.no_strict_loading)
    
    # -------------------------------
    # BN CALIBRATION SETUP + CHECK
    # -------------------------------
    if args.bn_calibrate:
        # Build a calibration dataset/loader.
        # IMPORTANT: you want "train-no-aug" here (no augmentation, no shuffle).
        # If your build_vessel_data(mode="train") already uses augment=True internally,
        # then you need to pass/force augment=False in your dataset builder.
        #
        # If build_vessel_data doesn't expose it, use the split that corresponds to
        # your training samples but with augmentation disabled in the dataset.
        calib_dataset = build_vessel_data(config, mode=args.bn_calib_mode, debug=False, max_samples=args.bn_calib_max_samples)
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            collate_fn=lambda x: image_graph_collate(x, gaussian_augment=False),
            pin_memory=True,
        )

        # 1) Check EXP1 diff BEFORE calibration (with BN frozen vs eval is usually smaller;
        # but we want to see your current eval vs train mismatch clearly)
        print("\n[Check] EXP1 diffs BEFORE BN calibration (eval vs train, dropout OFF):")
        dlog_before, dnodes_before = exp1_diff_on_first_batch(net, loader, device, args.seg, config)
        print("  pred_logits:", dlog_before)
        print("  pred_nodes :", dnodes_before)

        # 2) Calibrate BN running stats
        print(f"\n[Action] Running BN calibration on mode='{args.bn_calib_mode}' for {args.bn_calib_batches} batches ...")
        bn_calibrate(net, calib_loader, device, use_seg=args.seg, num_batches=args.bn_calib_batches)

        # 3) Check EXP1 diff AFTER calibration
        print("\n[Check] EXP1 diffs AFTER BN calibration (eval vs train, dropout OFF):")
        dlog_after, dnodes_after = exp1_diff_on_first_batch(net, loader, device, args.seg, config)
        print("  pred_logits:", dlog_after)
        print("  pred_nodes :", dnodes_after)
        print("\nExpected: the eval-vs-train gap should shrink a lot after calibration.\n")


    # Output dir and visualizer
    out_dir = f"/data/scavone/cross-dim_i2g_3d/visual di prova/test_vis/{args.exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    viz = DebugVisualizer3D(out_dir=out_dir, prob=config.display_prob)
    viz.start_epoch()

    # (Optional) metrics setup (same as your script)
    sinkhorn_distance = SinkhornDistance(eps=1e-7, max_iter=100)
    metrics = tuple([COCOMetric(classes=["Node"], per_class=False, verbose=False)])
    iou_thresholds = get_unique_iou_thresholds(metrics)
    iou_mapping = get_indices_of_iou_for_each_metric(iou_thresholds, metrics)
    box_evaluator = box_ap(box_iou_np, iou_thresholds, max_detections=40)

    mean_smd_eval, node_ap_eval, edge_ap_eval = [], [], []
    mean_smd_train, node_ap_train, edge_ap_train = [], [], []

    t_start = time.time()
    print("number of test samples:", len(dataset))

    for idx, batchdata in enumerate(tqdm(loader, desc="Batches", leave=True)):
        images, segs, nodes, edges, z_pos, domains = (
            batchdata[0],
            batchdata[1],
            batchdata[2],
            batchdata[3],
            batchdata[4],
            batchdata[5],
        )

        images = images.to(device, non_blocking=False).float()
        segs = segs.to(device, non_blocking=False).float()
        nodes = [node.to(device, non_blocking=False) for node in nodes]
        edges = [edge.to(device, non_blocking=False) for edge in edges]
        domains = domains.to(device, non_blocking=False)

        x = segs if args.seg else images

        # --- A) EVAL MODE ---
        net.eval()
        h_e, out_raw_e, out_pp_e = forward_raw_and_infer(net, x, z_pos, domains, config)

        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout) or "Dropout" in m.__class__.__name__:
                m.eval()
        h_t, out_raw_t, out_pp_t = forward_raw_and_infer(net, x, z_pos, domains, config)

        if idx == 0:
            print("RAW out keys:", list(out_raw_e.keys()))


        # Diff RAW tensors (these exist pre relation_infer)
        dlog  = tensor_diff_stats(out_raw_e["pred_logits"], out_raw_t["pred_logits"])
        dnodes = tensor_diff_stats(out_raw_e["pred_nodes"],  out_raw_t["pred_nodes"])

        tqdm.write(f"[batch {idx}] diff RAW pred_logits (eval vs train): {dlog} | diff RAW pred_nodes: {dnodes}")



        # Visualize both predictions for the first sample of the batch
        viz.maybe_save(
            segs=segs,
            images=images,
            gt_nodes_list=nodes,
            gt_edges_list=edges,
            pred_nodes_list=out_pp_e["pred_nodes"],
            pred_edges_list=out_pp_e["pred_rels"],
            epoch=0,
            step=idx,
            batch_index=0,
            tag="EVAL",
        )
        viz.maybe_save(
            segs=segs,
            images=images,
            gt_nodes_list=nodes,
            gt_edges_list=edges,
            pred_nodes_list=out_pp_t["pred_nodes"],
            pred_edges_list=out_pp_t["pred_rels"],
            epoch=0,
            step=idx,
            batch_index=0,
            tag="TRAIN",
        )

        # Optional: compute metrics for both modes
        if args.eval:
            for sample_num in range(images.shape[0]):
                sample_nodes = nodes[sample_num]
                sample_edges = edges[sample_num]

                # GT node boxes
                gt_boxes = [torch.cat([sample_nodes, 0.2 * torch.ones(sample_nodes.shape, device=sample_nodes.device)], dim=-1).cpu().numpy()]
                gt_boxes_class = [np.zeros(gt_boxes[0].shape[0])]

                # GT edge boxes
                gt_edge_boxes = torch.stack([sample_nodes[sample_edges[:, 0]], sample_nodes[sample_edges[:, 1]]], dim=2)
                gt_edge_boxes = torch.cat(
                    [torch.min(gt_edge_boxes, dim=2)[0] - 0.1, torch.max(gt_edge_boxes, dim=2)[0] + 0.1],
                    dim=-1,
                ).cpu().numpy()
                gt_edge_boxes = [gt_edge_boxes[:, [0, 1, 3, 4, 2, 5]]]
                gt_edge_boxes_class = [np.zeros(sample_edges.shape[0])]

                def eval_one_mode(out_mode, node_ap_list, edge_ap_list, smd_list, mode_name: str):
                    pred_nodes = torch.tensor(out_mode["pred_nodes"][sample_num], dtype=torch.float)
                    pred_edges = torch.tensor(out_mode["pred_rels"][sample_num], dtype=torch.int64)

                    # Nodes: use the same fields your script expects
                    pred_boxes = [out_mode["pred_boxes"][sample_num]]
                    pred_boxes_class = [out_mode["pred_boxes_class"][sample_num]]
                    pred_boxes_score = [out_mode["pred_boxes_score"][sample_num]]

                    # Edges: build boxes if any predicted edges
                    if pred_edges.shape[0] > 0:
                        pred_edge_boxes = torch.stack([pred_nodes[pred_edges[:, 0]], pred_nodes[pred_edges[:, 1]]], dim=2)
                        pred_edge_boxes = torch.cat(
                            [torch.min(pred_edge_boxes, dim=2)[0] - 0.1, torch.max(pred_edge_boxes, dim=2)[0] + 0.1],
                            dim=-1,
                        ).numpy()
                        pred_edge_boxes = [pred_edge_boxes[:, [0, 1, 3, 4, 2, 5]]]
                    else:
                        pred_edge_boxes = []

                    pred_rels_class = [out_mode["pred_rels_class"][sample_num]]
                    pred_rels_score = [out_mode["pred_rels_score"][sample_num]]

                    # AP
                    node_ap = box_evaluator(pred_boxes, pred_boxes_class, pred_boxes_score, gt_boxes, gt_boxes_class)
                    node_ap_list.extend(node_ap)

                    edge_ap = box_evaluator(
                        pred_edge_boxes,
                        pred_rels_class,
                        pred_rels_score,
                        gt_edge_boxes,
                        gt_edge_boxes_class,
                        convert_box=False,
                    )
                    edge_ap_list.extend(edge_ap)

                    # SMD (same check as your script)
                    A = torch.zeros((sample_nodes.shape[0], sample_nodes.shape[0]))
                    A[sample_edges[:, 0], sample_edges[:, 1]] = 1
                    A[sample_edges[:, 1], sample_edges[:, 0]] = 1
                    A = torch.tril(A)

                    pred_A = torch.zeros((pred_nodes.shape[0], pred_nodes.shape[0]))
                    if sample_nodes.shape[0] > 1 and pred_nodes.shape[0] > 1 and pred_edges.numel() != 0:
                        pred_A[pred_edges[:, 0], pred_edges[:, 1]] = 1.0
                        pred_A[pred_edges[:, 1], pred_edges[:, 0]] = 1.0
                        pred_A = torch.tril(pred_A)
                        smd_list.append(
                            compute_meanSMD(A, sample_nodes, pred_A, pred_nodes, sinkhorn_distance, n_points=100).numpy()
                        )

                eval_one_mode(out_pp_e, node_ap_eval, edge_ap_eval, mean_smd_eval, "EVAL")
                eval_one_mode(out_pp_t, node_ap_train, edge_ap_train, mean_smd_train, "TRAIN")

    print(f"\nDone in {(time.time() - t_start):.1f}s. Visualizations saved to: {out_dir}")

    if args.eval:
        # Compute final metrics for both modes (mAP/mAR across IoUs, same as your code)
        def summarize(node_ap_res, edge_ap_res, mean_smd_list, label: str):
            node_metric_scores = {}
            edge_metric_scores = {}
            for metric_idx, metric in enumerate(metrics):
                _filter = partial(iou_filter, iou_idx=iou_mapping[metric_idx])

                iou_filtered_results = list(map(_filter, node_ap_res))
                score, _ = metric(iou_filtered_results)
                if score is not None:
                    node_metric_scores.update(score)

                iou_filtered_results = list(map(_filter, edge_ap_res))
                score, _ = metric(iou_filtered_results)
                if score is not None:
                    edge_metric_scores.update(score)

            smd = float(torch.tensor(mean_smd_list).mean().item()) if len(mean_smd_list) else float("nan")
            print(f"\n=== {label} ===")
            print("smd:", smd)
            if "mAP_IoU_0.50_0.95_0.05_MaxDet_40" in node_metric_scores:
                print("node mAP:", node_metric_scores["mAP_IoU_0.50_0.95_0.05_MaxDet_40"][0])
                print("node mAR:", node_metric_scores["mAR_IoU_0.50_0.95_0.05_MaxDet_40"][0])
            if "mAP_IoU_0.50_0.95_0.05_MaxDet_40" in edge_metric_scores:
                print("edge mAP:", edge_metric_scores["mAP_IoU_0.50_0.95_0.05_MaxDet_40"][0])
                print("edge mAR:", edge_metric_scores["mAR_IoU_0.50_0.95_0.05_MaxDet_40"][0])

        summarize(node_ap_eval, edge_ap_eval, mean_smd_eval, "EVAL mode")
        summarize(node_ap_train, edge_ap_train, mean_smd_train, "TRAIN mode")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--general_transformer", action="store_true")
    parser.add_argument("--no_strict_loading", default=False, action="store_true")
    parser.add_argument("--out_path", default=None, required=False)  # not used (kept for compatibility)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--display_prob", type=float, default=0.0018)
    parser.add_argument("--bn_calibrate", action="store_true", help="Run BN calibration before evaluation.")
    parser.add_argument("--bn_calib_batches", type=int, default=200, help="How many batches to use for BN calibration.")
    parser.add_argument("--bn_calib_mode", type=str, default="train", help="Which split to use for BN calibration (e.g. train).")
    parser.add_argument("--bn_calib_max_samples", type=int, default=0, help="Cap calibration dataset size (0 = no cap).")


    args = parser.parse_args([
        '--exp_name', 'prova',
        '--config', '/home/scavone/cross-dim_i2g/3d/configs/synth_3D.yaml',
        '--model', '/data/scavone/cross-dim_i2g_3d/runs/finetuning_synth_1_20/models/checkpoint_epoch=100.pt',
        '--out_path', '/data/scavone/cross-dim_i2g_3d/test_results',
        '--max_samples', '5000',
        '--eval',
        '--no_strict_loading',
        '--display_prob', '0.0',
        # '--bn_calibrate',
        '--bn_calib_mode', 'train',
        '--bn_calib_batches', '200',

    ])
    
    main(args)
