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
import networkx as nx

from data.dataset_vessel3d import build_vessel_data
from data.dataset_mixed import build_mixed_data
from models import build_model
from training.inference import relation_infer
from utils.utils import image_graph_collate

from metrics.smd import compute_meanSMD, SinkhornDistance
from metrics.boxap import box_ap, iou_filter, get_unique_iou_thresholds, get_indices_of_iou_for_each_metric
from metrics.box_ops_np import box_iou_np
from metrics.coco import COCOMetric

from utils.vis_debug_3d import DebugVisualizer3D


debug_use_gt_as_pred = False  # set True only for debugging metrics


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def set_dropout_eval(model: torch.nn.Module):
    """Disable dropout randomness during BN calibration / controlled runs."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or "Dropout" in m.__class__.__name__:
            m.eval()


@torch.no_grad()
def bn_calibrate_on_loader(net, loader, device, use_seg: bool, num_batches: int = 200):
    """
    Forward-only BN calibration:
    - net.train() so BatchNorm updates running_mean/running_var
    - dropout disabled for determinism
    """
    net.train()
    set_dropout_eval(net)

    n = 0
    for batchdata in tqdm(loader, desc=f"BN calibration ({num_batches} batches)", leave=True):
        images, segs, nodes, edges, z_pos, domains = batchdata
        images = images.to(device, non_blocking=False).float()
        segs = segs.to(device, non_blocking=False)
        domains = domains.to(device, non_blocking=False)

        x = segs.float() if use_seg else images
        net(x, z_pos, domain_labels=domains)

        n += 1
        if n >= num_batches:
            break

    net.eval()


def build_loader(dataset, config, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=lambda x: image_graph_collate(x, gaussian_augment=False),
        pin_memory=True
    )

def cwhd_to_xyzxyz(boxes, clip01=True):
    # boxes: [N, 6] = [cx,cy,cz, sx,sy,sz]
    c = boxes[:, :3]
    s = boxes[:, 3:]
    lo = c - 0.5 * s
    hi = c + 0.5 * s
    out = np.concatenate([lo, hi], axis=1)
    if clip01:
        out = np.clip(out, 0.0, 1.0)
    return out  

def evaluate_on_test(net, test_loader, config, device, args):
    """
    Runs your existing inference + metric computation on the TEST set.
    This is a slightly cleaned version of your loop. :contentReference[oaicite:1]{index=1}
    """
    t_start = time.time()
    sinkhorn_distance = SinkhornDistance(eps=1e-7, max_iter=100)

    metrics = tuple([COCOMetric(classes=['Node'], per_class=False, verbose=False)])
    iou_thresholds = get_unique_iou_thresholds(metrics)
    iou_mapping = get_indices_of_iou_for_each_metric(iou_thresholds, metrics)
    box_evaluator = box_ap(box_iou_np, iou_thresholds, max_detections=40)

    mean_smd = []
    node_ap_result = []
    edge_ap_result = []

    folds_smd = []
    folds_node_mAP = []
    folds_node_mAR = []
    folds_edge_mAP = []
    folds_edge_mAR = []

    # Visualization (optional)
    out_dir = str(Path(args.out_path) / args.exp_name / "test_vis")
    os.makedirs(out_dir, exist_ok=True)
    viz = DebugVisualizer3D(out_dir=out_dir, prob=config.display_prob)
    viz.start_epoch()

    # Folding like your original script (5 folds)
    dataset_len = len(test_loader.dataset)
    fold_size = max(1, int(dataset_len / 5))
    print("number of test samples:", dataset_len)
    print("fold size:", fold_size)

    current_fold = 0

    net.eval()

    for idx, batchdata in enumerate(tqdm(test_loader, desc="Test batches", leave=True)):
        images, segs, nodes, edges, z_pos, domains = batchdata

        images = images.to(device, non_blocking=False).float()
        segs = segs.to(device, non_blocking=False)
        nodes = [node.to(device, non_blocking=False) for node in nodes]
        edges = [edge.to(device, non_blocking=False) for edge in edges]
        domains = domains.to(device, non_blocking=False)

        x = segs.float() if args.seg else images

        with torch.no_grad():
            h, out, *_ = net(x, z_pos, domain_labels=domains)
            out = relation_infer(
                h.detach(), out, net,
                config.MODEL.DECODER.OBJ_TOKEN,
                config.MODEL.DECODER.RLN_TOKEN,
                apply_nms=False
            )

        # Optional visualization
        viz.maybe_save(
            segs=segs,
            images=images,
            gt_nodes_list=nodes,
            gt_edges_list=edges,
            pred_nodes_list=out['pred_nodes'],
            pred_edges_list=out['pred_rels'],
            epoch=current_fold,
            step=idx,
            batch_index=0,
            tag="test",
        )

        if args.eval:
            # compute metrics per-sample (same as your original) :contentReference[oaicite:2]{index=2}
            for sample_num in range(images.shape[0]):
                pred_nodes = torch.tensor(out['pred_nodes'][sample_num], dtype=torch.float)
                pred_edges = torch.tensor(out['pred_rels'][sample_num], dtype=torch.int64)

                sample_nodes = nodes[sample_num]
                sample_edges = edges[sample_num]

                # GT boxes for nodes
                gt_boxes = [torch.cat([sample_nodes, 0.2*torch.ones(sample_nodes.shape, device=sample_nodes.device)], dim=-1).cpu().numpy()]
                gt_boxes_class = [np.zeros(gt_boxes[0].shape[0])]

                # GT boxes for edges
                edge_boxes = torch.stack([sample_nodes[sample_edges[:, 0]], sample_nodes[sample_edges[:, 1]]], dim=2)
                edge_boxes = torch.cat([torch.min(edge_boxes, dim=2)[0]-0.1, torch.max(edge_boxes, dim=2)[0]+0.1], dim=-1).cpu().numpy()
                edge_boxes = [edge_boxes[:, [0, 1, 3, 4, 2, 5]]]
                edge_boxes_class = [np.zeros(sample_edges.shape[0])]

                # Pred edge boxes
                if pred_edges.shape[0] > 0:
                    pred_edge_boxes = torch.stack([pred_nodes[pred_edges[:, 0]], pred_nodes[pred_edges[:, 1]]], dim=2)
                    pred_edge_boxes = torch.cat([torch.min(pred_edge_boxes, dim=2)[0]-0.1,
                                                 torch.max(pred_edge_boxes, dim=2)[0]+0.1], dim=-1).numpy()
                    pred_edge_boxes = [pred_edge_boxes[:, [0, 1, 3, 4, 2, 5]]]
                else:
                    pred_edge_boxes = []

                if debug_use_gt_as_pred:
                    pred_boxes_debug       = gt_boxes
                    pred_boxes_class_debug = gt_boxes_class
                    pred_boxes_score_debug = [np.ones_like(gt_boxes_class[0], dtype=np.float32)]

                    pred_edge_boxes_debug  = edge_boxes
                    pred_rels_class_debug  = edge_boxes_class
                    pred_rels_score_debug  = [np.ones_like(edge_boxes_class[0], dtype=np.float32)]
                else:
                    pred_boxes_debug       = [out["pred_boxes"][sample_num]]
                    pred_boxes_class_debug = [out["pred_boxes_class"][sample_num]]
                    pred_boxes_score_debug = [out["pred_boxes_score"][sample_num]]

                    pred_edge_boxes_debug  = pred_edge_boxes
                    pred_rels_class_debug  = [out["pred_rels_class"][sample_num]]
                    pred_rels_score_debug  = [out["pred_rels_score"][sample_num]]
                  
                    
                # ---- GT (list of numpy arrays) ----
                gt_boxes_cat = np.concatenate(gt_boxes, axis=0)   # [N, D]
                
                print('-'*50)
                print("GT first 3 boxes raw:\n", gt_boxes_cat[:3])
                

                D = gt_boxes_cat.shape[1]
                mid = D // 2

                gt_whd = gt_boxes_cat[:, mid:] - gt_boxes_cat[:, :mid]
                print("GT shape:", gt_boxes_cat.shape)
                print("GT min/max:", gt_boxes_cat.min(), gt_boxes_cat.max())
                print("GT mean size per dim:", gt_whd.mean(axis=0))
                print("GT mean size (scalar):", gt_whd.mean())

                # ---- PRED (already numpy) ----
                pred_boxes = out["pred_boxes"][sample_num]        # numpy array [M, D]
                print("PRED first 3 boxes raw:\n", pred_boxes[:3])
                D = pred_boxes.shape[1]
                mid = D // 2

                pred_whd = pred_boxes[:, mid:] - pred_boxes[:, :mid]
                print("PRED shape:", pred_boxes.shape)
                print("PRED min/max:", pred_boxes.min(), pred_boxes.max())
                print("PRED mean size per dim:", pred_whd.mean(axis=0))
                print("PRED mean size (scalar):", pred_whd.mean())
                print('-'*50)

                # Node AP
                node_ap = box_evaluator(
                    pred_boxes_debug,
                    pred_boxes_class_debug,
                    pred_boxes_score_debug,
                    gt_boxes,
                    gt_boxes_class
                )
                node_ap_result.extend(node_ap)

                # Edge AP
                edge_ap = box_evaluator(
                    pred_edge_boxes_debug,
                    pred_rels_class_debug,
                    pred_rels_score_debug,
                    edge_boxes,
                    edge_boxes_class,
                    convert_box=False
                )
                edge_ap_result.extend(edge_ap)

                # SMD
                A = torch.zeros((sample_nodes.shape[0], sample_nodes.shape[0]))
                pred_A = torch.zeros((pred_nodes.shape[0], pred_nodes.shape[0]))

                A[sample_edges[:, 0], sample_edges[:, 1]] = 1
                A[sample_edges[:, 1], sample_edges[:, 0]] = 1
                A = torch.tril(A)

                if sample_nodes.shape[0] > 1 and pred_nodes.shape[0] > 1 and pred_edges.numel() > 0:
                    pred_A[pred_edges[:, 0], pred_edges[:, 1]] = 1.0
                    pred_A[pred_edges[:, 1], pred_edges[:, 0]] = 1.0
                    pred_A = torch.tril(pred_A)

                    mean_smd.append(
                        compute_meanSMD(A, sample_nodes, pred_A, pred_nodes, sinkhorn_distance, n_points=100).numpy()
                    )

                # Fold boundary: compute intermediate fold results
                global_sample_idx = idx * config.DATA.BATCH_SIZE + sample_num
                if global_sample_idx % fold_size == (fold_size - 1):
                    tqdm.write("Calculating intermediate results")

                    node_metric_scores = {}
                    edge_metric_scores = {}
                    for metric_idx, metric in enumerate(metrics):
                        _filter = partial(iou_filter, iou_idx=iou_mapping[metric_idx])

                        iou_filtered_results = list(map(_filter, node_ap_result))
                        score, curve = metric(iou_filtered_results)
                        if score is not None:
                            node_metric_scores.update(score)

                        iou_filtered_results = list(map(_filter, edge_ap_result))
                        score, curve = metric(iou_filtered_results)
                        if score is not None:
                            edge_metric_scores.update(score)

                    folds_node_mAP.append(node_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_40'][0])
                    folds_node_mAR.append(node_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_40'][0])
                    folds_edge_mAP.append(edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_40'][0])
                    folds_edge_mAR.append(edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_40'][0])

                    smd_mean = torch.tensor(mean_smd).mean() if len(mean_smd) else torch.tensor(0.0)
                    folds_smd.append(smd_mean)

                    # Reset fold accumulators
                    mean_smd = []
                    node_ap_result = []
                    edge_ap_result = []

                    viz.start_epoch()
                    current_fold += 1

    # Aggregate final metrics
    smd = torch.tensor(folds_smd).mean()
    smd_std = torch.tensor(folds_smd).std() if len(folds_smd) > 1 else torch.tensor(0.0)

    node_mAP = torch.tensor(folds_node_mAP).mean()
    node_mAP_std = torch.tensor(folds_node_mAP).std() if len(folds_node_mAP) > 1 else torch.tensor(0.0)

    node_mAR = torch.tensor(folds_node_mAR).mean()
    node_mAR_std = torch.tensor(folds_node_mAR).std() if len(folds_node_mAR) > 1 else torch.tensor(0.0)

    edge_mAP = torch.tensor(folds_edge_mAP).mean()
    edge_mAP_std = torch.tensor(folds_edge_mAP).std() if len(folds_edge_mAP) > 1 else torch.tensor(0.0)

    edge_mAR = torch.tensor(folds_edge_mAR, dtype=torch.float32).mean()
    edge_mAR_std = torch.tensor(folds_edge_mAR, dtype=torch.float32).std() if len(folds_edge_mAR) > 1 else torch.tensor(0.0)

    print("smd: ", smd.item())
    print("smd std: ", smd_std.item())
    print("node mAP: ", node_mAP.item())
    print("node mAP std: ", node_mAP_std.item())
    print("node mAR: ", node_mAR.item())
    print("node mAR std: ", node_mAR_std.item())
    print("edge mAP: ", edge_mAP.item())
    print("edge mAP std: ", edge_mAP_std.item())
    print("edge mAR: ", edge_mAR.item())
    print("edge mAR std: ", edge_mAR_std.item())

    # Save CSV like your original
    dest_path = Path(args.out_path) / args.exp_name
    dest_path.mkdir(parents=True, exist_ok=True)
    csv_file = dest_path / "results.csv"

    csv_header_string = "smd;smd_std;node_mAP;node_mAP_std;node_mAR;node_mAR_std;edge_mAP;edge_mAP_std;edge_mAR;edge_mAR_std"
    csv_value_string = f"{smd.item()};{smd_std.item()};{node_mAP.item()};{node_mAP_std.item()};{node_mAR.item()};{node_mAR_std.item()};{edge_mAP.item()};{edge_mAP_std.item()};{edge_mAR.item()};{edge_mAR_std.item()}"

    # with open(csv_file, "w", encoding="utf-8") as f:
    #     f.write(csv_header_string + "\n")
    #     f.write(csv_value_string + "\n")

    print(f"[OK] Saved results to: {csv_file}")


def main(args):
    # Load config
    with open(args.config) as f:
        print("\n*** Config file")
        print(args.config)
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        print(cfg["log"]["exp_name"] if "log" in cfg and "exp_name" in cfg["log"] else "(exp_name not found)")
    config = dict2obj(cfg)

    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    config.display_prob = args.display_prob
    config.DATA.MIXED = False  # vessel only

    # --- Build train/val/test vessel datasets ---
    # Assumes your build_vessel_data supports these modes.
    if args.mixed:
        config.DATA.MIXED = True
        print('Mixed BN Calibration!')
        train_ds, val_ds, _ = build_mixed_data(config, mode="split", debug=False)
        test_ds  = build_vessel_data(config, mode="test",  debug=False, max_samples=args.max_samples_test)
    
    else:
        train_ds, val_ds, _ = build_vessel_data(config, mode="split", debug=False, max_samples=args.max_samples_train if args.max_samples_train > 0 else -1)
        test_ds  = build_vessel_data(config, mode="test",  debug=False, max_samples=args.max_samples_test)

    train_loader = build_loader(train_ds, config, shuffle=True)
    val_loader   = build_loader(val_ds,   config, shuffle=False)
    test_loader  = build_loader(test_ds,  config, shuffle=False)

    print(f"train/val/test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    # --- Build model ---
    if args.general_transformer:
        net = build_model(config, general_transformer=True, pretrain=False, output_dim=3).to(device)
    else:
        net = build_model(config).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    net.load_state_dict(checkpoint["net"], strict=not args.no_strict_loading)
    net.eval()

    # --- BN calibration on VAL ---
    if args.bn_calibrate:
        bn_calibrate_on_loader(
            net,
            val_loader,
            device=device,
            use_seg=args.seg,
            num_batches=args.bn_calib_batches
        )

    # --- Evaluate on TEST ---
    evaluate_on_test(net, test_loader, config, device, args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seg", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--general_transformer", action="store_true")
    parser.add_argument("--no_strict_loading", default=False, action="store_true")

    # dataset limits
    parser.add_argument("--max_samples_train", type=int, default=0)
    parser.add_argument("--max_samples_val", type=int, default=0)
    parser.add_argument("--max_samples_test", type=int, default=0)

    # BN calibration
    parser.add_argument("--bn_calibrate", action="store_true")
    parser.add_argument("--bn_calib_batches", type=int, default=100)

    # evaluation outputs
    parser.add_argument("--eval", action="store_true", help="Compute metrics on test set.")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--display_prob", type=float, default=0.0018)

    args = parser.parse_args([
        '--exp_name', 'finetuning_1_mixed_synth_vessel_calib',
        '--config', '/home/scavone/cross-dim_i2g/3d/configs/mixed_synth_3D.yaml',
        '--model', '/data/scavone/cross-dim_i2g_3d/runs/finetuning_mixed_synth_1_20/models/checkpoint_epoch=100.pt',
        '--out_path', '/data/scavone/cross-dim_i2g_3d/test_results',
        '--max_samples_test', '5000',
        '--max_samples_val', '1000',
        '--eval',
        '--no_strict_loading',
        '--display_prob', '0.0',
        '--bn_calibrate',
    ])
    
    main(args)
