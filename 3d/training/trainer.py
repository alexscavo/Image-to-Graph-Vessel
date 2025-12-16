import random
import sys
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.config import IgniteInfo
from monai.engines.utils import (
    default_metric_cmp_fn,
    default_prepare_batch,
)
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import min_version, optional_import

import os
from monai.engines import SupervisedTrainer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver, MeanDice
import torch
import gc
from metrics.similarity import SimilarityMetricPCA, SimilarityMetricTSNE, batch_cka, batch_cosine, batch_euclidean, downsample_examples, upsample_examples
from metrics.svcca import get_cca_similarity, robust_cca_similarity
import numpy as np
from ignite.engine import Events
from utils.vis_debug_3d import DebugVisualizer3D
from utils.open3d_tb import Open3DTensorboardLogger
from training.inference import relation_infer


if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")
    
debug_check = False


# define customized trainer
class RelationformerTrainer(SupervisedTrainer):
    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        seg: bool = False,
        alpha_coeff: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            train_data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_train_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            train_handlers=train_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network = network,
            optimizer = optimizer,
            loss_function = loss_function,
            inferer = SimpleInferer() if inferer is None else inferer,
            optim_set_to_none = optim_set_to_none,
        )
        self.config = kwargs.pop('config')
        self.seg = seg
        self.alpha_coeff = alpha_coeff

    def _iteration(self, engine, batchdata):
        images, segs, nodes, edges, z_pos, domains = batchdata[0], batchdata[1], batchdata[2], batchdata[3], batchdata[4], batchdata[5]
        
        # # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # # inputs = torch.cat(inputs, 1)
        images = images.to(engine.state.device,  non_blocking=False)
        segs = segs.to(engine.state.device,  non_blocking=False)
        nodes = [node.to(engine.state.device,  non_blocking=False) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        domains = domains.to(engine.state.device, non_blocking=False)
        target = {"nodes": nodes, "edges": edges, "domains": domains}

        self.network.train()
        self.optimizer.zero_grad()

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        
        if engine.state.iteration % engine.state.epoch_length == 1:
            print("epoch", engine.state.epoch,
                "iteration", engine.state.iteration,
                "epoch_length", engine.state.epoch_length)
            

        # --- PROVARE
        # global_iter = engine.state.iteration - 1  # 0-based global step
        # epoch_len = engine.state.epoch_length
        # total_iters = engine.state.max_epochs * epoch_len

        # warm_epochs = getattr(self.config.TRAIN, "ALPHA_WARMUP_EPOCHS", 0)
        # alpha_max = getattr(self.config.TRAIN, "ALPHA_MAX", self.alpha_coeff)

        # warm_iters = warm_epochs * epoch_len

        # if global_iter < warm_iters:
        #     alpha = 0.0
        # else:
        #     # progress only over the post-warmup part
        #     p = (global_iter - warm_iters) / float(max(total_iters - warm_iters, 1))
        #     alpha = (2. / (1. + np.exp(-10 * p)) - 1) * alpha_max

        
        p = float(iteration + epoch * engine.state.epoch_length) / engine.state.max_epochs / engine.state.epoch_length
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * self.alpha_coeff

        # ============= DIAGNOSTIC BLOCK START =============
        if debug_check:
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch} - ITERATION {iteration}")
            print(f"{'='*70}")
            
            # 1. Check input data
            print(f"\n[1. INPUT DATA CHECK]")
            for i in range(len(nodes)):
                n_nodes = nodes[i].shape[0]
                n_edges = edges[i].shape[0]
                fg_pixels = (segs[i] > 0).sum().item()
                total_pixels = segs[i].numel()
                
                print(f"  Sample {i}:")
                print(f"    GT: {n_nodes} nodes, {n_edges} edges")
                print(f"    Seg: {fg_pixels}/{total_pixels} foreground ({100*fg_pixels/total_pixels:.1f}%)")
                
                if n_nodes > 0:
                    coords = nodes[i]
                    print(f"    Node coords range: [{coords.min():.3f}, {coords.max():.3f}]")
                    if coords.min() < -0.01 or coords.max() > 1.01:
                        print(f"    ⚠️ WARNING: Coords outside [0,1]!")
                
                if n_edges > 0:
                    print(f"    Edge indices range: [{edges[i].min()}, {edges[i].max()}]")
                    if edges[i].max() >= n_nodes:
                        print(f"    ⚠️ WARNING: Edge indices out of bounds! max_idx={edges[i].max()}, n_nodes={n_nodes}")
                else:
                    print(f"    ⚠️ NO EDGES IN GT!")
        
        # ============= DIAGNOSTIC BLOCK END =============

        if self.seg:
            h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = self.network(segs.type(torch.FloatTensor).to(engine.state.device), z_pos, alpha, domain_labels=domains)
        else:
            h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = self.network(images.type(torch.FloatTensor).to(engine.state.device), z_pos, alpha, domain_labels=domains)

        # ============= DIAGNOSTIC BLOCK START =============
        # 2. Check raw model outputs
        if debug_check:
            print(f"\n[2. RAW MODEL OUTPUT]")
            print(f"  h shape: {h.shape}")  # (B, num_queries, hidden_dim)
            print(f"  pred_logits shape: {out['pred_logits'].shape}")  # (B, num_queries, num_classes)
            print(f"  pred_nodes shape: {out['pred_nodes'].shape}")  # (B, num_queries, 3 or 6)
            
            # 3. Check classification scores
            print(f"\n[3. CLASSIFICATION]")
            pred_logits = out['pred_logits']  # (B, num_queries, num_classes)
            pred_classes = torch.argmax(pred_logits, -1)  # (B, num_queries)
            probs = torch.softmax(pred_logits, dim=-1)    # (B, num_queries, num_classes)

            # Here we assume:
            #   class 0 = background
            #   class 1 = object
            obj_class = 1
            bg_class = 0

            for i in range(pred_classes.shape[0]):
                num_obj = (pred_classes[i] == obj_class).sum().item()
                num_bg  = (pred_classes[i] == bg_class).sum().item()
                num_other = pred_classes.shape[1] - num_obj - num_bg

                obj_probs = probs[i, pred_classes[i] == obj_class, obj_class] if num_obj > 0 else torch.tensor([])
                bg_probs  = probs[i, pred_classes[i] == bg_class, bg_class]   if num_bg  > 0 else torch.tensor([])

                print(f"  Sample {i}:")
                print(f"    Predicted classes: OBJ={num_obj}, BG={num_bg}, OTHER={num_other}")

                if num_obj > 0:
                    print(f"    OBJ probs: min={obj_probs.min():.3f}, max={obj_probs.max():.3f}, mean={obj_probs.mean():.3f}")
                else:
                    print(f"    ⚠️ NO OBJECT CLASS (1) PREDICTED!")
                if num_bg > 0:
                    print(f"    BG probs:  min={bg_probs.min():.3f}, max={bg_probs.max():.3f}, mean={bg_probs.mean():.3f}")
            
        # ============= DIAGNOSTIC BLOCK END =============

        infer = relation_infer(
            h.detach(),
            out,
            self.network, 
            self.config.MODEL.DECODER.OBJ_TOKEN, 
            self.config.MODEL.DECODER.RLN_TOKEN, 
            apply_nms=False
        )

        pred_nodes_list = infer["pred_nodes"] 
        pred_edges_list = infer["pred_rels"]

        # ============= DIAGNOSTIC BLOCK START =============
        # 4. Check inference results
        if random.random() < 0.01:
            print(f"\n[4. INFERENCE RESULTS]")
            N = 1  # number of samples you want to inspect
            for i in range(min(N, len(pred_nodes_list))):
                pred_n = pred_nodes_list[i]
                pred_e = pred_edges_list[i]
                
                n_pred_nodes = pred_n.shape[0] if isinstance(pred_n, np.ndarray) else 0
                n_pred_edges = pred_e.shape[0] if isinstance(pred_e, np.ndarray) else 0
                
                print(f"  Sample {i}:")
                print(f"    Predicted: {n_pred_nodes} nodes, {n_pred_edges} edges")
                
                if n_pred_nodes == 0:
                    print(f"    ⚠️ NO NODES PREDICTED!")
                else:
                    print(f"    Node coords range: [{pred_n.min():.3f}, {pred_n.max():.3f}]")
                
                if n_pred_edges == 0:
                    print(f"    ⚠️ NO EDGES PREDICTED!")
                
                # Compare with GT
                gt_n = nodes[i].shape[0]
                gt_e = edges[i].shape[0]
                print(f"    GT had: {gt_n} nodes, {gt_e} edges")
                print(f"    Recall: nodes={n_pred_nodes/max(gt_n,1):.2f}, edges={n_pred_edges/max(gt_e,1):.2f}")
            
            # print(f"{'='*70}\n")
        # ============= DIAGNOSTIC BLOCK END =============

        # Move to CPU & numpy
        seg_np = segs[0].detach().cpu().numpy()

        # print("seg min/max:", seg_np.min(), seg_np.max())

        non_empty = seg_np.max() > 0
        # print("seg contains foreground?", non_empty)

        if hasattr(self, "viz3d"):
            self.viz3d.maybe_save(
                segs=segs,
                images=images,
                gt_nodes_list=nodes,
                gt_edges_list=edges,
                pred_nodes_list=pred_nodes_list,
                pred_edges_list=pred_edges_list,
                epoch=epoch,
                step=iteration,
                batch_index=0,
                tag="train",
            )
    
        # if hasattr(self, "open3d_tb") and iteration % 1000 == 0:
        #     self.open3d_tb.maybe_log(
        #         segs=segs,
        #         gt_nodes_list=nodes,
        #         pred_nodes_list=pred_nodes_list,
        #         epoch=epoch,
        #         step=iteration,
        #         batch_index=0,
        #         tag="train",
        #     )
        
        target["interpolated_domains"] = interpolated_domains
        valid_token = torch.argmax(out['pred_logits'], -1)
        # valid_token = torch.sigmoid(nodes_prob[...,3])>0.5
        # print('valid_token number', valid_token.sum(1))

        # out1 = relation_matcher(h, out, self.network, self.config.MODEL.DECODER.OBJ_TOKEN, self.config.MODEL.DECODER.RLN_TOKEN)
        losses = self.loss_function(h, out, target, pred_backbone_domains, pred_instance_domains)
        # Clip the gradient
        # clip_grad_norm_(
        #     self.network.parameters(),
        #     max_norm=GRADIENT_CLIP_L2_NORM,
        #     norm_type=2,
        # )

        # ============= DIAGNOSTIC BLOCK START =============
        # 5. Check losses
        # print(f"[5. LOSSES]")
        # print(f"  Classification: {losses.get('class', 0):.4f}")
        # print(f"  Node loss: {losses.get('nodes', 0):.4f}")
        # print(f"  Edge loss: {losses.get('edges', 0):.4f}")
        # print(f"  Box loss: {losses.get('boxes', 0):.4f}")
        # print(f"  Total loss: {losses['total']:.4f}")
        # print(f"{'='*70}\n")
        # ============= DIAGNOSTIC BLOCK END =============

        losses['total'].backward()        
        self.optimizer.step()

        del images
        del segs
        del nodes
        del edges
        del target
        
        gc.collect()
        torch.cuda.empty_cache()

        return {"src": srcs, "loss": losses, "domains": domains}


def build_trainer(train_loader, net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False, seg=False):
    """[summary]

    Args:
        train_loader ([type]): [description]
        net ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        evaluator ([type]): [description]
        scheduler ([type]): [description]
        max_epochs ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=True,
            epoch_level=False,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config.TRAIN.VAL_INTERVAL,
            epoch_level=True,
            exec_at_start=True
        ),
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda x: x["loss"]["total"]
        ),
        CheckpointSaver(
            save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                  './models'),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_interval=5,
            n_saved=2,
            epoch_level=True,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: x["loss"]["nodes"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: x["loss"]["edges"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: x["loss"]["boxes"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: x["loss"]["cards"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: x["loss"]["total"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="domain_loss",
            output_transform=lambda x: x["loss"]["domain"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=1,
            # epoch_interval=1
        ),
    ]
    # train_post_transform = Compose(
    #     [AsDiscreted(keys=("pred", "label"),
    #     argmax=(True, False),
    #     to_onehot=True,
    #     n_classes=N_CLASS)]
    # )

    loss.to(device)

    if config.DATA.MIXED:
        # One metric is used for collecting the samples and performing tSNE so that other similarity metrics don't have to compute it by themselves
        base_metric = SimilarityMetricPCA(
            output_transform=lambda x: (x["src"], x["domains"]), 
            similarity_function=lambda X, Y: robust_cca_similarity(X,Y, threshold=0.98, compute_dirns=False, verbose=False, epsilon=1e-8)["mean"][0],
            base_metric=None
        )
        key_train_metric = {"train_cca_similarity": base_metric}
        additional_metrics = {
            "train_cka_similarity": SimilarityMetricPCA(
                output_transform=lambda x: (x["src"], x["domains"]), 
                similarity_function=batch_cka,
                base_metric=base_metric
            ),
        }
    else:
        key_train_metric=None
        additional_metrics=None


    trainer = RelationformerTrainer(
        config= config,
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        # post_transform=train_post_transform,
        # key_train_metric={
        #     "train_mean_dice": MeanDice(
        #         include_background=False,
        #         output_transform=lambda x: (x["pred"], x["label"]),
        #     )
        # },
        train_handlers=train_handlers,
        seg=seg,
        key_train_metric=key_train_metric,
        additional_metrics=additional_metrics,
        alpha_coeff=config.TRAIN.ALPHA_COEFF,
        # amp=fp16,
    )

    out_dir = os.path.join(
        '/data/scavone/cross-dim_i2g_3d/visual di prova',
        "runs",
        f"{config.log.exp_name}_{config.DATA.SEED}",
        "debug_vis"
    )
    # trainer.viz3d = DebugVisualizer3D(
    #     out_dir=out_dir,
    #     prob=config.display_prob,
    #     max_per_epoch=8,
    #     show_seg=True,
    # )
    
    trainer.viz3d = DebugVisualizer3D(
        out_dir=out_dir,
        prob=config.display_prob,
        max_per_epoch=8,
        show_seg=True,
    )
    
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda eng: trainer.viz3d.start_epoch())
    
    # trainer.open3d_tb = Open3DTensorboardLogger(
    #     writer=writer,
    #     prob=1.0,          # always try to log when allowed
    #     max_per_epoch=1,   # <= 1 image per epoch
    #     level=0.5,
    #     tag_prefix="3d_debug",
    # )
    # trainer.add_event_handler(Events.EPOCH_STARTED, lambda eng: trainer.open3d_tb.start_epoch())

    return trainer
