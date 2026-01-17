import os
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, \
    CheckpointSaver
import torch
import gc
import sys
from ignite.engine import Events
import numpy as np
from tqdm import tqdm

from training.inference import relation_infer
from utils.vis_debug import DebugVisualizer
from metrics.similarity import SimilarityMetricPCA, SimilarityMetricTSNE, batch_cka, batch_cosine, batch_euclidean, downsample_examples, upsample_examples
from metrics.svcca import get_cca_similarity, robust_cca_similarity


# define customized trainer
class RelationformerTrainer(SupervisedTrainer):

    def __init__(self, *args, config=None, alpha_coeff=1.0, ema_relation=None, **kwargs):
        super().__init__(*args, **kwargs)      

        self.config = config
        self.ema_relation = ema_relation
        self.alpha_coeff = float(alpha_coeff)

        # -------------------------
        # Alpha (gradient-based) state
        # -------------------------
        self.alpha_current = 0.0
        self.alpha_eps = 1e-6
        self.alpha_max = self.alpha_coeff  # cap alpha using existing coeff

        # -------------------------
        # Retrieve alpha-update knobs from existing EMA config
        # -------------------------
        self.use_alpha = bool(self.config.TRAIN.USE_ALPHA)
        self.alpha_ema_beta = float(self.config.TRAIN.EMA_DECAY)
        self.alpha_update_every = int(self.config.TRAIN.EMA_UPDATE_EVERY)
        self.alpha_warmup_iters = int(self.config.TRAIN.EMA_WARMUP_ITERS)


    def _iteration(self, engine, batchdata):
        alpha_new_logged = float("nan")
        g_task_logged = float("nan")
        g_da_base_logged = float("nan")

        images, seg, nodes, edges, domains = batchdata[0], batchdata[1], batchdata[2], batchdata[3], batchdata[4]
        # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # inputs = torch.cat(inputs, 1)

        images = images.to(engine.state.device,  non_blocking=False)
        seg = seg.to(engine.state.device,  non_blocking=False)
        nodes = [node.to(engine.state.device,  non_blocking=False) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        domains = domains.to(engine.state.device, non_blocking=False)
        target = {'nodes': nodes, 'edges': edges, 'domains': domains, 'seg': seg}

        self.network[0].train()
        self.network[1].eval()
        self.optimizer.zero_grad()

        # engine.state.epoch and engine.state.iteration exist in Ignite/Monai
        epoch = engine.state.epoch
        iteration = engine.state.iteration
        epoch_len = engine.state.epoch_length  # Monai sets this
        max_epochs = engine.state.max_epochs

        if hasattr(self.loss_function, "set_hnm_progress"):
            self.loss_function.set_hnm_progress(epoch, iteration, epoch_len, max_epochs)
        
        if engine.state.iteration % engine.state.epoch_length == 1:
            print("epoch", engine.state.epoch,
                "iteration", engine.state.iteration,
                "epoch_length", engine.state.epoch_length)
            
        
        alpha_used = float(self.alpha_current)

        h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains, conc_features_flat, domain_hs = self.network[0](images, seg=False, alpha=alpha_used, domain_labels=domains)
        target["interpolated_domains"] = interpolated_domains
        
        # ---- predictions for visualization ----
        pred_nodes, pred_edges = relation_infer(
            h.detach(), out, self.network[0],
            80, 5,
            nms=False, map_=False
        )
        

        # only call if visualizer exists
        if hasattr(self, "viz"):
            self.viz.maybe_save(
                images, nodes, edges, pred_nodes, pred_edges,
                epoch, iteration, batch_index=0, tag="train",
                # gt_seg=seg,           # Ground truth from batchdata
                # pred_seg=out.get("pred_seg") # Model output logits
            )  

        losses = self.loss_function(h, out, target, pred_backbone_domains, pred_instance_domains)

        # -------------------------------------------------
        # ALPHA COMPUTATION
        # -------------------------------------------------
        # Domain loss
        L_DA = losses.get("domain", None)

        # Task loss: sum everything except "total" and "domain"
        L_task = 0.0
        for k, v in losses.items():
            if k in ("total", "domain"):
                continue
            L_task = L_task + v
        
        # -------------------------------------------------
        # SEGMENTATION INFLUENCE (GRADIENT-BASED, SOURCE vs TARGET)
        # -------------------------------------------------
        pred_seg = out.get("pred_seg", None)

        if pred_seg is not None:
            grad = torch.autograd.grad(
                losses["total"],
                pred_seg,
                retain_graph=True,
                allow_unused=True
            )[0]

            if grad is not None:
                # domains: [B]
                keep = (domains == 0)

                if keep.any():
                    seg_influence_source = grad[keep].norm().detach()
                else:
                    seg_influence_source = torch.tensor(0.0, device=grad.device)

                if (~keep).any():
                    seg_influence_target = grad[~keep].norm().detach()
                else:
                    seg_influence_target = torch.tensor(0.0, device=grad.device)
            else:
                seg_influence_source = torch.tensor(0.0, device=engine.state.device)
                seg_influence_target = torch.tensor(0.0, device=engine.state.device)
        else:
            seg_influence_source = torch.tensor(0.0, device=engine.state.device)
            seg_influence_target = torch.tensor(0.0, device=engine.state.device)

        if (
            self.config.DATA.MIXED
            and self.use_alpha
            and (L_DA is not None)
            and (engine.state.iteration >= self.alpha_warmup_iters)
            and (engine.state.iteration % self.alpha_update_every == 0)
        ):
            grad_targets = []
            if conc_features_flat is not None:
                grad_targets.append(conc_features_flat)
            if domain_hs is not None:
                grad_targets.append(domain_hs)

            if len(grad_targets) > 0:
                g_task_list = torch.autograd.grad(
                    L_task, grad_targets, retain_graph=True, allow_unused=True
                )
                g_da_list = torch.autograd.grad(
                    L_DA, grad_targets, retain_graph=True, allow_unused=True
                )

                def summed_norm(grad_list, device):
                    total = torch.tensor(0.0, device=device)
                    for g in grad_list:
                        if g is None:
                            continue
                        total = total + g.detach().norm(p=2)
                    return total

                device = grad_targets[0].device
                g_task = summed_norm(g_task_list, device)
                g_da_measured = summed_norm(g_da_list, device)

                # IMPORTANT: DA grads are already scaled by GRL alpha_used.
                # Estimate "base" DA norm by dividing out alpha_used.
                g_da_base = g_da_measured / (abs(alpha_used) + self.alpha_eps)

                alpha_new = (g_task / (g_da_base + self.alpha_eps)).item()

                alpha_new_logged = float(alpha_new)
                g_task_logged = float(g_task.item())
                g_da_base_logged = float(g_da_base.item())

                # clamp
                alpha_new = max(0.0, min(alpha_new, self.alpha_max))

                # EMA smoothing
                self.alpha_current = (
                    self.alpha_ema_beta * self.alpha_current +
                    (1.0 - self.alpha_ema_beta) * alpha_new
                )


        losses['total'].backward()

        # if 0.1 > 0:
        #     _ = torch.nn.utils.clip_grad_norm_(self.network[0].parameters(), 0.1)
        # else:
        #     _ = get_total_grad_norm(self.network[0].parameters(), 0.1)
    
        self.optimizer.step()

        if self.ema_relation is not None:
            student_net = self.network[0]
            if hasattr(student_net, "module"):
                student_net = student_net.module
            self.ema_relation.update(student_net.relation_embed)


        del images
        del seg
        del nodes
        del edges
        del target
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "src": srcs[-1],
            "loss": losses,
            "domains": domains,
            "seg_influence_source": seg_influence_source,
            "seg_influence_target": seg_influence_target,

            # NEW: alpha logs
            "alpha_current": float(self.alpha_current),
            "alpha_used": float(alpha_used),
            "alpha_new": float(alpha_new_logged),
            "grad_task": float(g_task_logged),
            "grad_da_base": float(g_da_base_logged),
        }



def build_trainer(train_loader, net, seg_net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False, ema_relation=None):
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

    save_dict = {
        "net": net,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    if ema_relation is not None:
        save_dict["ema_relation"] = ema_relation.ema

    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=True,
            epoch_level=True,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config.TRAIN.VAL_INTERVAL,
            # interval=1,
            epoch_level=True
        ),
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda x: x["loss"]["total"]
        ),
        CheckpointSaver(
            save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                  './models'),
            save_dict=save_dict,
            save_interval=5,   # 5
            n_saved=2   # 5
        ),        
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,            # log every epoch; True also works
            # iteration_log=False   # optional: disable per-iteration logging
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: {"node_loss": float(x["loss"]["nodes"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,             # log every epoch
            # iteration_log=False,   # uncomment to disable per-iter logging
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: {"edge_loss": float(x["loss"]["edges"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
            # iteration_log=False,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: {"box_loss": float(x["loss"]["boxes"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
            # iteration_log=False,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: {"card_loss": float(x["loss"]["cards"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
            # iteration_log=False,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="domain_loss",
            output_transform=lambda x: {"domain_loss": float(x["loss"]["domain"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
            # iteration_log=False,
        ),
        # TensorBoardStatsHandler(
        #     writer,
        #     tag_name="seg_loss",
        #     output_transform=lambda x: {"seg_loss": float(x["loss"]["seg"])},
        #     global_epoch_transform=lambda _: scheduler.last_epoch,
        #     epoch_log=1,
        #     # iteration_log=False,
        # ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: {"total_loss": float(x["loss"]["total"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
            # iteration_log=False,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="alpha",
            output_transform=lambda x: {
                "alpha_current": float(x["alpha_current"]),
                "alpha_used": float(x["alpha_used"]),
                "alpha_new": float(x["alpha_new"]),
            },
            epoch_log=1,
            # iteration_log=1,  # log each iteration (you can set to 10 if you want less spam)
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
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=[net, seg_net],
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
        key_train_metric=key_train_metric,
        additional_metrics=additional_metrics,
        amp=fp16,
        ema_relation=ema_relation,
        alpha_coeff=config.TRAIN.ALPHA_COEFF,
        config=config
    )

    out_dir = os.path.join(
        'C:/Users/Utente/Desktop/tesi/cross-dim_i2g_2d/visual_di_prova',
        "runs",
        f"{config.log.exp_name}_{config.DATA.SEED}",
        "debug_vis"
    )
    
    hpo_enabled = bool(getattr(config, "HPO", getattr(config, "hpo", False)))

    if hpo_enabled:
        trainer.viz = DebugVisualizer(out_dir=out_dir, prob=0.0, max_per_epoch=8)
    else:
        trainer.viz = DebugVisualizer(out_dir=out_dir, prob=config.display_prob, max_per_epoch=8)
    
    trainer.add_event_handler(Events.EPOCH_STARTED, lambda eng: trainer.viz.start_epoch())

    return trainer