import os
import gc
import numpy as np
import torch

from ignite.engine import Events
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import (
    LrScheduleHandler,
    ValidationHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointSaver,
)

from training.inference import relation_infer
from utils.vis_debug import DebugVisualizer
from metrics.similarity import SimilarityMetricPCA, batch_cka
from metrics.svcca import robust_cca_similarity


class RelationformerTrainer(SupervisedTrainer):
    """
    Merged trainer:
      - supports seg output + seg visualization + seg_loss logging
      - supports EMA update (ema_relation) + saving EMA weights in checkpoint
      - supports HNM scheduling hook (loss_function.set_hnm_progress if present)
      - keeps seg influence gradient stats (source vs target) + TB logging
    """

    def __init__(self, *args, config=None, ema_relation=None, alpha_coeff=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.ema_relation = ema_relation
        self.alpha_coeff = alpha_coeff

    def _iteration(self, engine, batchdata):
        images, seg, nodes, edges, domains = (
            batchdata[0],
            batchdata[1],
            batchdata[2],
            batchdata[3],
            batchdata[4],
        )

        images = images.to(engine.state.device, non_blocking=False)
        seg = seg.to(engine.state.device, non_blocking=False)
        nodes = [n.to(engine.state.device, non_blocking=False) for n in nodes]
        edges = [e.to(engine.state.device, non_blocking=False) for e in edges]
        domains = domains.to(engine.state.device, non_blocking=False)

        target = {"nodes": nodes, "edges": edges, "domains": domains, "seg": seg}

        # student in train, seg_net in eval
        self.network[0].train()
        self.network[1].eval()
        self.optimizer.zero_grad()

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        epoch_len = engine.state.epoch_length
        max_epochs = engine.state.max_epochs

        # HNM scheduling hook (optional)
        if hasattr(self.loss_function, "set_hnm_progress"):
            self.loss_function.set_hnm_progress(epoch, iteration, epoch_len, max_epochs)

        # alpha schedule (same idea as original, but stable with total iters)
        total_iterations = max_epochs * epoch_len
        p = float(iteration) / float(total_iterations) if total_iterations > 0 else 0.0
        alpha = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0) * self.alpha_coeff

        # forward
        h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = self.network[0](
            images, seg=False, alpha=alpha, domain_labels=domains
        )
        target["interpolated_domains"] = interpolated_domains

        # visualization predictions
        pred_nodes, pred_edges = relation_infer(
            h.detach(), out, self.network[0], 80, 5, nms=False, map_=False
        )

        if hasattr(self, "viz"):
            self.viz.maybe_save(
                images,
                nodes,
                edges,
                pred_nodes,
                pred_edges,
                epoch,
                iteration,
                batch_index=0,
                tag="train",
                gt_seg=seg,
                pred_seg=out.get("pred_seg"),
            )

        # loss
        losses = self.loss_function(h, out, target, pred_backbone_domains, pred_instance_domains)

        # -----------------------------
        # seg influence (grad-based)
        # -----------------------------
        pred_seg = out.get("pred_seg", None)
        if pred_seg is not None:
            grad = torch.autograd.grad(
                losses["total"],
                pred_seg,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if grad is not None:
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

        # backward/step
        losses["total"].backward()
        self.optimizer.step()

        # EMA update (optional)
        if self.ema_relation is not None:
            student_net = self.network[0]
            if hasattr(student_net, "module"):
                student_net = student_net.module
            # expects student_net.relation_embed to exist (same as your EMA trainer)
            self.ema_relation.update(student_net.relation_embed)

        # cleanup
        del images, seg, nodes, edges, target
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "src": srcs[-1],
            "loss": losses,
            "domains": domains,
            "seg_influence_source": seg_influence_source,
            "seg_influence_target": seg_influence_target,
        }


def build_trainer(
    train_loader,
    net,
    seg_net,
    loss,
    optimizer,
    scheduler,
    writer,
    evaluator,
    config,
    device,
    fp16=False,
    ema_relation=None,
):
    # checkpoint contents
    save_dict = {"net": net, "optimizer": optimizer, "scheduler": scheduler}
    if ema_relation is not None:
        # matches your existing saving convention
        save_dict["ema_relation"] = ema_relation.ema

    train_handlers = [
        LrScheduleHandler(lr_scheduler=scheduler, print_lr=True, epoch_level=True),
        ValidationHandler(validator=evaluator, interval=config.TRAIN.VAL_INTERVAL, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]["total"]),
        CheckpointSaver(
            save_dir=os.path.join(
                config.TRAIN.SAVE_PATH,
                "runs",
                "%s_%d" % (config.log.exp_name, config.DATA.SEED),
                "./models",
            ),
            save_dict=save_dict,
            save_interval=5,
            n_saved=2,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: {"node_loss": float(x["loss"]["nodes"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: {"edge_loss": float(x["loss"]["edges"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: {"box_loss": float(x["loss"]["boxes"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: {"card_loss": float(x["loss"]["cards"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="domain_loss",
            output_transform=lambda x: {"domain_loss": float(x["loss"]["domain"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        # re-enabled seg loss logging (present in trainer.py)
        TensorBoardStatsHandler(
            writer,
            tag_name="seg_loss",
            output_transform=lambda x: {"seg_loss": float(x["loss"]["seg"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: {"total_loss": float(x["loss"]["total"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        # seg influence logs (present in EMA version)
        TensorBoardStatsHandler(
            writer,
            tag_name="seg_influence/source",
            output_transform=lambda x: float(x["seg_influence_source"]),
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="seg_influence/target",
            output_transform=lambda x: float(x["seg_influence_target"]),
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
        ),
    ]

    loss.to(device)

    if config.DATA.MIXED:
        base_metric = SimilarityMetricPCA(
            output_transform=lambda x: (x["src"], x["domains"]),
            similarity_function=lambda X, Y: robust_cca_similarity(
                X, Y, threshold=0.98, compute_dirns=False, verbose=False, epsilon=1e-8
            )["mean"][0],
            base_metric=None,
        )
        key_train_metric = {"train_cca_similarity": base_metric}
        additional_metrics = {
            "train_cka_similarity": SimilarityMetricPCA(
                output_transform=lambda x: (x["src"], x["domains"]),
                similarity_function=batch_cka,
                base_metric=base_metric,
            ),
        }
    else:
        key_train_metric = None
        additional_metrics = None

    trainer = RelationformerTrainer(
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=[net, seg_net],
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
        key_train_metric=key_train_metric,
        additional_metrics=additional_metrics,
        amp=fp16,
        config=config,
        ema_relation=ema_relation,
        alpha_coeff=config.TRAIN.ALPHA_COEFF,
    )

    out_dir = os.path.join(
        "C:/Users/Utente/Desktop/tesi/cross-dim_i2g_2d/visual_di_prova",
        "runs",
        f"{config.log.exp_name}_{config.DATA.SEED}",
        "debug_vis",
    )

    hpo_enabled = bool(getattr(config, "HPO", getattr(config, "hpo", False)))
    if hpo_enabled:
        trainer.viz = DebugVisualizer(out_dir=out_dir, prob=0.0, max_per_epoch=8)
    else:
        trainer.viz = DebugVisualizer(out_dir=out_dir, prob=config.display_prob, max_per_epoch=8)

    trainer.add_event_handler(Events.EPOCH_STARTED, lambda eng: trainer.viz.start_epoch())
    return trainer
