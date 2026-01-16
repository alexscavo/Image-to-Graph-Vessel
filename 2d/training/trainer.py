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

    def _iteration(self, engine, batchdata):
        images, seg, nodes, edges, domains = batchdata[0], batchdata[1], batchdata[2], batchdata[3], batchdata[4]
        # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # inputs = torch.cat(inputs, 1)

        images = images.to(engine.state.device,  non_blocking=False)
        seg = seg.to(engine.state.device,  non_blocking=False)
        nodes = [node.to(engine.state.device,  non_blocking=False) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        domains = domains.to(engine.state.device, non_blocking=False)
        target = {'nodes': nodes, 'edges': edges, 'domains': domains}

        self.network[0].train()
        self.network[1].eval()
        self.optimizer.zero_grad()

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        p = float(iteration + epoch * engine.state.epoch_length) / engine.state.max_epochs / engine.state.epoch_length
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * self.alpha_coeff
        h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = self.network[0](images, seg=False, alpha=alpha, domain_labels=domains)
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
                images,           # <- positional, corresponds to image_bchw
                nodes, edges,     # GT lists
                pred_nodes, pred_edges,
                epoch, iteration,
                batch_index=0,
                tag="train",
            )        

        losses = self.loss_function(h, out, target, pred_backbone_domains, pred_instance_domains)
        
        # loss_feat = 0
        # for i in range(len(seg_srcs)):
        #     loss_feat += F.l1_loss(srcs[i], seg_srcs[i].detach())

        # losses.update({'feature':loss_feat})
        # losses['total'] = losses['total']+loss_feat
        # Clip the gradient
        # clip_grad_norm_(
        #     self.network.parameters(),
        #     max_norm=GRADIENT_CLIP_L2_NORM,
        #     norm_type=2,
        # )
        losses['total'].backward()

        # if 0.1 > 0:
        #     _ = torch.nn.utils.clip_grad_norm_(self.network[0].parameters(), 0.1)
        # else:
        #     _ = get_total_grad_norm(self.network[0].parameters(), 0.1)
    
        self.optimizer.step()

        del images
        del seg
        del nodes
        del edges
        del target
        gc.collect()
        torch.cuda.empty_cache()

        return {"src": srcs[-1], "loss": losses, "domains": domains}


def build_trainer(train_loader, net, seg_net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False):
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
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
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
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: {"total_loss": float(x["loss"]["total"])},
            global_epoch_transform=lambda _: scheduler.last_epoch,
            epoch_log=1,
            # iteration_log=False,
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
    )

    out_dir = os.path.join(
        config.vis_path,
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
    
    trainer.alpha_coeff = config.TRAIN.ALPHA_COEFF

    return trainer

