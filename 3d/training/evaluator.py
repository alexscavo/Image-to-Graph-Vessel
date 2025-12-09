import os
import gc
import torch
import numpy as np
from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler, TensorBoardImageHandler
from utils.modified_library_functions import TensorBoardImageHandlerWithTag

from metrics.loss_metric import MeanLoss
from metrics.smd import MeanSMD
from metrics.boxap import MeanBoxAP, MeanSingleAP
from training.inference import relation_infer
from utils.utils import save_input, save_output

from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Union
from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import

from visualize_sample import create_sample_visual_3d, create_sample_visual_2d, create_gradcam_overlay_2d
from metrics.similarity import SimilarityMetricPCA, SimilarityMetricTSNE, batch_cka, batch_cosine, batch_euclidean, create_feature_representation_visual
from metrics.svcca import robust_cca_similarity


if TYPE_CHECKING:
    from ignite.engine import EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

# Define customized evaluator
class RelationformerEvaluator(SupervisedEvaluator):
    def __init__(
        self,
        device: torch.device,
        loss_function,
        val_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        seg: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network = network,
            inferer = SimpleInferer() if inferer is None else inferer,
        )

        self.config = kwargs.pop('config')
        self.loss_function = loss_function
        self.seg = seg
        
        self.encoder_feats = None     # will store [B, C, D, H, W]

        # ---- Grad-CAM hooks on encoder.layer4 ----
        self._enc_acts = None   # [B, C, D, H, W]
        self._enc_grads = None  # [B, C, D, H, W]

        def _fwd_hook(module, inp, out):
            # save activations
            self._enc_acts = out

        def _bwd_hook(module, grad_input, grad_output):
            # grad_output[0] is dY/d(out)
            self._enc_grads = grad_output[0]

        # attach to last conv block of SE-ResNet
        self.network.encoder.layer4.register_forward_hook(_fwd_hook)
        self.network.encoder.layer4.register_full_backward_hook(_bwd_hook)
        
    def _iteration(self, engine, batchdata):
        images, segs, nodes, edges, z_pos, domains = batchdata[0], batchdata[1], batchdata[2], batchdata[3], batchdata[4], batchdata[5]
        
        # # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # # inputs = torch.cat(inputs, 1)
        images = images.to(engine.state.device,  non_blocking=False)
        segs = segs.to(engine.state.device,  non_blocking=False)
        nodes = [node.to(engine.state.device,  non_blocking=False) for node in nodes]
        boxes = [torch.cat([node, 0.2*torch.ones(node.shape, device=node.device)], dim=-1) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        domains = domains.to(engine.state.device, non_blocking=False)
        self.network.eval()

        if self.seg:
            h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = self.network(segs.type(torch.FloatTensor).to(engine.state.device), z_pos, domain_labels=domains)
        else:
            h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = self.network(images.type(torch.FloatTensor).to(engine.state.device), z_pos, domain_labels=domains)

        # ---- Grad-CAM for sample 0, token with highest object logit ----
        gradcam_vol = None
        try:
            logits = out["pred_logits"]          # [B, num_queries, 2]
            obj_logits = logits[0, :, 1]         # object-class logit for sample 0
            top_idx = torch.argmax(obj_logits)
            y = obj_logits[top_idx]              # scalar target

            self.network.zero_grad()
            y.backward(retain_graph=True)

            # encoder feature maps & gradients
            A = self._enc_acts[0]    # [C, D', H', W']
            G = self._enc_grads[0]   # [C, D', H', W']

            # channel weights: global-average-pool gradients
            weights = G.mean(dim=(1, 2, 3))      # [C]

            # weighted sum of feature maps
            cam = (weights.view(-1, 1, 1, 1) * A).sum(dim=0)   # [D', H', W']
            cam = torch.relu(cam)

            # upsample to full input size [D, H, W]
            cam = cam.unsqueeze(0).unsqueeze(0)  # [1,1,D',H',W']
            cam = torch.nn.functional.interpolate(
                cam,
                size=images.shape[2:],           # (D, H, W)
                mode="trilinear",
                align_corners=False,
            )[0, 0]                              # [D, H, W]

            # normalize 0–1
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            gradcam_vol = cam.detach().cpu()     # [D, H, W]
        except Exception as e:
            # if anything goes wrong, just skip Grad-CAM for this iteration
            gradcam_vol = None

        with torch.no_grad():
            losses = self.loss_function(
                h.clone().detach(),
                {'pred_logits': out['pred_logits'].clone().detach(), 'pred_nodes': out["pred_nodes"].clone().detach()},
                {'nodes': [node.clone().detach() for node in nodes], 'edges': [edge.clone().detach() for edge in edges], 'domains': domains, "interpolated_domains": interpolated_domains},
                pred_backbone_domains,
                pred_instance_domains
            )

        out = relation_infer(h.detach(), out, self.network, self.config.MODEL.DECODER.OBJ_TOKEN, self.config.MODEL.DECODER.RLN_TOKEN, apply_nms=False)
        
        enc = self.encoder_feats
        if enc is not None:
            enc = enc.detach().cpu()   # [B, C, D, H, W]
        
        if self.config.TRAIN.SAVE_VAL:
            root_path = os.path.join(self.config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (self.config.log.exp_name, self.config.DATA.SEED), 'val_samples')
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            for i, (node, edge, pred_node, pred_edge) in enumerate(zip(nodes, edges, out['pred_nodes'], out['pred_rels'])):
                path = os.path.join(root_path, "ref_epoch_"+str(engine.state.epoch).zfill(3)+"_iteration_"+str(engine.state.iteration).zfill(5))
                save_input(path, i, images[i,0,...].cpu().numpy(), node.cpu().numpy(), edge.cpu().numpy())
                path = os.path.join(root_path, "pred_epoch_"+str(engine.state.epoch).zfill(3)+"_iteration_"+str(engine.state.iteration).zfill(5))
                save_output(path, i, pred_node, pred_edge)

        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            **{
                "images": images,
                "boxes": [box.to('cpu') for box in boxes],
                "boxes_class": [np.zeros(n.shape[0]) for n in boxes],
                "domains": domains,
                "nodes": nodes,
                "edges": edges,
                "loss": losses,
                "segs": segs,
                "z_pos": z_pos,
                "src": srcs,
                "gradcam": gradcam_vol,
            },
            **out}


def build_evaluator(val_loader, net, optimizer, scheduler, writer, config, device, loss_function, pre_2d=False, pretrain_general=False, gaussian_augment=False, seg=False):
    """[summary]

    Args:
        val_loader ([type]): [description]
        net ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                  './models'),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_key_metric=True,
            key_metric_n_saved=1,
            save_interval=0,
            key_metric_negative_sign=False,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="val_total_loss",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardImageHandlerWithTag(
            writer,
            epoch_level=True,
            interval=1,
            # max_channels=3,>
            output_transform=lambda x: create_sample_visual_2d(x) if pre_2d or pretrain_general else create_sample_visual_3d(x)
        )
    ]

    # scommentare
    '''mixed_ap_metric = MeanBoxAP(
        on_edges=False,
        output_transform=lambda x: (
            x["pred_boxes"],
            x["pred_boxes_class"],
            x["pred_boxes_score"],
            x["boxes"],
            x["boxes_class"],
            x["pred_nodes"],
            x['pred_rels'],
            x["pred_rels_class"],
            x["pred_rels_score"],
            x["nodes"],
            x["edges"]
        ),
        max_detections=40
    )'''

    additional_metrics={
        '''"val_node_ap": MeanSingleAP(mixed_ap_metric, lambda x: None, False),
        "val_edge_ap": MeanSingleAP(mixed_ap_metric, lambda x: None, True),'''  # scommentare
        "val_smd": MeanSMD(
            output_transform=lambda x: (x["boxes"], x["edges"], x["pred_boxes"], x["pred_rels"]),
        ),
        "val_total_loss": MeanLoss(
            output_transform=lambda x: x["loss"]["total"],
        ),
        "val_node_loss": MeanLoss(
            output_transform = lambda x: x["loss"]["nodes"],
        ),
        "val_class_loss": MeanLoss(
            output_transform=lambda x: x["loss"]["class"],
        ),
        "val_edge_loss": MeanLoss(
            output_transform=lambda x: x["loss"]["edges"],
        ),
        "val_card_loss": MeanLoss(
            output_transform=lambda x: x["loss"]["cards"],
        ),
        "val_box_loss": MeanLoss(
            output_transform=lambda x: x["loss"]["boxes"],
        ),
    }

    if config.DATA.MIXED:
        # One metric is used for collecting the samples and performing tSNE so that other similarity metrics don't have to compute it by themselves
        base_similarity = SimilarityMetricPCA(
            output_transform=lambda x: (x["src"], x["domains"]), 
            similarity_function=lambda X, Y: robust_cca_similarity(X,Y, threshold=0.98, compute_dirns=False, verbose=False, epsilon=1e-8)["mean"][0],
            base_metric=None
        )
        val_handlers.append(
            TensorBoardImageHandler(
                writer,
                epoch_level=True,
                interval=1,
                max_channels=3,
                output_transform=lambda x: create_feature_representation_visual(base_similarity),
            )
        )
        # Add base similarity metric to additional metrics
        additional_metrics["val_cca_similarity"] = base_similarity
        additional_metrics["val_cka_similarity"] = SimilarityMetricPCA(
            output_transform=lambda x: (x["src"], x["domains"]), 
                similarity_function=batch_cka,
                base_metric=base_similarity
        )
    else:
        # val_handlers.append(
        #     TensorBoardImageHandlerWithTag(
        #         writer,
        #         epoch_level=True,
        #         interval=1,
        #         output_transform=create_gradcam_overlay_2d,
        #         # index should be 0 (default), since we return [tensor] and want the first element
        #         output_tag='output_gradcam'
        #     )
        # )
        val_handlers.append(
            TensorBoardImageHandlerWithTag(
                writer,
                epoch_level=True,
                interval=1,
                max_channels=3,
                output_transform=lambda x: create_sample_visual_3d(x),
                output_tag='output_predictions'
            )
        )


    evaluator = RelationformerEvaluator(
        config= config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        # post_transform=val_post_transform,
        key_val_metric={
            # "val_smd": MeanSMD(
            #     output_transform=lambda x: (x["boxes"], x["edges"], x["pred_boxes"], x["pred_rels"]),
            # )pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes
            # "val_mixed_AP": mixed_ap_metric,  # usare questa?
            "val_total_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["total"],
            ),
        },
        additional_metrics=additional_metrics,
        val_handlers=val_handlers,
        amp=False,
        loss_function=loss_function,
        seg=seg
    )

    return evaluator
