import enum
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from utils import box_ops_2D
from utils.utils import downsample_edges, upsample_edges


class EDGE_SAMPLING_MODE(enum.Enum):
    NONE = "none"
    UP = "up"
    DOWN = "down"
    RANDOM_UP = "random_up"


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum() / num_boxes


class SetCriterion(nn.Module):
    """
    Merged criterion:
      - graph losses (class/nodes/boxes/edges/cards/domain)
      - segmentation loss (binary or multiclass) with source-only option
      - EMA-driven Hard Negative Mining (HNM) for negative edge sampling
      - trainer can call set_hnm_progress(epoch, iter, epoch_len, max_epochs)
    """

    def __init__(
        self,
        config,
        matcher,
        net,
        num_edge_samples=80,
        edge_sampling_mode=EDGE_SAMPLING_MODE.UP,
        domain_class_weight=None,
        ema_relation_embed=None,  # <--- EMA wrapper (same object your ema trainer passes)
    ):
        super().__init__()
        self.matcher = matcher
        self.net = net

        self.rln_token = config.MODEL.DECODER.RLN_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.losses = config.TRAIN.LOSSES

        self.num_edge_samples = num_edge_samples
        self.edge_sampling_mode = edge_sampling_mode
        self.sample_ratio = config.TRAIN.EDGE_SAMPLE_RATIO
        self.sample_ratio_interval = config.TRAIN.EDGE_SAMPLE_RATIO_INTERVAL

        # ---- segmentation options (from segmentation loss file)
        self.exclude_target_seg = getattr(config.TRAIN, "EXCLUDE_TARGET_SEG", True)

        # ---- domain / mixed setup
        if config.DATA.MIXED:
            self.domain_img_loss = nn.NLLLoss(domain_class_weight) if config.TRAIN.IMAGE_ADVERSARIAL else None
            self.domain_inst_loss = nn.NLLLoss(domain_class_weight) if config.TRAIN.GRAPH_ADVERSARIAL else None
            self.consistency_loss = nn.MSELoss() if config.TRAIN.CONSISTENCY_REGULARIZATION else None
            self.compute_target_graph_loss = config.TRAIN.COMPUTE_TARGET_GRAPH_LOSS
        else:
            self.domain_img_loss = None
            self.domain_inst_loss = None
            self.consistency_loss = None
            self.compute_target_graph_loss = True

        # ---- weights (merge: include seg)
        self.weight_dict = {
            "boxes": config.TRAIN.W_BBOX,
            "class": config.TRAIN.W_CLASS,
            "cards": config.TRAIN.W_CARD,
            "nodes": config.TRAIN.W_NODE,
            "edges": config.TRAIN.W_EDGE,
            "domain": config.TRAIN.W_DOMAIN,
            "seg": getattr(config.TRAIN, "W_SEG", 0.0),
        }

        # ---- EMA + HNM scheduling (from EMA loss file)
        self.ema_relation_embed = ema_relation_embed
        self.hard_negative_mining = getattr(config.TRAIN, "HARD_NEGATIVE_MINING", False)
        self.hnm_hard_fraction_end = float(getattr(config.TRAIN, "HNM_HARD_FRACTION_END", 0.6))
        self.hnm_hard_fraction_start = float(getattr(config.TRAIN, "HNM_HARD_FRACTION_START", 0.2))
        self.hnm_ramp_epochs = int(getattr(config.TRAIN, "HNM_RAMP_EPOCHS", 10))
        self.hnm_warmup_epochs = int(getattr(config.TRAIN, "HNM_WARMUP_EPOCHS", 0))
        self.hnm_pool_mult = getattr(config.TRAIN, "HNM_POOL_MULT", 2)
        self.hnm_mode = getattr(config.TRAIN, "HNM_MODE", "top_k")  # top_k | top_p_uniform | weighted
        self.hnm_top_p = float(getattr(config.TRAIN, "HNM_TOP_P", 0.3))
        self.hnm_temp = float(getattr(config.TRAIN, "HNM_TEMP", 0.5))

        # progress state updated by trainer
        self._hnm_epoch = 1
        self._hnm_iter = 0
        self._hnm_epoch_len = 1
        self._hnm_max_epochs = 1

    # --------------------
    # HNM progress plumbing
    # --------------------
    def set_hnm_progress(self, epoch: int, iteration: int, epoch_len: int, max_epochs: int):
        self._hnm_epoch = int(epoch)
        self._hnm_iter = int(iteration)
        self._hnm_epoch_len = max(1, int(epoch_len))
        self._hnm_max_epochs = max(1, int(max_epochs))

    def _get_scheduled_hard_fraction(self) -> float:
        e = self._hnm_epoch
        if e <= self.hnm_warmup_epochs:
            return 0.0
        ramp_e = max(1, self.hnm_ramp_epochs)
        t = min(1.0, (e - self.hnm_warmup_epochs - 1) / ramp_e)
        return float(self.hnm_hard_fraction_start + t * (self.hnm_hard_fraction_end - self.hnm_hard_fraction_start))

    @torch.no_grad()
    def select_hard_negative_edges(
        self,
        neg_edges: torch.Tensor,
        take_neg: int,
        rearranged_object_token: torch.Tensor,
        relation_token_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # fallback / quick exits
        n_all = int(neg_edges.size(0))
        if take_neg <= 0 or n_all == 0:
            return neg_edges.new_zeros((0, 2))
        take_neg = min(int(take_neg), n_all)

        # if EMA missing or HNM disabled -> random sampling (same behavior as baseline)
        if self.ema_relation_embed is None or not self.hard_negative_mining:
            idx = torch.randperm(n_all, device=neg_edges.device)[:take_neg]
            sel = neg_edges[idx]
            shuffle = torch.rand((sel.size(0),), device=sel.device) > 0.5
            sel[shuffle] = sel[shuffle][:, [1, 0]]
            return sel

        device = neg_edges.device

        # candidate pool
        pool_mult = max(1.0, float(self.hnm_pool_mult))
        candidate_pool_size = min(n_all, int(round(pool_mult * take_neg)))
        cand_idx = torch.randperm(n_all, device=device)[:candidate_pool_size]
        cand_edges = neg_edges[cand_idx]

        # undirected shuffle
        shuffle = torch.rand((cand_edges.size(0),), device=device) > 0.5
        cand_edges[shuffle] = cand_edges[shuffle][:, [1, 0]]

        tok_i = rearranged_object_token[cand_edges[:, 0], :]
        tok_j = rearranged_object_token[cand_edges[:, 1], :]

        if self.rln_token > 0:
            assert relation_token_batch is not None, "relation_token_batch required when self.rln_token > 0"
            rel_flat = relation_token_batch.reshape(-1).unsqueeze(0).repeat(cand_edges.size(0), 1)
            relation_feature = torch.cat([tok_i, tok_j, rel_flat], dim=1)
        else:
            relation_feature = torch.cat([tok_i, tok_j], dim=1)

        # EMA scoring: higher score means "more likely positive" -> harder negative
        logits = self.ema_relation_embed.ema(relation_feature)
        scores = torch.softmax(logits, dim=1)[:, 1]

        hard_frac = self._get_scheduled_hard_fraction()
        hard_k = int(round(hard_frac * take_neg))
        hard_k = max(0, min(hard_k, take_neg))
        soft_k = take_neg - hard_k

        if self.hnm_mode == "weighted":
            T = max(1e-6, float(self.hnm_temp))
            w = torch.exp((scores - scores.max()) / T) + 1e-12
            sel_idx = torch.multinomial(w, num_samples=take_neg, replacement=False)
            selected = cand_edges[sel_idx]
        else:
            # hard selection
            if hard_k > 0:
                if self.hnm_mode == "top_k":
                    hard_idx = torch.topk(scores, k=min(hard_k, candidate_pool_size), largest=True).indices
                elif self.hnm_mode == "top_p_uniform":
                    top_p = max(0.0, min(1.0, float(self.hnm_top_p)))
                    hard_set_size = max(1, int(round(top_p * candidate_pool_size)))
                    hard_set_size = min(hard_set_size, candidate_pool_size)
                    hard_set_idx = torch.topk(scores, k=hard_set_size, largest=True).indices
                    if hard_set_size >= hard_k:
                        perm = torch.randperm(hard_set_size, device=device)[:hard_k]
                        hard_idx = hard_set_idx[perm]
                    else:
                        hard_idx = hard_set_idx
                else:
                    raise ValueError(f"Unknown HNM_MODE: {self.hnm_mode}")

                hard_edges = cand_edges[hard_idx]
                mask = torch.ones((candidate_pool_size,), dtype=torch.bool, device=device)
                mask[hard_idx] = False
                remaining_edges = cand_edges[mask]
            else:
                hard_edges = cand_edges.new_zeros((0, 2))
                remaining_edges = cand_edges

            # soft selection
            if soft_k > 0:
                n_rem = int(remaining_edges.size(0))
                if n_rem >= soft_k:
                    soft_idx = torch.randperm(n_rem, device=device)[:soft_k]
                    soft_edges = remaining_edges[soft_idx]
                else:
                    soft_edges = remaining_edges
                    need = soft_k - n_rem
                    filler_src = hard_edges if hard_edges.size(0) > 0 else cand_edges
                    fill_idx = torch.randperm(int(filler_src.size(0)), device=device)[:need]
                    soft_edges = torch.cat([soft_edges, filler_src[fill_idx]], dim=0)
            else:
                soft_edges = cand_edges.new_zeros((0, 2))

            selected = torch.cat([hard_edges, soft_edges], dim=0)

        # exact size safety
        if selected.size(0) > take_neg:
            selected = selected[:take_neg]
        elif selected.size(0) < take_neg:
            need = take_neg - selected.size(0)
            extra = cand_edges[torch.randperm(candidate_pool_size, device=device)[:need]]
            selected = torch.cat([selected, extra], dim=0)

        # final undirected shuffle
        shuffle2 = torch.rand((selected.size(0),), device=device) > 0.5
        selected[shuffle2] = selected[shuffle2][:, [1, 0]]
        return selected

    # --------------------
    # segmentation loss
    # --------------------
    def loss_seg(self, pred_seg, target_seg, eps=1e-6):
        """
        pred_seg: [B, Cseg, H, W] logits
        target_seg:
          - binary: [B, 1, H, W] or [B, H, W] with {0,1}
          - multiclass: [B, H, W] with {0..Cseg-1}
        """
        if pred_seg is None or pred_seg.numel() == 0:
            return torch.tensor(0.0, device=target_seg.device)

        target_seg = target_seg.to(pred_seg.device)

        if pred_seg.shape[1] == 1:
            if target_seg.dim() == 3:
                target_seg = target_seg.unsqueeze(1)
            target_seg = target_seg.float()

            pos = target_seg.sum()
            neg = target_seg.numel() - pos
            pos_weight = (neg / (pos + eps)).clamp(1.0, 50.0)

            return F.binary_cross_entropy_with_logits(pred_seg, target_seg, pos_weight=pos_weight)
        else:
            if target_seg.dim() == 4 and target_seg.shape[1] == 1:
                target_seg = target_seg[:, 0, ...]
            target_seg = target_seg.long()
            return F.cross_entropy(pred_seg, target_seg)

    # --------------------
    # original losses
    # --------------------
    def loss_class(self, outputs, indices):
        weight = torch.tensor([0.2, 0.8], device=outputs.device)
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long, device=outputs.device)
        targets[idx] = 1
        return F.cross_entropy(outputs.permute(0, 2, 1), targets, weight=weight, reduction="mean")

    def loss_cardinality(self, outputs, indices):
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long, device=outputs.device)
        targets[idx] = 1
        tgt_lengths = torch.as_tensor([t.sum() for t in targets], device=outputs.device)
        card_pred = (outputs.argmax(-1) == outputs.shape[-1] - 1).sum(1)
        return F.l1_loss(card_pred.float(), tgt_lengths.float(), reduction="sum") / (outputs.shape[0] * outputs.shape[1])

    def loss_nodes(self, outputs, targets, indices):
        num_nodes = sum(len(t) for t in targets)
        idx = self._get_src_permutation_idx(indices)
        pred_nodes = outputs[idx]
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss = F.l1_loss(pred_nodes, target_nodes, reduction="none")
        return loss.sum() / max(1, num_nodes)

    def loss_boxes(self, outputs, targets, indices):
        num_boxes = sum(len(t) for t in targets)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = torch.cat([target_boxes, 0.15 * torch.ones_like(target_boxes)], dim=-1)
        loss = 1 - torch.diag(
            box_ops_2D.generalized_box_iou(
                box_ops_2D.box_cxcywh_to_xyxy(src_boxes),
                box_ops_2D.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        return loss.sum() / max(1, num_boxes)

    def loss_edges(self, h, target_nodes, target_edges, indices):
        object_token = h[..., : self.obj_token, :]
        relation_token = None
        if self.rln_token > 0:
            relation_token = h[..., self.obj_token : self.rln_token + self.obj_token, :]

        # map gt edges by matcher ordering
        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
        target_edges = [
            torch.stack(t, 0) if len(t) > 0 else torch.zeros((0, 2), dtype=torch.long, device=h.device)
            for t in target_edges
        ]

        new_target_edges = []
        for t, (_, i) in zip(target_edges, indices):
            tx = t.clone().detach()
            for idx, k in enumerate(i):
                t[tx == k] = idx
            new_target_edges.append(t)

        edge_labels = []
        relation_feature = []

        for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
            rearranged_object_token = object_token[batch_id, indices[batch_id][0], :]

            full_adj = torch.ones((n.shape[0], n.shape[0]), device=h.device) - torch.diag(torch.ones(n.shape[0], device=h.device))
            if pos_edge.numel() > 0:
                full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
                full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0

            neg_edges = torch.nonzero(torch.triu(full_adj)).to(h.device)

            # shuffle positive for undirected
            if pos_edge.numel() > 0:
                shuffle = torch.rand((pos_edge.shape[0],), device=pos_edge.device) > 0.5
                pos_edge = pos_edge.clone()
                pos_edge[shuffle] = pos_edge[shuffle][:, [1, 0]]

            if self.edge_sampling_mode == EDGE_SAMPLING_MODE.NONE and pos_edge.shape[0] > 40:
                pos_edge = pos_edge[:40, :]

            # random permute negatives before HNM pool selection
            if neg_edges.numel() > 0:
                neg_edges = neg_edges[torch.randperm(neg_edges.shape[0], device=neg_edges.device)]

            # how many negatives to take
            if self.num_edge_samples - pos_edge.shape[0] < neg_edges.shape[0]:
                take_neg = self.num_edge_samples - pos_edge.shape[0]
                total_edge = self.num_edge_samples
            else:
                take_neg = neg_edges.shape[0]
                total_edge = pos_edge.shape[0] + neg_edges.shape[0]

            selected_neg_edges = self.select_hard_negative_edges(
                neg_edges.to(pos_edge.device),
                take_neg,
                rearranged_object_token,
                relation_token_batch=(relation_token[batch_id, :] if self.rln_token > 0 else None),
            )
            all_edges_ = torch.cat((pos_edge, selected_neg_edges), 0)

            edge_labels.append(
                torch.cat(
                    (
                        torch.ones(pos_edge.shape[0], dtype=torch.long, device=h.device),
                        torch.zeros(take_neg, dtype=torch.long, device=h.device),
                    ),
                    0,
                )
            )

            if self.rln_token > 0:
                relation_feature.append(
                    torch.cat(
                        (
                            rearranged_object_token[all_edges_[:, 0], :],
                            rearranged_object_token[all_edges_[:, 1], :],
                            torch.flatten(relation_token[batch_id, ...]).repeat(total_edge, 1),
                        ),
                        1,
                    )
                )
            else:
                relation_feature.append(
                    torch.cat(
                        (rearranged_object_token[all_edges_[:, 0], :], rearranged_object_token[all_edges_[:, 1], :]),
                        1,
                    )
                )

        relation_feature = torch.cat(relation_feature, 0)
        edge_labels = torch.cat(edge_labels, 0)
        relation_pred = self.net.relation_embed(relation_feature)

        if self.edge_sampling_mode == EDGE_SAMPLING_MODE.UP:
            relation_pred, edge_labels = upsample_edges(relation_pred, edge_labels, self.sample_ratio, self.sample_ratio_interval)
        elif self.edge_sampling_mode == EDGE_SAMPLING_MODE.DOWN:
            relation_pred, edge_labels = downsample_edges(relation_pred, edge_labels, self.sample_ratio, self.sample_ratio_interval)
        elif self.edge_sampling_mode == EDGE_SAMPLING_MODE.RANDOM_UP:
            ratio = np.random.uniform(self.sample_ratio, 1)
            relation_pred, edge_labels = upsample_edges(relation_pred, edge_labels, ratio, self.sample_ratio_interval)

        return F.cross_entropy(relation_pred, edge_labels, reduction="mean")

    def loss_domains(self, img_preds, img_labels, instance_preds, instance_labels):
        domain_loss = 0
        img_preds, non_reverse_img_pred = img_preds
        instance_preds, non_reverse_instance_preds = instance_preds

        if self.domain_img_loss is not None:
            domain_loss += self.domain_img_loss(img_preds, torch.flatten(img_labels))
        if self.domain_inst_loss is not None:
            domain_loss += self.domain_inst_loss(instance_preds, instance_labels)
        if self.domain_img_loss is not None and self.domain_inst_loss is not None and self.consistency_loss is not None:
            non_reverse_img_pred = torch.reshape(non_reverse_img_pred, (instance_labels.shape[0], -1, 2))
            mean_non_reverse_img_pred = torch.mean(non_reverse_img_pred, dim=1)
            domain_loss += self.consistency_loss(mean_non_reverse_img_pred, non_reverse_instance_preds)

        return domain_loss

    # --------------------
    # helpers
    # --------------------
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # --------------------
    # forward
    # --------------------
    def forward(self, h, out, target, pred_backbone_domains, pred_instance_domains):
        """
        Segmentation loss is applied on:
          - source only if exclude_target_seg == True
          - otherwise on all samples
        Graph loss computation can optionally exclude target graphs if compute_target_graph_loss == False
        """
        device = h.device if hasattr(h, "device") else h.get_device()

        # Keep mask for original batch (used for seg-only on source)
        keep = (target["domains"] == 0)

        # Optionally remove target-domain samples for graph supervision
        if not self.compute_target_graph_loss:
            out["pred_logits"] = out["pred_logits"][keep]
            out["pred_nodes"] = out["pred_nodes"][keep]
            target["nodes"] = [node_list for i, node_list in enumerate(target["nodes"]) if keep[i]]
            target["edges"] = [edge_list for i, edge_list in enumerate(target["edges"]) if keep[i]]
            # after filtering, everything left is source (for graph terms)
            keep = torch.ones(len(target["nodes"]), dtype=torch.bool, device=keep.device)

        # If nothing left, return all zeros (include seg key)
        if len(target["nodes"]) == 0:
            z = torch.tensor(0.0, device=device)
            return {"total": z, "class": z, "nodes": z, "boxes": z, "edges": z, "cards": z, "domain": z, "seg": z}

        # matching (on whatever set remains for graph loss)
        indices = self.matcher(out, target)

        losses = {}
        losses["class"] = self.loss_class(out["pred_logits"], indices)
        losses["nodes"] = self.loss_nodes(out["pred_nodes"][..., :2], target["nodes"], indices)
        losses["boxes"] = self.loss_boxes(out["pred_nodes"], target["nodes"], indices)
        losses["edges"] = self.loss_edges(h, target["nodes"], target["edges"], indices)
        losses["cards"] = self.loss_cardinality(out["pred_logits"], indices)

        if self.domain_img_loss is not None:
            losses["domain"] = self.loss_domains(
                pred_backbone_domains,
                target["interpolated_domains"],
                pred_instance_domains,
                target["domains"],
            )
        else:
            losses["domain"] = torch.tensor(0.0, device=device)

        # segmentation loss (may use original keep if exclude_target_seg)
        pred_seg = out.get("pred_seg", None)
        gt_seg = target.get("seg", None)

        if pred_seg is None or gt_seg is None:
            losses["seg"] = torch.tensor(0.0, device=device)
        else:
            if self.exclude_target_seg:
                if keep.any():
                    losses["seg"] = self.loss_seg(pred_seg[keep], gt_seg[keep])
                else:
                    losses["seg"] = torch.tensor(0.0, device=device)
            else:
                losses["seg"] = self.loss_seg(pred_seg, gt_seg)

        # total (use config.TRAIN.LOSSES ordering)
        losses["total"] = sum(losses[k] * self.weight_dict[k] for k in self.losses)

        return losses
