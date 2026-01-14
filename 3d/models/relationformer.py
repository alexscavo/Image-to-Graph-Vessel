# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RelationFormer model and criterion classes.
"""
import random
import torch
import torch.nn.functional as F
from torch import nn

from models.domain_adaptation.domain_classifier import Discriminator

from .deformable_detr_backbone import build_backbone
from .deformable_transformer_3D import build_def_detr_transformer
from .seresnet import build_seresnet
from .position_encoding import PositionEmbeddingSine3D
from .utils import nested_tensor_from_tensor_list, NestedTensor
from .segmentation_head import SegHead3D


class RelationFormer(nn.Module):
    """ This is the RelationFormer module that performs object detection """

    def __init__(self, encoder, decoder, config, cls_token=False, use_proj_in_dec=False, device='cuda', pre2d=False):
        """[summary]

        Args:
            encoder ([type]): [description]
            decoder ([type]): [description]
            num_classes (int, optional): [description]. Defaults to 8.
            num_queries (int, optional): [description]. Defaults to 100.
            imsize (int, optional): [description]. Defaults to 64.
            cls_token (bool, optional): [description]. Defaults to False.
            use_proj_in_dec (bool, optional): [description]. Defaults to False.
            with_box_refine (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = config.MODEL.DECODER.OBJ_TOKEN+config.MODEL.DECODER.RLN_TOKEN+config.MODEL.DECODER.DUMMY_TOKEN
        # self.imsize = imsize
        self.cls_token = cls_token
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM
        self.device = device
        self.init = True
        self.use_proj_in_dec = use_proj_in_dec
        self.position_embedding = PositionEmbeddingSine3D(channels=config.MODEL.DECODER.HIDDEN_DIM)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.config = config
        self.segmentation_head = None
        
        self.input_proj = nn.Conv3d(encoder.num_features, config.MODEL.DECODER.HIDDEN_DIM, kernel_size=1)
        
        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.coord_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 6, 3)
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        if config.MODEL.DECODER.RLN_TOKEN>0:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*(2+config.MODEL.DECODER.RLN_TOKEN), config.MODEL.DECODER.HIDDEN_DIM, 2, 3)
        else:
            self.relation_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM*2, config.MODEL.DECODER.HIDDEN_DIM, 2, 3)

        self.pre2d = pre2d
        if self.pre2d:
            self.input_proj_2d = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
            self.padding_mode = config.MODEL.PRE2D_BACKBONE.PADDING_MODE
            self.mark_padding = config.MODEL.PRE2D_BACKBONE.MARK_PADDING

        if config.DATA.MIXED:
            self.backbone_domain_discriminator = Discriminator(in_size=self.hidden_dim)
            self.instance_domain_discriminator = Discriminator(in_size=self.hidden_dim*self.num_queries)
            
        print("-"*50)
        print('Segmentation configs:')
        print(f"  Enabled: {self.config.MODEL.SEGMENTATION.ENABLED}")
        print(f"  In channels: {self.config.MODEL.SEGMENTATION.IN_CHANS}")
        print(f"  Mid channels: {self.config.MODEL.SEGMENTATION.MID_CHANS}")
        print(f"  Num classes: {self.config.MODEL.SEGMENTATION.NUM_CLASSES}")
        print("-"*50)
        
        if self.config.MODEL.SEGMENTATION.ENABLED:
            seg_cfg = self.config.MODEL.SEGMENTATION

            self.segmentation_head = SegHead3D(
                in_ch=seg_cfg.IN_CHANS,          # e.g. 512
                mid_ch=seg_cfg.MID_CHANS,        # e.g. 64
                out_ch=seg_cfg.NUM_CLASSES       # e.g. 1 or K
            )
        else:
            self.segmentation_head = None


        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(2) * bias_value
        # nn.init.constant_(self.coord_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.coord_embed.layers[-1].bias.data, 0)

    def forward(self, samples, z_pos=None, alpha=1, domain_labels=None, seg=True):

        # Always define pred_seg so it's available at the end
        pred_seg = None

        # If samples isn't a NestedTensor, build one (works whether samples is a list or a batched tensor)
        if not isinstance(samples, NestedTensor):
            # If samples is a 5D batch tensor [B,C,D,H,W], iterating yields [C,D,H,W] per item
            samples = nested_tensor_from_tensor_list(list(samples))

        # ----------------------------
        # Encoder / feature extraction
        # ----------------------------
        if self.pre2d:
            # pre2d path expects 2D images; expand single-channel 2D to 3 channels
            # NOTE: this only makes sense if each element is [1,H,W] (2D). If you have volumes, don't use pre2d.
            samples2d = nested_tensor_from_tensor_list(
                [tensor.expand(3, -1, -1).contiguous() for tensor in samples.tensors]  # iter over batch -> [C,H,W]
            )

            features, _pos = self.encoder(samples2d)
            src, mask = features[-1].decompose()
            srcs = self.input_proj_2d[-1](src)

            # Padding (your existing code)
            z_coords = round((srcs.shape[-1] - 1) * z_pos)
            padding_before = z_coords
            padding_after = srcs.shape[-1] - padding_before - 1

            srcs = srcs[:, :, :, :, None]
            srcs = F.pad(srcs, (padding_before, padding_after, 0, 0, 0, 0), self.padding_mode, value=0)
            mask = mask[:, :, :, None]
            mask = F.pad(mask, (padding_before, padding_after), "constant", self.mark_padding)

            mask_list = [mask]
            pos_list = [self.position_embedding(mask_list[-1]).to(srcs.device)]

        else:
            # 3D SEResNet path: backbone expects a Tensor, not a NestedTensor
            x = samples.tensors  # [B, C, D, H, W]

            # IMPORTANT: SEResNet first conv expects C==1 (as your error showed)
            # So DO NOT expand to 3 channels here.
            feat = self.encoder(x)  # should output [B, C', D', H', W'] (Tensor)
            srcs = self.input_proj(feat)

            # If you have padding and want correct masking, you need to downsample samples.mask to feat resolution.
            # For fixed-size volumes, an all-false mask is fine:
            mask = torch.zeros(feat.shape[0], feat.shape[2], feat.shape[3], feat.shape[4],
                            dtype=torch.bool, device=feat.device)

            mask_list = [mask]
            pos_list = [self.position_embedding(mask_list[-1]).to(feat.device)]

        # ----------------------------
        # Mixed-domain heads (domain + seg)
        # ----------------------------
        
        # Segmentation head (only if present AND seg flag is True)
            if seg and (self.segmentation_head is not None):
                seg_level = self.config.MODEL.SEGMENTATION.FROM_LEVEL

                # For 3D, out_size should be (D,H,W). For 2D, it’s (H,W).
                if samples.tensors.dim() == 5:
                    out_size = samples.tensors.shape[-3:]  # (D,H,W)
                else:
                    out_size = samples.tensors.shape[-2:]  # (H,W)
 
                pred_seg = self.segmentation_head(srcs, out_size)
                
        if self.config.DATA.MIXED:
            flat_srcs = torch.flatten(srcs.clone().permute(0, 2, 3, 4, 1), end_dim=3)
            domain_labels = domain_labels.unsqueeze(1).repeat_interleave(
                srcs.shape[2] * srcs.shape[3] * srcs.shape[4], dim=1
            ).flatten()
            backbone_domain_classifications = self.backbone_domain_discriminator(flat_srcs, alpha)
        else:
            backbone_domain_classifications = torch.tensor(-1, device=srcs.device)
            domain_labels = None

        # ----------------------------
        # Decoder + outputs
        # ----------------------------
        query_embed = self.query_embed.weight
        h = self.decoder(srcs, mask_list[-1], query_embed, pos_list[-1])
        object_token = h[...,:self.obj_token,:]

        class_prob = self.class_embed(object_token)
        
        # DEBUG START
        # coord_loc = self.bbox_embed(object_token).sigmoid()        
        # pre = self.coord_embed(object_token)          # BEFORE sigmoid
        # post = pre.sigmoid()                         # AFTER sigmoid
        # if random.random() < 0.01:  # ~10% chance
        #     print("\npre min/max:", pre.min().item(), pre.max().item())
        #     print("post min/max:", post.min().item(), post.max().item())
        #     q = torch.quantile(pre.flatten(), torch.tensor([0.01,0.5,0.99], device=pre.device))
        #     sat_lo = (post < 0.01).float().mean().item()
        #     sat_hi = (post > 0.99).float().mean().item()
        #     print(f"pre q1/med/q99=({q[0]:.2f},{q[1]:.2f},{q[2]:.2f}), "
        #         f"post sat low={sat_lo:.2f}, high={sat_hi:.2f}")
        # coord_loc = post
        #DEBUG END
        
        coord_loc = self.coord_embed(object_token).sigmoid()

        if self.config.DATA.MIXED:
            # Flatten the tensor but keep batch dimension
            domain_hs = torch.flatten(h.clone(), start_dim=1)
            instance_domain_classifications = self.instance_domain_discriminator(domain_hs, alpha)
        else:
            instance_domain_classifications = torch.tensor(-1)

        
        out = {'pred_logits': class_prob, 'pred_nodes': coord_loc, 'pred_seg': pred_seg}
        return h, out, srcs, backbone_domain_classifications, instance_domain_classifications, domain_labels


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_relationformer(config, pre2d=False, **kwargs):

    if pre2d:
        encoder = build_backbone(config)
    else:
        encoder = build_seresnet(config)

    decoder = build_def_detr_transformer(config)

    model = RelationFormer(
        encoder,
        decoder,
        config,
        pre2d=pre2d,
        **kwargs
    )

    # losses = ['labels', 'boxes', 'cardinality']

    # criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
    #                          eos_coef=args.eos_coef, losses=losses, loss_type = args.loss_type, use_fl=args.use_fl, loss_transform=args.loss_transform)
    # criterion.to(device)
    
    return model
