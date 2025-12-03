# extract_activations.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models import build_model
from data.dataset_road_network import build_road_network_data
from utils.utils import image_graph_collate

from argparse import ArgumentParser


def dict2obj(d):
    import json
    class Obj:
        def __init__(self, d): self.__dict__.update(d)
    return json.loads(json.dumps(d), object_hook=Obj)


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return dict2obj(cfg)


def main(args):
    device = torch.device(args.device)

    config = load_config(args.config)
    config.DATA.MIXED = False  # or True depending on your setup

    # --- dataset (mirror what you do in train.py) ---
    train_ds, val_ds, _ = build_road_network_data(
        config,
        mode='split',
        debug=False,
        max_samples=config.DATA.NUM_SOURCE_SAMPLES,
        gaussian_augment=True,
        rotate=True,
        continuous=True,
    )

    loader = DataLoader(
        val_ds if args.split == "val" else train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=lambda x: image_graph_collate(
            x,
            pre2d=args.pre2d,
            gaussian_augment=config.DATA.MIXED,
        ),
        pin_memory=True,
    )

    # --- model ---
    net = build_model(config, pre2d=args.pre2d).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(ckpt["net"], strict=False)

    net.eval()

    all_feats = []     # flattened srcs, for Overcomplete
    all_volumes = []   # [B, C, D, H, W], for visualization

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, segs, nodes, edges, z_pos, domains = batch
            images = images.to(device)    # from collate: [B*D, C, H, W] or [B, C, D, H, W]
            domains = domains.to(device)

            # --- reconstruct [B, C, D, H, W] from collate output ---
            if images.dim() == 4:
                # collate gives [B*D, C, H, W]; recover B and D
                B = domains.shape[0]
                N, C_in, H, W = images.shape
                assert N % B == 0, f"Cannot infer depth: N={N}, B={B}"
                D = N // B

                images_5d = images.view(B, D, C_in, H, W).permute(0, 2, 1, 3, 4).contiguous()
            elif images.dim() == 5:
                # already [B, C, D, H, W]
                images_5d = images
                B, C_in, D, H, W = images_5d.shape
            else:
                raise ValueError(f"Unexpected images.dim()={images.dim()}")

            # --- forward pass ---
            h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = net(
                images_5d.float(), z_pos, domain_labels=domains
            )

            # srcs: [B, C_feat, D, H, W]
            B, C_feat, D, H, W = srcs.shape

            # flatten spatial dims → [B*D*H*W, C_feat]
            feats = srcs.permute(0, 2, 3, 4, 1).reshape(-1, C_feat)

            # (optional) subsample to keep size manageable
            if feats.shape[0] > args.max_tokens_per_batch:
                idx = torch.randperm(feats.shape[0], device=feats.device)[:args.max_tokens_per_batch]
                feats = feats[idx]

            all_feats.append(feats.cpu())

            # store volumes used for this batch (for visualization later)
            if args.volumes_out_path is not None and len(all_volumes) < args.max_volumes:
                all_volumes.append(images_5d.cpu())

            if (batch_idx + 1) >= args.max_batches:
                break

    # --- save activations ---
    Activations = torch.cat(all_feats, dim=0)  # [N, d]
    out_dir = os.path.dirname(args.out_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    torch.save(Activations, args.out_path)
    print("Saved activations:", Activations.shape, "→", args.out_path)

    # --- save volumes (optional) ---
    if args.volumes_out_path is not None and len(all_volumes) > 0:
        volumes = torch.cat(all_volumes, dim=0)  # [N_volumes, C, D, H, W]
        vol_dir = os.path.dirname(args.volumes_out_path)
        if vol_dir != "":
            os.makedirs(vol_dir, exist_ok=True)
        torch.save(volumes, args.volumes_out_path)
        print("Saved volumes:", volumes.shape, "→", args.volumes_out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pre2d", action="store_true")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--max_tokens_per_batch", type=int, default=4096)
    parser.add_argument("--out_path", default="activations/relformer_srcs.pt")
    # new: where to save volumes & how many to keep
    parser.add_argument("--volumes_out_path", default=None)
    parser.add_argument("--max_volumes", type=int, default=512)

    # you can keep this hard-coded call or use CLI
    args = parser.parse_args([
        '--config', '/home/scavone/cross-dim_i2g/3d/configs/roads_only.yaml',
        '--checkpoint', '/data/scavone/checkpoint_epoch=50.pt',
        '--out_path', '/data/scavone/features_3d.pt',
        '--volumes_out_path', '/data/scavone/volumes_3d.pt',
    ])

    main(args)
