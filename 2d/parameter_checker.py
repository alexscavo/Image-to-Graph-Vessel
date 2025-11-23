import torch
import yaml
from models import build_model   # same import used during training

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

# ------------------------
# paths to fill in
# ------------------------
CONFIG_PATH = "/home/scavone/cross-dim_i2g/2d/configs/pretrained_config_2d_synth.yaml"
CHECKPOINT_PATH = "/data/scavone/cross-dim_i2g_2d/trained_weights/runs/finetune_mixed_synth1_10/models/checkpoint_epoch=100.pt"

# ------------------------
# 1. load cfg
# ------------------------
with open(CONFIG_PATH) as f:
    cfg_raw = yaml.load(f, Loader=yaml.FullLoader)

# convert cfg dict into object-like access
class Obj:
    def __init__(self, d):
        for k,v in d.items():
            setattr(self, k, Obj(v) if isinstance(v, dict) else v)

cfg = Obj(cfg_raw)
cfg.DATA.MIXED = True

# ------------------------
# 2. rebuild the model
# ------------------------
net = build_model(cfg)     # SAME as in train.py
net.cpu()

# ------------------------
# 3. load checkpoint
# ------------------------
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

net.load_state_dict(ckpt["net"], strict=False)

param_dicts = [
        {
            "params":
                [p for n, p in net.named_parameters()
                 if not match_name_keywords(n, ["encoder.0"]) and not match_name_keywords(n, ['reference_points', 'sampling_offsets']) and not match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
            "lr": float(cfg.TRAIN.LR),
            "weight_decay": float(cfg.TRAIN.WEIGHT_DECAY)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["encoder.0"]) and p.requires_grad],
            "lr": float(cfg.TRAIN.LR_BACKBONE),
            "weight_decay": float(cfg.TRAIN.WEIGHT_DECAY)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(cfg.TRAIN.LR)*0.1,
            "weight_decay": float(cfg.TRAIN.WEIGHT_DECAY)
        },
        {
            "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ['domain_discriminator']) and p.requires_grad],
            "lr": float(cfg.TRAIN.LR_DOMAIN),
            "weight_decay": float(cfg.TRAIN.WEIGHT_DECAY)
        }
    ]

# ------------------------
# 4. build dummy optimizer & load its state
# ------------------------
optimizer = torch.optim.AdamW(param_dicts, lr=float(cfg.TRAIN.LR), weight_decay=float(cfg.TRAIN.WEIGHT_DECAY))
# optimizer.load_state_dict(ckpt["optimizer"])

# ------------------------
# 5. inspect LRs for transformer/backbone
# ------------------------
KEYS = ["transformer", "decoder", "encoder", "detr", "backbone"]

# Build a lookup table from parameter id to lr
param_to_lr = {}
for group in optimizer.param_groups:
    lr = group["lr"]
    for p in group["params"]:
        param_to_lr[id(p)] = lr

# Now inspect
for name, p in net.named_parameters():
    if any(k in name.lower() for k in KEYS):
        lr = param_to_lr.get(id(p), None)
        if lr is not None:
            print(f"{name:60s}  -->  LR = {lr}")

