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
CONFIG_PATH = "/home/scavone/cross-dim_i2g/3d/configs/synth_3D.yaml"

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

param_dicts = [
    {
        "params": [p for n, p in net.named_parameters() if match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
        "lr": float(cfg.TRAIN.DOMAIN_LR),
        "weight_decay": float(cfg.TRAIN.WEIGHT_DECAY)
    },
    {
        "params": [p for n, p in net.named_parameters() if not match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
        "lr": float(cfg.TRAIN.BASE_LR),
        "weight_decay": float(cfg.TRAIN.WEIGHT_DECAY)
    },
]


optimizer = torch.optim.AdamW(
    param_dicts, lr=float(cfg.TRAIN.BASE_LR), weight_decay=float(cfg.TRAIN.WEIGHT_DECAY)
    )
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

