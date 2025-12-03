import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import yaml

# Example import — adjust as needed
from data.dataset_road_network import build_road_network_data


# -------------------------------------------------------------
# Save 2D → 3D visualization for a sample
# -------------------------------------------------------------
def save_2d_and_3d_visualization(dataset, idx, out_dir="viz_output"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Original 2D raw image (from path in dataset.data)
    sample_meta = dataset.data[idx]
    img_path = sample_meta["img"]

    img_2d = Image.open(img_path).convert("RGB")
    img_2d.save(os.path.join(out_dir, f"sample_{idx}_2d_raw.png"))

    # 2) 3D embedded image from __getitem__
    sample = dataset[idx]          # tuple of 6 lists
    img_3d = sample[0][0]          # [img_data] → img_data

    # now it's a tensor: [C, D, H, W]
    img_3d = img_3d.detach().cpu().numpy()
    C, D, H, W = img_3d.shape
    vol = img_3d[0]                # single channel → [D, H, W]

    mid_z = D // 2
    mid_slice = vol[mid_z]         # [H, W]
    mip = vol.max(axis=0)          # depth MIP: [H, W]

    def to_uint8(x):
        # typically x is in [-0.5, 0.5]; normalize to [0, 255]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return (x_norm * 255).astype(np.uint8)

    mid_uint8 = to_uint8(mid_slice)
    mip_uint8 = to_uint8(mip)

    Image.fromarray(mid_uint8).save(
        os.path.join(out_dir, f"sample_{idx}_3d_midslice.png")
    )
    Image.fromarray(mip_uint8).save(
        os.path.join(out_dir, f"sample_{idx}_3d_mip.png")
    )

    print(f"[✓] Saved sample {idx} visualizations to {out_dir}")


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

with open('3d/configs/roads_only.yaml') as f:
    print('\n*** Config file')
    config = yaml.load(f, Loader=yaml.FullLoader)

config = dict2obj(config)


train_ds, val_ds, sampler = build_road_network_data(
            config, 
            mode='split', 
            debug=False, 
            max_samples=config.DATA.NUM_SOURCE_SAMPLES, 
            domain_classification=-1, 
            gaussian_augment=True, 
            rotate=True, 
            continuous=True
        )



save_2d_and_3d_visualization(train_ds, idx=4)
