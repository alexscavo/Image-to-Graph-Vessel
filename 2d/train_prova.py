from functools import partial
import os
from pathlib import Path
import sys
from matplotlib import pyplot as plt
import yaml
import json
import shutil
from argparse import ArgumentParser
from monai.handlers import EarlyStopHandler
import torch
from monai.data import DataLoader
from data.dataset_mixed import build_mixed_data
from data.dataset_road_network import build_road_network_data
from data.prova_synth import build_synthetic_vessel_network_data
from data.prova import build_real_vessel_network_data
from training.evaluator import build_evaluator
from training.trainer import build_trainer
from models import build_model
from utils.utils import image_graph_collate_road_network
from torch.utils.tensorboard import SummaryWriter
from models.matcher import build_matcher
from training.losses import EDGE_SAMPLING_MODE, SetCriterion
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from PIL import Image

from utils.vis_debug_old import visualize_graph_batch

if not hasattr(Image, "ANTIALIAS"):
    from PIL import Image as _PILImage
    try:
        # Pillow ≥10
        Image.NEAREST   = _PILImage.Resampling.NEAREST
        Image.BILINEAR  = _PILImage.Resampling.BILINEAR
        Image.BICUBIC   = _PILImage.Resampling.BICUBIC
        Image.LANCZOS   = _PILImage.Resampling.LANCZOS
        Image.ANTIALIAS = _PILImage.Resampling.LANCZOS
    except Exception:
        pass

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def main(args):
    
    import torch, numpy as np   # RIMUOVERE

    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    if config.DATA.DATASET == 'road_dataset':
        build_dataset_function = build_road_network_data
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'synthetic_eye_vessel_dataset':
        build_dataset_function = build_synthetic_vessel_network_data
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'real_eye_vessel_dataset':
        build_dataset_function = build_real_vessel_network_data
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'mixed_road_dataset' or config.DATA.DATASET == 'mixed_synthetic_eye_vessel_dataset' or config.DATA.DATASET == "mixed_real_eye_vessel_dataset":
        build_dataset_function = partial(build_mixed_data, upsample_target_domain=config.TRAIN.UPSAMPLE_TARGET_DOMAIN)
        config.DATA.MIXED = True

    # check if the val set is already provided, or if we need to use the random split with 0.8
    root = Path(config.DATA.TARGET_DATA_PATH)
    val_root = root / "val"
    has_val = val_root.exists()
    
    print('val_root:', val_root)
    print('has_val? ',has_val)

    train_ds, val_ds, sampler = build_dataset_function(
        config, mode='split', max_samples=config.DATA.NUM_SOURCE_SAMPLES, split=0.8, has_val=has_val
    )

    batch_size = 1

    train_loader = DataLoader(train_ds,
                              batch_size=config.DATA.BATCH_SIZE,
                              shuffle=not sampler,
                              num_workers=config.DATA.NUM_WORKERS,
                              collate_fn=image_graph_collate_road_network,
                              pin_memory=True,
                              sampler=sampler)
    

    val_loader = DataLoader(val_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)
    
    img, seg, nodes, edges, domain = train_ds[5]
    

    print('-'*50)
    print(f"Image shape: {img.shape}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Seg shape: {seg.shape}")
    print(f"Nodes shape: {nodes.shape}")
    print(f"Node coordinates range: [{nodes.min():.3f}, {nodes.max():.3f}]")
    print(f"First 3 nodes:\n{nodes[:3]}")
    print(f"Edges shape: {edges.shape}")
    print(f"Edge indices range: [{edges.min()}, {edges.max()}]")
    print('-'*50)

    print("Sample nodes (first few):")
    print(nodes[:5])

    print("Sample edges (first few):")
    print(edges[:5])

    print('-'*50)
    
    sample = {
        "images":     [img],       # (C,H,W) tensor
        "nodes":      [nodes],     # (N,2) or (N,3) tensor
        "edges":      [edges],     # (E,2) tensor
        "pred_nodes": [nodes],     # optional — reuse GT if you just want to see overlay
        "pred_edges": [edges],
    }

    # Visualize it
    fig = visualize_graph_batch(sample, n=1)
    fig.savefig('/data/scavone/overlay_prova/modello_octa500', dpi=180, bbox_inches="tight")
    plt.close(fig)



if __name__ == '__main__':

    
    
    # --- PRE-TRAINING --- 
    args = parser.parse_args(['--config', '/home/scavone/2d/cross-dim_i2g/configs/prova.yaml'])
    

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)
