import sys
from data.dataset_synth_octa_network import build_octa_network_data
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from data.dataset_road_network import build_road_network_data
import math
from data.dataset_vessel3d import build_vessel_data

def build_mixed_data(config, mode='split', split=0.95, use_grayscale=False, debug=False, rotate=False, continuous=False, upsample_target_domain=True):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
    

    Returns:
        [type]: [description]
    """
    config.DATA.DATA_PATH = config.DATA.SOURCE_DATA_PATH
    if config.DATA.DATASET == "mixed_real_vessels" or config.DATA.DATASET == "mixed_synth_3d":
        source_train_data, source_val_data, _ = build_road_network_data(config, mode, split, debug=debug, max_samples=config.DATA.NUM_SOURCE_SAMPLES, domain_classification=0, gaussian_augment=True, rotate=rotate, continuous=continuous)
    elif config.DATA.DATASET == "mixed_real_vessels_octa" or config.DATA.DATASET == "mixed_synth_3d_octa":
        source_train_data, source_val_data, _ = build_octa_network_data(config, mode, split, debug=debug, max_samples=config.DATA.NUM_SOURCE_SAMPLES, domain_classification=0, gaussian_augment=True, rotate=rotate, continuous=continuous)
    
    config.DATA.DATA_PATH = config.DATA.TARGET_DATA_PATH
    target_train_data, target_val_data, _ = build_vessel_data(config, mode, split, debug=debug, max_samples=config.DATA.NUM_TARGET_SAMPLES, domain_classification=1)

    # Calculate the number of samples in each dataset
    num_samples_A = len(source_train_data)
    num_samples_B = len(target_train_data)

    # Calculate the weights for each sample in each dataset
    # TODO: if sample proportion remains ~ 99000 to 4000, this weighting possibly leads to high overfitting
    target_multiplier = num_samples_A / num_samples_B
    weights_A = torch.ones(num_samples_A)
    weights_B = torch.ones(num_samples_B) * target_multiplier

    train_ds = ConcatDataset([source_train_data, target_train_data])
    val_ds = ConcatDataset([source_val_data, target_val_data])

    print(f"--- Dataset Balance Info ---")
    print(f"Source samples (A) - train: {num_samples_A}, val: {len(source_val_data)}")
    print(f"Target samples (B) - train: {num_samples_B}, val: {len(target_val_data)}")
    print(f"Oversampling Factor for Target: {target_multiplier:.2f}x")
    print(f"Total Combined Samples in ConcatDataset - train: {len(train_ds)}, val: {len(val_ds)}")
    
    if upsample_target_domain:
        print("upsampling")
        sampler = WeightedRandomSampler(torch.cat([weights_A, weights_B]), num_samples=math.floor(torch.sum(weights_A) + torch.sum(weights_B)))
    else:
        print("not upsampling")
        sampler = None

    return train_ds, val_ds, sampler
