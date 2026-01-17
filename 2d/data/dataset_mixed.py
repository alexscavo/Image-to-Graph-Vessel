import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from data.dataset_real_eye_vessels import build_real_vessel_network_data
from data.dataset_road_network import build_road_network_data
from data.dataset_synthetic_eye_vessels import build_synthetic_vessel_network_data
from data.dataset_plants import build_plants_network_data
import math


SOURCE_BUILDERS = {
    "roads": build_road_network_data,
    "plants": build_plants_network_data,
}

TARGET_BUILDERS = {
    "roads": build_road_network_data,
    "octa-synth": build_synthetic_vessel_network_data,
    "octa-real": build_real_vessel_network_data,
    "plants": build_plants_network_data,
}

def build_mixed_data(config, mode, split, use_grayscale, has_val, upsample_target_domain):
    # ---- SOURCE (option A: explicit per-dataset counts) ----
    # For mixed sources, we need a per-dataset path mapping (SOURCE_DATA_PATHS).

    mixed_source = bool(config.DATA.MIXED_SOURCE)
    source_names = config.DATA.SOURCE_DATASET_NAME
    if not mixed_source:
        # normalize to list for unified logic
        source_names = [source_names]
    elif isinstance(source_names, str):
        # defensive: YAML should give a list when MIXED_SOURCE=True, but handle it anyway
        source_names = [source_names]

    counts_by_ds = getattr(config.DATA, "NUM_SOURCE_SAMPLES_BY_DATASET", None)
    if counts_by_ds is None:
        raise ValueError("Option A selected but config.DATA.NUM_SOURCE_SAMPLES_BY_DATASET is missing.")
    if not isinstance(counts_by_ds, dict) and hasattr(counts_by_ds, "__dict__"):
        counts_by_ds = vars(counts_by_ds)

    source_paths_by_ds = getattr(config.DATA, "SOURCE_DATA_PATHS", None)
    if source_paths_by_ds is not None and not isinstance(source_paths_by_ds, dict):
        if hasattr(source_paths_by_ds, "__dict__"):
            source_paths_by_ds = vars(source_paths_by_ds)

    # If MIXED_SOURCE is enabled, enforce SOURCE_DATA_PATHS so each source can point to its own root.
    if mixed_source and source_paths_by_ds is None:
        raise ValueError(
            "config.DATA.MIXED_SOURCE is True, but config.DATA.SOURCE_DATA_PATHS is missing. "
            "Provide a dict mapping dataset name -> path (e.g., {roads: /data/roads, plants: /data/plants})."
        )

    source_train_parts, source_val_parts = [], []
    for name in source_names:
        if name not in SOURCE_BUILDERS:
            raise ValueError(f"Unknown source dataset '{name}'. Available: {list(SOURCE_BUILDERS.keys())}")

        # Set path for this source dataset
        if source_paths_by_ds is None:
            raise ValueError("config.DATA.SOURCE_DATA_PATHS is missing.")
        if name not in source_paths_by_ds:
            raise ValueError(
                f"Missing path for source dataset '{name}' in SOURCE_DATA_PATHS. "
                f"Available keys: {list(source_paths_by_ds.keys())}"
            )
        config.DATA.DATA_PATH = source_paths_by_ds[name]

        if name not in counts_by_ds:
            raise ValueError(
                f"Missing sample count for source dataset '{name}' in NUM_SOURCE_SAMPLES_BY_DATASET."
            )

        n_samples = int(counts_by_ds[name])
        if n_samples <= 0:
            # skip datasets with 0 samples to allow easy ablations in YAML
            continue

        builder = SOURCE_BUILDERS[name]
        train_i, val_i, _ = builder(
            config,
            mode,
            split,
            n_samples,
            use_grayscale=(config.DATA.TARGET_DATA_PATH != "mixed_road_dataset"),
            domain_classification=0,
            has_val=has_val,
        )
        source_train_parts.append(train_i)
        source_val_parts.append(val_i)

    if len(source_train_parts) == 0:
        raise ValueError("No source datasets produced any samples (check NUM_SOURCE_SAMPLES_BY_DATASET).")

    source_train_data = source_train_parts[0] if len(source_train_parts) == 1 else ConcatDataset(source_train_parts)
    source_val_data   = source_val_parts[0]   if len(source_val_parts) == 1 else ConcatDataset(source_val_parts)

    # ---- TARGET (single only) ----
    config.DATA.DATA_PATH = config.DATA.TARGET_DATA_PATH

    target_name = config.DATA.TARGET_DATASET_NAME
    if target_name not in TARGET_BUILDERS:
        raise ValueError(f"Unknown target dataset '{target_name}'. Available: {list(TARGET_BUILDERS.keys())}")

    builder_t = TARGET_BUILDERS[target_name]
    target_train_data, target_val_data, _ = builder_t(
        config,
        mode,
        split,
        config.DATA.NUM_TARGET_SAMPLES,
        use_grayscale,
        domain_classification=1,
        mixed=True,
        has_val=has_val,
    )

    # ---- CONCAT SOURCE + TARGET ----
    num_samples_A = len(source_train_data)
    num_samples_B = len(target_train_data)

    weights_A = torch.ones(num_samples_A)
    weights_B = torch.ones(num_samples_B) * (num_samples_A / max(1, num_samples_B))

    train_ds = ConcatDataset([source_train_data, target_train_data])
    val_ds   = ConcatDataset([source_val_data, target_val_data])

    print(f"samples A (source): {num_samples_A}")
    print(f"samples B (target): {num_samples_B}")
    print(f"weight sum A: {torch.sum(weights_A)}")
    print(f"weight sum B: {torch.sum(weights_B)}")

    if upsample_target_domain:
        print("upsampling")
        sampler = WeightedRandomSampler(
            torch.cat([weights_A, weights_B]),
            num_samples=math.floor(torch.sum(weights_A) + torch.sum(weights_B)),
        )
    else:
        print("not upsampling")
        sampler = None

    return train_ds, val_ds, sampler
