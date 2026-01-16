from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

from utils.utils import load_graph_from_json


# ----------------------------
# 2) Rasterize edges into a mask
# ----------------------------
@dataclass
class RasterizeConfig:
    line_width_px: int = 3                 # base stroke width for drawing edges
    thickening_method: str = "none"        # "none" | "sat_conv"
    sat_conv_passes: int = 4               # matches generate_sat_data.py (1 + 3 extra passes)
    return_soft: bool = False              # if True, also return soft distance map
    soft_sigma_px: float = 6.0             # controls softness if return_soft is True


def rasterize_graph_mask(
    image_size_wh: Tuple[int, int],
    nodes_xy: np.ndarray,
    edges_ix: np.ndarray,
    cfg: RasterizeConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Args:
      image_size_wh: (W,H)
      nodes_xy: (N,2) x,y
      edges_ix: (E,2) node indices

    Returns:
      mask_u8: (H,W) uint8 in {0,255}
      soft_f32: (H,W) float32 in [0,1] (or None if cfg.return_soft=False)
    """
    W, H = image_size_wh

    # draw lines with PIL
    canvas = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(canvas)

    for a, b in edges_ix:
        x1, y1 = nodes_xy[a]
        x2, y2 = nodes_xy[b]
        draw.line(
            [(float(x1), float(y1)), (float(x2), float(y2))],
            fill=255,
            width=max(1, int(cfg.line_width_px)),
        )

    mask = np.array(canvas, dtype=np.uint8)

    # Thickening: match the road pipeline thickening (iterated 3x3 conv + clamp)
    if cfg.thickening_method.lower() == "sat_conv":
        mask = thicken_like_generate_sat_data(mask, passes_total=cfg.sat_conv_passes)
    elif cfg.thickening_method.lower() in ("none", ""):
        pass
    else:
        raise ValueError(f"Unknown thickening_method='{cfg.thickening_method}'. Use 'none' or 'sat_conv'.")

    soft = None
    if cfg.return_soft:
        soft = mask_to_soft(mask, sigma_px=cfg.soft_sigma_px)

    return mask, soft


# ----------------------------
# 3) Thickening + soft map helpers
# ----------------------------
def thicken_like_generate_sat_data(mask_u8: np.ndarray, passes_total: int = 4) -> np.ndarray:
    """
    Reproduces the road thickening from generate_sat_data.py:
      x = conv2d(mask, ones3x3, padding=1) repeated passes_total times,
      clamp to 1, then *255.

    Args:
      mask_u8: (H,W) uint8 in 0..255 (binary recommended)
      passes_total: 4 matches their implementation (1 + 3 extra passes)

    Returns:
      thick_u8: (H,W) uint8 in 0..255
    """
    if mask_u8.ndim != 2:
        raise ValueError(f"Expected (H,W) mask, got shape {mask_u8.shape}")

    # Import locally so this script can still run for "none" mode without torch installed.
    import torch
    from torch.nn.functional import conv2d

    inp = torch.from_numpy(mask_u8).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1,1,H,W)
    w = torch.ones((1, 1, 3, 3), dtype=inp.dtype, device=inp.device)

    x = inp
    for _ in range(int(passes_total)):
        x = conv2d(x, weight=w, padding=1)

    x = x.clamp(max=1.0) * 255.0
    return x[0, 0].byte().cpu().numpy()


def mask_to_soft(mask_u8: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Convert a binary-ish mask to a soft map in [0,1] based on distance transform.
    soft = exp(-d^2 / (2*sigma^2))
    """
    bin01 = (mask_u8 > 0).astype(np.uint8)

    try:
        import cv2  # type: ignore
        dist = cv2.distanceTransform(1 - bin01, distanceType=cv2.DIST_L2, maskSize=3)
    except Exception:
        try:
            from scipy.ndimage import distance_transform_edt  # type: ignore
            dist = distance_transform_edt(bin01 == 0).astype(np.float32)
        except Exception as e:
            raise RuntimeError("Need either cv2 or scipy for distance transform.") from e

    dist = dist.astype(np.float32)
    sigma = max(1e-6, float(sigma_px))
    soft = np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32)
    soft[bin01 > 0] = 1.0
    return soft


# ----------------------------
# 4) End-to-end utility for one sample
# ----------------------------
def build_graph_derived_inputs(
    image_path: Path,
    graph_json_path: Path,
    out_dir: Path,
    cfg: RasterizeConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image only to get spatial size (W,H)
    img = Image.open(image_path)
    W, H = img.size

    nodes_xy, edges_ix = load_graph_from_json(graph_json_path)

    mask_u8, soft_f32 = rasterize_graph_mask((W, H), nodes_xy, edges_ix, cfg)

    stem = image_path.stem

    mask_path = out_dir / f"{stem}_seg.png"
    Image.fromarray(mask_u8).save(mask_path)

    if soft_f32 is not None and args.use_soft_f32:
        soft_u8 = (np.clip(soft_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        soft_path = out_dir / f"{stem}_graphsoft_seg.png"
        Image.fromarray(soft_u8).save(soft_path)
        
        
def main(args):
    
    raw_path = Path(args.dataset_path) / "raw"
    graph_path = Path(args.dataset_path) / "graphs"
    
    for file in tqdm(list(raw_path.iterdir())):
        if not file.name.endswith(".jpeg") and not file.name.endswith(".jpg"):
            continue
        
        img_path = file
        json_path = graph_path / f"{file.stem}_annotation.json"
        out = Path(args.dataset_path) / "seg"

        cfg = RasterizeConfig(
            line_width_px=3,
            thickening_method="sat_conv",  # <- comparable to roads thickening
            sat_conv_passes=4,             # <- same as generate_sat_data.py
            return_soft=False,
        )
    
        build_graph_derived_inputs(img_path, json_path, out, cfg)
    


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the dataset")
    parser.add_argument("--use_soft_f32", action="store_true", help="use soft mask: A soft distance map (0-255) with gradual falloff from the graph edges...") 
    
    args = parser.parse_args([
        '--dataset_path', '/data/scavone/plants_3d2cut'
    ])
    
    main(args)

    
