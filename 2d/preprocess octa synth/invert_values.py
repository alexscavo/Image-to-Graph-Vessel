import argparse
from pathlib import Path
from monai.transforms import Compose
from monai.data import Dataset
from monai.transforms import (
    LoadImaged, Lambdad, SaveImaged, EnsureChannelFirstd
)
from tqdm import tqdm

def main(args):
    
    root_path = Path(args.root)
    
    image_path = root_path / "raw"
    label_path = root_path / "labels"
    
    out_img_path = root_path / "inverted_images"
    # out_lbl_path = root_path / "output/inverted_labels"
    out_img_path.mkdir(exist_ok=True, parents=True)
    # out_lbl_path.mkdir(exist_ok=True, parents=True)
    
    image_files = sorted((image_path).glob("*"))
    label_files = sorted((label_path).glob("*"))
    
    files = [{"image": str(i), "label": str(l)} for i, l in zip(image_files, label_files)]
    
    transforms = Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(
                keys=["image", "label"],
                func=lambda x: 255 - x
            )
    ])
    
    dataset = Dataset(files, transforms)
    
    for i, item in enumerate(tqdm(dataset, total=len(dataset), desc="Processing items")):
        image = item["image"]
        label = item["label"]

        # Save images
        SaveImaged(
            keys="image",
            output_dir=str(out_img_path),
            separate_folder=False,
            print_log=False,
            output_ext=".png",
            output_postfix="",
        )(item)

        # SaveImaged(
        #     keys="label",
        #     output_dir=str(out_lbl_path),
        #     separate_folder=False,
        #     print_log=False,
        #     output_ext=".png",
        #     output_postfix="",
        # )(item)
    
   
   
    
    
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    
    args = ap.parse_args(['--root', '/data/scavone/octa-synth-packed_bigger_inverted',
                            ]) 
    
    main(args)