import os
from pathlib import Path
import numpy as np
import random
from medpy.io import load
import torch
import pyvista
from torch.utils.data import Dataset
from torch.nn.functional import conv2d
from torchvision.transforms import Grayscale
import torch.nn.functional as F
from utils.utils import rotate_image, rotate_coordinates
from monai.transforms import GaussianSmooth, RandGaussianNoise, ScaleIntensity, Resize, SpatialPad, ScaleIntensityRange
from PIL import Image

train_transform = []
val_transform = []


class Sat2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, transform, gaussian_augment=False, rotate=False, continuous=True, size=64, padding=5, real_set_augment=False, growth_range=[1, 3], domain_classification=-1):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform
        self.gaussian_augment = gaussian_augment
        self.rotate = rotate
        self.continuous = continuous

        weights = torch.tensor([[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]])
        self.growth_tensor = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.growth_lower_bound, self.growth_upper_bound = growth_range

        self.resize = Resize(spatial_size=(size-2*padding, size-2*padding))
        self.size = size
        self.padding = padding
        self.real_set_augment = real_set_augment

        self.pad_transform = SpatialPad([size,size,size], value=-0.5)
        self.pad_transform2d = SpatialPad([size, size], value=-0.5)

        self.transform_gray = Grayscale()

        if real_set_augment:
            self.gaussian_noise = RandGaussianNoise(prob=1, std=0.2, mean=0.55)
            self.gaussian_smooth = GaussianSmooth(sigma=1)
        else:
            self.gaussian_noise = RandGaussianNoise(prob=1, std=0.05, mean=0.)
            self.gaussian_smooth = GaussianSmooth(sigma=1.2)

        self.domain_classification = domain_classification

        self.scaling = ScaleIntensityRange(
            a_min=-0.5, a_max=0.5,
            b_min=-0.5, b_max=0.5,
            clip=True
        )

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)
    
    def _project_image_3d(self, sliced_data, z_pos):
        """
        Project a 2D slice into a 3D space.

        sliced_data: torch tensor (C,H,W) in [-0.5, 0.5]
        returns: (C,H,W,D) in [-0.5, 0.5], background = -0.5
        """

        C, H, W = sliced_data.shape

        # Put everything at background value
        projected_data = torch.full(
            (C, H, W, W),  # keep your original D = W convention
            -0.5,
            dtype=sliced_data.dtype,
            device=sliced_data.device,
        )

        z = int(round(z_pos * H))  # keep your original choice (uses H)
        # optional: clamp to avoid edge issues
        z = max(2, min(W - 3, z))

        # Add the slice on a few neighboring planes
        for dz in (-2, -1, 0, 1, 2):
            projected_data[:, :, :, z + dz] = sliced_data

        return projected_data


    def __getitem__(self, idx):
        data = self.data[idx]
        seg_np, _ = load(data["seg"])
        img_np, _ = load(data["img"])
        vtk_data = pyvista.read(data["vtp"])

        # ---- Load seg: [0,1], then binarize, then shift to {-0.5, +0.5} ----
        seg_data = torch.from_numpy(seg_np).float() / 255.0   # H,W (or already H,W)
        seg_data = seg_data.unsqueeze(0)                      # 1,H,W

        # ---- Load image: keep as REAL image, not seg. Grayscale is OK. ----
        # If img_np is (C,H,W), convert to (H,W,C) for PIL then grayscale
        img_np_hwc = np.moveaxis(img_np, 0, -1)               # H,W,C
        img_gray = np.array(Image.fromarray(img_np_hwc).convert("L"))
        img_data = torch.from_numpy(img_gray).float() / 255.0 # H,W in [0,1]
        img_data = img_data.unsqueeze(0)                      # 1,H,W

        # ---- Resize 2D inputs consistently ----
        img_data = self.resize(img_data)
        seg_data = self.resize(seg_data)

        # ---- Threshold seg in 2D, then map to {-0.5, +0.5} BEFORE 3D projection ----
        seg_data = (seg_data >= 0.3).float()  # {0,1}
        seg_data = seg_data - 0.5             # {-0.5, +0.5}

        # ---- Graph data ----
        coordinates = torch.tensor(np.asarray(vtk_data.points), dtype=torch.float32)[:, :2]
        lines = torch.tensor(vtk_data.lines.reshape(-1, 3), dtype=torch.int64)[:, 1:]
        z_pos = None

        if self.gaussian_augment:
            # Choose z (normalized) for 3D embedding
            z_pos = random.randint(3, seg_data.shape[-1] - 4) / (seg_data.shape[-1] - 1) if not self.rotate else 0.5

            # Project to 3D. NOTE:
            # - seg_data is already centered {-0.5,+0.5}
            # - img_data is still [0,1]; _project_image_3d() will center it internally
            seg_data = self._project_image_3d(seg_data, z_pos)
            img_data = self._project_image_3d(img_data, z_pos)

            # Coordinates -> 3D
            coordinates = F.pad(coordinates, (0, 1), "constant", z_pos)

            if self.rotate:
                if self.continuous:
                    alpha = random.randint(0, 270)
                    beta  = random.randint(0, 270)
                    gamma = random.randint(0, 270)
                else:
                    alpha = random.randint(0, 3) * 90
                    beta  = random.randint(0, 3) * 90
                    gamma = random.randint(0, 3) * 90

                seg_data = rotate_image(seg_data, alpha, beta, gamma)
                img_data = rotate_image(img_data, alpha, beta, gamma)
                coordinates = rotate_coordinates(coordinates, alpha, beta, gamma)

                # Re-binarize seg after interpolation artifacts from rotation
                seg_data = torch.where(seg_data >= 0, torch.tensor(0.5, device=seg_data.device), torch.tensor(-0.5, device=seg_data.device))

            # Photometric aug on img only (safe). Keep it mild while debugging.
            img_data = self.gaussian_noise(img_data)

            # IMPORTANT:
            # Do NOT apply self.scaling here if it expects input in [0,1].
            # After _project_image_3d(), img_data is typically in [-0.5, 0.5].
            # So: either skip scaling, or set self.scaling to an identity mapping for [-0.5,0.5].
            # img_data = self.scaling(img_data)  # <-- keep OFF unless it's identity in [-0.5,0.5]

            # Pad 3D volumes with background = -0.5
            img_data = self.pad_transform(img_data)
            seg_data = self.pad_transform(seg_data)

        else:
            # 2D path (if you ever use it)
            img_data = img_data - 0.5  # center to match padding/value convention
            img_data = self.gaussian_noise(img_data)

            # Here scaling can be OK if it's consistent with your chosen range.
            # If self.scaling expects [0,1], don't center before scaling.
            # If you want [-0.5,0.5], use an identity scaling or a fixed range scaler.
            # img_data = self.scaling(img_data)

            img_data = self.pad_transform2d(img_data)
            seg_data = self.pad_transform2d(seg_data)
            coordinates = coordinates[:, [1, 0]]

        coordinates = (coordinates * (self.size - 2 * self.padding) + self.padding) / self.size

        return [img_data], [seg_data], [coordinates], [lines], [z_pos], [self.domain_classification]



def build_road_network_data(config, mode='train', split=0.95, debug=False, gaussian_augment=False, rotate=False, continuous=False, max_samples=0, domain_classification=-1):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    if mode == 'train':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=train_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels") ,
            growth_range=config.DATA.GROWTH_RANGE
        )
        return ds
    elif mode == 'test':
        img_folder = os.path.join(config.DATA.SOURCE_DATA_PATH, 'test/raw')
        seg_folder = os.path.join(config.DATA.SOURCE_DATA_PATH, 'test/seg')
        vtk_folder = os.path.join(config.DATA.SOURCE_DATA_PATH, 'test/vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=val_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels")  ,
            growth_range=config.DATA.GROWTH_RANGE
        )
        return ds
    elif mode == 'split':
        
        data_root = Path(config.DATA.SOURCE_DATA_PATH)
        
        # ---- TRAIN FOLDERS ----
        train_root_path = data_root / "train"
        
        img_folder_train = train_root_path / "raw"
        seg_folder_train = train_root_path / "seg"
        vtk_folder_train = train_root_path / "vtp"
        img_files_train = []
        vtk_files_train = []
        seg_files_train = []

        for file_ in os.listdir(img_folder_train):
            file_ = file_[:-8]
            img_files_train.append(os.path.join(str(img_folder_train), file_+'data.png'))
            vtk_files_train.append(os.path.join(str(vtk_folder_train), file_+'graph.vtp'))
            seg_files_train.append(os.path.join(str(seg_folder_train), file_+'seg.png'))

        data_dicts_train = [
            {"img": img_file_train, "vtp": vtk_file_train, "seg": seg_file_train} for img_file_train, vtk_file_train, seg_file_train in zip(img_files_train, vtk_files_train, seg_files_train)
        ]
        
        # ---- VAL FOLDERS ----
        val_root = data_root / "val"
        
        img_folder_val = val_root / "raw"
        seg_folder_val = val_root / "seg"
        vtk_folder_val = val_root / "vtp"
        img_files_val = []
        vtk_files_val = []
        seg_files_val = []

        for file_ in os.listdir(img_folder_val):
            file_ = file_[:-8]
            img_files_val.append(os.path.join(str(img_folder_val), file_+'data.png'))
            vtk_files_val.append(os.path.join(str(vtk_folder_val), file_+'graph.vtp'))
            seg_files_val.append(os.path.join(str(seg_folder_val), file_+'seg.png'))

        data_dicts_val = [
            {"img": img_file_val, "vtp": vtk_file_val, "seg": seg_file_val} for img_file_val, vtk_file_val, seg_file_val in zip(img_files_val, vtk_files_val, seg_files_val)
        ]
        
        train_files = data_dicts_train
        val_files = data_dicts_val

        random.shuffle(train_files)
        random.shuffle(val_files)
        
        N_train_total = len(train_files)
        N_val_total = len(val_files)

        if max_samples > 0:
            # ratio of how much of training you keep
            r = min(1.0, max_samples / max(1, N_train_total))

            # slice train
            train_keep = min(max_samples, N_train_total)
            train_files = train_files[:train_keep]

            # slice val with the same retention ratio
            val_keep = int(round(N_val_total * r))

            # safety: ensure at least 1 val sample if you want validation to run
            # (set to 0 if you explicitly want to allow "no validation")
            val_keep = max(1, val_keep) if N_val_total > 0 else 0

            val_files = val_files[:val_keep]

        train_ds = Sat2GraphDataLoader(
            data=train_files,
            transform=train_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels") ,
            growth_range=config.DATA.GROWTH_RANGE,
            domain_classification=domain_classification
        )
        val_ds = Sat2GraphDataLoader(
            data=val_files,
            transform=val_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels") ,
            growth_range=config.DATA.GROWTH_RANGE,
            domain_classification=domain_classification
        )
        return train_ds, val_ds, None