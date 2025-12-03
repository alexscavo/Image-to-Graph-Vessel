import os
from pathlib import Path
from pprint import pprint
import sys
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
from monai.transforms import GaussianSmooth, RandGaussianNoise, ScaleIntensity, Resize, SpatialPad, Zoom
from PIL import Image

train_transform = []
val_transform = []



def clip_segment_to_box3d(p0, p1, x0, y0, z0, x1, y1, z1):
    """
    p0, p1: np.array shape (3,)
    box: [x0, x1] x [y0, y1] x [z0, z1]
    returns None if segment is entirely outside,
            or (c0, c1) clipped segment endpoints (np.array(3,))
    """
    dx, dy, dz = p1 - p0
    p = np.array([-dx, dx, -dy, dy, -dz, dz], dtype=float)
    q = np.array([
        p0[0] - x0, x1 - p0[0],
        p0[1] - y0, y1 - p0[1],
        p0[2] - z0, z1 - p0[2]
    ], dtype=float)

    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u1: 
                    return None
                if t > u0: 
                    u0 = t
            else:
                if t < u0: 
                    return None
                if t < u1: 
                    u1 = t

    c0 = p0 + u0 * np.array([dx, dy, dz])
    c1 = p0 + u1 * np.array([dx, dy, dz])
    return c0, c1


def clip_graph_to_unit_cube(coordinates, lines, eps=1e-6):
    """
    coordinates: [N, >=3] torch tensor (x,y,z, ...)
                 assumed normalized ~[0,1] before rotation; after rotation may go out of [0,1]
    lines      : [M, 2] torch tensor of indices into coordinates

    Returns:
        new_coordinates: [N', >=3] torch tensor, all x,y,z in [0,1]
        new_lines      : [M', 2] torch tensor, valid indices into new_coordinates
    """
    device = coordinates.device

    coords_np = coordinates.detach().cpu().numpy()
    lines_np = lines.detach().cpu().numpy()

    xyz = coords_np[:, :3]
    extra = coords_np[:, 3:] if coords_np.shape[1] > 3 else None

    def inside(p):
        return (-eps <= p[0] <= 1.0 + eps and
                -eps <= p[1] <= 1.0 + eps and
                -eps <= p[2] <= 1.0 + eps)

    def almost_equal(a, b, tol=1e-5):
        return np.linalg.norm(a - b) < tol

    new_xyz = xyz.tolist()
    new_extra = extra.tolist() if extra is not None else None
    new_lines = []

    def add_point(p, src_idx):
        # clamp to [0,1] for safety
        px = float(np.clip(p[0], 0.0, 1.0))
        py = float(np.clip(p[1], 0.0, 1.0))
        pz = float(np.clip(p[2], 0.0, 1.0))
        new_xyz.append([px, py, pz])
        if extra is not None:
            new_extra.append(extra[src_idx].tolist())
        return len(new_xyz) - 1

    for i, j in lines_np:
        p0 = xyz[i]
        p1 = xyz[j]

        clipped = clip_segment_to_box3d(p0, p1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        if clipped is None:
            # no intersection with unit cube
            continue

        c0, c1 = clipped
        p0_in = inside(p0)
        p1_in = inside(p1)

        if p0_in and p1_in:
            # both endpoints inside → keep original edge
            new_lines.append([i, j])

        elif p0_in and not p1_in:
            # p0 in, p1 out → one new boundary point
            border = c1 if almost_equal(c0, p0) else c0
            k = add_point(border, src_idx=j)
            new_lines.append([i, k])

        elif p1_in and not p0_in:
            # p1 in, p0 out
            border = c0 if almost_equal(c1, p1) else c1
            k = add_point(border, src_idx=i)
            new_lines.append([k, j])

        else:
            # both endpoints outside, but the segment crosses the cube
            k0 = add_point(c0, src_idx=i)
            k1 = add_point(c1, src_idx=j)
            if k0 != k1:
                new_lines.append([k0, k1])

    # If nothing intersects the cube:
    if len(new_lines) == 0:
        return (torch.empty((0, coords_np.shape[1]), dtype=torch.float32, device=device),
                torch.empty((0, 2), dtype=torch.int64, device=device))

    # Stack and build full coord array
    new_xyz = np.asarray(new_xyz, dtype=np.float32)
    if extra is not None:
        new_extra = np.asarray(new_extra, dtype=np.float32)
        all_coords_np = np.concatenate([new_xyz, new_extra], axis=1)
    else:
        all_coords_np = new_xyz

    new_lines_np = np.asarray(new_lines, dtype=np.int64)

    # ---- prune unused nodes & renumber indices ----
    used = np.unique(new_lines_np.flatten())

    mapping = -np.ones(all_coords_np.shape[0], dtype=np.int64)
    mapping[used] = np.arange(len(used), dtype=np.int64)

    pruned_coords = all_coords_np[used]
    remapped_lines = mapping[new_lines_np]

    # final clamp x,y,z to [0,1]
    pruned_coords[:, 0] = np.clip(pruned_coords[:, 0], 0.0, 1.0)
    pruned_coords[:, 1] = np.clip(pruned_coords[:, 1], 0.0, 1.0)
    pruned_coords[:, 2] = np.clip(pruned_coords[:, 2], 0.0, 1.0)

    new_coordinates = torch.from_numpy(pruned_coords).to(device)
    new_lines = torch.from_numpy(remapped_lines).to(device)

    return new_coordinates, new_lines


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
            self.gaussian_noise = RandGaussianNoise(prob=1, std=0.4, mean=0.)
            self.gaussian_smooth = GaussianSmooth(sigma=1.2)

        self.domain_classification = domain_classification

        self.scaling = ScaleIntensity(minv=-0.5, maxv=0.5)

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)
    
    def _project_image_3d(self, sliced_data, z_pos):
        """
        Project a 2D slice into a 3D space
        Paramaters:
            sliced_data: 2D image slice. Must be a torch tensor with values in the range [-0.5, 0.5]
        Returns:
            Projected 3D image
        """

        projected_data = torch.zeros(size=(
            sliced_data.shape[0],
            sliced_data.shape[1],
            sliced_data.shape[2],
            sliced_data.shape[-1]
        ))
        cutoff_slice_data = sliced_data.clone()
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) + 1] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) + 2] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) - 1] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) - 2] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * (sliced_data.shape[1]))] += cutoff_slice_data
        projected_data -= 0.5

        return projected_data
    
    def _scale_volume_around_center(self, volume: torch.Tensor, scale: float, mode: str = "bilinear") -> torch.Tensor:
        """
        Uniformly scale a 3D volume around its center, keeping the same voxel grid size.

        volume: (C, H, W, D)
        scale:  scalar > 0
        mode:   "bilinear" for images, "nearest" for labels
        """
        if abs(scale - 1.0) < 1e-6:
            return volume

        # volume: (C, H, W, D) -> (N=1, C, D, H, W) for grid_sample
        vol5 = volume.unsqueeze(0).permute(0, 1, 3, 2, 1)  # (1, C, D, H, W)

        theta = torch.zeros(1, 3, 4, dtype=volume.dtype, device=volume.device)
        theta[0, 0, 0] = scale
        theta[0, 1, 1] = scale
        theta[0, 2, 2] = scale
        # translation components remain 0 (scale around center in [-1,1] space)

        grid = F.affine_grid(theta, size=vol5.size(), align_corners=True)
        vol5_scaled = F.grid_sample(
            vol5,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )

        # back to (C, H, W, D)
        volume_scaled = vol5_scaled.permute(0, 1, 4, 3, 2).squeeze(0)
        return volume_scaled

    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.data[idx]
        seg_data, _ = load(data['seg'])
        img_data, _ = load(data['img'])

        vtk_data = pyvista.read(data['vtp'])

        seg_data = (torch.from_numpy(seg_data).float()/255.).unsqueeze(0)

        img_data = np.moveaxis(img_data, 0, -1)
        img_data = torch.from_numpy(np.array(Image.fromarray(img_data).convert('L')) / 255.).unsqueeze(0)
        
        # invert image values
        # img_data = 1.0 - img_data

        img_data = self.resize(img_data)
        seg_data = self.resize(seg_data)
        seg_data[seg_data < 0.3] = 0
        seg_data[seg_data >= 0.3] = 1

        if self.real_set_augment:
            growth_factor = random.randint(self.growth_lower_bound, self.growth_upper_bound)

            x = seg_data.unsqueeze(0).unsqueeze(0).float()
            for i in range(growth_factor):
                x = conv2d(x, weight=self.growth_tensor, padding=1)

            x[x > 1] = 1
            x[x < 1] = 0
            seg_data = x - 0.5

            seg_data = self.resize(seg_data[0])
            img_data = seg_data


        orig_coordinates = torch.tensor(np.float32(
            np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]
        orig_lines = torch.tensor(np.asarray(
            vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]

        z_pos = None

        if self.gaussian_augment:
            z_pos = random.randint(3, seg_data.shape[-1] - 4) / (seg_data.shape[-1] - 1) if not self.rotate else 0.5

            seg_data = self._project_image_3d(seg_data, z_pos)
            img_data = self._project_image_3d(img_data, z_pos)

            orig_coordinates = F.pad(orig_coordinates, (0, 1), "constant", z_pos)

            if self.rotate:
                if self.continuous:
                    alpha = random.randint(0, 360)
                    beta = random.randint(0, 360)
                    gamma = random.randint(0, 360)
                else:
                    alpha = random.randint(0, 3) * 90
                    beta = random.randint(0, 3) * 90
                    gamma = random.randint(0, 3) * 90
                
                candidate_coordinates = rotate_coordinates(orig_coordinates, alpha, beta, gamma)
                
                candidate_coordinates, candidate_lines = clip_graph_to_unit_cube(candidate_coordinates, orig_lines)
                
                if candidate_coordinates.numel() > 0:
                    coordinates = candidate_coordinates
                    lines = candidate_lines
                else:
                    alpha = random.randint(0, 3) * 90
                    beta = random.randint(0, 3) * 90
                    gamma = random.randint(0, 3) * 90
                    coordinates = rotate_coordinates(orig_coordinates, alpha, beta, gamma)
                    lines = orig_lines
                
                seg_data = rotate_image(seg_data, alpha, beta, gamma)
                img_data = rotate_image(img_data, alpha, beta, gamma)

                seg_data[seg_data >= 0] = 0.5
                seg_data[seg_data < 0] = -0.5

            '''if self.real_set_augment:
                img_data = seg_data.clone()
                img_data[seg_data > 0] -= 0.5
                img_data[seg_data <= 0] += 0.25

                img_data = self.gaussian_noise(img_data)

                img_data = self.gaussian_smooth(img_data)'''

            img_data = self.scaling(img_data)

            # Add padding to the borders
            img_data = self.pad_transform(img_data)
            seg_data = self.pad_transform(seg_data)
        else:
            img_data = self.gaussian_noise(img_data)
            """image_data[image_data > 0.5] = 0.5
            image_data = self.gaussian_smooth(image_data)"""
            img_data = self.scaling(img_data)
            img_data = self.pad_transform2d(img_data)
            seg_data = self.pad_transform2d(seg_data)
            coordinates = orig_coordinates
            lines = orig_lines
            # coordinates = coordinates[:, [1, 0]]
        
        coordinates = (coordinates * (self.size - 2 * self.padding) + self.padding) / self.size
        
        if coordinates.min() < 0.0 or coordinates.max() > 1.0:
            print('-'*50)
            print('ROAD COORDINATES:')
            print("⚠️ Coords outside [0,1]!")
        
        if len(lines) == 0:
            print("⚠️ NO EDGES IN GT!")   


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
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
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
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} 
            for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
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
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            if "rgb" in file_ or "sat" in file_:
                continue
            
            base_name, ext = os.path.splitext(file_)
            base_name_region = base_name[:-3]
            region_number, patch_number = base_name.split('_')[1], base_name.split('_')[2]
            
            img_files.append(os.path.join(img_folder, 'region_' + region_number + '_' + patch_number +'_sat.png'))
            seg_files.append(os.path.join(seg_folder, base_name + '.png'))

            # Check for both .vtp and .pickle files
            vtp_path = os.path.join(vtk_folder, 'region_' + region_number + '_' + patch_number + 'gt_graph.vtp')
            pickle_path = os.path.join(vtk_folder, 'region_' + region_number + '_' + patch_number + '_gt_graph.pickle')
            vtk_files.append(vtp_path)
            
            
        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} 
            for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        
        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]
            
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
        
        train_root = data_root / "train"
        
        img_folder_train = train_root / 'raw'
        seg_folder_train = train_root / 'raw'
        vtk_folder_train = train_root / 'vtp'
        
        img_files_train = []
        vtk_files_train = []
        seg_files_train = []

        for file_ in os.listdir(img_folder_train):
            
            if "rgb" in file_ or "sat" in file_:
                continue
            
            base_name, ext = os.path.splitext(file_)
            base_name_region = base_name[:-3]
            region_number, patch_number = base_name.split('_')[1], base_name.split('_')[2]
            
            img_files_train.append(str(img_folder_train / f"region_{region_number}_{patch_number}_sat.png"))
            seg_files_train.append(str(seg_folder_train / f"{base_name}.png"))
            
            vtp_path = str(vtk_folder_train / f"{base_name}_graph.vtp")
            pickle_path = str(vtk_folder_train / f"{base_name}_graph.pickle")
            
            vtk_files_train.append(vtp_path)       

        data_dicts_train = [
            {"img": img_file_train, "vtp": vtk_file_train, "seg": seg_file_train} 
            for img_file_train, vtk_file_train, seg_file_train in zip(img_files_train, vtk_files_train, seg_files_train)
        ]
        
        # print(f"---- Number of 20cities data_dicts_train: {len(data_dicts_train)}")
        # print("---- Data Dicts:")
        # pprint(data_dicts_train[:2])
        
        # ---- VALIDATION FOLDERS ----
        
        val_root = data_root / "val"
        
        img_folder_val = val_root / 'raw'
        seg_folder_val = val_root / 'raw'
        vtk_folder_val = val_root / 'vtp'
        
        img_files_val = []
        vtk_files_val = []
        seg_files_val = []

        for file_ in os.listdir(img_folder_val):
            
            if "rgb" in file_ or "sat" in file_:
                continue
            
            base_name, ext = os.path.splitext(file_)
            base_name_region = base_name[:-3]
            region_number, patch_number = base_name.split('_')[1], base_name.split('_')[2]
            
            img_files_val.append(str(img_folder_val / f"region_{region_number}_{patch_number}_sat.png"))
            seg_files_val.append(str(seg_folder_val / f"{base_name}.png"))
            
            vtp_path = str(vtk_folder_val / f"{base_name}_graph.vtp")
            pickle_path = str(vtk_folder_val / f"{base_name}_graph.pickle")
            
            vtk_files_val.append(vtp_path)   
            

        data_dicts_val = [
            {"img": img_file_val, "vtp": vtk_file_val, "seg": seg_file_val} 
            for img_file_val, vtk_file_val, seg_file_val in zip(img_files_val, vtk_files_val, seg_files_val)
        ]
        
        # print(f"---- Number of 20cities data_dicts_val: {len(data_dicts_val)}")
        # print("---- Data Dicts:")
        # pprint(data_dicts_val[:2])
        
        train_files = data_dicts_train
        val_files   = data_dicts_val
        
        if max_samples > 0:
            train_files = train_files[:max_samples]
            val_files = val_files[:round(max_samples * (1 - split))]


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
