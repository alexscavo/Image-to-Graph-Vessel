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
from monai.transforms import GaussianSmooth, RandGaussianNoise, ScaleIntensity, Resize, SpatialPad
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


        coordinates = torch.tensor(np.float32(
            np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]
        lines = torch.tensor(np.asarray(
            vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]

        z_pos = None

        if self.gaussian_augment:
            z_pos = random.randint(3, seg_data.shape[-1] - 4) / (seg_data.shape[-1] - 1) if not self.rotate else 0.5

            seg_data = self._project_image_3d(seg_data, z_pos)
            img_data = self._project_image_3d(img_data, z_pos)

            coordinates = F.pad(coordinates, (0, 1), "constant", z_pos)
            coordinates = coordinates[:, [1, 0, 2]]

            if self.rotate:
                if self.continuous:
                    alpha = random.randint(0, 270)
                    beta = random.randint(0, 270)
                    gamma = random.randint(0, 270)
                else:
                    alpha = random.randint(0, 3) * 90
                    beta = random.randint(0, 3) * 90
                    gamma = random.randint(0, 3) * 90

                seg_data = rotate_image(seg_data, alpha, beta, gamma)
                img_data = rotate_image(img_data, alpha, beta, gamma)
                coordinates = rotate_coordinates(coordinates, alpha, beta, gamma)

                seg_data[seg_data >= 0] = 0.5
                seg_data[seg_data < 0] = -0.5

            if self.real_set_augment:
                img_data = seg_data.clone()
                img_data[seg_data > 0] -= 0.5
                img_data[seg_data <= 0] += 0.25

                img_data = self.gaussian_noise(img_data)

                img_data = self.gaussian_smooth(img_data)

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

        for file_ in os.listdir(img_files_train):
            
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
            
        print(vtk_files_train[:5])
        sys.exit()
        

        data_dicts_train = [
            {"img": img_file_train, "vtp": vtk_file_train, "seg": seg_file_train} 
            for img_file_train, vtk_file_train, seg_file_train in zip(img_files_train, vtk_files_train, seg_files_train)
        ]
        
        print(f"---- Number of 20cities data_dicts_train: {len(data_dicts_train)}")
        print("---- Data Dicts:")
        pprint(data_dicts_train[:2])
        
        # ---- VALIDATION FOLDERS ----
        
        val_root = data_root / "val"
        
        img_folder_val = val_root / 'raw'
        seg_folder_val = val_root / 'raw'
        vtk_folder_val = val_root / 'vtp'
        
        img_files_val = []
        vtk_files_val = []
        seg_files_val = []

        for file_ in os.listdir(img_files_val):
            
            if "rgb" in file_ or "sat" in file_:
                continue
            
            base_name, ext = os.path.splitext(file_)
            base_name_region = base_name[:-3]
            region_number, patch_number = base_name.split('_')[1], base_name.split('_')[2]
            
            img_files_val.append(str(img_folder_val / f"region_{region_number}_{patch_number}_sat.png"))
            seg_files_val.append(str(seg_folder_val / f"{base_name}.png"))
            
            vtp_path = str(vtk_folder_val / f"{base_name}_graph_gt.vtp")
            pickle_path = str(vtk_folder_val / f"{base_name}_graph.pickle")
            
            vtk_files_val.append(vtp_path)   
            

        data_dicts_val = [
            {"img": img_file_val, "vtp": vtk_file_val, "seg": seg_file_val} 
            for img_file_val, vtk_file_val, seg_file_val in zip(img_files_val, vtk_files_val, seg_files_val)
        ]
        
        print(f"---- Number of 20cities data_dicts_val: {len(data_dicts_val)}")
        print("---- Data Dicts:")
        pprint(data_dicts_val[:2])
        
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
