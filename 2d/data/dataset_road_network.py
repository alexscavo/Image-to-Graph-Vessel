import os
from pathlib import Path
import numpy as np
import random
import imageio
import torch
import pyvista
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from torchvision.transforms import Grayscale
from PIL import Image
from overlay.prova_overlay_csv_graphs import load_graph_pickle_generic
from utils.utils import rotate_coordinates
from torchvision.transforms.functional import rotate
import pickle
from pprint import pprint
import sys

def normalize_nodes(nodes_xy, seg_np):
    """
    Convert pixel coords into [0,1] relative coords.
    nodes_xy: (N,2) in pixels (x,y)
    seg_np: (H,W) numpy array, used to get image size
    """
    H, W = seg_np.shape
    nodes_norm = nodes_xy.copy().astype(np.float32)
    nodes_norm[:, 0] /= float(W)
    nodes_norm[:, 1] /= float(H)
    return nodes_norm


class Sat2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, augment, use_grayscale=False, domain_classification=-1):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.augment = augment

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.domain_classification = domain_classification

        self.use_grayscale = use_grayscale
        self.grayscale = Grayscale(num_output_channels=3)

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)

    # def __getitem__(self, idx):
    #     """[summary]

    #     Args:
    #         idx ([type]): [description]

    #     Returns:
    #         [type]: [description]
    #     """
    #     data = self.data[idx]
    #     vtk_data = pyvista.read(data['vtp'])
    #     raw_seg_data = Image.open(data['seg'])

    #     seg_data = np.array(raw_seg_data)
    #     seg_data = np.array(seg_data)/np.max(seg_data)
    #     seg_data = torch.tensor(seg_data, dtype=torch.int).unsqueeze(0)

    #     image_data = Image.open(data['img'])

    #     if self.use_grayscale:
    #         image_data = np.array(self.grayscale(image_data))
    #         image_data = torch.tensor(
    #             image_data, dtype=torch.float).permute(2, 0, 1)
    #         image_data = image_data / 255.0
    #         image_data -= 0.5
    #     else:
    #         image_data = np.array(image_data)
    #         image_data = torch.tensor(
    #             image_data, dtype=torch.float).permute(2, 0, 1)
    #         image_data = image_data / 255.0
    #         image_data = tvf.normalize(image_data.clone().detach(), mean=self.mean, std=self.std)

    #     nodes = torch.tensor(np.float32(
    #         np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]
    #     lines = torch.tensor(np.asarray(
    #         vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)

    #     if self.augment:
    #         angle = random.randint(0, 3) * 90
    #         image_data = rotate(image_data, angle)
    #         seg_data = rotate(seg_data, angle)
    #         nodes = rotate_coordinates(nodes, angle)

    #     return image_data, seg_data-0.5, nodes, lines[:, 1:], self.domain_classification
    
    
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
    
        data = self.data[idx]
        graph_path = data["vtp"] 
        name = os.path.basename(graph_path)

        # Handle .vtp or .pickle files
        if data['vtp'].endswith('.vtp'):
            vtk_data = pyvista.read(data['vtp'])
            nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]
            lines = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]
        elif data["vtp"].endswith(".pickle"):
            nodes_xy, edges_ix, _ = load_graph_pickle_generic(data["vtp"])  # <— your working function

            # If your pickles store (y,x), swap once here:
            # nodes_xy = nodes_xy[:, [1, 0]]
            # nodes_xy[:, 1] = 128 - 1 - nodes_xy[:, 1]

            # Convert to torch
            nodes = torch.from_numpy(nodes_xy.astype(np.float32))
            lines = torch.from_numpy(edges_ix.astype(np.int64))
            if lines.ndim == 1:
                lines = lines.view(-1, 2)

        nodes = nodes[:, [1, 0]]

        # Load segmentation data
        raw_seg_data = Image.open(data['seg'])
        seg_data = np.array(raw_seg_data)
        seg_data = np.array(seg_data) / np.max(seg_data)
        seg_data = torch.tensor(seg_data, dtype=torch.int).unsqueeze(0)

        # Load image data
        image_data = Image.open(data['img'])
        if self.use_grayscale:
            image_data = np.array(self.grayscale(image_data))
            image_data = torch.tensor(image_data, dtype=torch.float).permute(2, 0, 1)
            image_data = image_data / 255.0
            image_data -= 0.5
        else:
            image_data = np.array(image_data)
            image_data = torch.tensor(image_data, dtype=torch.float).permute(2, 0, 1)
            image_data = image_data / 255.0
            image_data = tvf.normalize(image_data.clone().detach(), mean=self.mean, std=self.std)
            
        H, W = seg_data.shape[-2:]   # (H,W) from segmentation
        nodes_norm = nodes.clone().float()
        # nodes_norm[:, 0] /= float(W)
        # nodes_norm[:, 1] /= float(H)
        
        # Apply augmentation if enabled
        if self.augment:
            angle = random.randint(0, 3) * 90
            image_data = rotate(image_data, angle)
            seg_data = rotate(seg_data, angle)  
            nodes_norm = rotate_coordinates(nodes_norm, angle)      

         
        
        return image_data, seg_data - 0.5, nodes_norm, lines, self.domain_classification

def build_road_network_data(config, mode='train', split=0.95, max_samples=0, use_grayscale=False, domain_classification=-1, mixed=False, has_val=False):
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
        dds = Sat2GraphDataLoader(
            data=data_dicts,
            augment=True,
            use_grayscale=use_grayscale
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
            use_grayscale=use_grayscale,
            augment=False
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
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=True
        )
        val_ds = Sat2GraphDataLoader(
            data=val_files,
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=False
        )
        return train_ds, val_ds, None

# def build_road_network_data(config, mode='train', split=0.95, max_samples=0, use_grayscale=False, domain_classification=-1, mixed=False, has_val=False):
    """Build road network dataset.

    Args:
        config: Configuration object.
        mode (str): 'train', 'test', or 'split'.
        split (float): Train/validation split ratio.
        max_samples (int): Maximum number of samples to load.
        use_grayscale (bool): Whether to use grayscale images.
        domain_classification (int): Domain classification (-1 for none, 0 for source, 1 for target).
        mixed (bool): Whether to use mixed datasets.

    Returns:
        Dataset or train/validation datasets.
    """
    
    print(f"---- build_road_network_data: mode {mode}")
    
    
    if mode == 'train':
        if not mixed:
            img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
            seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
            vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        else:
            img_folder = os.path.join(config.DATA.TARGET_DATA_PATH, 'raw')
            seg_folder = os.path.join(config.DATA.TARGET_DATA_PATH, 'seg')
            vtk_folder = os.path.join(config.DATA.TARGET_DATA_PATH, 'vtp')

        img_files = []
        vtk_files = []
        seg_files = []
        
        

        for file_ in os.listdir(img_folder):
            
            
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_ + 'data.png'))
            seg_files.append(os.path.join(seg_folder, file_ + 'seg.png'))

            # Check for both .vtp and .pickle files
            vtp_path = os.path.join(vtk_folder, file_ + 'graph.vtp')
            pickle_path = os.path.join(vtk_folder, file_ + 'graph.pickle')
            if os.path.exists(vtp_path):
                vtk_files.append(vtp_path)
            elif os.path.exists(pickle_path):
                vtk_files.append(pickle_path)
                
        
        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file}
            for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
    

        ds = Sat2GraphDataLoader(
            data=data_dicts,
            augment=True,
            use_grayscale=use_grayscale
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
            vtp_path = os.path.join(vtk_folder, '_graph_gt.vtp')
            pickle_path = os.path.join(vtk_folder, 'region_' + region_number + '_' + patch_number + '_gt_graph.pickle')
            
            # if os.path.exists(vtp_path):
            #     vtk_files.append(vtp_path)
            # elif os.path.exists(pickle_path):
            #     vtk_files.append(pickle_path)

            vtk_files.append(pickle_path)


        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file}
            for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]

        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]

        ds = Sat2GraphDataLoader(
            data=data_dicts,
            use_grayscale=use_grayscale,
            augment=False
        )
        return ds

    elif mode == 'split':
        
        data_root = Path(config.DATA.SOURCE_DATA_PATH)
        target_root = Path(config.DATA.TARGET_DATA_PATH)
        
        
        # ----- TRAIN FOLDERS -----
        
        if not mixed:
            train_root = data_root / "train"
        else:
            train_root = target_root / "train"
            
        img_folder_train = train_root / 'raw'
        seg_folder_train = train_root / 'raw'
        vtk_folder_train = train_root / 'vtp'

        img_files_train = []
        vtk_files_train = []
        seg_files_train = []
        
        i = 0

        for file_ in os.listdir(img_folder_train):
            # print('------------------------', file_, '-----------', file_.split('.'))
            
            if "rgb" in file_ or "sat" in file_:
                continue
            
            base_name, ext = os.path.splitext(file_)
            base_name_region = base_name[:-3]
            region_number, patch_number = base_name.split('_')[1], base_name.split('_')[2]
    
            # Construct paths for image and segmentation files
            img_files_train.append(str(img_folder_train / f"region_{region_number}_{patch_number}_sat.png"))
            seg_files_train.append(str(seg_folder_train / f"{base_name}.png"))
        
            # Construct paths for graph files
            vtp_path = str(vtk_folder_train / f"{base_name}_graph_gt.vtp")
            pickle_path = str(vtk_folder_train / f"{base_name}_graph.pickle")
            
            vtk_files_train.append(pickle_path)    
                
            i += 1
                        
        # print(f'---- iterated over {i} file:')
        print('len of img train files', len(img_files_train))
        print('len of vtk train files', len(vtk_files_train))
        print('len of seg train files', len(seg_files_train))
        

        data_dicts_train = [
            {"img": img_file_train, "vtp": vtk_file_train, "seg": seg_file_train}
            for img_file_train, vtk_file_train, seg_file_train in zip(img_files_train, vtk_files_train, seg_files_train)
        ]
        print(f"---- Number of data_dicts_train: {len(data_dicts_train)}")
        print("---- Data Dicts:")
        pprint(data_dicts_train[:2])
        
        if has_val:
            
            if not mixed:
                val_root = data_root / "val"
            else:
                val_root = target_root / "val"
                
            img_folder_val = val_root / 'raw'
            seg_folder_val = val_root / 'raw'
            vtk_folder_val = val_root / 'vtp'

            img_files_val = []
            vtk_files_val = []
            seg_files_val = []
        
            i = 0

            for file_ in os.listdir(img_folder_val):
                # print('------------------------', file_, '-----------', file_.split('.'))
                
                if "rgb" in file_ or "sat" in file_:
                    continue
                
                base_name, ext = os.path.splitext(file_)
                base_name_region = base_name[:-3]
                region_number, patch_number = base_name.split('_')[1], base_name.split('_')[2]
        
                # Construct paths for image and segmentation files
                img_files_val.append(str(img_folder_val / f"region_{region_number}_{patch_number}_sat.png"))
                seg_files_val.append(str(seg_folder_val / f"{base_name}.png"))
            
                # Construct paths for graph files
                vtp_path = str(vtk_folder_val / f"{base_name}_graph_gt.vtp")
                pickle_path = str(vtk_folder_val / f"{base_name}_graph.pickle")
                
                vtk_files_val.append(pickle_path)    
                    
                i += 1
                            
            # print(f'---- iterated over {i} file:')
            print('len of img val files', len(img_files_val))
            print('len of vtk val files', len(vtk_files_val))
            print('len of seg val files', len(seg_files_val))
            

            data_dicts_val = [
                {"img": img_file_val, "vtp": vtk_file_val, "seg": seg_file_val}
                for img_file_val, vtk_file_val, seg_file_val in zip(img_files_val, vtk_files_val, seg_files_val)
            ]
            print(f"---- Number of data_dicts_val: {len(data_dicts_val)}")
            print("---- Data Dicts:")
            pprint(data_dicts_val[:2])
            
            train_files = data_dicts_train
            val_files   = data_dicts_val

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
                
        else:    
            random.seed(config.DATA.SEED)
            random.shuffle(data_dicts_train)
            train_split = int(split * len(data_dicts_train))
            print('train_split', train_split)
            train_files, val_files = data_dicts_train[:train_split], data_dicts_train[train_split:]

            if max_samples > 0:
                train_files = train_files[:max_samples]
                val_files = val_files[:round(max_samples * (1 - split))]

        

        train_ds = Sat2GraphDataLoader(
            data=train_files,
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=True
        )
        val_ds = Sat2GraphDataLoader(
            data=val_files,
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=False
        )
        
        return train_ds, val_ds, None