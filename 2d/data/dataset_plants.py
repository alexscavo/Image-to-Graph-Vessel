import os
from pathlib import Path
from networkx import nodes
import numpy as np
import random
import imageio
import torch
import pyvista
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from torchvision.transforms import Grayscale
from PIL import Image
from utils.utils import rotate_coordinates, load_graph_from_json
from overlay.prova_overlay_csv_graphs import load_graph_pickle_generic
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

def vtk_lines_to_edges_e2(vtk_lines) -> torch.Tensor:
    """
    Convert VTK 'lines' connectivity to standard edge list (E,2).
    Handles both:
      - simple 2-point lines: [2, i, j, 2, k, l, ...]
      - polylines: [n, p0, p1, ..., p(n-1), ...]  -> edges between consecutive points
    """
    arr = np.asarray(vtk_lines).ravel()
    edges = []
    i = 0
    L = len(arr)
    while i < L:
        n = int(arr[i]); i += 1
        pts = arr[i:i+n].astype(np.int64)
        i += n
        if n >= 2:
            edges.append(np.stack([pts[:-1], pts[1:]], axis=1))
    if len(edges) == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    edges = np.concatenate(edges, axis=0)
    return torch.from_numpy(edges).to(torch.int64)



class Plants2GraphDataLoader(Dataset):
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

        # TODO: check mean and std for plant dataset
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

    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
    
        data = self.data[idx]
        graph_path = data["graph"] 
        
        # Handle .graph or .pickle files
        if data['graph'].endswith('.json'):
            nodes_xy, edges_ix, feat_ids, types = load_graph_from_json(Path(data['graph']))
            nodes = torch.as_tensor(nodes_xy, dtype=torch.float32)
            edges = torch.as_tensor(edges_ix, dtype=torch.int64)
            if edges.ndim == 1:
                edges = edges.view(-1, 2)

        elif data['graph'].endswith('.vtp'):
            vtk_data = pyvista.read(data['graph'])
            pts = np.asarray(vtk_data.points, dtype=np.float32)
            nodes = torch.as_tensor(pts[:, [1, 0]], dtype=torch.float32)  # swap x <-> y
            edges = vtk_lines_to_edges_e2(vtk_data.lines)  # (E,2)

        elif data["graph"].endswith(".pickle"):
            nodes_xy, edges_ix, _ = load_graph_pickle_generic(data["graph"])
            nodes = torch.as_tensor(nodes_xy.astype(np.float32), dtype=torch.float32)
            edges = torch.as_tensor(edges_ix.astype(np.int64), dtype=torch.int64)
            if edges.ndim == 1:
                edges = edges.view(-1, 2)
                
        else:
            raise NotImplementedError("Only .json, .vtp or .pickle graph formats are supported in this implementation.")
        
        # Load segmentation data
        seg_pil = Image.open(data['seg'])
        seg_data = torch.from_numpy(np.array(seg_pil, dtype=np.float32)) / 255.0
        seg_data = seg_data.unsqueeze(0)  # (1,H,W)


        image_pil = Image.open(data['img'])
        image_np = np.array(image_pil)

        if self.use_grayscale:
            image_data = np.array(self.grayscale(image_pil))
            image_data = torch.tensor(image_data, dtype=torch.float).permute(2, 0, 1)
            image_data = image_data / 255.0
            image_data -= 0.5
        else:
            image_data = torch.tensor(image_np, dtype=torch.float).permute(2, 0, 1)
            image_data = image_data / 255.0
            image_data = tvf.normalize(
                image_data.clone().detach(),
                mean=self.mean,
                std=self.std,
            )
            
        H, W = seg_data.shape[-2:]
        nodes_norm = torch.as_tensor(nodes, dtype=torch.float32)    # nodes are already normalized

        # Apply augmentation if enabled
        if self.augment:
            angle = random.randint(0, 3) * 90
            image_data = rotate(image_data, angle)
            seg_data = rotate(seg_data, angle)  
            nodes_norm = rotate_coordinates(nodes_norm, angle)         
        
        return image_data, seg_data - 0.5, nodes_norm, edges, self.domain_classification



def build_plants_network_data(config, mode='train', split=0.95, max_samples=0, use_grayscale=False, domain_classification=-1, mixed=False, has_val=False):
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
    print(f"---- build_plants_network_data: mode {mode}")
    
    if mode == 'train':
        if not mixed:
            img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
            seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
            graph_folder = os.path.join(config.DATA.DATA_PATH, 'graph')
        else:
            img_folder = os.path.join(config.DATA.TARGET_DATA_PATH, 'raw')
            seg_folder = os.path.join(config.DATA.TARGET_DATA_PATH, 'seg')
            graph_folder = os.path.join(config.DATA.TARGET_DATA_PATH, 'graph')

        img_files = []
        graph_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_ + 'data.png'))
            seg_files.append(os.path.join(seg_folder, file_ + 'seg.png'))

            # Check for both .graph and .pickle files
            graph_path = os.path.join(graph_folder, file_ + 'graph.graph')
            pickle_path = os.path.join(graph_folder, file_ + 'graph.pickle')
            if os.path.exists(graph_path):
                graph_files.append(graph_path)
            elif os.path.exists(pickle_path):
                graph_files.append(pickle_path)
                
        
        data_dicts = [
            {"img": img_file, "graph": graph_file, "seg": seg_file}
            for img_file, graph_file, seg_file in zip(img_files, graph_files, seg_files)
        ]
    

        ds = Plants2GraphDataLoader(
            data=data_dicts,
            augment=True,
            use_grayscale=use_grayscale
        )
        return ds

    elif mode == 'test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
        graph_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'graphs')

        img_files = []
        graph_files = []
        seg_files = []

        for file_ in img_folder_train.iterdir():
            
            # file_ contains the full path of the raw image            
            file_name = file_.stem
            
            img_files.append(str(file_))
            seg_files.append(str(seg_folder_train / f"{file_name}_seg.png"))

            # Check for both .graph and .pickle files
            graph_files.append(str(graph_folder_train / f"{file_name}_gt_graph.vtp")) 
            

        data_dicts = [
            {"img": img_file, "graph": graph_file, "seg": seg_file}
            for img_file, graph_file, seg_file in zip(img_files, graph_files, seg_files)
        ]

        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]

        ds = Plants2GraphDataLoader(
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
        seg_folder_train = train_root / 'seg'
        graph_folder_train = train_root / 'graphs'

        img_files_train = []
        graph_files_train = []
        seg_files_train = []
        
        i = 0

        for file_ in img_folder_train.iterdir():
            
            # file_ contains the full path of the raw image            
            file_name = file_.stem
    
            # Construct paths for image and segmentation files
            img_files_train.append(str(file_))
            seg_files_train.append(str(seg_folder_train / f"{file_name}_seg.png"))
        
            # Construct paths for graph files            
            graph_files_train.append(str(graph_folder_train / f"{file_name}_gt_graph.vtp"))    
                
            i += 1
                        
        # print(f'---- iterated over {i} file:')
        print('len of img train files', len(img_files_train))
        print('len of graph train files', len(graph_files_train))
        print('len of seg train files', len(seg_files_train))

        data_dicts_train = [
            {"img": img_file_train, "graph": graph_file_train, "seg": seg_file_train}
            for img_file_train, graph_file_train, seg_file_train in zip(img_files_train, graph_files_train, seg_files_train)
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
            seg_folder_val = val_root / 'seg'
            graph_folder_val = val_root / 'graphs'

            img_files_val = []
            graph_files_val = []
            seg_files_val = []
        
            i = 0

            for file_ in img_folder_val.iterdir():
                
                file_name = file_.stem
                
                # Construct paths for image and segmentation files
                img_files_val.append(str(file_))
                seg_files_val.append(str(seg_folder_val / f"{file_name}_seg.png"))
            
                # Construct paths for graph files                
                graph_files_val.append(str(graph_folder_val / f"{file_name}_gt_graph.vtp"))  
                    
                i += 1
                            
            # print(f'---- iterated over {i} file:')
            print('len of img val files', len(img_files_val))
            print('len of graph val files', len(graph_files_val))
            print('len of seg val files', len(seg_files_val))
            
            data_dicts_val = [
                {"img": img_file_val, "graph": graph_file_val, "seg": seg_file_val}
                for img_file_val, graph_file_val, seg_file_val in zip(img_files_val, graph_files_val, seg_files_val)
            ]
            print(f"---- Number of data_dicts_val: {len(data_dicts_val)}")
            print("---- Data Dicts:")
            pprint(data_dicts_val[:2])
            
            train_files = data_dicts_train
            val_files   = data_dicts_val
            
            if max_samples > 0:
                train_files = train_files[:max_samples]
                val_files = val_files[:round(max_samples * (1 - split))]
                
        else:    
            random.seed(config.DATA.SEED)
            random.shuffle(data_dicts_train)
            train_split = int(split * len(data_dicts_train))
            print('train_split', train_split)
            train_files, val_files = data_dicts_train[:train_split], data_dicts_train[train_split:]

            if max_samples > 0:
                train_files = train_files[:max_samples]
                val_files = val_files[:round(max_samples * (1 - split))]

        train_ds = Plants2GraphDataLoader(
            data=train_files,
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=True
        )
        val_ds = Plants2GraphDataLoader(
            data=val_files,
            use_grayscale=use_grayscale,
            domain_classification=domain_classification,
            augment=False
        )
        
        return train_ds, val_ds, None