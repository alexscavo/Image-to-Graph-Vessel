import csv
import os
from pprint import pprint
import re
import sys
import numpy as np
import random
import pandas as pd
import torch
import pyvista
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as tvf
from utils.utils import rotate_coordinates
from torchvision.transforms.functional import rotate
from pathlib import Path



class Vessel2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, augment, max_nodes, domain_classification=-1):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.augment = augment
        self.max_nodes = max_nodes

        self.domain_classification = domain_classification

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
        nodes = pd.read_csv(data['nodes'], sep=",", index_col="id")
        edges = pd.read_csv(data['edges'], sep=",", index_col="id")
        image_data = Image.open(data['img'])
        seg_data = Image.open(data['seg'])

        image_data = image_data.convert('RGB')
        image_data = torch.tensor(
            np.array(image_data), dtype=torch.float).permute(2, 0, 1)
        image_data = image_data/255.0

        seg_data = torch.tensor(
            np.array(seg_data), dtype=torch.float).unsqueeze(0)
        seg_data = seg_data / 255.0

        # swap x,y for the coordinates
        nodes = torch.tensor(
            nodes.to_numpy()[:, [1, 0]].astype(np.float32)
        )
        edges = torch.tensor(edges.to_numpy()[:, :2].astype(int))

        if self.augment:
            angle = random.randint(0, 3) * 90
            image_data = rotate(image_data, angle)
            seg_data = rotate(seg_data, angle)
            nodes = rotate_coordinates(nodes, angle)

        return image_data-0.5, seg_data-0.5, nodes, edges, self.domain_classification


def build_real_vessel_network_data(config, mode='train', split=0.8, max_samples=0, use_grayscale=False, domain_classification=-1, mixed=False, has_val=False):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    if mode == 'train':
        
        img_folder = os.path.join(config.DATA.DATA, 'raw')
        lbl_folder = os.path.join(config.DATA.DATA, 'labels')
        graphs_folder = os.path.join(config.DATA.DATA, 'graphs')

        img_files = []
        seg_files = []
        node_files = []
        edge_files = []

        for file_ in os.listdir(img_folder):
            if not file_.endswith(".png"):
                continue

            base = os.path.splitext(file_)[0][:-4]  
            graph_subdir = os.path.join(graphs_folder, base)
            
            img_files.append(os.path.join(img_folder,  file_))
            seg_files.append(os.path.join(lbl_folder,  base + ".png"))
            node_files.append(os.path.join(graph_subdir, "nodes.csv"))
            edge_files.append(os.path.join(graph_subdir, "edges.csv"))

        data_dicts = [
            {"img": img_file, "seg": seg_file, "nodes": node_file, "edges": edge_file} for
            img_file, seg_file, node_file, edge_file in zip(img_files, seg_files, node_files, edge_files)
        ]
        
        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]
            
        ds = Vessel2GraphDataLoader(
            data=data_dicts,
            augment=True,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN,
            domain_classification=domain_classification
        )
        return ds
    
    elif mode == 'test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        graphs_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'graphs')
        lbl_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'labels')
        
        img_files = []
        node_files, edge_files = [], []
        seg_files = []


        for file_ in os.listdir(img_folder):
            file_ = file_[:-4]  # keep your original slicing logic
            graph_subdir = Path(graphs_folder, file_)
            
            img_files.append(os.path.join(img_folder, file_+'.png'))
            seg_files.append(os.path.join(lbl_folder, file_+'.png'))
            edge_files.append(os.path.join(graph_subdir, "edges.csv"))
            node_files.append(os.path.join(graph_subdir, "nodes.csv"))
            

        data_dicts = [
            {"img": img_file, "seg": seg_file, "nodes": node_file, "edges": edge_file} for
            img_file, seg_file, node_file, edge_file in zip(img_files, seg_files, node_files, edge_files)
        ]
        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]

        ds = Vessel2GraphDataLoader(
            data=data_dicts,
            augment=False,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        return ds
    
    elif mode == 'split':   
        
        root = Path(config.DATA.TARGET_DATA_PATH)  
                
        # ----- TRAIN FOLDERS -----
        train_root = root / "train"
        
        if config.DATA.TRAIN_WITH_LABELS:
            img_folder_train = train_root / 'labels'
        else:
            img_folder_train = train_root / 'raw'

        graph_folder_train = train_root /'graphs'
        lbl_folder_train = train_root /'labels'
        
        print('-'*50)
        print(graph_folder_train)
        print(lbl_folder_train)
        print(img_folder_train)

        img_files_train, node_files_train, edge_files_train, lbl_files_train = [], [], [], []

        for file_ in os.listdir(img_folder_train):
            file_ = file_[:-4]  # keep your original slicing logic
            graph_subdir = Path(graph_folder_train, file_)
            
            img_files_train.append(str(img_folder_train / (file_ + '.png')))
            lbl_files_train.append(str(lbl_folder_train / (file_ + '.png')))
            node_files_train.append(str(graph_subdir / 'nodes.csv'))
            edge_files_train.append(str(graph_subdir / 'edges.csv'))
            
            

        data_dicts_train = [
            {"img": img_file, "seg": seg_file, "nodes": node_file, "edges": edge_file} for
            img_file, seg_file, node_file, edge_file in zip(img_files_train, lbl_files_train, node_files_train, edge_files_train)
        ]

        # if the validation set is already provided
        if has_val:
            val_root = root / "val"
    
            # ----- VAL FOLDERS -----
            if config.DATA.TRAIN_WITH_LABELS:
                img_folder_val = val_root / 'labels'
            else:
                img_folder_val = val_root / 'raw'

            graph_folder_val = val_root / 'graphs'
            lbl_folder_val = val_root / 'labels'

            img_files_val, nodes_files_val, edges_files_val, lbl_files_val = [], [], [], []

            for file_ in os.listdir(img_folder_val):
                file_ = file_[:-4]  # keep your original slicing logic
                graph_subdir = Path(graph_folder_val, file_)
                
                img_files_val.append(str(img_folder_val / (file_ + '.png')))
                lbl_files_val.append(str(lbl_folder_val / (file_ + '.png')))
                nodes_files_val.append(str(graph_subdir / 'nodes.csv'))
                edges_files_val.append(str(graph_subdir / 'edges.csv'))

            data_dicts_val = [
                {"img": img_file, "seg": seg_file, "nodes": node_file, "edges": edge_file} for
                img_file, seg_file, node_file, edge_file in zip(img_files_val, lbl_files_val, nodes_files_val, edges_files_val)
            ]

            train_files = data_dicts_train
            val_files   = data_dicts_val
            
            if max_samples > 0:
                train_files = train_files[:max_samples]
                val_files = val_files[:round(max_samples * (1 - split))]
        
        else:
            random.seed(config.DATA.SEED)
            random.shuffle(data_dicts_train)
            train_split = int(split * len(data_dicts_train))
            train_files = data_dicts_train[:train_split]
            val_files   = data_dicts_train[train_split:]

            if max_samples > 0:
                train_files = train_files[:max_samples]
                val_files   = val_files[:round(max_samples * (1 - split))]

        train_ds = Vessel2GraphDataLoader(
            data=train_files,
            augment=True,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN,
            domain_classification=domain_classification,
        )
        val_ds = Vessel2GraphDataLoader(
            data=val_files,
            augment=False,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN,
            domain_classification=domain_classification,
        )
        return train_ds, val_ds, None
