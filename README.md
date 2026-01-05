# README

## Overview
This repository provides a complete framework for training and evaluating **Image-to-Graph Transformers (RelationFormer)** in both **2D** and **3D**.  
The model takes an input image or volume and predicts a graph structure composed of:

- **Nodes** – keypoints or anatomical/structural junctions  
- **Edges** – connectivity between nodes  

The architecture is based on the ideas introduced in **“Cross-Domain and Cross-Dimension Learning for Image-to-Graph Transformers (WACV 2025)”**, extended with:

- Mixed-domain training  
- Domain-adversarial learning  
- Optional segmentation supervision  
- 2D and 3D unified pipelines  
- Extensive visualization and debugging tools  

Both pipelines are self-contained and include all modules needed for data loading, model definition, training, and evaluation.

## Key Features

### Image-to-Graph Transformer Architecture
- Multi-scale CNN backbone  
- Deformable DETR encoder/decoder  
- Object tokens for node prediction  
- Relation tokens for edge prediction    

### Mixed-Domain Learning
Optional adversarial modules include:

- **Backbone domain discriminator**  
- **Instance-level domain discriminator**  
- **Grad-reverse scheduling**  

Useful when combining data from:

- Synthetic + real worlds  
- Road networks + retinal vessels  
- 2D OCTA + 3D angiography  
- Multiple imaging modalities  


## Installation
```
conda create -n relationformer python=3.10
conda activate relationformer
pip install -r requirements.txt
```

## Running Experiments

### 2D
```
cd 2d
python train.py --config configs/config_2d.yaml --exp_name my_exp
```

### 3D
```
cd 3d
python train.py --config configs/config_3d.yaml --exp_name my_exp
```

### Resume Training
```
python train.py --resume path/to/checkpoint.pt --restore_state
```
