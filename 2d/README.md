# 2D Img-to-Graph Prediction 🕸️ ➜ 📈

Transform 2D images into graph structures with style, order, and the
occasional questionable life choice.

This repository guides you through the full pipeline: - dataset
preparation
- split creation
- patch extraction
- training
- evaluation
- inference

Everything is designed to be intuitive, reproducible, and just chaotic
enough to keep things interesting.

------------------------------------------------------------------------

# 🚀 Quickstart Overview

1.  Create dataset splits
2.  Extract patches
3.  Train
4.  Evaluate
5.  Predict on new images

If you follow the numbered sections below, you're basically unstoppable.

------------------------------------------------------------------------

# 0️⃣ Installation

I suggest to use uv for the package installation

``` bash
git clone https://github.com/your-user/your-repo.git
cd your-repo

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

# 1️⃣ Create Train / Val / Test Splits 🧪

Script location:

    2d/preprocess/create_splits.py

Modify the bottom of the script:

``` python
args = parser.parse_args([
    '--root', "C:/Users/Utente/Desktop/tesi/datasets/octa-synth",
    '--ratio',
    '--train_patches_num', '480',
    '--val_patches_num', '220',
    '--test_patches_num', '2000',
])

main(args)
```

Run:

``` bash
cd 2d/preprocess
python create_split.py
```

------------------------------------------------------------------------

# 2️⃣ Extract Dataset Patches 🩹📦

Extractors:

- 2d/preprocess/patch_extractor_20cities.py
- 2d/preprocess/patch_extractor_octasynth.py
- 


------------------------------------------------------------------------

# 3️⃣ Train the Model 🧠

``` bash
python 2d/train_2d.py   --data_root path/to/dataset   --splits_file path/to/splits.json   --patches_root path/to/output_patches   --epochs 100   --batch_size 8   --lr 1e-4   --out_dir runs/exp_001
```

------------------------------------------------------------------------

# 4️⃣ Evaluate the Model 📊

``` bash
python 2d/eval_2d.py   --data_root path/to/dataset   --splits_file path/to/splits.json   --patches_root path/to/output_patches   --checkpoint runs/exp_001/best.ckpt
```

------------------------------------------------------------------------

# 5️⃣ Predict on New Images 🔮

``` bash
python 2d/predict_2d.py   --checkpoint runs/exp_001/best.ckpt   --input_dir path/to/new/images   --output_dir predictions/
```

------------------------------------------------------------------------

# 🗂 Project Structure

    .
    ├─ 2d/
    │  ├─ preprocess/
    │  │  ├─ create_split.py
    │  │  ├─ patch_extractor_20cities.py
    │  │  ├─ patch_extractor_octasynth.py
    │  │  └─ ...
    │  ├─ train_2d.py
    │  ├─ eval_2d.py
    │  ├─ predict_2d.py
    │  └─ models/
    ├─ configs/
    ├─ runs/
    ├─ datasets/
    └─ README.md

------------------------------------------------------------------------

# 🧯 Troubleshooting

-   Wrong dataset path → split script fails\
-   Wrong extractor → patch folder empty\
-   Missing patches → training refuses to start

------------------------------------------------------------------------

# 📚 Citation

``` text
@article{yourname2025img2graph,
  title   = {2D Image-to-Graph Prediction for Something Very Important},
  author  = {Your Name and Someone Else},
  journal = {Some Journal},
  year    = {2025}
}
```
