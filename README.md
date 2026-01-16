# 🧠 Image to Graph Vessel

Clean setup and run instructions with minimal friction ✨

---

## 📥 Download the Code

```bash
git clone https://github.com/alexscavo/Image-to-Graph-Vessel.git
```

---

## 🧩 Environment Setup

Using **uv** is strongly recommended—it handles dependency conflicts automatically.

### 1️⃣ Install uv

**Windows (PowerShell)**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal, then verify:

```bash
uv --version
```

**Linux (bash / zsh)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell and verify:

```bash
uv --version
```

---

### 2️⃣ Create and Activate the Virtual Environment

```bash
uv venv --python 3.10
source .venv/bin/activate
```

---

### 3️⃣ Install Python Dependencies

```bash
uv pip install -r requirements.txt
```

Some dependencies are missing from `requirements.txt`.  
Run `train.py`, inspect the error, and install them manually:

```bash
uv pip install package_name
```

---

### 4️⃣ Install MultiScaleDeformableAttention (2D / 3D)

If you encounter:

```bash
ModuleNotFoundError: No module named 'MultiScaleDeformableAttention2D'
```

Open **x64 Native Tools Command Prompt for VS**, activate the venv, then:

```bash
set DISTUTILS_USE_SDK=1
set MSSdk=1
```

#### 2D
```bash
cd 2d/models/ops
python -m pip install -U pip setuptools wheel ninja
python -m pip install -e . --no-build-isolation -v
```

#### 3D
Repeat the same steps in:

```bash
3d/models/ops
```

Ensure CUDA and Visual Studio Build Tools are installed.

---

## 🧱 Preprocessing (Dataset Construction)

Before training, datasets must be **constructed and preprocessed**.  
This step is required for **each dataset** (source and target, 2D and/or 3D).

### 1️⃣ Create Dataset Splits

Run `create_splits.py` and set the **total number of patches** to generate.  
The script computes all required statistics and produces `splits.csv`.

```bash
python create_splits.py
```

---

### 2️⃣ Extract Patches

After creating the splits, extract the patches using the appropriate script.

#### 🔹 2D Datasets

Scripts are named:

```text
preprocess_<dataset_name>.py
```

They are located in the same folder as `create_splits.py`.

You must specify:
- the path to the corresponding `splits.csv`
- dataset folders (check names like `raw`, `seg`, `labels`, `vtp`, `graphs`, etc.)
- number of patches to generate
- overlap between patches

🚫 **Do not change output folder names**, as the training pipeline depends on them.

---

#### 🔹 3D Datasets

Patch extractors are located in:

- **Roads (satellite)**  
  ```text
  3d/data/generate_sat_data.py
  ```

- **Vessels (synthetic)**  
  ```text
  3d/data/generate_synth_data.py
  ```

Specify:
- the correct `splits.csv`
- dataset root folders
- number of patches
- overlap

Ensure folder names match the dataset structure and  
**do not modify output folder names**.

📝 **Note (Roads – 2D)**  
For the roads patch extractor, it is **expected behavior** that the **raw images and segmentation masks are saved in the same output folder**.  
This is intentional and required by the downstream loading pipeline.

---

## 🚀 Running the Code

### 🏋️ Training

All arguments are configured in `train.py`.

**Pretraining**
- `exp_name`
- `config`
- `continuous`

**Finetuning**
- `exp_name`
- `config`
- `continuous`
- `resume`
- `restore_state`
- `no_strict_loading`
- `vis_path`

---

### ⚠️ Configuration Checklist

Verify:
- `SOURCE_DATA_PATH`
- `TARGET_DATA_PATH`
- `NUM_SOURCE_SAMPLES`, `NUM_TARGET_SAMPLES`
- `DATASET`
- `IMG_SIZE`, `PAD_SIZE`
- `ALPHA_COEFF`
- `EDGE_SAMPLING_MODE` → "up"
- `UPSAMPLING_TARGET_DOMAIN`
- `EDGE_SAMPLING_RATIO` → `0.15`
- `NUM_EDGE_SAMPLES` → `9999`
- loss weights

---

### 🧪 Testing

```bash
python test.py
```

Ensure the correct config file is selected before running.
