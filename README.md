# 🧠 Image to Graph Vessel

Clean setup and run instructions with minimal friction and a bit of visual structure ✨

---

## 📥 Download the Code

Clone the repository:

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

Reload the shell (or open a new terminal), then verify:

```bash
uv --version
```

> uv is installed in `~/.cargo/bin` (Linux) or `%USERPROFILE%\.cargo\bin` (Windows) and added to PATH automatically.

---

### 2️⃣ Create and Activate the Virtual Environment

Move into the folder where you want the environment and run:

```bash
uv venv --python 3.10
```

Activate it:

```bash
source .venv/bin/activate
```

---

### 3️⃣ Install Python Dependencies

```bash
uv pip install -r requirements.txt
```

⚠️ Not all dependencies are listed.  
Run `train.py`, check missing-module errors, and install them manually:

```bash
uv pip install package_name
```

---

### 4️⃣ Install MultiScaleDeformableAttention (2D / 3D)

You will eventually see:

```bash
ModuleNotFoundError: No module named 'MultiScaleDeformableAttention2D'
```

Follow these steps on **Windows**:

#### 4.1 Open
**x64 Native Tools Command Prompt for VS**  
(Visual Studio Build Tools must be installed)

#### 4.2 Activate the Same Virtual Environment

```bash
cd C:\...\Image-to-Graph-Vessel
.\.venv\Scripts\activate
```

#### 4.3 Set Required Variables

```bash
set DISTUTILS_USE_SDK=1
set MSSdk=1
```

#### 4.4 Build 2D Ops

```bash
cd 2d/models/ops
python -m pip install -U pip setuptools wheel ninja
python -m pip install -e . --no-build-isolation -v
```

For **3D**, repeat the same steps in:

```bash
3d/models/ops
```

#### 🔧 Requirements
Make sure you have:
- CUDA installed
- Visual Studio Build Tools with:
  - Desktop development with C++
  - MSVC v143 (VS 2022) or v142 (VS 2019)
  - Windows 10/11 SDK

---

## 🚀 Running the Code

### 🏋️ Training

All arguments are configured inside `train.py`.

**Pretraining requires:**
- `exp_name`
- `config`
- `continuous` (recommended)

**Finetuning requires:**
- `exp_name`
- `config`
- `continuous`
- `resume`
- `restore_state`
- `no_strict_loading`
- `vis_path`

---

### ⚠️ Configuration Checklist

Always double-check your config files:

- `SOURCE_DATA_PATH`
- `TARGET_DATA_PATH`
- `NUM_SOURCE_SAMPLES`, `NUM_TARGET_SAMPLES`
- `DATASET`
- `IMG_SIZE`, `PAD_SIZE`
- `ALPHA_COEFF`
- `EDGE_SAMPLING_MODE` (use "up" to match the paper)
- `UPSAMPLING_TARGET_DOMAIN` (critical for 3D)
- `EDGE_SAMPLING_RATIO` → `0.15`
- `NUM_EDGE_SAMPLES` → `9999`
- All loss weights

---

### 🧪 Testing

Run:

```bash
python test.py
```

Make sure the correct config file is set before execution and all the necessary arguments.

---

Everything should now be ready. Happy experimenting 🧪✨
