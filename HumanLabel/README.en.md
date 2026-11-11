# HumanLabel

[简体中文](README.md) | [English](README.en.md)

Annotate head-camera video with **wrist epose in the camera frame and thumb–index fingertip distance**.

## 1. Annotation pipeline

**EgoHand → native smoothing → temporal-context crop recovery → epose export**

This is the project's only pipeline. `label` runs the complete process for each episode and supports both a single episode and a batch of `episode_*` directories.

| Item | Details |
|---|---|
| Inference device | NVIDIA GPU / CUDA |
| Input | `head.mp4`, `timestamps.csv`, and `manifest.json` per episode |
| Wrist pose | Head optical camera frame; meters; `xyzw` quaternions |
| Gripper range | `0–0.07 m`, following PiperCapture configuration |
| Occlusion recovery | Borrow hand boxes from surrounding frames to infer on the current RGB frame; recovered frames retain review flags |
| Temporal boundaries | Pose smoothing does not cross capture-grid gaps |

## 2. Environment setup

Requirements: **Linux, an NVIDIA GPU, a driver compatible with the selected PyTorch CUDA build**, and Git. Conda (Miniconda / Anaconda) is recommended; a venv alternative is provided below.
The verified configuration is **Python 3.13, PyTorch 2.7.1/cu126, and an RTX 4090 D with 24 GB VRAM**. Other GPUs and the minimum VRAM requirement have not been verified.

### 2.1 Get the code and enter the module directory

For a fresh clone:

```bash
git clone https://github.com/alivecat05/PiperX_WS.git
cd PiperX_WS/HumanLabel
```

If you already cloned the repository, enter its `HumanLabel/` directory directly.
**Run every command below from `HumanLabel/`**, where `humanlabel/` and `requirements.txt` are present.
The module currently runs from the source directory. Installing dependencies alone does not make `humanlabel` available from other working directories.

### 2.2 Create your own environment and install dependencies

Complete all installations in this section and the model resource preparation in section 2.3, then perform the final checks in section 2.4.

The project uses `opencv-python-headless` and the detector's official `ultralytics-opencv-headless` distribution for servers without a desktop.
Installation requires neither root nor OpenCV's `libGL` / Qt GUI runtime libraries. Video reading, writing, and preview MP4 export remain available.

Each user must create an environment on their own computer; the developer's virtual environment is not included.
Conda is recommended. The following commands start with a new environment; **do not install directly into `(base)`**:

```bash
conda create -n humanlabel python=3.13 pip -y
conda activate humanlabel
# Ignore other projects' Python packages in ~/.local to avoid mixing OpenCV and other dependencies.
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate humanlabel

python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r requirements-egohand.txt
```

Once you have created and installed this Conda environment, later sessions only require `conda activate humanlabel`.
Choose a compatible PyTorch CUDA build for your GPU and driver.

Both requirements files pin the headless dependencies, which the command above installs together. The detector still imports as `from ultralytics import YOLO`.
See the [OpenCV headless installation instructions](https://pypi.org/project/opencv-python-headless/4.11.0.86/) and the [Ultralytics headless distribution](https://pypi.org/project/ultralytics-opencv-headless/8.3.242/).

<details>
<summary>Without Conda: create a new venv instead</summary>

First install Python 3.13 with `venv` support. Create the environment before activating it:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Then run the dependency installation commands above, starting with `python -m pip install --upgrade pip`.
`.venv/` is created on your computer and is not distributed through Git. `.venv/bin/activate` exists only after creation succeeds.
In a new terminal, run `source .venv/bin/activate` again from `HumanLabel/`.

</details>

### 2.3 Prepare model resources (required before first use)

In the environment you just activated, run:

```bash
python -m humanlabel prepare-sources
```

`prepare-sources` prepares third-party source code only; **it does not download model weights**.
Continue with [steps 2 and 3 of the model resource setup guide (Chinese)](docs/SETUP.md#2-权重和-soma-资源) to download the EgoHand checkpoint, SOMA/MHR resources, and hand detector. Check the directory layout before annotation.
Third-party sources, model weights, and SOMA/MHR resources all reside in this module's `model/` directory. Conda manages its own environments; a venv stays in the directory you created. None of these resources are committed to Git.

### 2.4 Check the environment after all installation steps

After installing dependencies and preparing model resources, run these checks in the same environment:

```bash
python -m pip check
python -m humanlabel --help
python -c "import sys, cv2; print(sys.executable); print(cv2.__version__, cv2.__file__)"
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable; check the NVIDIA driver and PyTorch installation'; print(torch.cuda.get_device_name(0))"
```

Confirm that dependencies have no conflicts, OpenCV imports, CUDA is available, and the model files match the resource guide before running annotation in section 3.

## 3. Single-episode and batch processing

After installing dependencies and model resources, enter `HumanLabel/` and activate your environment.
The command format is **`python -m humanlabel label "input directory" --output "output directory"`**.
The module is `humanlabel`; `label` is its subcommand. Use the actual input directory on your computer.
Do not add extra spaces inside quotes: `"/path/to/episodes"` and `" /path/to/episodes"` are different paths.

```bash
# Batch: the input directory contains episode_001/, episode_002/, etc.
python -m humanlabel label "/path/to/episodes" --output "/path/to/result"

# Single episode.
python -m humanlabel label "human_demonstration/episode_004" --output ./result

# Re-export existing geometry without running GPU inference again.
python -m humanlabel export result/human_demonstration
```

The input may be a parent directory containing multiple `episode_*` directories or a single episode directory. Each episode needs `head.mp4`, `timestamps.csv`, and `manifest.json`.
Recordings are not included in the repository; supply your own. `./result` is relative to the current directory; `/result` is at the filesystem root and requires permission to write there.

Final results are saved as **`output directory/original parent folder name/episode_XXX/`**. For example, use either `human_demonstration/` or its `episode_004/` subdirectory as input with `--output ./result`:

```text
result/
└── human_demonstration/
    ├── episode_004/          # Final result of the complete pipeline
    ├── episode_005/
    ├── summary.json
    └── validation_all.json
```

Each episode retains only the final result after crop recovery and epose export. Completed episodes are reused directly.
Extracted frames, detection records, and native predictions are temporarily written to `.cache/`. Each run removes its own temporary directory on completion or failure; the empty `.cache/` directory is removed when all jobs finish.
Models and upstream sources remain in `model/` for future batch processing.

## 4. Output format

Each episode is saved to `result/original parent folder name/episode_XXX/`:

| File | Contents |
|---|---|
| `epose.csv` | 17 columns of wrist poses and gripper widths for all source frames |
| `pinch.csv` | Unclipped fingertip distances and clipped gripper widths |
| `wrist_epose.npz` | `T_camera_wrist[side,frame,4,4]` |
| `action.csv` / `action_raw.csv` | Rows where both hands are valid / all source frames, in the same 17-column format |
| `quality.csv` | Detection, recovery source, validity, and review flags |
| `frame_map.csv` | Mapping of valid rows to original frames, grid indices, and contiguous segments |
| `soma.npz` / `manifest.json` | Native model parameters / method, coordinates, and configuration records |
| `preview.mp4` | Preview with mesh, joint, and wrist-point overlays |

**CSV: timestamp + eight dimensions for the left hand + eight for the right hand**, with the same column order as PiperCapture.

```text
timestamp_unix_s,
left.x,left.y,left.z,left.qx,left.qy,left.qz,left.qw,left.gripper_m,
right.x,right.y,right.z,right.qx,right.qy,right.qz,right.qw,right.gripper_m
```

| Definition | Convention |
|---|---|
| Wrist origin | `joint 0`, rather than the camera translation parameter or the midpoint between fingertips |
| Camera axes | `+X` right, `+Y` down, `+Z` forward |
| SOMA rotation | Converted through official FK: `A.T @ R_native @ A` |
| Gripper value | `clip(fingertip distance, 0, gripper_max_m)`, without extra scaling |
| Upper limit | Prefer `PiperCapture/config.json`; fall back to `config.example.json`. The current hardware configuration uses `0.07 m` |
| Timestamp meaning | Human wrist at the current image frame; original timestamps are preserved |

PiperCapture's raw `action` represents the robot TCP at the next grid time. Only the column format and gripper-width range are aligned here.
Depth, scale, and uncalibrated intrinsics are model estimates. Review recovered occlusions using `quality.csv`.

## 5. Code layout

```text
humanlabel/
├── cli.py                 # Command entry point and routing
├── export.py              # Camera-frame epose, PiperCapture format, and checks
├── workflows/             # Inference, recovery, and the complete pipeline
├── validation.py          # Completeness, timestamps, capture gaps, and preview checks
└── utils/                 # Cameras, timestamps, crop tracking, video export, and resource preparation
model/                     # EgoHand weights, SOMA/MHR, detector, and required upstream sources (excluded from Git)
configs/                   # Model versions, checksums, and SOMA reference axes
docs/                      # EgoHand environment and model resource instructions
patches/                   # Compatibility patches for pinned upstream versions
tests/                     # Geometry, format, tracking, and export tests
```

## 6. Tests and resource licenses

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check humanlabel tests
```

Weights, SOMA/MHR, and third-party sources are subject to their respective upstream licenses; see the [model resource setup guide (Chinese)](docs/SETUP.md).
Recordings, `model/`, runtime temporary directories, virtual environments, and `result/` are not uploaded to GitHub.
