# HumanLabel

[简体中文](README.md) | [English](README.en.md)

将 head 视频标注为 **腕部在相机系下的 epose＋拇指/食指距离**。

## 1. 标注链路

**EgoHand → native smoothing → 前后帧裁剪恢复 → epose 导出**

本项目只提供这一条链路。`label` 对每个片段执行完整流程，支持单片段和整批 `episode_*` 目录。

| 项目 | 说明 |
|---|---|
| 推理设备 | NVIDIA GPU / CUDA |
| 输入 | 每段的 `head.mp4`、`timestamps.csv`、`manifest.json` |
| 腕部位姿 | head 光学相机系；米；四元数 `xyzw` |
| 夹爪范围 | `0–0.07 m`，参考 PiperCapture 配置 |
| 遮挡恢复 | 借用前后帧手框，对当前 RGB 推理；恢复帧保留复核标记 |
| 时序边界 | 姿态平滑不跨采集 grid 缺口 |

## 2. 环境配置

运行条件：**Linux、NVIDIA GPU、支持所选 PyTorch CUDA 构建的驱动**，以及 Git。环境管理推荐 Conda（Miniconda / Anaconda），也可使用下文的 venv 方案。
已有验证配置为 **Python 3.13、PyTorch 2.7.1/cu126、RTX 4090 D 24 GB**；这不代表已验证所有 GPU 或最低显存需求。

### 2.1 获取代码并进入模块所在目录

首次获取本仓库：

```bash
git clone https://github.com/alivecat05/PiperX_WS.git
cd PiperX_WS/HumanLabel
```

如果已经克隆，直接进入其中的 `HumanLabel/`，不必再次克隆。
**下文所有命令都在 `HumanLabel/` 执行**，该目录下应能看到 `humanlabel/` 和 `requirements.txt`。
目前按源码目录运行模块；仅安装依赖后，从其他目录执行仍可能提示找不到 `humanlabel`。

### 2.2 创建自己的环境并安装依赖

按顺序完成本节的全部安装、第 2.3 节的模型资源准备，最后在第 2.4 节统一校验。

本项目使用 `opencv-python-headless` 和检测器的官方 `ultralytics-opencv-headless` 发行版，适用于没有桌面的服务器。
安装不需要 root，也不需要安装 OpenCV 的 `libGL` / Qt 图形界面运行库；视频读写和预览 MP4 导出仍然可用。

每位使用者都需要创建自己的环境，仓库不包含开发者的虚拟环境。
推荐使用 Conda；以下命令从新建环境开始，**不要直接在 `(base)` 中安装**：

```bash
conda create -n humanlabel python=3.13 pip -y
conda activate humanlabel
# 忽略 ~/.local 中其他项目的 Python 包，防止串用 OpenCV 等依赖。
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate humanlabel

python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r requirements-egohand.txt
```

如果已经自行创建并安装过这个 Conda 环境，后续只需 `conda activate humanlabel`。
不同 GPU / 驱动需要选择兼容的 PyTorch CUDA 构建。

两个 requirements 已固定 headless 依赖，上述命令会统一安装。检测器的 Python 导入仍是 `from ultralytics import YOLO`。
参见 [OpenCV headless 安装说明](https://pypi.org/project/opencv-python-headless/4.11.0.86/)和 [Ultralytics headless 发行版](https://pypi.org/project/ultralytics-opencv-headless/8.3.242/)。

<details>
<summary>不用 Conda：从零创建 venv（二选一）</summary>

需要预先安装 Python 3.13 及其 `venv` 支持。先创建，再激活：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

然后执行上面从 `python -m pip install --upgrade pip` 开始的依赖安装命令。
`.venv/` 在你的机器上生成，不随 Git 分发；只有创建成功后才会存在 `.venv/bin/activate`。
以后打开新终端时，在 `HumanLabel/` 中重新执行 `source .venv/bin/activate`。

</details>

### 2.3 准备模型资源（首次运行必做）

在刚才激活的环境中执行：

```bash
python -m humanlabel prepare-sources
```

`prepare-sources` 只准备第三方源码，**不会下载模型权重**。
继续执行 [模型资源配置第 2、3 步](docs/SETUP.md#2-权重和-soma-资源)，下载 EgoHand checkpoint、SOMA/MHR 资源及手部检测器，检查目录后再运行标注。
第三方源码、模型权重及 SOMA/MHR 资源统一保存在本仓库的 `model/` 中；Conda 环境由 Conda 管理，venv 则保存在你创建的目录中。这些内容均不提交到 Git。

### 2.4 全部安装完成后统一校验

完成上述依赖安装和模型资源准备后，在同一环境中执行：

```bash
python -m pip check
python -m humanlabel --help
python -c "import sys, cv2; print(sys.executable); print(cv2.__version__, cv2.__file__)"
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA 不可用，请检查 NVIDIA 驱动和 PyTorch 安装'; print(torch.cuda.get_device_name(0))"
```

确认没有依赖冲突、OpenCV 能导入、CUDA 可用，并按资源说明核对模型文件目录后，进入第 3 节运行标注。

## 3. 单片段与批量处理

确认已完成依赖和模型资源安装，进入 `HumanLabel/` 并激活自己的环境。
命令格式为 **`python -m humanlabel label "输入目录" --output "输出目录"`**。
模块名是 `humanlabel`，`label` 是子命令。输入路径使用你自己的实际目录名。
引号内不要额外添加空格：`"/path/to/episodes"` 与 `" /path/to/episodes"` 是不同的路径。

```bash
# 批量处理：输入目录下包含 episode_001/、episode_002/ 等片段。
python -m humanlabel label "/path/to/episodes" --output "/path/to/result"

# 单个片段。
python -m humanlabel label "human_demonstration/episode_004" --output ./result

# 只从已有几何结果重新导出；不重新执行 GPU 推理。
python -m humanlabel export result/human_demonstration
```

输入可以是包含多个 `episode_*` 的父目录，也可以是单个片段目录；每个片段需要 `head.mp4`、`timestamps.csv`、`manifest.json`。
录像不随仓库提供，需要自行准备。`./result` 表示当前目录下的输出文件夹；`/result` 则是系统根目录下的文件夹，需要相应写入权限。

最终结果按 **`输出目录/原母文件夹名/episode_XXX/`** 保存。例如，输入 `human_demonstration/` 或其中的 `episode_004/`，并指定 `--output ./result`：

```text
result/
└── human_demonstration/
    ├── episode_004/          # 完整链路的最终结果
    ├── episode_005/
    ├── summary.json
    └── validation_all.json
```

每个片段只保留裁剪恢复及 epose 导出后的最终结果。已完成片段会直接复用。
运行中的抽帧、检测记录和原生预测临时写入 `.cache/`；本次运行结束或失败时自动清除自己的临时目录，全部任务结束后移除空的 `.cache/`。
模型和上游源码保留在 `model/`，供后续批量处理使用。

## 4. 输出格式

每段保存到 `result/原母文件夹名/episode_XXX/`：

| 文件 | 内容 |
|---|---|
| `epose.csv` | 所有源帧的 17 列腕部位姿与夹爪宽度 |
| `pinch.csv` | 未截断的指尖距离，以及限制后的夹爪宽度 |
| `wrist_epose.npz` | `T_camera_wrist[side,frame,4,4]` |
| `action.csv` / `action_raw.csv` | 双手有效行 / 所有源帧；同样的 17 列格式 |
| `quality.csv` | 检测、恢复来源、有效性、需复核标记 |
| `frame_map.csv` | 有效行到原始帧、grid 及连续片段的映射 |
| `soma.npz` / `manifest.json` | 原生模型参数 / 方法、坐标和配置记录 |
| `preview.mp4` | 网格、关节与腕点叠加预览 |

**CSV：时间戳＋左手 8 维＋右手 8 维**，列顺序与 PiperCapture 一致。

```text
timestamp_unix_s,
left.x,left.y,left.z,left.qx,left.qy,left.qz,left.qw,left.gripper_m,
right.x,right.y,right.z,right.qx,right.qy,right.qz,right.qw,right.gripper_m
```

| 定义 | 约定 |
|---|---|
| 腕部原点 | `joint 0`，不是相机平移参数或两指中点 |
| 相机轴 | `+X` 向右、`+Y` 向下、`+Z` 向前 |
| SOMA 旋转 | 通过官方 FK 转换：`A.T @ R_native @ A` |
| 夹爪数值 | `clip(指尖距离, 0, gripper_max_m)`，不额外缩放 |
| 上限来源 | 优先 `PiperCapture/config.json`；没有时用 `config.example.json`，本机为 `0.07 m` |
| 时间含义 | 当前图像帧的人体腕部；保留原始时间戳 |

PiperCapture 的原始 `action` 是下一 grid 的机器人 TCP；这里只对齐列格式及宽度范围。
深度、尺度和未标定内参属于模型估计；遮挡恢复需按 `quality.csv` 复核。

## 5. 代码目录

```text
humanlabel/
├── cli.py                 # 命令入口与路由
├── export.py              # 相机系 epose、PiperCapture 格式及校验
├── workflows/             # 当前方案：推理、恢复、完整流水线
├── validation.py          # 完整性、时间戳、采集缺口和预览校验
└── utils/                 # 相机、时间戳、裁剪跟踪、视频导出和资源准备
model/                     # EgoHand 权重、SOMA/MHR、检测器和必需的上游源码（不提交 Git）
configs/                   # 模型版本、校验值与 SOMA 参考轴
docs/                      # EgoHand 环境与模型资源说明
patches/                   # 固定上游版本的兼容性补丁
tests/                     # 几何、格式、跟踪与导出测试
```

## 6. 测试与资源许可

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check humanlabel tests
```

权重、SOMA/MHR 及第三方源码适用各自上游许可，见 [模型资源配置](docs/SETUP.md)。
录制数据、`model/`、运行时临时目录、虚拟环境和 `result/` 不上传 GitHub。
