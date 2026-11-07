# HumanLabel

将 head 视频标注为 **腕部在相机系下的 epose＋拇指/食指距离**。

## 1. 当前使用的方案

**EgoHand → native smoothing → 前后帧裁剪恢复 → epose 导出**

| 项目 | 说明 |
|---|---|
| 推理设备 | NVIDIA GPU / CUDA |
| 输入 | 每段的 `head.mp4`、`timestamps.csv`、`manifest.json` |
| 腕部位姿 | head 光学相机系；米；四元数 `xyzw` |
| 夹爪范围 | `0–0.07 m`，参考 PiperCapture 配置 |
| 遮挡恢复 | 借用前后帧手框，对当前 RGB 推理；恢复帧保留复核标记 |
| 时序边界 | 姿态平滑不跨采集 grid 缺口 |

## 2. 环境配置

本机已验证：**Linux、Python 3.13、PyTorch 2.7.1/cu126、RTX 4090 D 24 GB**。
以下命令在 `HumanLabel/` 执行；默认流程不需要 MediaPipe、OpenVINO 或 HaWoR 环境。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -r requirements-egohand.txt

# 检查 GPU，再准备固定版本源码。
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m humanlabel prepare-sources --models egohand
```

**还需模型权重和 SOMA 资源**。下载地址、版本及目录见 [模型资源配置](docs/SETUP.md)。
源码、权重与运行环境均保留在 `.cache/` / 虚拟环境中，不提交到 Git。

> 本机继续使用已有 `.ace-venv/bin/python` 即可，无需重复安装。不同 GPU/驱动需安装兼容的 PyTorch CUDA 构建。

## 3. 运行当前方案

```bash
# 全部视频：默认读取 human demonstration/，结果写入 result/。
python -m humanlabel label

# 指定数据及输出目录。
python -m humanlabel label /path/to/episodes --output /path/to/result

# 只从已有几何结果重新导出；不重新执行 GPU 推理。
python -m humanlabel export result/best
```

已完成片段会复用。`result/best/` 指向最终的 EgoHand 增强结果。

## 4. 输出格式

每段保存到 `result/best/episode_XXX/`：

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
├── backends/              # 可选对比模型适配
├── utils/                 # 相机、时间戳、跟踪、几何与资源工具
└── visualization/         # 算法对比、视频合成、重绘与可选优化
configs/                   # 模型版本、校验值与 SOMA 参考轴
docs/                      # 环境/资源细节及算法选型记录
patches/                   # 固定上游版本的兼容性补丁
tests/                     # 几何、格式、跟踪与导出测试
```

## 6. 其他方法与对比视频

| 方法 | 用途 |
|---|---|
| WiLoR / MANO | 单帧基线 |
| HaWoR＋infiller | 时序估计与学习补全 |
| ACE-Ego-Hand K-free | 不依赖检测框的视频手部恢复 |
| ACE＋本地时序拟合 | 可选 CUDA 优化；不是官方 Dyn-HaMR |

```bash
# 对已有输出生成完整对比视频，无需加载模型权重。
python -m humanlabel compare --render-only --enhanced

# 重新执行全部对比模型；每个模型的独立环境见 docs/SETUP.md。
python -m humanlabel compare --enhanced \
  --wilor-python /path/to/wilor/bin/python \
  --hawor-python /path/to/hawor/bin/python
```

对比产物位于 `result/comparison/` 或 `result/comparison_all/`。
[选型记录](docs/model_comparison.json)包含可用性、来源和引用数快照；无可用代码或必要权重的方法不纳入实测。
连续性指标不等于三维精度排名。

## 7. 测试与资源许可

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check humanlabel tests
```

权重、SOMA/MANO 及第三方源码适用各自上游许可，见 [模型资源配置](docs/SETUP.md)。
录制数据、模型、缓存、虚拟环境和 `result/` 不上传 GitHub。
