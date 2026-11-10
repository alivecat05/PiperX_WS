# EgoHand 模型资源

## 当前方案：EgoHand

先完成 [README 的环境安装](../README.md#2-环境配置)，创建并激活你自己的 Conda 或 venv 环境，安装 Python 依赖。
以下命令均在克隆所得的 `HumanLabel/` 目录及该环境中执行，不依赖开发者机器上的环境或缓存。
本流程使用独立的手部检测器；不需要额外安装 RT-DETR、MediaPipe 或完整 IsaacCapture 工具链。

### 1. 源码

```bash
python -m humanlabel prepare-sources
```

自动准备固定版本的 IsaacCapture、DINOv3，并应用 `patches/IsaacCapture.patch`。
已有上游目录存在用户修改时，脚本不会强制清理或重置。

### 2. 权重和 SOMA 资源

源码准备完成后，仍需单独下载以下全部资源。`hf` 命令随 `requirements-egohand.txt` 中的 `huggingface-hub` 安装。
使用上游公开仓库；如服务要求，先按上游说明接受许可并运行 `hf auth login`。

```bash
hf download nvidia/EgoHand-v1 egohand_s060000.ckpt \
  --revision a153616c5192e6a8b09dbd302f682093080c5b83 \
  --local-dir model/egohand

hf download nvidia/SOMA-X \
  --revision 75acbe89a5082d2d36f1150bde8fb7d7b8d77f42 \
  --include SOMAHand.npz SOMA_neutral.npz SOMA_template_rig.usda \
    SOMA_procedural_transforms.json MHR/mhr_model_lod1.pt \
    MHR/base_body_lod1.obj MHR/SOMA_wrap_lod1.obj \
  --local-dir model/egohand/soma

hf download ThunderVVV/HaWoR external/detector.pt \
  --revision da6335f47f9806308992d5ae1002a4cc5f7252c2 \
  --local-dir model/weights
```

这里只从 `ThunderVVV/HaWoR` 下载 `detector.pt`，用于 EgoHand 的手框检测；无需该仓库的源码、手部重建模型或 infiller 权重。

### 3. 检查目录

```text
model/
├── IsaacCapture/
├── dinov3/
├── egohand/
│   ├── egohand_s060000.ckpt
│   └── soma/
│       ├── SOMAHand.npz
│       ├── SOMA_neutral.npz
│       ├── SOMA_template_rig.usda
│       ├── SOMA_procedural_transforms.json
│       └── MHR/
│           ├── mhr_model_lod1.pt
│           ├── base_body_lod1.obj
│           └── SOMA_wrap_lod1.obj
└── weights/external/detector.pt
```

| 资源 | 来源 |
|---|---|
| EgoHand 源码与许可 | [NVIDIA IsaacCapture](https://github.com/NVIDIA/IsaacCapture) |
| EgoHand 权重 | [nvidia/EgoHand-v1](https://huggingface.co/nvidia/EgoHand-v1) |
| SOMA/MHR 资源 | [nvidia/SOMA-X](https://huggingface.co/nvidia/SOMA-X) |
| 手部检测器权重（仅检测框） | [ThunderVVV/HaWoR](https://huggingface.co/ThunderVVV/HaWoR) |
| 固定版本及 SHA-256 | [configs/model_sources.json](../configs/model_sources.json) |
| SOMA 腕轴及资源校验值 | [configs/soma_wrist_axes.json](../configs/soma_wrist_axes.json) |

> 单目输出使用模型尺度。未提供真实相机标定时，默认 `fx=fy=max(width,height)`、中心主点、零畸变；它们不是实测内参。

## 数据与环境约定

| 项目 | 约定 |
|---|---|
| 数据入口 | `human_demonstration/episode_XXX/`，或显式传入其他目录 |
| 原始录像 | `HumanCapture` 生成；禁止用缺失帧伪造连续采集 |
| 默认环境 | 按 README 自行创建 Conda 环境 `humanlabel`，或选择自行创建的 `.venv`；环境不随仓库分发 |
| 本地配置 | `PiperCapture/config.json` 不存在时使用其配置样例 |
| 临时文件 | 抽帧和原生预测临时写入 `.cache/`，运行结束或失败后自动清理 |
| 最终结果 | `result/原母文件夹名/episode_XXX/` |
| Git 排除项 | `model/`、`.cache/`、虚拟环境、录像、权重、`result/` |
