# 模型资源与可选环境

## 当前方案：EgoHand

先完成 [README 的环境安装](../README.md#2-环境配置)。本流程使用独立的手部检测器；不需要额外安装 RT-DETR、MediaPipe 或完整 IsaacCapture 工具链。

### 1. 源码

```bash
python -m humanlabel prepare-sources --models egohand
```

自动准备固定版本的 IsaacCapture、DINOv3，并应用 `patches/IsaacCapture.patch`。
已有上游目录存在用户修改时，脚本不会强制清理或重置。

### 2. 权重和 SOMA 资源

使用上游公开仓库；如服务要求，先按上游说明接受许可并运行 `hf auth login`。

```bash
hf download nvidia/EgoHand-v1 egohand_s060000.ckpt \
  --revision a153616c5192e6a8b09dbd302f682093080c5b83 \
  --local-dir .cache/egohand

hf download nvidia/SOMA-X \
  --revision 75acbe89a5082d2d36f1150bde8fb7d7b8d77f42 \
  --include SOMAHand.npz SOMA_neutral.npz SOMA_template_rig.usda \
    SOMA_procedural_transforms.json MHR/mhr_model_lod1.pt \
    MHR/base_body_lod1.obj MHR/SOMA_wrap_lod1.obj \
  --local-dir .cache/egohand/soma

hf download ThunderVVV/HaWoR external/detector.pt \
  --revision da6335f47f9806308992d5ae1002a4cc5f7252c2 \
  --local-dir .cache/weights
```

### 3. 检查目录

```text
.cache/
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
| 共享手部检测器 | [ThunderVVV/HaWoR](https://huggingface.co/ThunderVVV/HaWoR) |
| 固定版本及 SHA-256 | [configs/model_sources.json](../configs/model_sources.json) |
| SOMA 腕轴及资源校验值 | [configs/soma_wrist_axes.json](../configs/soma_wrist_axes.json) |

> 单目输出使用模型尺度。未提供真实相机标定时，默认 `fx=fy=max(width,height)`、中心主点、零畸变；它们不是实测内参。

## 可选：其他模型对比

这些依赖不属于默认 EgoHand 环境。

| 模型 | 独立环境及资源 |
|---|---|
| WiLoR | Python 3.10；本机验证 torch 2.2.0/cu118；按 WiLoR-mini 上游安装依赖 |
| HaWoR | Python 3.10；本机验证 torch 1.13.0/cu117；按 HaWoR 上游安装依赖 |
| ACE-Ego-Hand | 默认 GPU 环境另装 `requirements-ace.txt`；需要 Wan2.2 5B、VAE、文本编码器、K-free checkpoint |

```bash
python -m humanlabel prepare-sources --models wilor hawor ace
# WiLoR 资源准备：独立环境中执行，遵守 MANO 许可。
/path/to/wilor/bin/python -m humanlabel.utils.assets
# ACE 的额外 Python 依赖。
python -m pip install -r requirements-ace.txt
```

| 模型 | 资源位置 / 上游说明 |
|---|---|
| WiLoR | `.cache/WiLoR-mini`、`.cache/wilor/pretrained_models/`；[上游](https://github.com/warmshao/WiLoR-mini) |
| HaWoR | `.cache/weights/hawor/{model_config.yaml,checkpoints/hawor.ckpt,checkpoints/infiller.pt}`；[上游](https://github.com/ThunderVVV/HaWoR) |
| MANO 模板 | 合法获取的右手模板放 `.cache/mano/MANO_RIGHT.pkl`；[MANO 许可](https://mano.is.tue.mpg.de/license.html) |
| ACE | `.cache/ACE-Ego-Hand/{checkpoints,ckpt,cache}/`；按[上游](https://github.com/ggxxii/ACE-Ego-Hand)准备 K-free 模型及 caption embedding |

可比较方法使用同一批 head RGB；SOMA 和 MANO 的原生参数分别保存，不能直接混用。
`patches/` 保留已验证上游兼容修改；没有复制完整第三方源码或模型到本仓库。

## 本地数据与环境

| 项目 | 约定 |
|---|---|
| 数据入口 | `human demonstration/episode_XXX/`，或显式传入其他目录 |
| 原始录像 | `HumanCapture` 生成；禁止用缺失帧伪造连续采集 |
| 已有本机环境 | `.ace-venv`：当前方案；`.wilor-venv`：WiLoR |
| 自定义 HaWoR 环境 | `--hawor-python /path/to/python` 或 `HAWOR_PYTHON` |
| 本地配置 | `PiperCapture/config.json` 不存在时使用其配置样例 |
| Git 排除项 | `.cache/`、虚拟环境、录像、权重、`result/` |
