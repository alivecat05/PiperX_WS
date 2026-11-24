# HumanLabel

HumanLabel 把 D555 拍摄的人手操作视频转成双臂 Piper X 的 `action.csv`。

## 环境配置

在 `HumanLabel/` 目录运行。首次安装需要 Linux 和 NVIDIA GPU。

```bash
conda create -n humanlabel python=3.13 pip -y
conda activate humanlabel
export PYTHONNOUSERSITE=1
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r requirements-egohand.txt
```

首次安装还需[下载 EgoHand 源码、权重、SOMA 和检测器](docs/SETUP.md)。在 `humanlabel` 环境中安装双臂转换所用的三个包。

```bash
python -m pip install -r requirements-piperx.txt
```

MuJoCo 计算机器人位姿，OSQP 求平滑的关节轨迹，PyYAML 读取外参文件。

## 使用方法

`label-parallel` 生成双臂基座系 `action.csv`。输入可以是一个 episode，也可以是整个数据目录。

```bash
python -m humanlabel label-parallel "/path/to/Human_Data" \
  --intrinsics-json "/path/to/intrinsics.json" \
  --extrinsics-yaml "/path/to/extrinsics.yaml" \
  --workers-per-gpu 2
```

默认跳过已有 `action.csv` 的片段，加 `--relabel` 重算。

默认只写 `action.csv`。加 `--keep-intermediates` 会在原 episode 目录同时保留 `soma.npz`、相机系腕部位姿、夹爪距离、质量表和预览。标注的元数据写到 `humanlabel_manifest.json`。

需要相机系腕部位姿或 SOMA 网格时，用 `label` 导出。

```bash
python -m humanlabel label "/path/to/episode_001" \
  --intrinsics-json "/path/to/intrinsics.json" --output ./result
```

## 验证方法

先准备[回放环境和 STL 网格](docs/SETUP.md#mujoco-回放资源)。输入同一片段的 Action CSV、双臂关节轨迹 NPZ 和 SOMA 归档。

```bash
conda run --no-capture-output -n sim_capture env MUJOCO_GL=disable \
  python -m humanlabel replay-piperx "/path/to/episode_001" \
  --action "/path/to/action.csv" --joint-trace "/path/to/joint_trace.npz" \
  --soma "/path/to/soma.npz" --intrinsics-json "/path/to/intrinsics.json" \
  --extrinsics-yaml "/path/to/extrinsics.yaml" --output "/path/to/replay.mp4"
```

回放检查两臂的法兰位姿、夹爪开口、关节角度和速度，并输出视频和逐帧诊断 CSV。

## 数据格式

**输入** 每个 episode 包含 `head.mp4`、`timestamps.csv` 和 `manifest.json`。内参文件提供 D555 的相机矩阵和畸变，外参文件提供 `T_Head_LeftBase` 和 `T_Head_RightBase`。

**输出** `action.csv` 写在原 episode 目录，每帧一行，共 17 列。

```csv
timestamp_unix_s,left.x,left.y,left.z,left.qx,left.qy,left.qz,left.qw,left.gripper_m,right.x,right.y,right.z,right.qx,right.qy,right.qz,right.qw,right.gripper_m
```

时间戳沿用原视频，单位为秒。左右臂各用自己的基座系。法兰位置和夹爪开口单位为米，四元数顺序为 `xyzw`。

`label` 则在 `result/任务目录/episode_001/` 导出相机系 `epose.csv`、手部模型 `soma.npz` 和预览 `preview.mp4`。
