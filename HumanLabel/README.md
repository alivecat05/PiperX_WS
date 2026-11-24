# HumanLabel

[English](README.en.md)

**D555 视频 → EgoHand 腕部 epose 和 SOMA → 左右基座系 → 虎口和指尖姿态优化 → `action.csv`**

## 批量标注

在 `HumanLabel/` 下运行。每个 episode 需要 `head.mp4`、`timestamps.csv` 和 `manifest.json`。

```bash
conda activate humanlabel
python -m pip install -r requirements-piperx.txt

python -m humanlabel label-parallel "/path/to/Human_Data" \
  --intrinsics-json "/path/to/intrinsics.json" \
  --extrinsics-yaml "/path/to/extrinsics.yaml" \
  --workers-per-gpu 2
```

安装命令只需执行一次。它安装 MuJoCo 来计算法兰位姿和检查关节限位，安装 OSQP 来求平滑的关节轨迹，安装 PyYAML 来读取外参。

每个 episode 只写一个 `action.csv`。每帧一行，保存双臂各自基座系下的法兰位姿和夹爪开口。单位为米，四元数顺序为 `xyzw`。默认跳过已有 CSV，加 `--relabel` 可重新生成。

## 导出相机系 epose 和 SOMA

```bash
python -m humanlabel label "/path/to/episode_001" \
  --intrinsics-json "/path/to/intrinsics.json" --output ./result
```

在 `result/任务目录/episode_001/` 查看 `epose.csv`、`soma.npz` 和 `preview.mp4`。

## MuJoCo 回放

[首次下载回放网格](docs/SETUP.md#mujoco-回放资源)。传入同一片段的 Action CSV、双臂关节轨迹和 SOMA。

```bash
conda run --no-capture-output -n sim_capture env MUJOCO_GL=disable \
  python -m humanlabel replay-piperx "/path/to/episode_001" \
  --action "/path/to/action.csv" --joint-trace "/path/to/joint_trace.npz" \
  --soma "/path/to/soma.npz" --intrinsics-json "/path/to/intrinsics.json" \
  --extrinsics-yaml "/path/to/extrinsics.yaml" --output "/path/to/replay.mp4"
```

输出正中视角视频、逐帧诊断 CSV 和截图。

## 从零安装

需要 Linux 和 NVIDIA GPU。下面是已验证的 Python 3.13 和 PyTorch CUDA 12.6 配置。

```bash
conda create -n humanlabel python=3.13 pip -y
conda activate humanlabel
export PYTHONNOUSERSITE=1
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r requirements-egohand.txt
```

继续按[模型资源配置](docs/SETUP.md)准备 EgoHand 源码、权重、SOMA 和手部检测器。
