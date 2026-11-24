# 三相机外参标定

`read_intrinsics.py` 读内参，`calibrate.py` 求外参，`project_tcp.py` 导出验证视频，
`camera_action_to_base.py` 将相机系 action 转到各臂 Base 系。
固定使用 OpenCV **Park**，没有多算法选择、随机重试、FK 或预览分支。

## 1. 读三路 RGB 内参

先关闭占用相机的采集程序，在安装了 RealSense SDK Python 包的环境中运行：

```powershell
python calibration/read_intrinsics.py --config PiperX_Teleop/config.json
```

也可将 `--config` 指向录制目录中的 `manifest.json`，以使用录制时的配置。
脚本按配置中的序列号、RGB 分辨率、帧率和 DDS 设置读取三台相机，保存
`calibration/intrinsics.json`。缺少 SDK 时安装 `pyrealsense2`；D555 的网络连接
需使用与现有采集程序相同、支持 DDS 的 SDK 环境。

## 2. 标定

```powershell
python -m pip install -r calibration/requirements.txt
python calibration/calibrate.py --data path/to/episode_001 --intrinsics calibration/intrinsics.json --board 7 5 0.04 0.03
```

`--board` 依次是横向方格数、纵向方格数、方格边长、marker 边长，长度单位 **米**。
默认字典 `DICT_4X4_50`，可用 `--dictionary` 修改。

输入是现有录制目录中的 `head.mp4`、`left_wrist.mp4`、`right_wrist.mp4` 和
`samples.jsonl`。无需导出或转换录制文件。
**这里使用 JSONL 的实测 `observation.state`，不是遥操作的目标 `action`**。
仓库的 `action.csv` 不是同一时刻的实测反馈，不能直接替代。

流程：每五帧检测一次 ChArUco → PnP → Park 手眼 → 固定板位姿融合 → 两个最终矩阵。
保留基本质量检查：图像与位姿时间差 ≤20 ms、重投影 RMS <2 pixel、姿态不退化；
每臂至少 20 组有效姿态，20% 留出验证；板位姿一致性 RMS ≤15 mm / 3 degree。
默认参数写在代码里，不再增加一堆命令行开关。
旋转融合使用 SO(3) 平均。支持常规 OpenCV 和 RealSense Brown 畸变模型。

输出默认 `calibration/extrinsics.yaml`，也可用 `--output` 指定。
文件仅包含 `T_Head_LeftBase`、`T_Head_RightBase` 两个 4×4 矩阵：

```python
T_Head_LeftEEF = T_Head_LeftBase @ T_LeftBase_LeftEEF
T_Head_RightEEF = T_Head_RightBase @ T_RightBase_RightEEF
```

`T_A_B` 将 B 系坐标转换到 A 系，平移单位米。EEF 是录制时配置的 TCP。
两臂均验证通过后才写输出；报错时已有输出文件不会更新。

## 3. 导出末端投影视频

```powershell
python calibration/project_tcp.py --data path/to/episode_001
```

默认读取 `calibration/intrinsics.json` 和 `calibration/extrinsics.yaml`，
输出 `calibration/tcp_projection.mp4`。可通过 `--intrinsics`、`--extrinsics`、`--output` 修改。
视频由 **ffmpeg / libx264** 编码为 H.264，要求 `ffmpeg` 在 PATH 中；OpenCV 仅负责读帧和画点。

直接计算 `p_Head = T_Head_Base @ [x_TCP, y_TCP, z_TCP, 1]`，再用中间相机内参及畸变
投影到 `head.mp4`。左 TCP 为绿色十字，右 TCP 为紫色十字。
输出保持原视频帧数和帧率；补帧、与头部图像时间差超过 20 ms 的反馈不会画点，画面注明原因。

观察十字是否始终贴着对应的 TCP 位置、是否随末端同步运动；建议使用另一段运动录制验证。
TCP 指录制配置定义的位置，不一定是夹爪尖端。视频用于目视验证，不能单独给出毫米级精度。

## 4. 相机系 action 转为基座系 action

```powershell
python calibration/camera_action_to_base.py --input camera_action.csv --output base_action.csv
```

输入 CSV 与仓库 `action.csv` 同列名：`timestamp_unix_s`，以及左右各自的
`left.x,left.y,left.z,left.qx,left.qy,left.qz,left.qw,left.gripper_m`，右臂对应 `right.*`。
输入位置是 **中间相机光学坐标系中的三维坐标（米）**，姿态是该坐标系下的绝对末端姿态，
四元数顺序 **xyzw**；不是图像像素或相对位姿增量。

使用标定结果进行完整位姿转换，位置和旋转都会转换：

```python
T_LeftBase_LeftEEF = inv(T_Head_LeftBase) @ T_Head_LeftEEF
T_RightBase_RightEEF = inv(T_Head_RightBase) @ T_Head_RightEEF
```

时间戳、夹爪值、额外列及行顺序保持不变。默认读取 `calibration/extrinsics.yaml`，
可用 `--extrinsics` 指定；不需要相机内参。
