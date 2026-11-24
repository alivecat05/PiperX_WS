# Calibration

读取三路 RGB 内参，标定 D555 到左右臂基座的外参，并验证末端投影。

## 环境配置

在仓库根目录运行。读内参使用采集电脑上支持相机连接方式的 RealSense SDK Python 环境，投影视频需要 PATH 中的 FFmpeg。

```powershell
python -m pip install -r Calibration/requirements.txt
```

## 使用方法

关闭占用相机的程序，再按采集配置读取内参。也可以把 `--config` 指向录制时的 `manifest.json`。

```powershell
python Calibration/read_intrinsics.py --config PiperX_Teleop/config.json --output Calibration/intrinsics.json
python Calibration/calibrate.py --data path/to/episode_001 --intrinsics Calibration/intrinsics.json --board 8 6 0.045 0.033 --legacy-pattern --output Calibration/extrinsics.yaml
```

`--board` 依次是横纵方格数、格边长和 marker 边长，长度单位为米。示例使用仓库内的 [ChArUco 标定板](calib.io_CHARUCO_420x297_6x8_45_33_DICT_4X4.pdf)，OpenCV 新版布局用 `--no-legacy-pattern`。

相机系 CSV 可以转换到左右臂各自的基座系。

```powershell
python Calibration/camera_action_to_base.py --input camera_action.csv --extrinsics Calibration/extrinsics.yaml --output base_action.csv
```

## 验证方法

```powershell
python Calibration/project_tcp.py --data path/to/another_episode --intrinsics Calibration/intrinsics.json --extrinsics Calibration/extrinsics.yaml --output Calibration/tcp_projection.mp4
```

播放视频，检查左臂绿十字和右臂紫十字是否随各自的 TCP 同步运动。标定脚本使用留出的样本做验证，通过后写入外参。

## 数据格式

标定读取三路 MP4 和 `samples.jsonl` 中实测的 `observation.state`。外参包含 `T_Head_LeftBase` 和 `T_Head_RightBase`，均为从对应基座系到头部相机系的 4×4 矩阵。

转换 CSV 时使用 `timestamp_unix_s` 和左右臂的 `x,y,z,qx,qy,qz,qw,gripper_m`。位置与开口单位为米，姿态使用 `xyzw` 四元数。
