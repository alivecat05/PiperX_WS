# PiperCapture

[English](README.en.md)

录制两组主从 Piper X 的示教动作及头部、双腕三路 RGB 视频。

## 环境配置

在 Windows 的 `PiperCapture/` 目录安装环境并查找设备。

```powershell
.\setup.cmd
.\run.cmd discover
.\run.cmd discover-arms
```

在 `config.json` 核对相机序列号和左右 CAN 编号，并安装 USB-CAN 适配器驱动。同一套机械臂运行采集时先退出 PiperX_Teleop。

## 使用方法

```powershell
.\run.cmd
```

用机械臂自身的示教控制进入可拖动状态。在窗口填写任务名及采集序号，点击「开始录制」，完成后点「结束保存」。保存结束后再录下一段。主从共用 CAN 的配置可以保存从臂参考位，自定义回位按钮在该配置下禁用。

首次导出 LeRobot 时安装导出环境。

```powershell
.\setup.cmd -Export
.\export.cmd
```

默认读取 `outputs/`，输出到 `datasets/pipercapture/`，可用 `--source` 指定录制目录。

## 验证方法

```powershell
.\run.cmd --mock
```

模拟录一段，检查三路画面、保存状态和输出文件。实机界面显示相机 FPS、反馈频率和有效 action 频率。

## 数据格式

每段写到 `outputs/<任务>/episode_001/`，保存 `head.mp4`、`left_wrist.mp4`、`right_wrist.mp4`、`action.csv`、`samples.jsonl` 和 `manifest.json`。

三路视频、CSV、JSONL 逐帧对应。CSV 为 Unix 秒时间戳加左右臂各 8 维 `x,y,z,qx,qy,qz,qw,gripper`，位姿在各自基座系，位置和开口单位为米。`state` 是当前实测状态，`action` 是下一采样时刻的实测状态，动作时间记在 JSONL 的 `action_timestamp_unix_s`。
