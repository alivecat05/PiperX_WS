# Piper 工作空间

[简体中文](README.md) | [English](README.en.md)

- [PiperCapture](PiperCapture/README.md)：主从机械臂示教采集，三视角预览、RGB/action 同步、MP4/CSV/JSONL、LeRobot 导出。
- [PiperX_Teleop](PiperX_Teleop/README.md)：Quest 3 双臂遥操，支持 USB / Wi-Fi。
- [PiperX_GO](PiperX_GO/README.md)：ACT / Diffusion Policy 远程推理与双臂本地执行，支持 30Hz 控制、RTC / NO-RTC 调度和动作后处理。
- [HumanCapture](HumanCapture/README.md)：人类示范三相机同步采集，白色界面、实时预览、任务序号、MP4 与逐帧时间戳。
- [HumanLabel](HumanLabel/README.md)：EgoHand＋native smoothing＋前后帧裁剪恢复；输出 head 相机系腕部 epose、PiperCapture 格式夹爪距离和完整对比视频。

## 采集

Windows 10/11，首次运行安装项目内 Conda：

```powershell
cd PiperCapture
.\setup.cmd
# 在 config.json 填相机序列号和 CAN 编号
.\run.cmd
```

在窗口填写任务名、采集序号，点击“开始录制 / 结束保存”：

```text
PiperCapture/outputs/pick_cup/episode_001/
  head.mp4
  left_wrist.mp4
  right_wrist.mp4
  action.csv
  samples.jsonl
  manifest.json
```

序号自动建议下一号，也可手填；不覆盖已有数据。当前主从共线接法暂不支持自定义回位。

## 导出

```powershell
.\setup.cmd -Export
.\export.cmd
```

默认读取 `PiperCapture/outputs/`，输出到 `PiperCapture/datasets/pipercapture/`。

## 策略推理与执行

`PiperX_GO` 的 Server 在模型环境中加载 checkpoint，PC Client 采集三相机与双臂反馈，通过 WebSocket + MessagePack 请求动作块，并在本地完成时间对齐、平滑、限速与 SDK 下发。

服务端启动后，在已配置依赖的 `PiperX_Teleop` 环境中，从仓库根目录运行只读检查：

```powershell
python PiperX_GO/piperx_execution/client.py --server ws://127.0.0.1:8000 --teleop-root PiperX_Teleop --max-steps 60
```

默认启用 RTC，添加 `--execute` 后才使能机械臂并执行回位及模型动作；`--no-rtc` 切换同步调度。服务端启动、依赖安装、协议格式与后处理参数见 [PiperX_GO README](PiperX_GO/README.md)。

仓库包含代码、测试、SDK 源码及许可证、配置示例和安装脚本，以及 HumanCapture 和 PiperX_Teleop 当前的 `config.json`（相机序列号、采集参数、踏板绑定及遥操配置），方便迁移到新电脑继续使用同一套硬件。PiperCapture 的本机配置仍单独保留。不包含本地 Conda、录制数据、证书及缓存。
