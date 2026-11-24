# PiperX_Teleop

[English](README.en.md)

用 Quest 3 遥操双臂 Piper X，并录制三路 RGB、动作和状态。

## 环境配置

在 Windows 的 `PiperX_Teleop/` 目录安装环境并查找设备。

```powershell
.\setup.cmd
.\teleop.cmd discover
```

安装 USB-CAN 适配器驱动，在 `config.json` 中核对相机序列号、CAN 通道、固件、TCP 偏移、工作区和回位航点。

## 使用方法

通过 **USB** 连接前，在 Quest 开启开发者模式并允许调试。运行后在 Quest 打开 `http://localhost:8012`。

```powershell
.\teleop.cmd
```

通过 **Wi-Fi** 连接时，把电脑和 Quest 接入同一网络，将示例 IP 换成电脑的 WLAN 地址并放行 TCP 8012。Quest 打开终端显示的 HTTPS 地址。

```powershell
.\teleop.cmd cert --ip 192.168.1.100
.\teleop.cmd --transport wifi --cert certs/cert.pem --key certs/key.pem
```

### 启动时对齐头显方向

头显朝向机械臂基座 +X 后再点「启动手柄追踪」。按住中指键接管对应臂，松开后保持最后下发的位姿。食指扳机独立控制夹爪。

| 按键 | 操作 |
| --- | --- |
| B / A | 开始录制 / 停止并保存 |
| X 长按 2 秒 | 保存当前双臂位姿为初始位 |
| Y 长按 1.5 秒 | 按已配置航点将两臂回到初始位 |

保存初始位和回位前，松开中指键并等待录制保存完成。回放 CSV 时先点「加载 CSV」，确认路径畅通后点「到起点并回放」。用「停止回放」或空格停止。

### 挂脖采集

连接 USB 执行 `.\teleop.cmd awake`，让 Quest 摄像头持续看到手柄。结束后执行 `.\teleop.cmd sleep`。

导出 LeRobot 时，首次运行 `.\setup.cmd -Export`，之后用 `.\export.cmd`。结果位于 `datasets/piperx/`。

## 验证方法

```powershell
.\teleop.cmd --mock
.\teleop.cmd --transport usb --input-test
```

第一条使用模拟设备。第二条检查 Quest 和真实相机输入，机器人使用模拟反馈。[实机前检查和验证记录](docs/validation.md)

## 数据格式

`outputs/episodes/<任务>/episode_001/` 保存三路 30 FPS MP4、`action.csv`、`samples.jsonl` 和 `manifest.json`。三路视频和数据逐帧对应。

CSV 保存 Unix 秒时间戳以及左右臂各 8 维 `x,y,z,qx,qy,qz,qw,gripper`。位姿在各臂基座系，位置和开口单位为米。`action` 是软件下发的目标，JSONL 中的 `state` 是实测反馈。缺帧用保持画面或黑帧保留时间轴，标记为 `sample_valid=false`。
