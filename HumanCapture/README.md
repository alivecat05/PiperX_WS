# HumanCapture

[English](README.en.md)

同步录制人类示范的头部 D555 和双腕 D405i RGB 视频。

## 环境配置

在 Windows 安装带 Tk 的 Python 3.12，再进入 `HumanCapture/` 安装依赖并查找设备。

```powershell
.\setup.cmd
.\capture.cmd discover
```

`setup.cmd` 创建 `.venv`。按 `discover` 的结果在 `config.json` 中核对相机序列号和 RGB 模式。

## 使用方法

```powershell
.\capture.cmd
```

三路画面就绪后输入任务名和采集序号，点击「开始录制」，完成后点「结束录制并保存」。勾选自动递增可接着录下一段。使用键盘型踏板时，在采集窗口点击「识别踏板」完成绑定，前台窗口中的每次踩踏切换开始和保存。

## 验证方法

```powershell
.\capture.cmd --mock --smoke-seconds 6
.\capture.cmd --diagnose-seconds 300
```

第一条用模拟相机检查写盘，第二条读取实机相机并记录诊断到 `outputs/logs/capture.log`。[验证说明](docs/validation.md)

## 数据格式

数据写入 `outputs/episodes/<任务>/episode_001/`。三路 MP4 以 30 FPS 保存，与 `timestamps.csv` 和 `samples.jsonl` 逐帧对应。`manifest.json` 记录完成状态、有效帧数和丢帧统计。

`sample_valid=false` 表示占位帧。缺帧时使用上一组画面或黑帧，视频时间轴连续，片段标记为 `incomplete`。
