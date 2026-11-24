# PiperX_GO

远程运行 ACT 或 Diffusion Policy，在采集电脑上以 30 Hz 控制双臂 Piper X。

## 环境配置

以下命令在 `PiperX_GO/` 执行。服务端使用原训练环境，保留 ACT 的 `policy` 或 DP 的 `diffusion_policy` 导入路径。客户端使用已配置 CAN 和 RealSense 的 [PiperX_Teleop 环境](../PiperX_Teleop/README.md)。

```bash
# 服务端环境
python -m pip install -r piperx_execution/requirements-server.txt
# 客户端环境
python -m pip install -r piperx_execution/requirements-client.txt
```

## 使用方法

在服务端选一条命令启动策略。

```bash
python piperx_execution/server.py --policy act --checkpoint /path/to/last.pt --device cuda:0 --port 8000
python piperx_execution/server.py --policy diffusion --checkpoint /path/to/model.ckpt --capture-config piperx_execution/config.pc.json --device cuda:0 --port 8001
```

ACT 若缺少训练精度记录，指定 `--precision fp32` 或 `bf16`。客户端把地址改成实际服务端地址，DP 使用上例的 `8001`。

```powershell
python piperx_execution/client.py --server ws://127.0.0.1:8000 --teleop-root ../PiperX_Teleop --max-steps 60
```

上面的客户端命令只检查读图、反馈和推理。确认回位航点与工作区后，加 `--execute` 才使能双臂、回位并执行模型动作。默认启用 RTC，`--no-rtc` 切换为同步调度。

## 验证方法

客户端加 `--mock` 可通过模拟硬件检查推理连接。离线测试使用客户端环境。

```bash
python -m pytest piperx_execution/tests -q
```

## 数据格式

客户端上传头部、左右腕 RGB 和 16 维状态，服务端返回 `[H, 16]` 动作块。每臂 8 维依次为 `x,y,z,qx,qy,qz,qw,gripper`，位姿使用各臂基座系，位置和开口单位为米。

[通信协议、RTC 和动作后处理](docs/protocol.md)
