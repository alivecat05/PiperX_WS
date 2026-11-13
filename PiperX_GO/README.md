# PiperX_GO

双 PiperX 的远程策略推理与本地执行，支持 ACT 和 Diffusion Policy（DP）。推理与硬件控制分离，策略执行频率固定为训练使用的 **30Hz**。

## 1. 文件结构

```text
PiperX_GO/
├── README.md                     # 使用、通信协议、RTC 与动作后处理
├── piperx_execution/              # 部署运行代码，模块平铺
│   ├── client.py                 # PC 入口、参数、连接与硬件生命周期
│   ├── remote_policy.py          # 观测历史、图像编码、握手校验、远程推理
│   ├── execution.py              # 30Hz 控制循环、RTC / NO-RTC 调度
│   ├── control.py                # chunk 平滑、衔接、时间轴、TCP 命令约束
│   ├── hardware.py               # 真实 / 模拟硬件、同步观测、启用与关闭
│   ├── camera_capture.py         # RealSense 采集与逐帧耗时
│   ├── homing.py                 # 回位配置、航点与到位判定
│   ├── server.py                 # 服务入口、鉴权、请求分发与串行推理
│   ├── act_policy.py             # ACT checkpoint 加载与推理
│   ├── dp_policy.py              # DP checkpoint 加载与推理
│   ├── image_inputs.py           # 服务端图像解码与输入校验
│   ├── protocol.py               # v2 消息、NumPy 编解码与数据校验
│   ├── timing.py                 # 时钟校准与单程延迟估算
│   ├── config.pc.json            # CAN、相机、TCP、同步与后处理配置
│   ├── requirements-*.txt        # Client / Server 依赖
│   ├── run_client*.cmd           # Windows ACT / DP 快捷入口
│   └── tests/                    # 离线回归与 WebSocket 集成测试
└── ace_training/                 # 保留原有路径的训练兼容模块
    ├── preprocessing.py          # 训练 / 部署共享的像素与状态变换
    ├── data.py                   # LeRobot 索引与视频读取
    ├── act_dataset.py            # ACT 数据适配
    ├── dp_dataset.py             # DP 数据适配
    ├── frame_cache.py            # 训练帧缓存（使用 Linux fcntl）
    └── run_artifacts.py          # 训练产物同步与进度监测
```

保留原有 `client.py` / `server.py` 启动方式、脚本和配置路径。共享预处理源码的哈希属于输入契约，两端必须使用同一份 `ace_training/preprocessing.py`。

## 2. Server / Client 与通信协议

```text
PC Client：三相机 + 双臂反馈 → 同步观测 / 图像预处理
               │ infer：状态 + 图像（或连续观测历史）
               ▼
推理 Server：输入校验 → ACT / DP 模型 → 物理量动作块
               │ action：[H,16]
               ▼
PC Client：时间对齐 → 平滑与衔接 → 命令约束 → 本地 SDK / CAN
```

- **Server**：在策略环境中加载 checkpoint、归一化参数并预热，以单工作线程串行推理。ACT 要求 CUDA，DP 按 `--device` 运行；服务端不连接相机或机械臂。
- **Client**：在 PC 采集与同步观测、收发推理消息、按 30Hz 执行，通过 `PiperX_Teleop` 调用 SDK；PC 不需要 Torch / CUDA。
- **协议**：WebSocket 二进制帧 + MessagePack，自定义 NumPy 数组表示；应用版本 **v2**，单消息上限 **16MiB**，关闭 WebSocket 压缩。线上不是 JSON，也不是可直接替换 OpenPI 服务的接口。
- **连接**：默认 `ws://127.0.0.1:8000`，可经 SSH 转发。两端设置相同 `ACE_INFERENCE_TOKEN` 时，通过 `Authorization: Bearer <token>` 握手鉴权；未设置则不校验。`GET /healthz` 返回 `{"ready":true}`，也受相同鉴权约束。

## 3. 启动

### Server

在原 ACT / DP 训练环境中运行；本目录不包含上游模型实现和 checkpoint。需让 `ace_training`、ACT 的 `policy` 或 DP 的 `diffusion_policy` 可被导入，保持既有训练工程布局或设置对应的 `PYTHONPATH`。

以下命令均在 `PiperX_GO` 目录执行，checkpoint 路径按实际位置替换：

```bash
python -m pip install -r piperx_execution/requirements-server.txt

# ACT：auto 精度读取 checkpoint 同目录的 training_config.json
python piperx_execution/server.py --policy act --checkpoint /path/to/last.pt --device cuda:0 --port 8000

# DP：没有原训练数据的 provenance 时，显式提供采集配置
python piperx_execution/server.py --policy diffusion --checkpoint /path/to/model.ckpt --capture-config piperx_execution/config.pc.json --device cuda:0 --port 8001
```

ACT 的 `last.pt` 提供策略配置、权重、图像尺寸与归一化统计；缺少训练精度信息时须指定 `--precision fp32` 或 `bf16`。DP 默认按 checkpoint 选择 EMA / model 权重与去噪步数，可用 `--weights`、`--inference-steps` 覆盖。

### Client

在已配置 `PiperX_Teleop`、CAN 与 RealSense 的 PC Python 环境中执行：

```powershell
cd D:\PiperX_WS\PiperX_GO
python -m pip install -r piperx_execution/requirements-client.txt

# 读取真实反馈和图像并推理；不使能、不下发动作
python piperx_execution/client.py --server ws://127.0.0.1:8000 --teleop-root D:\PiperX_WS\PiperX_Teleop --max-steps 60

# RTC 执行
python piperx_execution/client.py --server ws://127.0.0.1:8000 --teleop-root D:\PiperX_WS\PiperX_Teleop --rtc --execute

# NO-RTC 执行，显式选择较短的 4 步前缀
python piperx_execution/client.py --server ws://127.0.0.1:8000 --teleop-root D:\PiperX_WS\PiperX_Teleop --no-rtc --action-steps 4 --execute
```

DP 将地址改为端口 `8001`。Windows 的 `run_client.cmd` / `run_client_dp.cmd` 分别默认使用 `8000` / `8001`，可用 `SERVER_URL` 环境变量覆盖。`--mock --max-steps 60` 使用模拟反馈与画面，不连接硬件，但仍需推理服务。

经 SSH 连接时，在另一窗口保持转发运行，例如 `ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 Exp`；DP 改为 `8001`，`Exp` 替换为实际 SSH 别名。

真实执行顺序：校验元数据 → 等待新鲜同步观测与使能确认 → 按**当前 `PiperX_Teleop/config.json`** 航点回位 → 清除回位前历史 → 策略控制。回位不采用导出的旧航点。退出或通信异常时保持新鲜实测关节位置并关闭设备，不自动回位。

## 4. 消息格式

### 数据约定

- 相机固定为 `head`、`left_wrist`、`right_wrist`；`state` 为 `float32[16]`，`actions` 为 `float32[H,16]`，`H` 来自握手。
- 16 维顺序：`[左 x,y,z,qx,qy,qz,qw,gripper, 右 x,y,z,qx,qy,qz,qw,gripper]`。位置、夹爪开口单位为米，姿态为 `xyzw` 四元数；每臂都是**自身基座坐标系中的绝对 TCP 目标**，不是关节角或相对增量。
- 默认 PC 只执行一次共享的 `YUV420p → 等比缩放 / RGB24 → 黑边补齐`，再传 PNG bytes 或 raw `uint8[height,width,3]` RGB 数组；尺寸由 `hello.image_size` 指定。JPEG 需显式传 `--allow-lossy-images`。
- NumPy 数组编码为 `{b'__ndarray__': true, b'data': <连续原始字节>, b'dtype': <dtype.str>, b'shape': <维度列表>}`，不使用 Base64。

以下是**解码后的结构示意**，`float32[...]`、`<...>` 表示类型或实际值，线上仍为 MessagePack。

### 握手：Server → Client

```text
{
  type: "hello", protocol_version: 2, policy: "act" | "diffusion",
  checkpoint: <路径>, checkpoint_step: <int>, precision: "fp32" | "bf16",
  action_dim: 16, horizon: H, control_hz: 30,
  action_semantics: "absolute_tcp_xyz_quaternion_xyzw_gripper_m_each_arm_base",
  camera_names: ["head", "left_wrist", "right_wrist"], image_color: "RGB",
  image_size: [height,width], camera_shapes: {<相机>: [height,width,3]},
  tcp_offsets: {left: [6 个数], right: [6 个数]},
  input_contract: {preprocess, preprocess_sha256, pyav_version, ...},
  normalization_sha256: <归一化参数摘要>,
  rtc_algorithm: "client_async_delay_compensation_overlap_blend",
  prefix_conditioning: false, ...
}
```

DP 另含 `n_obs_steps`、`action_steps`、`training_horizon`、`inference_steps`、权重与 epoch；ACT 当前不发送 `n_obs_steps` / `action_steps`。客户端校验版本、语义、频率、TCP 偏移、采集尺寸、预处理源码哈希与 PyAV 版本。

### 推理：Client → Server

```text
{
  type: "infer", protocol_version: 2, request_id: <递增正整数>,
  observation_tick: <PC 观测在控制时间轴上的整数 tick>,
  state: float32[16],
  images: {head: <PNG bytes 或 RGB ndarray>, left_wrist: <同上>, right_wrist: <同上>},
  images_preprocessed: true, input_contract: <hello 中的输入契约>,
  allow_lossy_images: false
}
```

当 `n_obs_steps > 1` 时，顶层 `state` / `images` 替换为 `observation_history: [{state, images}, ...]`，从旧到新排列，长度严格等于 `n_obs_steps`；`observation_tick` 锚定最新帧。历史按 PC 控制 tick 采集，重复 / 倒退时间戳不追加，间隔超过两个控制周期则重新积累。

### 动作：Server → Client

```text
{
  type: "action", request_id: <原请求 ID>, observation_tick: <原观测 tick>,
  actions: float32[H,16], server_receive_s: <服务端单调时钟秒>,
  server_timing: {request_decode_ms, queue_ms, preprocess_ms,
                 model_infer_ms, model_wall_ms, postprocess_ms, backend_total_ms}
}
```

`actions[0]` 对应观测锚点 tick。客户端校验 ID、锚点、形状与数值。当前 `action` 不另带 `protocol_version`，沿用握手版本。错误响应为 `{type: "error", error: <说明>}`，随后服务端关闭连接。

### 时钟与二进制封套

- 时钟请求：`{type: "clock_sync", protocol_version: 2, request_id}`；响应增加 `server_receive_s`。
- `infer`、`action` 与双向 `clock_sync` 带封套：`b"PX02" + 8 字节大端 float64 发送时间 + MessagePack 消息体`。发送时间为发送端 `time.perf_counter()` 秒，解包后以 `_sent_at` 暴露。`hello` / `error` 直接发送 MessagePack。
- 客户端先做 7 次校准，约每 30 秒重新校准，以最小网络 RTT 样本估计时钟偏移。单程延迟带误差范围；动作过期判定只用 PC 自己的 tick，不直接比较两台机器的时钟。

## 5. RTC 与 NO-RTC

**RTC（默认 `--rtc`）**：执行现有动作块时，后台请求下一块。按观测 tick 裁掉过期预测，新块放到 PC 时间轴；仅融合新旧块中**同一未来 tick** 的预测。推理尚未返回时，继续执行有效旧预测。

**NO-RTC（`--no-rtc`）**：等待推理响应，裁掉过期预测，执行接下来最多 `action_steps` 个有效动作；队列耗尽后才请求下一块。等待期间进入实测关节保持，推理与动作执行不重叠。

两者都不重放旧动作、不补发错过的 tick。整块过期则清空计划、保持并请求最新观测，会话与 watchdog 继续运行。例如观测 tick 为 100、收到响应时为 106，跳过前 6 个预测；若 `H=30` 且已到 130，则整块丢弃。

- **实际 CLI 默认 `--action-steps 10`，两种模式相同**。RTC 用它确定重规划间隔，仍保留整段有效 horizon；NO-RTC 用它限制执行前缀。短视界需显式指定 `--action-steps 4`。
- 仅程序调用传入 `args.action_steps=None` 时，才用服务元数据的 `action_steps`（缺省 15）；NO-RTC 再限制到 `inference.sync_action_steps`（默认 4）。CLI 默认 10 不触发这条回退。
- RTC 提前量为 `max(rtc_lookahead_steps, ceil(最近 roundtrip_s × 30)+1)`，`rtc_lookahead_steps` 默认 3。安装新块后，在 `max(1, action_steps - 提前量)` 个 tick 后请求下一块；最多一个推理请求在途。
- `--blend-steps` 默认 5，`--control-hz` 必须为 30。配置顶层的 `control_hz=60` 来自采集 / 遥操作配置，不决定此处策略执行频率。

这里的 RTC 是客户端异步调度、延迟补偿和重叠融合；ACT / DP 均未实现去噪前缀条件或 inpainting。30Hz 是控制频率，不代表每秒运行模型 30 次。

## 6. Action 后处理

处理顺序如下：

1. **恢复物理量**：服务端 ACT 用 checkpoint 的 `action_std / action_mean` 反归一化；DP 的 `predict_action` 使用自身 normalizer，并裁掉 `n_obs_steps - 1` 个历史位置。两者检查输出、单位化四元数，返回 `[H,16]`。
2. **校验原始块**：客户端检查 horizon、有限值与非零四元数；先判断是否整块过期，再检查原始预测工作空间，避免平滑掩盖越界点。
3. **块内平滑**：默认 5 点三角 FIR 处理 xyz；姿态在局部 SO(3) 旋转向量中平滑，处理 `q/-q` 等价性。此步不改变夹爪时序，仅对预测补边，不补造观测。
4. **对齐与衔接**：丢弃过期前缀；RTC 融合同 tick 预测，NO-RTC 从参考位姿渐入。采用五次平滑权重，xyz / 夹爪线性插值，姿态走最短四元数路径。
5. **逐 tick 约束**：模型位置目标先限制到实测 TCP 的超前半径（默认 0.1m），再经过持续低通、反向确认、速度 / 加速度与离散刹停约束；夹爪另做低通、速度 / 加速度与开口范围限制。最终命令再次检查工作空间与反馈距离，成功下发后才提交滤波状态。
6. **保持与恢复**：无有效目标时只在进入等待的一刻切换关节保持；恢复后从新鲜反馈重置速度状态，保留已确认方向。每周期仍检查 CAN 反馈与相机新鲜度。

主要参数在 `config.pc.json.inference`：位置 / 姿态 / 夹爪低通时间常数 `0.12 / 0.15 / 0.04s`；反向确认 `3` 步；线速度 `0.08m/s`、角速度 `0.5rad/s`、夹爪速度 `0.08m/s`；对应加速度上限 `0.25m/s²`、`1.5rad/s²`、`0.5m/s²`。SDK 速度取 `sdk_speed_percent=20` 与每臂配置的较小值。这些是 TCP 命令约束，不等于实际关节加速度保证。

## 7. 日志与验证

- `connected`、`action_filter_config`、`home_started / home_done`：连接、配置与回位。
- `inference` 与 `[ms]`：模型、编解码、上传 / 下载估算及相机 SDK 耗时；`roundtrip_ms` 是 PC 请求准备至响应处理的总耗时。
- `control`：丢弃动作、过期块、保持 tick 与滤波计数；整块过期另记 `action_chunk_expired`。

在含客户端依赖、`pytest` 及可导入的 `piperx_teleop` 的环境中，从 `PiperX_GO` 执行：

```powershell
python -m pytest piperx_execution/tests -q
```

测试覆盖平滑与命令约束、RTC / NO-RTC、过期恢复、启动回位，以及本地真实 WebSocket 的握手、时钟同步、PNG / raw 和单帧 / 连续历史传输；使用模拟硬件与确定性后端，不连接真实设备或加载真实模型。
