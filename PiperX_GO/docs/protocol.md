# PiperX_GO 协议参考

## 消息格式

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

## RTC 与 NO-RTC

**RTC（默认 `--rtc`）**：执行现有动作块时，后台请求下一块。按观测 tick 裁掉过期预测，新块放到 PC 时间轴；仅融合新旧块中**同一未来 tick** 的预测。推理尚未返回时，继续执行有效旧预测。

**NO-RTC（`--no-rtc`）**：等待推理响应，裁掉过期预测，执行接下来最多 `action_steps` 个有效动作；队列耗尽后才请求下一块。等待期间进入实测关节保持，推理与动作执行不重叠。

两者都不重放旧动作、不补发错过的 tick。整块过期则清空计划、保持并请求最新观测，会话与 watchdog 继续运行。例如观测 tick 为 100、收到响应时为 106，跳过前 6 个预测；若 `H=30` 且已到 130，则整块丢弃。

- **实际 CLI 默认 `--action-steps 10`，两种模式相同**。RTC 用它确定重规划间隔，仍保留整段有效 horizon；NO-RTC 用它限制执行前缀。短视界需显式指定 `--action-steps 4`。
- 仅程序调用传入 `args.action_steps=None` 时，才用服务元数据的 `action_steps`（缺省 15）；NO-RTC 再限制到 `inference.sync_action_steps`（默认 4）。CLI 默认 10 不触发这条回退。
- RTC 提前量为 `max(rtc_lookahead_steps, ceil(最近 roundtrip_s × 30)+1)`，`rtc_lookahead_steps` 默认 3。安装新块后，在 `max(1, action_steps - 提前量)` 个 tick 后请求下一块；最多一个推理请求在途。
- `--blend-steps` 默认 5，`--control-hz` 必须为 30。配置顶层的 `control_hz=60` 来自采集 / 遥操作配置，不决定此处策略执行频率。

这里的 RTC 是客户端异步调度、延迟补偿和重叠融合；ACT / DP 均未实现去噪前缀条件或 inpainting。30Hz 是控制频率，不代表每秒运行模型 30 次。

## 动作后处理

处理顺序如下：

1. **恢复物理量**：服务端 ACT 用 checkpoint 的 `action_std / action_mean` 反归一化；DP 的 `predict_action` 使用自身 normalizer，并裁掉 `n_obs_steps - 1` 个历史位置。两者检查输出、单位化四元数，返回 `[H,16]`。
2. **校验原始块**：客户端检查 horizon、有限值与非零四元数；先判断是否整块过期，再检查原始预测工作空间，避免平滑掩盖越界点。
3. **块内平滑**：默认 5 点三角 FIR 处理 xyz；姿态在局部 SO(3) 旋转向量中平滑，处理 `q/-q` 等价性。此步不改变夹爪时序，仅对预测补边，不补造观测。
4. **对齐与衔接**：丢弃过期前缀；RTC 融合同 tick 预测，NO-RTC 从参考位姿渐入。采用五次平滑权重，xyz / 夹爪线性插值，姿态走最短四元数路径。
5. **逐 tick 约束**：模型位置目标先限制到实测 TCP 的超前半径（默认 0.1m），再经过持续低通、反向确认、速度 / 加速度与离散刹停约束；夹爪另做低通、速度 / 加速度与开口范围限制。最终命令再次检查工作空间与反馈距离，成功下发后才提交滤波状态。
6. **保持与恢复**：无有效目标时只在进入等待的一刻切换关节保持；恢复后从新鲜反馈重置速度状态，保留已确认方向。每周期仍检查 CAN 反馈与相机新鲜度。

主要参数在 `config.pc.json.inference`：位置 / 姿态 / 夹爪低通时间常数 `0.12 / 0.15 / 0.04s`；反向确认 `3` 步；线速度 `0.08m/s`、角速度 `0.5rad/s`、夹爪速度 `0.08m/s`；对应加速度上限 `0.25m/s²`、`1.5rad/s²`、`0.5m/s²`。SDK 速度取 `sdk_speed_percent=20` 与每臂配置的较小值。这些是 TCP 命令约束，不等于实际关节加速度保证。
