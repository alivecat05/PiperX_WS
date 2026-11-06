# HumanLabel

读取 `HumanCapture/outputs/episodes/` 的同步视频：**head 估计腕部位姿，left_wrist/right_wrist 补充各自手的夹爪距离**。结果写到本目录的 `result/<任务>/episode_XXX/`。沿用已有目录名 **HumanLabel/result**。

继续使用当前 MediaPipe 流程，不引入新模型。模型内部检测 21 个关键点，CSV 输出腕部 epose、拇指尖到食指尖距离和质量信息；预览用青色圆圈及十字标出 `WRIST #0`，坐标轴也从腕部出发。

## 模型选择（查阅日期 2026-11-06）

采用 [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python) 的 Python Tasks API。三路视频各使用独立的 VIDEO 跟踪状态及各自源时间戳，没有把 wrist 相机中的 XYZ 混入 head 的坐标系。

当前 wrist 近景常有掌根出画、手指遮挡和左右类别误判。只有匹配操作者解剖侧别的检测才可用于同侧夹爪；不会把 left_wrist 中出现的另一只手当成左手。

## 运行

```powershell
cd D:\Piper_WS\PiperX_WS\HumanLabel
.\setup.cmd
.\label.cmd
```

使用独立 `.venv`，不会修改 Teleop 的 Python 依赖。首次自动下载 Google 官方模型到 `.cache/`，以后离线使用。默认跳过模拟录制，处理所有真实片段，包括有采集缺口的片段，保留真实时间及网格索引；不覆盖现有结果。源 manifest 或视频异常时停止并留下 `status: failed`，重新尝试请指定新的 `--output`。

只处理一段，或调整捏合闭合距离：

```powershell
.\label.cmd "..\HumanCapture\outputs\episodes\human demonstration\episode_003"
.\label.cmd --output result_v2 --pinch-closed-m 0.015 --pinch-open-m 0.10
```

默认 `gripper_m = clip(距离米, 0, 0.10)`，最大开口与当前 `PiperX_Teleop/config.json` 一致。调整阈值后是 `0.10 * clip((d-closed)/(open-closed),0,1)`。更改机器人最大开口时同步传 `--gripper-max-m`。

wrist 补充规则：同侧检测不能有类别歧义，左右类别分数至少 0.8，拇指和食指两条关节链的预测像素点均须在图像内，腕点位于图像下部（当前佩戴方式的近景手）。head 与 wrist 都有效且距离相差不超过 **25 mm** 时取两者均值；超过该值保留 head，避免误判直接改写夹爪。head 距离缺失且 wrist 通过以上检查时才使用 wrist 补缺；两路均不可用时保留 NaN。

这只是两个模型距离估计的一致性筛选，不是经过外参标定的多视角三角化，也不能证明被遮挡的指尖真实可见。`quality.csv` 保存原始两路距离、来源与拒绝原因，方便检查是否有所改善。可用 `--head-only` 关闭补充，用 `--max-wrist-disagreement-m` 调整一致性阈值。不会插值或沿用旧帧补空缺。

## epose 的严格定义

这里 epose 指 **腕部坐标系 W 相对于固定 head 彩色相机光学坐标系 C 的绝对位姿**，记作 `T_C_W(t)`。**位置原点是 `P0=WRIST`，不再是拇指尖与食指尖连线的中点。** 它不是帧间增量、机器人 joint action 或机器人实发命令。

**参考系 C**：用户已确认 head 固定在支架上。C 是右手系；原点为 head 彩色相机光心，`+X` 朝图像右、`+Y` 朝图像下、`+Z` 朝相机前方，长度单位为米，遵循 [OpenCV PnP 的光学坐标约定](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)。移动支架后会改变这个参考系。

**腕部系 W**（左右手使用同一套几何定义）：

- 原点 `p = P0`，即腕部关键点。
- `z = normalize(P9-P0)`：腕部到中指掌指关节，表示沿手掌朝手指的方向。
- `a = P5-P17`：小指掌指关节指向食指掌指关节。
- `x = normalize(a-dot(a,z)*z)`，`y = normalize(z×x)`，再取 `x=y×z`。
- `R_C_W=[x y z]`：三列是 W 的三个轴在 C 中的表达，满足 `x×y=z`；`P_C = R_C_W P_W + p_C`。
- CSV 保存腕点 `p_C.x,p_C.y,p_C.z,qx,qy,qz,qw`，四元数是 `R_C_W` 的 **xyzw** 顺序，单位四元数，不是欧拉角。同一侧连续观测保持四元数符号一致。

`P0/4/5/8/9/17` 分别为腕、拇指尖、食指掌指关节、食指尖、中指掌指关节、小指掌指关节。轴方向来自掌部，捏合距离接近 0 时仍可定义姿态。它是模型定义的腕部/掌部方向，不是解剖学关节角，也不等同机器人法兰。

**位置如何获得**：MediaPipe 的 `hand_world_landmarks` 原点在手部中心，不能直接写为机器人 XYZ。脚本沿用当前全部 21 点的 `solvePnP`，得到 `P_C=R_pnp P_hand+t_pnp`，再取变换后的腕点 P0 及掌部方向。腕部位置/姿态仅来自 head；指尖距离可由同步 wrist 相机补充。所有米制尺度来自模型先验；PnP 没有消除单目物理尺度的不确定性。

现有 RGB 录像没有保存内参/深度，所以默认 **假设** `fx=fy=max(width,height)`、主点在图像中心、零畸变；该假设写入每段 manifest，不代表 D555 的标定参数。位置、距离及旋转都只是估计，特别是相机前后方向和严重遮挡。`--hand-scale` 可按已知手尺寸统一缩放模型先验，仍不是深度真值。

如已有与录制分辨率相同的 head **彩色流** OpenCV 针孔标定，可提供 JSON 并运行 `--intrinsics head_intrinsics.json`：

```json
{
  "width": 1280,
  "height": 800,
  "camera_matrix": [[1000, 0, 640], [0, 1000, 400], [0, 0, 1]],
  "distortion_coeffs": [0, 0, 0, 0, 0]
}
```

以上数值仅展示格式，**不是 D555 实参**。畸变参数顺序为 OpenCV 标准 `k1,k2,p1,p2,k3`；不要直接填深度流内参、逆 Brown 系数或 fisheye 系数。录制后查询设备内参时必须使用相同流/分辨率并确认录制时标定未变。

## 与 PiperX_Teleop 对齐的部分及训练使用

`action.csv` 的表头、左右顺序、维度和单位与 `piperx_teleop/core.py::VECTOR_NAMES` 一致：

```text
timestamp_unix_s,left.x,left.y,left.z,left.qx,left.qy,left.qz,left.qw,left.gripper_m,right.x,right.y,right.z,right.qx,right.qy,right.qz,right.qw,right.gripper_m
```

**坐标与时间语义不同**：Teleop 是每侧机器人自身 base 下的绝对 TCP target，并选择观测时刻之前最后发出的指令；这里是同一观测帧的人手估计状态，属于伪动作监督。manifest 用独立 schema，不能直接冒充 `piperx.rgb_action.v2` 或输入 Teleop replay/export。这里不会构造机器人的 `observation.state`、关节角或不存在的执行指令。

每段结果包含：

- `action.csv`：仅双手几何检查均通过的行，17 列且全部数值有限；不存在有效双手帧时只有表头。
- `frame_map.csv`：与 action.csv 一一对应，记录原视频 `frame_index`、源 `grid_index`、`segment_id` 及时间戳。**action 第 N 行不一定等于视频第 N 帧**。必须按映射取视频；构建 action chunk 时不能跨 `segment_id`。
- `action_raw.csv`：与源 MP4 逐帧对应的完整 17 列；不可用分量为 `nan`。如需只使用一侧监督，从此文件读取并使用该侧有效性掩码。两指可检测而腕部出画时，夹爪值仍保留，位置/四元数可为 `nan`。
- `quality.csv`：增加 head 画面腕点 `wrist_u_px/wrist_v_px`、`wrist_in_image`、独立的 `pose_valid/pinch_valid`、`pinch_source/pinch_reason`、head/wrist 原始距离和 wrist 来源时间戳。`valid` 表示腕部位姿和最终夹爪值均通过检查，`pose_valid` 只表示腕部位姿有效。`pinch_distance_px` 始终是 head 的像素距离，`wrist_pinch_distance_px` 属于同侧 wrist 画面。分类分数及图像内检查都**不是遮挡可见性真值**。
- `preview.mp4`：与源视频帧数相同；标出腕点及从腕部出发的坐标轴，红/绿/蓝对应 W 的 x/y/z，并显示夹爪值的来源。
- `wrist_preview.mp4`：两路 wrist 的并排预览，标出检测到的两指尖、距离和本帧是否参与补充。`--no-preview` 同时关闭两种预览。
- `manifest.json`：算法、参数、相机假设、坐标/时间语义、原始片段完整性与处理状态。这里 `complete` 只表示标注跑完，不表示原始采集无掉帧或标签已人工审核。

左右手指的是操作者的解剖左右手，不按图像左右排序。查看预览确认左右标签；只对本来被水平镜像的视频使用 `--swap-handedness`。它只交换类别，不会修改图像/坐标。

时间戳完全继承源 `timestamps.csv` 的 `timestamp_unix_s`；另存 head 实际曝光来源时间。完整片段也存在软件同步容差，不声称三路视频同时曝光。不会插值补手、重复旧姿态、平滑跨越缺口或补齐已丢失的采集帧。

要与机器人动作共用监督空间，仍需要已标定的 `T_B_C` 及每侧腕部到 TCP 的对齐 `T_W_TCP`：

```text
T_B_TCP(t) = T_B_C @ T_C_W(t) @ T_W_TCP
```

左右臂 B 各不相同。head 固定，`T_B_C` 可为常量；但仓库目前没有这些实测标定，所以仍写 `robot_base_calibrated: false`。本轮按用户要求只修改当前版本的腕部原点、可视化及夹爪补充，没有新增标定/大模型流程。最终真实机器人 base 标签仍需标定及人体到机器人工作空间对齐；VLA/WAM 的未来动作步长、动作块和归一化也需按目标模型处理。

## 验证

```powershell
uv pip install --python .venv\Scripts\python.exe pytest
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.cache/pytest
```

测试检查腕点原点不随指尖移动、坐标轴/四元数方向、PnP 恢复、实际 Teleop 表头、双视角距离一致性/拒绝规则，以及同步多相机、缺手和采集缺口的映射。真实视频仍需抽查，重投影误差不能代替手部精度评估。
