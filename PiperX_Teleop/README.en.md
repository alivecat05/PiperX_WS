# PiperX Dual-Arm Teleoperation

[简体中文](README.md) | [English](README.en.md)

Windows · Meta Quest 3 · Official pyAgxArm · D555 head camera + two D405 wrist cameras

## Features

- Relative-pose dual-arm teleoperation, proportional trigger control of grippers, and custom home poses.
- A three-view RGB window **on the computer**, showing camera FPS, action rate, and recording status.
- Software-aligned 30 Hz RGB and dual-arm action/state, with one-command LeRobot v3 export.
- Quest connection over USB or Wi-Fi; recording is independent of robot movement.

## 1. Installation (first use)

Windows 10/11 x64 with internet access. A project path without spaces is recommended.

```powershell
git clone https://github.com/alivecat05/PiperX_WS.git
cd PiperX_WS/PiperX_Teleop
.\setup.cmd
```

Setup automatically installs project-local Conda, Python 3.12, RealSense, the robot SDK, and ADB. Existing `config.json` is preserved. Install the hardware vendor's Windows driver for USB-CAN adapters.

## 2. Configure hardware

```powershell
.\teleop.cmd discover
```

The repository includes the current hardware's `config.json`, preserving camera serial numbers, CAN mapping, reset waypoints, and teleoperation parameters. Reuse it on a new computer with the same hardware; setup preserves existing configuration. Edit `config.json` when changing hardware:

- `cameras.*.serial`: head, left wrist, and right wrist camera serial numbers.
- `arms.*.channel`: verify left/right USB-CAN channels; the example uses left 1 and right 0.
- `arms.*.firmware`: match the SDK firmware profile; the current configuration uses `v189`.
- `tcp_offset / xr_to_base / workspace_*`: adjust for the tool and arm installation orientation.
- `arms.left.position_scale` and `arms.right.position_scale`: independent controller displacement scales, applied after saving and restarting. With `0.7`, a 10 cm hand movement produces a 7 cm robot target displacement; `1.0` produces 10 cm, and `1.5` produces 15 cm. The allowed range is `(0, 5]`; the window footer shows `Scale L/R`.
- `arms.*.pose_smoothing_s`: additional fixed smoothing, default `0`. `tremor_smoothing_s` defaults to `0.04` seconds and enables adaptive tremor filtering: small changes use a time constant up to 40 ms, shortened to approximately 10 ms for clear movement, without changing displacement scale. `position_deadband_m` defaults to `0.001` m and `rotation_deadband_rad` to `0.008726646` (0.5°), retaining a small-motion deadband. Index-trigger gripper control is independent.

Only RGB camera streams are enabled; depth streams are disabled. Each camera warms up with automatic exposure for two seconds, then locks its current exposure and gain. Reconnection within the same run reuses the locked values. These values are recorded in native sensor SDK units in camera metadata, logs, and detailed diagnostics. The current hardware configuration uses displacement scale 1.0, a rotation limit of 2 rad/s, and a translation limit of 0.5 m/s. Engagement no longer waits for an extra 5 mm movement or 3° rotation. A 1:1 scale describes displacement only; deadbands, speed limits, and robot response still affect instantaneous error.

D405 preferentially reads auto-exposure values from frame metadata; `get_option(exposure)` may still report the manual default of 33000. For sensors using microseconds, locked exposure is capped at 90% of the frame period (30000 µs at 30 FPS), avoiding the observed D405 drop to 15 FPS at a fixed 33000 µs. D555 retains its native units without applying D405 unit conversions.

Each middle-finger grip engagement uses the current position as the reference center, reads the controller orientation and actual TCP orientation, and establishes an orientation offset for that engagement. Engagement itself keeps the robot in place; after release, the next engagement records a new reference. This removes the initial orientation difference but does not by itself determine the axes from the headset reference frame to the robot base.

Releasing the grip holds **the last successfully sent filtered TCP position and orientation before release**. The new controller position on the release frame is ignored; lagging robot feedback does not replace the hold target. Any remaining filter lag is discarded immediately. While idle, the filter does not advance, joint mode and speed remain unchanged, and the hold target updates only upon deliberate re-engagement. The index trigger still controls the gripper independently, while recording and the other arm continue. Re-engagement establishes a new reference using the actual pose at that time, without replaying inputs received during release.

The current setting is `joint_guard_enabled=false`, disabling the previously added local joint-prediction projection so it does not alter following direction or response. Controller hard limits, workspace limits, and Cartesian speed limits remain active. When the target leads the actual TCP by more than `max_position_tracking_error_m`, following continues with a capped target and a 0.1 m allowance, without an additional 3 cm limit or global interruption. Expanded diagnostics distinguish mapped-target error from actual robot tracking error.

### Align headset direction at startup

The actual input is `frame.getPose(gripSpace, local-floor)`, using WebXR's right-handed coordinates: reference +X right, +Y up, and −Z forward. The platform initializes the horizontal direction of `local-floor`, which may not align with the workbench. The controller's local `gripSpace` axes are not the position reference axes. See the [WebXR coordinate and reference-space specification](https://immersive-web.github.io/webxr/#xrreferencespace).

Section 2.2.2 of the PiperX manual (PDF page 17) defines user coordinates as the robot base frame. Use the base diagram for axis directions; the room's “forward” direction cannot always be assumed to be base +X. See the [official PiperX manual](https://cdn.shopify.com/s/files/1/0551/0630/6141/files/PIPER_X_USER_MANUAL.pdf?v=1791861496#page=17). SDK end-effector poses use meters/radians and fixed-axis XYZ Euler angles, `R=Rz(yaw)Ry(pitch)Rx(roll)`, matching `Rotation.from_euler("xyz", ...)` in the code. See the [official SDK definition](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/drivers/piper/default/driver.py).

Verified on site: **with the current hardware and dual-arm installation, facing the headset along the robot base X-axis before clicking “Start controller tracking” (启动手柄追踪) in the Quest browser produces correct following directions, without three-point axis calibration.**

1. Before tracking starts, orient the headset in the direction verified on site, with its forward direction aligned to robot base +X.
2. Click “Start controller tracking” (启动手柄追踪) in Quest. The fixed mapping is WebXR −Z → base +X, −X → base +Y, and +Y → base +Z: forward / left / up map to base +X / +Y / +Z.
3. Hold the middle-finger grip to engage. The application automatically records the current controller position and actual TCP orientation as the reference; no manual samples are needed. B always starts recording, and A stops and saves.

Tracking uses the stable `local-floor` reference frame; turning your head does not rotate the following axes every frame. Keep the same startup orientation when restarting tracking. A reference-space reset ends the XR session through the existing logic; restart tracking and re-engage. The fixed matrix still comes from `xr_to_base`; each engagement's reference orientation is recorded in `/status`, logs, and the episode's `teleop_coordinate_calibration` configuration snapshot. Three-point calibration controls and sampling state have been removed.

Displacement scale determines target distance, `max_speed_m_s` limits host-side target speed, `speed_percent` controls SDK point-motion speed percentage, and `workspace_min/max` define the reachable workspace. The updated following configuration uses `0.5 m/s`, `2 rad/s`, and SDK `100%`; reset motion retains a separate `reset_speed_percent=10`. The old `0.12 m/s` plus SDK `15%` configuration limited both targets and physical movement, causing the robot to keep catching up after the hand stopped. The footer's `Follow` display reports target tracking error, command age, and `limited`; `limited=1` means mapped hand movement reached the host-side speed limit. Physical acceleration, inverse kinematics, and CAN latency still need hardware measurement; zero delay at arbitrary hand speeds is not guaranteed.

First place both arms in an extended working pose rather than a folded zero pose. After launching, hold **X** to save the home pose. Camera parameters and saved home poses are kept locally.

## 3. Connect Quest (choose one)

**USB (default):** enable developer mode on Quest, connect USB, and allow USB debugging in the headset. The following command is equivalent to adding `--transport usb`:

```powershell
.\teleop.cmd
```

Open **http://localhost:8012** in the Quest browser. Face along the robot base X-axis, then click “Start controller tracking” (启动手柄追踪).

**Wi-Fi:** connect both devices to the same Wi-Fi network. Replace the address below with the computer's **WLAN adapter IPv4 address** and allow TCP 8012 through its firewall:

```powershell
.\teleop.cmd cert --ip 192.168.1.100
.\teleop.cmd --transport wifi --cert certs/cert.pem --key certs/key.pem
```

On Quest, open **https://address:8012** as printed in the terminal. No token is required. Generate the certificate initially and regenerate it when the IP changes. A self-signed certificate may trigger a trust warning; USB does not have this warning.

**Chest-mounted use:** first connect USB and run `.\teleop.cmd awake`, then start tracking. Keep cameras facing forward and unobstructed. Run `.\teleop.cmd sleep` afterward; repeat `awake` after a headset reboot. MQDH is unnecessary, but the headset must still see the controllers. Chest-mounted stability requires hardware testing.

## 4. Controller operation

- **Left/right middle-finger grips:** hold to make the corresponding arm follow. Release freezes the last sent filtered pose and immediately discards filter lag. Re-engagement uses the current actual pose as a new reference.
- **Left/right index triggers:** half-press for half closure, fully press to close, and release to open; grip engagement is not required.
- **B:** start recording.
- **A:** stop and save; wait for `SAVING` to finish before starting another episode.
- **Hold X for two seconds:** save the current dual-arm positions as the home pose.
- **Hold Y for 1.5 seconds:** both arms begin returning home simultaneously, each through its own waypoints. Completion requires both arms to arrive. Release and press Y again to cancel both resets.

Before X/Y operations, release both grips, stop recording, and wait for saving. Press **Q / Esc** in the computer window to exit.

Normal exit (Q / Esc, closing the preview window, or Ctrl+C) uses fresh feedback to hold the current arm positions and gripper widths before disconnecting CAN, preserving the existing enable state. Faults, stale feedback, or a failed hold still trigger emergency-stop protection. After startup, holding a grip enables its corresponding arm for control.

### Capture window

The interface uses a white theme. A green banner indicates “Ready · Not recording,” and a red banner indicates recording. Saving, protective hold, and replay have separate status messages.

- “Task name” (任务名), “Episode number” (采集序号), and “Auto-increment” (自动递增) synchronize automatically after edits, without an extra apply button. Quest B uses synchronized settings; the desktop “Start capture” (开始采集) button also applies current input. Invalid task names or numbers block capture. Old status responses do not overwrite fields being edited.
- Episode numbers are editable, for example starting at 42. With auto-increment enabled, completing the episode selects 43, skipping existing directories. Incomplete episodes also reserve their numbers to prevent overwrite. With auto-increment disabled, change the number manually before the next episode.
- “Next available number” (下一个可用序号) suggests a number based on existing data for the entered task. Task and number fields are locked during capture and saving.
- The default task comes from `task` in `config.json`; window edits apply to the current run, and output directories retain task names.
- Protective hold displays its reason in Chinese, both arms' hold states, and recovery steps. With v189 firmware, “Resume teleoperation” (恢复遥操) can explicitly exit teaching and request a hold at the current joint positions. It waits for fresh CAN feedback to confirm this, then checks tracking, controls, enable state, and limits. Reset is not used to exit teaching; other firmware requires manual exit. Expand “Show detailed diagnostics” (显示详细诊断) for raw state.

### CSV trajectory replay

1. Paste the full path of an `action.csv` generated by this application into “Action CSV path” (Action CSV 路径), then click “Load CSV” (加载 CSV). Background loading validates format, timestamps, workspace, gripper widths, and quaternions without starting motion.
2. Set playback speed (`0.1`–`1.0`, default original speed `1.0`), confirm the path is clear, and click “Move to start and replay” (到起点并回放). Replay cannot start during capture/saving, reset, or fault recovery.
3. Both arms first move slowly to the initial frame: SDK speed 10%, target translation at most 0.05 m/s and rotation at most 0.35 rad/s. Gripper widths remain unchanged during positioning. Once both arms' position errors are ≤3 mm and orientation errors ≤2° for 0.3 seconds, following speeds are restored and replay begins at CSV time zero.
4. Translation and rotation are interpolated using actual timestamps, preserving original timing by default. If speed exceeds configured limits, startup is rejected with a request to lower playback speed. Quest does not need to be online for replay.
5. Stop either positioning or replay with “Stop replay” (停止回放) or Space; Quest Y also stops it when connected. Stopping retains the last target; completion retains the final frame. To resume teleoperation, release the controls and re-engage.

Limit violations, tracking errors, and positioning timeouts use the existing hold/recovery procedure. Faults are not skipped automatically, and stale queued commands are not replayed. Real-hardware replay and positioning paths still require on-site verification.

## 5. Export data

```powershell
.\setup.cmd -Export    # Install the export environment on first use.
.\export.cmd           # Convert.
```

Raw data: `outputs/episodes/<task>/episode_001/` (task and number set in the window). LeRobot output: `datasets/piperx/`. By default, nothing is uploaded and incomplete/simulated recordings are excluded. Older JPEG recordings can still be exported.

Each episode contains `head.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, `action.csv`, `samples.jsonl`, and `manifest.json`, with the same video/CSV row alignment as PiperCapture: frame N in all three videos corresponds to data row N in CSV and row N in JSONL. CSV contains `timestamp_unix_s` plus 16-dimensional action. JSONL retains actual source timestamps, frame numbers, state, and time bases.

Recording uses a **fixed 30 Hz time grid**. The maximum source timestamp span across all three images and both arms' action/state is **one frame (33.33 ms)**. RealSense SDK Global Timestamp is preferred; SDK `system_time` also retains its frame timestamp. If only raw hardware clocks are available, preview is allowed but Python receive times are not used to claim synchronized capture. D555 DDS Global Timestamp support depends on SDK/firmware; inspect actual `camera_timestamps`. Global Timestamp aligns clocks; exposure differences still require measurement.

MP4 files retain the complete capture timeline instead of shortening playback by reducing frame counts at gaps. Missing positions repeat the previous image group or use black frames before the first observation. Rows are marked `sample_valid=false`, `frame_status=held/blank`, and `held_from_grid`, with original source timestamps retained. Only new real synchronized frames are marked `captured`. The manifest records valid frames, placeholders, and capture/video durations; episodes with placeholders remain `incomplete`. For example, if 30 seconds contain only 150 real image groups, the video still has 900 frames and lasts 30 seconds. Missing observations or actions cannot be recovered. Placeholders are invalid for training/demonstration; default export and replay reject them.

The matcher uses only unused real camera frames as valid samples, preserving source frame order. Background encoding writes the complete 30 FPS timeline without blocking the control thread on disk. A full writer queue records a gap, later filled with explicitly marked placeholders, without stopping teleoperation. Missing grid frames are continuously logged with reasons and mark the entire episode `incomplete`. `manifest.json` retains `effective_fps`, `dropped_grid_ticks`, `drop_reasons`, and `observed_max_synchronization_span_s`, so incomplete recordings at an effective 20–25 Hz are not mistaken for valid 30 Hz data.

`action` is the sent target; `state` is actual feedback. Each contains `[x,y,z,qx,qy,qz,qw,gripper width]` for each side, for 16 dimensions total, using meters and each arm's base frame. Synchronization is software time alignment, not hardware exposure synchronization.

## Troubleshooting

- **Connection unexpectedly closes during installation:** rerun `.\setup.cmd` (add `-Export` for the export environment). Setup tries the Tsinghua mirror first, then GitHub, with retries and installer checksums. Valid complete downloads are reused; interrupted or invalid files are downloaded again. If it still fails, check proxy/network access to both sources.
- **Test controllers without moving robots:** `.\teleop.cmd --transport usb --input-test`; inspect tracking, grip, and trigger states in the window.
- **Simulate without hardware:** `.\teleop.cmd --transport usb --mock`.
- **paused:** restore the required controller's tracking, release buttons/triggers for 0.3 seconds, then press again. Re-squeeze the trigger for gripper control. Losing an idle controller does not block the other hand.
- **Camera status messages:** low frame rate, stale frames, warm-up, unsynchronized timestamps, and SDK errors appear in the footer. Cameras reconnect automatically while teleoperation and recording continue. Failed synchronization positions are explicitly invalid placeholders. Cameras no longer participate in robot readiness or recovery conditions.
- **One arm rejects a target:** if controller state is 4 with no error bits, only that arm receives a request to hold its current joints, and Cartesian targets to it stop. The other arm and recording continue. Once its controller is normal, release and press the grip to establish a new reference. No reset or disable is issued. The interface reports a hold request and does not claim an unconfirmed hold has succeeded.
- **recovery:** states requiring explicit recovery, such as teaching, retain the recovery procedure. No movement commands are sent during teaching. After manual exit, fresh measured joints replace old targets. When both arms are NORMAL and enabled, Quest input is valid, and buttons/triggers have been released for 0.3 seconds, automatic recovery runs and requires grip re-engagement. Recovery is not forced if an arm is disabled or faults remain. Communication and motor hardware fault protection remain active.
- **fault:** inspect `outputs/logs/teleop.log`. To clear supported recoverable controller faults, exit the application, **physically support both arms**, then run `.\teleop.cmd recover --supported`. Recovery removes motor power; it does not return the arms home.

Code: `piperx_teleop/`; tests: `tests/`; third-party SDKs: `vendor/` (with their licenses retained). See [verification and limitations (Chinese)](docs/validation.md).
