# PiperCapture

[简体中文](README.md) | [English](README.en.md)

**Demonstration capture with two PiperX leader–follower pairs**, a D555, and two D405 cameras. No Quest is required.

The current wiring uses two USB-CAN adapters, each connected to a leader–follower pair on the CAN side. Existing hardware following behavior is preserved.
On real hardware, “Save follower reference pose” saves only the follower positions; **returning both leaders and followers to a custom home pose is not implemented, and its button is disabled**.
The official `move_leader_follower_to_home()` returns the pair to zero and cannot substitute for a custom working pose. Recording, preview, synchronization, and export remain available.

## Usage

```powershell
cd D:\PiperX_WS\PiperCapture
.\setup.cmd        # Install a local environment and dependencies on first use.
.\run.cmd --mock    # Simulate without hardware.
.\run.cmd           # Capture with real hardware.
```

Run `setup.cmd` before first use to install a local Conda environment and dependencies. Set camera serial numbers and left/right CAN channels in `config.json`; use `run.cmd discover` to list cameras. The repository does not contain the developer's environment. The launcher can also reuse a sibling `PiperX_Teleop` Conda environment already installed on your computer.

`run.cmd discover-arms` lists USB-CAN channels and adapter identifiers without moving the arms. Detecting two channels is expected for the current two leader–follower pairs.

**Do not run PiperX_Teleop at the same time.** Use the robot's own teaching controls to enter a manually movable state. During recording, the application only reads feedback; it does not automatically disable motors or switch teaching modes.

## Desktop controls

Before recording, enter a **task name and episode number** (a positive integer). The next available number is suggested and can be edited; existing directories are never overwritten. These fields are locked during recording and saving, and can be changed before the next episode.

- **Start recording / Stop and save** (开始录制 / 结束保存): record one episode; wait until saving finishes before starting another.
- **Save follower reference pose** (保存从臂参考位): save the joint angles and gripper openings of both followers, excluding the leaders.
- **Return home / Cancel return** (返回初始位 / 取消返回): available only in simulation or configurations with two independently controlled arms. These commands are not sent to real hardware with the current shared-bus leader–follower wiring.
- **Exit** (退出): save an active recording before closing; cancel an active return motion first.

Click the Chinese buttons below the previews. Results or rejection reasons appear above them. R/S/H/G/C/Q shortcuts, in either case, also work while the preview window has focus; terminal keystrokes are not control inputs.

Stop recording and wait for saving to finish before saving a reference pose. The application does not automatically release the motors or change leader–follower roles.

The window shows the three RGB streams and FPS, robot feedback rate, valid action rate, and saved frame count. A `fault` stops capture and marks the episode incomplete; check the connections and robot state before restarting.

## Data and synchronization

- Raw data: `outputs/<task_name>/episode_001/`. Home pose: `home.json` (simulation uses a separate file).
- Each episode contains `head.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, `action.csv`, `samples.jsonl`, and `manifest.json`. JPG files are no longer written.
- Task names and episode numbers name directories and are not additionally written into data files. Existing recordings under `outputs/episodes/` are preserved.
- Frame N in all three videos corresponds to data row N in the CSV and row N in JSONL. Dropped video frames are not filled; use timestamps to determine actual intervals.
- `action.csv` has exactly 17 columns: `timestamp_unix_s`, eight dimensions for the left arm, and eight for the right arm. It excludes joint angles and state.
- All saved capture timestamps use **Unix seconds** (UTC); CSV values have six decimal places. Internal synchronization uses a monotonic clock with a fixed mapping to Unix time for each episode.
- Software alignment runs at 30 Hz and jointly selects the three camera frames, with per-camera deviation ≤20 ms and a three-camera span ≤25 ms. This is not hardware exposure synchronization.
- Cameras preferentially use `global_time` (device time mapped to the computer clock); `system_time` uses SDK frame time. The source is recorded as `timestamp_basis` in JSONL, with `host_receive` used only as a fallback when unavailable.
- The window displays drop reasons; `manifest.json` records `drop_reasons` and the longest consecutive gap in seconds. `complete` means saving finished normally, not that no frames were dropped.
- `state[t]` is the current measured state; `action[t]` is the **measured state at the next sampling time**, not an SDK command.
- Each side contains `[x,y,z,qx,qy,qz,qw,gripper width]`, for 16 dimensions total, in the respective base frame, using meters and `xyzw` quaternions.
- CSV timestamps correspond to the current RGB frames. The next-time action timestamp is stored separately as `action_timestamp_unix_s` in JSONL. JSONL retains state, joint angles, and source Unix timestamps. Action independently matches robot feedback at the next sampling time and does not depend on the next set of camera frames. Tail frames whose actions would exceed the recording endpoint are discarded.

## Convert to LeRobot

```powershell
.\setup.cmd -Export  # Required on a new computer; can be skipped after setup.
.\export.cmd
```

Output: `datasets/pipercapture/`, using LeRobot 0.4.2 / v3. Both new MP4 and older JPG recordings are supported. The exporter splits at gaps, skips incomplete data, rejects simulated data, and does not upload anything.

The default source is `outputs/`, searched recursively. Use `--source outputs/pick_cup` for one task; task descriptions come from directory names. For older recordings, use `--source outputs/episodes`.

## Python API

```python
import time
from piper_capture.capture import Capture, load_config

with Capture(load_config("config.json")) as cap:
    while cap.status()["mode"] == "warming":
        time.sleep(0.05)
    cap.record(task="pick_cup", capture_index=1)
    input("Move the arms to demonstrate; press Enter to stop recording")
    cap.stop()
    while cap.status()["saving"]:
        time.sleep(0.05)
    print(cap.status())
```

`set_home()`, `go_home()`, and `cancel_home()` are also available. Operations return status dictionaries. `status()` includes `mode/message/recording/saving/frames/dropped/episode/fps`.

Code layout: `capture` for orchestration, `core` for synchronization, `cameras/robot` for devices, `recording/export` for data, and `app` for the window.
