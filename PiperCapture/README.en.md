# PiperCapture

[简体中文](README.md)

Records dual-arm leader–follower Piper X demonstrations and synchronized head and wrist RGB video.

## Setup

Set up the environment and discover devices on Windows from `PiperCapture/`.

```powershell
.\setup.cmd
.\run.cmd discover
.\run.cmd discover-arms
```

Install the USB-CAN adapter driver and check camera serials and CAN channels in `config.json`. Close PiperX_Teleop before using the same arms.

## Usage

```powershell
.\run.cmd
```

Use the robot's teaching controls to make the arms movable by hand. Enter a task name and episode number, then start recording. Stop and wait for saving to finish before the next episode. The shared-CAN leader–follower setup can save follower reference positions and has custom homing disabled.

Set up the export environment on the first LeRobot export.

```powershell
.\setup.cmd -Export
.\export.cmd
```

The exporter reads `outputs/` and writes to `datasets/pipercapture/`. Use `--source` to select a recording directory.

## Verification

```powershell
.\run.cmd --mock
```

Record a mock episode and check its previews, save status, and output files. With real hardware, the window shows camera FPS, feedback rate, and valid action rate.

## Data format

Each `outputs/<task>/episode_001/` contains `head.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, `action.csv`, `samples.jsonl`, and `manifest.json`.

The three videos, CSV, and JSONL align by frame. The CSV stores a Unix timestamp in seconds plus `x,y,z,qx,qy,qz,qw,gripper` for each arm. Poses use each arm's base frame, with positions and gripper widths in meters. The state is the current measurement. The action is the measurement at the next sample time, recorded in JSONL as `action_timestamp_unix_s`.
