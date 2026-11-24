# PiperX_Teleop

[简体中文](README.md)

Teleoperates two Piper X arms with a Quest 3 and records synchronized RGB, actions, and states.

## Setup

Set up the environment and discover devices on Windows from `PiperX_Teleop/`.

```powershell
.\setup.cmd
.\teleop.cmd discover
```

Install the USB-CAN adapter driver. Check camera serials, CAN channels, firmware, TCP offsets, workspaces, and homing waypoints in `config.json`.

## Usage

**USB**. Enable developer mode on Quest, connect USB, and authorize debugging in the headset. Run the command and open `http://localhost:8012` on Quest.

```powershell
.\teleop.cmd
```

**Wi-Fi**. Connect both devices to the same network, replace the example IP with the PC's Wi-Fi address, and allow TCP port 8012. Open the HTTPS address printed in the terminal.

```powershell
.\teleop.cmd cert --ip 192.168.1.100
.\teleop.cmd --transport wifi --cert certs/cert.pem --key certs/key.pem
```

### Align the headset before tracking

Face the headset along robot-base +X before starting controller tracking. Hold the middle-finger grip button to move the corresponding arm. Release it to hold the last commanded pose. The index trigger controls the gripper independently.

| Button | Action |
| --- | --- |
| B / A | Start recording / stop and save |
| Hold X for 2 s | Save the current dual-arm pose as home |
| Hold Y for 1.5 s | Home both arms along the configured waypoints |

Release the grip buttons and finish saving before saving home or homing. For CSV replay, load the file in the desktop window, check the path, and start replay. Use the stop button or Space to stop.

### Chest-mounted capture

With USB connected, run `.\teleop.cmd awake` and keep the controllers visible to Quest. Run `.\teleop.cmd sleep` when finished.

For LeRobot export, run `.\setup.cmd -Export` once, then `.\export.cmd`. The dataset is saved under `datasets/piperx/`.

## Verification

```powershell
.\teleop.cmd --mock
.\teleop.cmd --transport usb --input-test
```

The first command uses mock devices. The second checks Quest and real camera inputs with simulated robot feedback. [Hardware checks and validation notes](docs/validation.md)

## Data format

Each `outputs/episodes/<task>/episode_001/` contains three 30 FPS MP4s, `action.csv`, `samples.jsonl`, and `manifest.json`. Video and data align by frame.

The CSV contains a Unix timestamp in seconds and `x,y,z,qx,qy,qz,qw,gripper` for each arm. Poses use each arm's base frame, with positions and gripper widths in meters. The action is the dispatched target and the JSONL state is measured feedback. Held or blank frames preserve the timeline and carry `sample_valid=false`.
