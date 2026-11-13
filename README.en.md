# Piper Workspace

[简体中文](README.md) | [English](README.en.md)

- [PiperCapture](PiperCapture/README.en.md): leader–follower robot demonstration capture, three-view previews, synchronized RGB/action, MP4/CSV/JSONL, and LeRobot export.
- [PiperX_Teleop](PiperX_Teleop/README.en.md): Quest 3 dual-arm teleoperation over USB or Wi-Fi.
- [PiperX_GO](PiperX_GO/README.md): remote ACT / Diffusion Policy inference and local dual-arm execution, with 30Hz control, RTC / NO-RTC scheduling, and action postprocessing. Detailed documentation is in Chinese.
- [HumanCapture](HumanCapture/README.en.md): synchronized three-camera human demonstration capture, with a white interface, live previews, episode numbering, MP4 video, and per-frame timestamps.
- [HumanLabel](HumanLabel/README.en.md): EgoHand + native smoothing + temporal-context crop recovery; exports wrist epose in the head camera frame, PiperCapture-format gripper distances, and full comparison videos.

## Capture

On Windows 10/11, install the project-local Conda environment before the first run:

```powershell
cd PiperCapture
.\setup.cmd
# Set camera serial numbers and CAN channels in config.json.
.\run.cmd
```

Enter a task name and episode number in the window, then use “Start recording / Stop and save” (开始录制 / 结束保存):

```text
PiperCapture/outputs/pick_cup/episode_001/
  head.mp4
  left_wrist.mp4
  right_wrist.mp4
  action.csv
  samples.jsonl
  manifest.json
```

The next episode number is suggested automatically and can be edited. Existing data is never overwritten. Returning to a custom home pose is currently unsupported with the shared-bus leader–follower wiring.

## Export

```powershell
.\setup.cmd -Export
.\export.cmd
```

By default, the exporter reads `PiperCapture/outputs/` and writes to `PiperCapture/datasets/pipercapture/`.

## Policy inference and execution

The `PiperX_GO` server loads a checkpoint in the model environment. The PC client captures three camera views and dual-arm feedback, requests action chunks over WebSocket + MessagePack, and handles time alignment, smoothing, command limits, and SDK dispatch locally.

Once the server is running, use the configured `PiperX_Teleop` environment to run a read-only check from the repository root:

```powershell
python PiperX_GO/piperx_execution/client.py --server ws://127.0.0.1:8000 --teleop-root PiperX_Teleop --max-steps 60
```

RTC is enabled by default. Add `--execute` to enable the arms, run the home sequence, and execute model actions; use `--no-rtc` for synchronous scheduling. See the [PiperX_GO README](PiperX_GO/README.md) for server startup, dependencies, message formats, and postprocessing parameters.

The repository includes code, tests, SDK sources and licenses, example configurations, and setup scripts. It also includes the current `config.json` files for HumanCapture and PiperX_Teleop (camera serial numbers, capture parameters, pedal bindings, and teleoperation settings), so the same hardware can be used on a new computer. PiperCapture's machine-specific configuration is kept separately. Local Conda installations, recordings, certificates, and caches are excluded.
