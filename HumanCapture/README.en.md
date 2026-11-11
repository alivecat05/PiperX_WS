# HumanCapture

[简体中文](README.md) | [English](README.en.md)

Three-camera RGB capture for human demonstrations: two D405i wrist cameras and one D555 head camera. Runs independently, without Quest, CAN, or a robot SDK. The white interface shows each live stream's FPS, camera frame number, and timestamp source. A green banner means ready but idle; red means recording.

## Start

Before first use, install Python 3.12 with Tk support, or set up the sibling `PiperX_Teleop` environment. In PowerShell, enter this repository's `HumanCapture/` directory:

```powershell
cd D:\PiperX_WS\HumanCapture
.\setup.cmd
.\capture.cmd discover
.\capture.cmd
```

`setup.cmd` creates a separate local `.venv` and installs dependencies; the developer's environment is not included in the repository. For later launches, run `.\capture.cmd`. The launcher can also directly reuse a sibling PiperX_Teleop environment already installed on your computer.

The repository includes the current hardware's `config.json`, preserving the three camera serial numbers, capture parameters, and detected pedal key (currently F8). Reuse it when moving the same hardware to a new computer; setup preserves existing configuration. Update serial numbers under `cameras` when changing cameras. If configuration is missing, copy `config.example.json` (setup does this automatically). `discover` lists SDK-visible devices and actual RGB modes. The example uses DDS domain 0, D555 at 896×504@30, and both wrists at 640×480@30; the current hardware configuration uses D555 at 1280×800@30 and both wrists at 1280×720@30. `model: D405` matches device model names containing D405. Actual enumerated modes take precedence; the application does not silently switch to lower frame rates. All three cameras must be running before recording can start.

To test the software without connected devices:

```powershell
.\capture.cmd --mock
.\capture.cmd --mock --smoke-seconds 6
```

Simulated recordings have `simulated: true` in their manifest. Real operation reports a configuration error if camera serial numbers are missing.

## Operation

1. Enter a task name and episode number. Valid edits synchronize automatically; the start button also uses the current input directly.
2. Once the green banner appears, click “Start recording” (开始录制). Task and episode fields are locked during recording and saving.
3. Click “Stop recording and save” (结束录制并保存). The application aligns tail frames before the cutoff, finishes writing MP4 files, and checks frame counts. Wait for saving to finish before starting another episode.
4. With automatic increment enabled, saving selects the next available number, skipping occupied directories. You can also click “Next available number” (下一个可用序号). Existing data, including incomplete episodes, is never overwritten.

Closing the window first stops and saves the active recording. Camera drops, timeouts, brief outages, and a full writer queue do not automatically end recording. Missing synchronized groups are counted; cameras reconnect automatically and continue into the same episode. Only an explicit stop, closing the application, or an actual write failure ends recording. Episodes with gaps are retained and marked `incomplete`. Missing positions repeat the previous image group, or use black frames before the first group, and are explicitly marked invalid. UI settings apply to the current run; the default task comes from configuration.

At startup, SDK default stream settings are cleared and only RGB is enabled; actual streams are checked to exclude depth or other streams. Each camera warms up with automatic exposure for two seconds, then its exposure and gain are read and locked. Reconnection within the same run reuses those values instead of running automatic exposure again. Unsupported locking or mismatched readback produces an explicit error. Locked values use the sensor's native SDK units and appear in the preview, per-frame metadata, and logs.

D405 preferentially reads actual exposure/gain from auto-exposed frame metadata, avoiding stale manual option values of 33000. Its exposure in microseconds is capped at 90% of the frame period: 30000 µs at 30 FPS. On the tested hardware, a manual exposure of 33000 µs reduced output to 15 FPS; 30000 µs restored approximately 30 FPS. D555 uses its own native units.

## Start / stop with a foot pedal

USB pedals that emulate keyboard keys are supported without extra libraries. Click “Detect pedal” (识别踏板), then press the pedal once. This detects the key without recording. After release, each press toggles “Start recording / Stop and save.” The binding is written automatically to local `config.json` and survives restarts. Press Esc to cancel detection.

Holding the pedal, keyboard auto-repeat, and contact bounce within 250 ms do not cause repeated toggles. Presses during saving are ignored; press again after saving to begin the next episode. Starting is rejected with a reason if cameras are not ready or input is invalid. Pedal and manual controls can be mixed; starting uses the current task and episode fields.

The capture window must have focus. The application intercepts the bound key so it does not insert spaces into the task field or activate a focused button. Pressing the same key on a keyboard also toggles recording. For programmable pedals, a dedicated key such as F13 is recommended. This interface supports keyboard-emulating pedals; serial or other HID protocols require device-specific integration.

Configuration example (normally, use the detection button):

```json
"pedal": {
  "key": "F13",
  "debounce_ms": 250
}
```

## Data and synchronization

Output directory: `outputs/episodes/<task_name>/episode_001/`, containing:

- `head.mp4`, `left_wrist.mp4`, `right_wrist.mp4`: three 30 FPS RGB videos.
- `timestamps.csv`: frame index, 30 Hz grid index, unified observation Unix time, three source Unix timestamps, original camera frame numbers, and maximum time difference.
- `samples.jsonl`: per-frame timestamps, clock sources, receive times, and camera metadata.
- `manifest.json`: task, episode number, configuration, simulation flag, completion status, frame counts, drop reasons, and synchronization statistics.

Frame N in each MP4 corresponds to data entry N in CSV and JSONL (zero-based). This application captures video; it does not generate robot actions or estimate human joints.

Synchronization follows PiperX_Teleop: each camera has an independent capture thread. RealSense Global Timestamp is enabled when supported by the SDK; `global_time` / `system_time` are accepted and mapped to a shared `perf_counter()` monotonic clock. After a 120 ms buffer, unused real frames are jointly matched to a fixed 30 Hz grid, with the total source timestamp span capped at 33.33 ms. The buffer waits for transmission arrivals without relaxing that span. Unsynchronized raw hardware clocks allow preview only and block recording. Saved timestamps are mapped to Unix seconds.

This is software alignment based on SDK timestamps, not hardware exposure-trigger synchronization across the three cameras. Valid samples must use real, unique frames. MP4 files retain the complete 30 Hz timeline; gaps repeat the previous frames or use black placeholders. Dropped frames do not compress the timeline: a 30-second capture produces approximately 30 seconds of video, within one frame. For example, 150 valid frame groups still occupy 900 time points instead of shrinking 30 seconds into five.

Each CSV / JSONL row corresponds to an MP4 frame. Only `sample_valid=true` with `frame_status=captured` indicates a new real synchronized frame. `sample_valid=false` with `frame_status=held/blank` indicates a placeholder; `held_from_grid` points to the reused valid frame, and source timestamps retain their original values or are null. Do not treat placeholders as new observations. The manifest separates `valid_frames/placeholder_frames`; `video_duration_s/capture_duration_s` allow duration checks. `effective_fps` still counts only valid source frames.

GUI preview, synchronization, camera acquisition, and video encoding run separately; preview resizing does not reduce saved resolution. Auto-exposure priority frame-rate reduction is disabled during exposure warm-up as well. Check the UI and saved statistics to confirm whether the hardware sustains 30 FPS. Camera history is limited to the synchronization buffer plus 0.5 seconds, and the writer queue is capped at approximately 128 MiB, avoiding the previous roughly 1.3 GB image buffering at the current full-resolution settings.

To investigate gaps, inspect `manifest.json`. Under `camera_diagnostics`, `source_gaps` reports SDK frame-number gaps, `max_receive_gap_ms` the peak receive interval, and `arrival_delay_ms` recent transmission/clock-mapped arrival delay. `timeouts/reconnects/timestamp_rejections` distinguish outages from clock problems. `writer_diagnostics` records encoding times and peak queue usage; `drop_reasons.writer_queue_full` indicates an encoding or disk backlog. Complete source frames with synchronization gaps can reflect source timestamp spans or arrival delays; not every synchronization gap is USB packet loss.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests -q
```

Tests use synthetic timestamps and real MP4 encoding. They cover continuous 30 Hz matching, no frame reuse, excessive time differences, invalid clocks, task numbering and overwrite prevention, stop-tail handling, and camera loss. They do not operate real hardware.
