# HumanCapture

[简体中文](README.md)

Records synchronized RGB videos from a head-mounted D555 and two wrist-mounted D405i cameras during human demonstrations.

## Setup

Install Python 3.12 with Tk on Windows. Then set up the environment and discover cameras from `HumanCapture/`.

Choose **Record** or **Breakpoints** in the capture window. In Record mode, pedal presses start and stop episodes. In Breakpoints mode, start continuous recording with the on-screen button; pedal presses then alternate `start` and `end` markers in `clip.csv` inside the episode directory. The `State,Timestamps` CSV stores Unix seconds on the same global synchronized time axis as `timestamps.csv`. The pedal mode is saved in `config.json`.

```powershell
.\setup.cmd
.\capture.cmd discover
```

Setup creates a local `.venv`. Check the camera serials and RGB modes in `config.json` against discovery.

## Usage

```powershell
.\capture.cmd
```

When all three views are ready, enter a task name and episode number, then start recording. Stop and save before starting the next episode. To use a keyboard-style pedal, bind it in the capture window and keep that window in the foreground to start and stop recording.

## Verification

```powershell
.\capture.cmd --mock --smoke-seconds 6
.\capture.cmd --diagnose-seconds 300
```

The first command checks saving with mock cameras. The second reads the real cameras and writes diagnostics to `outputs/logs/capture.log`. [Validation notes](docs/validation.md)

## Data format

Episodes are saved under `outputs/episodes/<task>/episode_001/`. The three 30 FPS MP4s align frame by frame with `timestamps.csv` and `samples.jsonl`. The `manifest.json` records completion, valid frames, and dropped frames.

Rows with `sample_valid=false` are placeholders. Held or blank images preserve the timeline, and the episode is marked `incomplete`.
