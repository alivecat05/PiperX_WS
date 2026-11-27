"""Stitch synchronized camera videos into one labeled MP4 (FFmpeg required).

Examples:
  python stitch_videos.py --report results/multi_video_sync_head/group3/sync_report.json
  python stitch_videos.py --input-dir results/multi_video_sync_head/group3/synced
  python stitch_videos.py --duration 2 --output results/group3_preview.mp4

Report mode reads original videos and applies the exact frame mapping, so an
intermediate export of the nine synchronized videos is not required.
"""
import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "results/multi_video_sync_head/group3/sync_report.json"
CAMERA_ORDER = [f"j{i}" for i in range(6)] + ["head", "left_wrist", "right_wrist"]


@dataclass
class Clip:
    name: str
    path: Path
    start_frame: int
    fps: Fraction
    total_frames: int


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=avg_frame_rate,nb_frames,width,height", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if not streams:
        raise ValueError(f"No video stream: {path}")
    stream = streams[0]
    try:
        fps = Fraction(stream["avg_frame_rate"])
        frames = int(stream["nb_frames"])
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        raise ValueError(f"Cannot read frame rate/count from {path}") from exc
    if fps <= 0 or frames <= 0:
        raise ValueError(f"Invalid video timing: {path}")
    return fps, frames, stream


def validate_mapping(mapping_file, clips, expected_count):
    """Contiguous equal-FPS pairing is required for frame-exact FFmpeg trim."""
    count = 0
    with mapping_file.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"sync_frame"} | {f"{clip.name}_frame" for clip in clips}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing camera columns in {mapping_file}")
        for i, row in enumerate(reader):
            if int(row["sync_frame"]) != i:
                raise ValueError(f"Non-sequential sync_frame at row {i}")
            for clip in clips:
                if int(row[f"{clip.name}_frame"]) != clip.start_frame + i:
                    raise ValueError(
                        f"{clip.name}: non-contiguous mapping at row {i}; "
                        "this script does not resample or remove held frames"
                    )
            count += 1
    if count != expected_count or count <= 0:
        raise ValueError(f"Mapping has {count} rows; report expects {expected_count}")


def load_inputs(report_file=None, input_dir=None):
    if report_file:
        report_file = report_file.resolve()
        report = json.loads(report_file.read_text(encoding="utf-8"))
        cameras = {camera["name"]: camera for camera in report["cameras"]}
        unknown = set(cameras) - set(CAMERA_ORDER)
        if unknown:
            raise ValueError(f"Unsupported camera names: {sorted(unknown)}")
        clips = []
        for name in CAMERA_ORDER:
            if name not in cameras:
                continue
            camera = cameras[name]
            path = Path(camera["path"])
            if not path.is_absolute():
                path = report_file.parent / path
            fps, frames, _ = probe(path)
            if abs(float(fps) - camera["fps"]) > 1e-6:
                raise ValueError(f"{name}: actual FPS differs from the sync report")
            clips.append(Clip(name, path.resolve(), int(camera["trim_start_frame"]), fps, frames))
        count = int(report["mapped_frame_count"])
        mapping_file = Path(report.get("frame_mapping_file", "frame_mapping.csv"))
        if not mapping_file.is_absolute():
            mapping_file = report_file.parent / mapping_file
        if not mapping_file.is_file():
            mapping_file = report_file.parent / "frame_mapping.csv"
        validate_mapping(mapping_file, clips, count)
        output_dir, group_name = report_file.parent, report.get("recording", report_file.parent.name)
    else:
        input_dir = input_dir.resolve()
        clips = []
        for name in CAMERA_ORDER:
            path = input_dir / f"{name}.mp4"
            if path.is_file():
                fps, frames, _ = probe(path)
                clips.append(Clip(name, path, 0, fps, frames))
        count = min((c.total_frames for c in clips), default=0)
        output_dir = input_dir.parent if input_dir.name == "synced" else input_dir
        group_name = output_dir.name
    required = {f"j{i}" for i in range(6)} | {"left_wrist", "right_wrist"}
    if not required.issubset(c.name for c in clips):
        raise ValueError("Expected j0..j5, left_wrist, right_wrist, and optional head")
    if any(clip.fps != clips[0].fps for clip in clips):
        raise ValueError("All camera videos must have the same FPS; no implicit resampling")
    for clip in clips:
        if clip.start_frame < 0 or count <= 0 or clip.start_frame + count > clip.total_frames:
            raise ValueError(f"{clip.name}: requested frames are outside the source video")
    return clips, count, output_dir / f"{group_name}_{len(clips)}views.mp4"


def build_command(clips, count, output, cols, width, height, start_frame=0, font_file=None):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-filter_complex_threads", "2"]
    filters = []
    # Known labels contain no FFmpeg filter syntax. A Windows drive colon must
    # be escaped inside a quoted font path (the outer subprocess uses no shell).
    font = ""
    if font_file:
        value = font_file.resolve().as_posix()
        if "'" in value:
            raise ValueError("Font path cannot contain a single quote")
        font = "fontfile='" + value.replace(":", "\\:") + "':"
    for i, clip in enumerate(clips):
        command += ["-threads", "1", "-i", str(clip.path)]
        first = clip.start_frame + start_frame
        label = clip.name.upper().replace("_", " ")
        filters.append(
            f"[{i}:v]trim=start_frame={first}:end_frame={first+count},setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"drawtext={font}text='{label}':fontsize=24:fontcolor=white:"
            f"box=1:boxcolor=black@0.65:boxborderw=6:x=12:y=12[v{i}]"
        )
    layout = "|".join(f"{i % cols * width}_{i // cols * height}" for i in range(len(clips)))
    inputs = "".join(f"[v{i}]" for i in range(len(clips)))
    filters.append(f"{inputs}xstack=inputs={len(clips)}:layout={layout}:fill=black:shortest=1[out]")
    command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "4",
                "-pix_fmt", "yuv420p", "-frames:v", str(count), "-fps_mode", "passthrough",
                "-movflags", "+faststart", str(output)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--report", type=Path, help="Sync JSON and its frame_mapping.csv; reads original videos")
    source.add_argument("--input-dir", type=Path, help="Directory of already synchronized MP4 files")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cols", type=int, help="Default: 3 columns for nine cameras, 4 for eight")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480)
    parser.add_argument("--start", type=float, default=0, help="Start seconds within the synchronized interval")
    parser.add_argument("--duration", type=float, help="Optional output duration in seconds")
    parser.add_argument("--font-file", type=Path)
    args = parser.parse_args()
    if any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")):
        parser.error("ffmpeg and ffprobe must be on PATH")
    if any(v < 2 or v % 2 for v in (args.tile_width, args.tile_height)):
        parser.error("Tile width and height must be positive even integers")
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("--start must be a finite nonnegative value")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    report = args.report if args.report else (None if args.input_dir else DEFAULT_REPORT)
    try:
        clips, count, default_output = load_inputs(report, args.input_dir)
        fps = clips[0].fps
        first = math.floor(args.start * fps + 0.5)
        count -= first
        if args.duration is not None:
            count = min(count, math.floor(args.duration * fps + 1e-8))
        if count <= 0:
            raise ValueError("Requested interval contains no frames")
        cols = args.cols if args.cols is not None else (3 if len(clips) == 9 else 4)
        if not 1 <= cols <= len(clips):
            raise ValueError("--cols must be between 1 and the number of cameras")
        output = (args.output or default_output).resolve()
        if output in {clip.path.resolve() for clip in clips}:
            raise ValueError("Output path must differ from every source video")
        output.parent.mkdir(parents=True, exist_ok=True)
        font = args.font_file
        if font is None:
            font = next((p for p in (Path("C:/Windows/Fonts/arial.ttf"),
                                     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) if p.is_file()), None)
        elif not font.is_file():
            raise ValueError(f"Font file not found: {font}")
        rows = math.ceil(len(clips) / cols)
        print(f"{len(clips)} cameras, {cols}x{rows}, {cols*args.tile_width}x{rows*args.tile_height}, "
              f"{float(fps):g} fps, {count} frames ({float(count/fps):.3f}s)", flush=True)
        print("Order: " + ", ".join(clip.name for clip in clips), flush=True)
        subprocess.run(build_command(clips, count, output, cols, args.tile_width, args.tile_height, first, font), check=True)
        actual_fps, actual_count, stream = probe(output)
        if (actual_fps != fps or actual_count != count or stream["width"] != cols*args.tile_width
                or stream["height"] != rows*args.tile_height):
            raise ValueError("Exported video timing or dimensions do not match the requested mosaic")
        print(f"Saved: {output}", flush=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
