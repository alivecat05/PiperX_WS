"""QR-only alignment of a six-camera rig and a synchronized RealSense group.

The RealSense group contains left/right wrists and optionally head. Only wrists
are scanned for QR codes. Head inherits the same frame mapping and offset.
Frame indices are zero-based. Positive offset means RealSense starts later:
rig_time_seconds = realsense_time_seconds + offset_seconds.
"""
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import zxingcpp

DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "results" / "multi_video_sync_head"
NS = 1_000_000_000
DAY_NS = 86_400 * NS


@dataclass
class ParsedTimestamp:
    timestamp_ns: int
    normalized_text: str
    fraction_digits: int
    has_date: bool


@dataclass
class QRHit:
    frame_idx: int
    source: str
    timestamp_ns: int
    normalized_text: str
    fraction_digits: int
    has_date: bool
    decoder: str
    preprocessing: str


def log_progress(message):
    print(message, flush=True)


def parse_timestamp_text(text: str) -> ParsedTimestamp | None:
    text = text.strip().translate(str.maketrans({"：": ":", "。": ".", "，": "."}))
    compact = re.sub(r"\s+", "", text)
    match = re.search(
        r"(?:(?P<year>20\d{2})[-./]?(?P<month>\d{2})[-./]?(?P<day>\d{2})\D*?)?"
        r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
        r"(?:\.(?P<fraction>\d{1,9}))?(?!\d)", compact,
    )
    if not match:
        return None
    hour, minute, second = (int(match[k]) for k in ("hour", "minute", "second"))
    if hour > 23 or minute > 59 or second > 59:
        return None
    fraction = match["fraction"] or ""
    timestamp = (hour * 3600 + minute * 60 + second) * NS + int(fraction.ljust(9, "0") or 0)
    has_date = match["year"] is not None
    if has_date:
        try:
            timestamp += date(*(int(match[k]) for k in ("year", "month", "day"))).toordinal() * DAY_NS
        except ValueError:
            return None
    return ParsedTimestamp(timestamp, compact, len(fraction), has_date)


def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise RuntimeError(f"Invalid video timing: {path}")
    return {"path": str(path.resolve()), "name": path.stem, "fps": fps,
            "total_frames": count, "duration_seconds": count / fps}


def group_paths(group):
    # Both flat group3/*.mp4 and legacy groupN/{rig,episode}/*.mp4 layouts.
    directories = [group] + sorted(child for child in group.iterdir() if child.is_dir())
    rigs = [child for child in directories
            if all((child / f"j{i}.mp4").is_file() for i in range(6))]
    wrists = [child for child in directories
              if all((child / f"{side}_wrist.mp4").is_file() for side in ("left", "right"))]
    if len(rigs) != 1 or len(wrists) != 1:
        raise ValueError(f"{group}: expected exactly one six-camera folder and one wrist folder")
    sdk_paths = [wrists[0] / f"{side}_wrist.mp4" for side in ("left", "right")]
    if (wrists[0] / "head.mp4").is_file():
        sdk_paths.append(wrists[0] / "head.mp4")
    return [rigs[0] / f"j{i}.mp4" for i in range(6)], sdk_paths


def validate_realsense_grid(videos):
    """Validate the existing output grid, without re-aligning source exposures.

    Source timestamps may repeat on held frames. Their row positions must be
    preserved; using those timestamps to shift individual cameras would undo
    the SDK synchronization already performed by the recording application.
    """
    timestamp_file = Path(videos[0]["path"]).parent / "timestamps.csv"
    info = {"cameras": [v["name"] for v in videos],
            "basis": "user_confirmed_SDK_synchronized_common_frame_grid",
            "timestamp_file": None}
    if not timestamp_file.is_file():
        return info
    repeats = {v["name"]: 0 for v in videos}
    previous_sources = {}
    previous_grid = None
    row_count, invalid_rows = 0, 0
    period_ns = Decimal(NS) / Decimal(str(videos[0]["fps"]))
    with timestamp_file.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"frame_index", "timestamp_unix_s"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{timestamp_file}: missing common-grid columns {sorted(required)}")
        for index, row in enumerate(reader):
            try:
                frame_index = int(row["frame_index"])
                grid_ns = Decimal(row["timestamp_unix_s"]) * NS
            except (ValueError, InvalidOperation) as exc:
                raise RuntimeError(f"{timestamp_file}: invalid timestamp at row {index}") from exc
            if frame_index != index or not grid_ns.is_finite():
                raise RuntimeError(f"{timestamp_file}: row {index} is not a valid MP4 frame-grid row")
            if previous_grid is not None and abs(grid_ns - previous_grid - period_ns) > 1_000_000:
                raise RuntimeError(f"{timestamp_file}: non-CFR grid at frame {index}")
            previous_grid = grid_ns
            for name in repeats:
                timestamp = row.get(f"{name}_timestamp_unix_s", "")
                if timestamp and timestamp == previous_sources.get(name):
                    repeats[name] += 1
                previous_sources[name] = timestamp
            invalid_rows += row.get("sample_valid", "true").lower() == "false"
            row_count += 1
    for video in videos:
        if video["total_frames"] != row_count:
            raise RuntimeError(f"{video['name']}: {video['total_frames']} MP4 frames but {row_count} timestamp rows")
    info.update({"timestamp_file": str(timestamp_file.resolve()), "rows": row_count,
                 "repeated_source_timestamp_rows": repeats, "sample_invalid_rows": invalid_rows,
                 "frame_alignment": "MP4 frame N = timestamps.csv row N; keep held/invalid rows"})
    return info


def discover_recordings(root):
    try:
        group_paths(root)
        return [root]
    except ValueError:
        return [child for child in sorted(root.glob("group*")) if child.is_dir()]


def build_green_binary(image):
    b, g, r = cv2.split(image.astype(np.int16))
    return 255 - (((g - r > 10) & (g - b > 10) & (g > 50)).astype(np.uint8) * 255)


def decode_qr_from_frame(frame):
    def decode(image, label):
        # ZXing handles all rotations internally.
        for result in zxingcpp.read_barcodes(image, formats=zxingcpp.BarcodeFormat.QRCode, try_denoise=True):
            parsed = parse_timestamp_text(result.text)
            if parsed is not None:
                return parsed, label
        return None

    direct = decode(frame, "raw")
    if direct:
        return direct
    # A small dark screen is lost by full-frame decoder downsampling. Locate
    # green components, crop with a quiet-zone margin, then enlarge the green
    # channel before decoding. Closing joins the individual QR modules.
    b, g, r = cv2.split(frame.astype(np.int16))
    green = ((g-r > 5) & (g-b > 5) & (g > 20)).astype(np.uint8) * 255
    joined = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    regions, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for region in sorted(regions, key=cv2.contourArea, reverse=True)[:3]:
        x, y, w, h = cv2.boundingRect(region)
        if min(w, h) < 60 or w*h > frame.shape[0]*frame.shape[1]*0.6:
            continue
        roi = frame[max(0, y-20):y+h+20, max(0, x-20):x+w+20, 1]
        for label, candidate in (
            ("green-roi", roi),
            ("green-roi-2x", cv2.resize(roi, None, fx=2, fy=2)),
            ("green-roi-clahe", cv2.createCLAHE(2, (4, 4)).apply(roi)),
        ):
            decoded = decode(candidate, label)
            if decoded:
                return decoded
    binary = build_green_binary(frame)
    decoded = decode(binary, "green-mask")
    if decoded:
        return decoded
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        rect = cv2.minAreaRect(contour)
        if min(rect[1]) < 40:
            continue
        box = cv2.boxPoints(rect).astype(np.float32)
        sums, diffs = box.sum(axis=1), np.diff(box, axis=1)[:, 0]
        ordered = box[[np.argmin(sums), np.argmin(diffs), np.argmax(sums), np.argmax(diffs)]]
        width = int(max(np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])))
        height = int(max(np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])))
        if width < 40 or height < 40 or len(set(map(tuple, ordered))) < 4:
            continue
        dest = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        warped = cv2.warpPerspective(binary, cv2.getPerspectiveTransform(ordered, dest), (width, height))
        warped = cv2.copyMakeBorder(warped, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
        decoded = decode(warped, "green-warp")
        if decoded:
            return decoded
    return None


def scan_video_frame_indices(video, indices):
    path = Path(video["path"])
    cap = cv2.VideoCapture(str(path))
    hits = []
    previous = -2
    try:
        for idx in sorted(set(indices)):
            if not 0 <= idx < video["total_frames"]:
                continue
            if idx != previous + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            previous = idx
            if not ok:
                continue
            decoded = decode_qr_from_frame(frame)
            if decoded:
                parsed, preprocessing = decoded
                hits.append(QRHit(idx, path.name, parsed.timestamp_ns, parsed.normalized_text,
                                  parsed.fraction_digits, parsed.has_date, "zxingcpp", preprocessing))
    finally:
        cap.release()
    return hits


def estimate_timeline(hits, fps):
    if len(hits) < 3:
        raise RuntimeError("At least three decoded QR frames are required")
    hits = sorted(hits, key=lambda h: h.frame_idx)
    # Use time-of-day for mixed dated/time-only payloads and unwrap midnight.
    # Integer arithmetic preserves nanoseconds without large calendar floats.
    base = hits[0].timestamp_ns % DAY_NS
    values = []
    for hit in hits:
        display = hit.timestamp_ns % DAY_NS
        expected = base + round((hit.frame_idx - hits[0].frame_idx) / fps * NS)
        display += round((expected - display) / DAY_NS) * DAY_NS
        values.append(display - round(hit.frame_idx / fps * NS))
    center = int(median(values))
    mad = median([abs(v - center) for v in values])
    precision_ns = max(10 ** (9 - min(h.fraction_digits, 9)) for h in hits)
    threshold = max(3 * NS / fps, 6 * mad, precision_ns)
    keep = [i for i, v in enumerate(values) if abs(v - center) <= threshold]
    if len(keep) < 3:
        raise RuntimeError("Too few mutually consistent QR observations")
    center = int(median([values[i] for i in keep]))
    residuals = np.array([(values[i] - center) / NS for i in keep])
    span = (hits[keep[-1]].frame_idx - hits[keep[0]].frame_idx) / fps
    unique = len({hits[i].timestamp_ns for i in keep})
    if unique < 3 or span < 0.5:
        raise RuntimeError("Need three changing QR timestamps spanning at least 0.5 seconds")
    stats = {"start_display_clock_ns": center, "decoded_frames": len(hits),
             "inlier_frames": len(keep), "rejected_frames": len(hits) - len(keep),
             "unique_timestamps": unique, "observation_span_seconds": span,
             "median_absolute_residual_ms": float(np.median(np.abs(residuals)) * 1000),
             "p95_absolute_residual_ms": float(np.percentile(np.abs(residuals), 95) * 1000),
             "timestamp_resolution_ms": precision_ns / 1e6}
    return center, stats


def find_anchor(videos, label, scan_seconds, sample_seconds):
    candidates = []
    for video in videos:
        fps = video["fps"]
        limit = min(video["total_frames"], math.ceil(scan_seconds * fps))
        indices = range(0, limit, max(1, round(sample_seconds * fps)))
        hits = scan_video_frame_indices(video, indices)
        log_progress(f"  [{label}] {video['name']}: {len(hits)} sparse QR hits")
        candidates.append((video, hits))
        if len(hits) >= 5:
            try:
                estimate_timeline(hits, fps)
                break
            except RuntimeError:
                pass
    for video, hits in sorted(candidates, key=lambda pair: len(pair[1]), reverse=True):
        if not hits:
            continue
        centers = [hits[i].frame_idx for i in np.linspace(0, len(hits)-1, min(6, len(hits)), dtype=int)]
        radius = max(2, round(video["fps"] * 0.5))
        refined = scan_video_frame_indices(video, [idx for c in centers for idx in range(c-radius, c+radius+1)])
        merged = {h.frame_idx: h for h in hits + refined}
        hits = sorted(merged.values(), key=lambda h: h.frame_idx)
        try:
            start, stats = estimate_timeline(hits, video["fps"])
            return {"video": video["name"], "start_ns": start, "statistics": stats,
                    "hits": [asdict(h) for h in hits]}
        except RuntimeError as exc:
            log_progress(f"  [{label}] {video['name']}: {exc}")
            # Brief QR visibility can fall between sparse samples. Rescue the
            # candidate over its full scan window before giving up.
            log_progress(f"  [{label}] {video['name']}: dense fallback")
            limit = min(video["total_frames"], math.ceil(scan_seconds * video["fps"]))
            dense = scan_video_frame_indices(video, range(limit))
            try:
                start, stats = estimate_timeline(dense, video["fps"])
                return {"video": video["name"], "start_ns": start, "statistics": stats,
                        "hits": [asdict(h) for h in dense]}
            except RuntimeError:
                pass
    raise RuntimeError(f"[{label}] No reliable QR timeline in first {scan_seconds:g}s; increase --scan-seconds or reduce --sample-seconds")


def build_sync_report(group, output_root, scan_seconds=120.0, sample_seconds=0.5):
    rig_paths, sdk_paths = group_paths(group)
    rig = [probe_video(p) for p in rig_paths]
    sdk = [probe_video(p) for p in sdk_paths]
    for label, videos in (("six_camera", rig), ("realsense", sdk)):
        if any(abs(v["fps"] - videos[0]["fps"]) > 1e-6 for v in videos):
            raise RuntimeError(f"{label}: inconsistent FPS inside synchronized subsystem")
    sdk_grid = validate_realsense_grid(sdk)
    # Never require head to see the QR display: wrists already share its grid.
    wrists = [v for v in sdk if v["name"] in ("left_wrist", "right_wrist")]
    rig_anchor = find_anchor(list(reversed(rig)), f"{group.name}/six_camera", scan_seconds, sample_seconds)
    wrist_anchor = find_anchor(wrists, f"{group.name}/wrist", scan_seconds, sample_seconds)
    delta_ns = wrist_anchor["start_ns"] - rig_anchor["start_ns"]
    delta_ns -= round(delta_ns / DAY_NS) * DAY_NS
    offset = delta_ns / NS
    common_start = max(0.0, offset)
    common_end = min(min(v["duration_seconds"] for v in rig),
                     offset + min(v["duration_seconds"] for v in sdk))
    if common_end <= common_start:
        raise RuntimeError(f"{group}: QR timelines have no common video window")
    cameras = []
    for kind, videos, start in (("six_camera", rig, 0.0), ("realsense", sdk, offset)):
        for video in videos:
            trim = common_start - start
            first = math.floor(trim * video["fps"] + 0.5)
            cameras.append({**video, "system": kind, "system_start_on_rig_seconds": start,
                            "trim_start_seconds": trim, "trim_start_frame": first,
                            "rounding_error_ms": (first / video["fps"] - trim) * 1000})
    report_dir = output_root / group.name
    report_dir.mkdir(parents=True, exist_ok=True)
    rig_first = math.floor(common_start * rig[0]["fps"] + 0.5)
    mappings = []
    for rig_idx in range(rig_first, min(v["total_frames"] for v in rig)):
        t = rig_idx / rig[0]["fps"]
        if t >= common_end - 1e-9:
            break
        wrist_idx = math.floor((t - offset) * sdk[0]["fps"] + 0.5)
        if not 0 <= wrist_idx < min(v["total_frames"] for v in sdk):
            continue
        mappings.append([len(mappings), t-rig_first/rig[0]["fps"]]
                        + [rig_idx]*len(rig) + [wrist_idx]*len(sdk)
                        + [(offset + wrist_idx / sdk[0]["fps"] - t)*1000])
    if not mappings:
        raise RuntimeError(f"{group}: no valid frame pairs")
    with (report_dir / "frame_mapping.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sync_frame", "time_seconds"] + [f"{c['name']}_frame" for c in cameras]
                        + ["realsense_minus_rig_ms"])
        writer.writerows(mappings)
    for i, camera in enumerate(cameras):
        camera["trim_start_frame"] = mappings[0][i+2]
        camera["output_frames"] = len(mappings)
        camera["rounding_error_ms"] = (
            camera["trim_start_frame"] / camera["fps"] - camera["trim_start_seconds"]
        ) * 1000
    report = {"recording": group.name, "method": "QR_only_between_two_pre_synchronized_systems",
              "offset_definition": "rig_time_seconds = realsense_time_seconds + offset_seconds; positive means RealSense starts later",
              "offset_seconds": offset, "offset_in_rig_frames": offset * rig[0]["fps"],
              "offset_in_realsense_frames": offset * sdk[0]["fps"],
              "common_start_on_rig_seconds": common_start,
              "common_duration_seconds": common_end-common_start, "mapped_frame_count": len(mappings),
              "frame_mapping_file": str((report_dir / "frame_mapping.csv").resolve()),
              "assumptions": ["Six camera files are internally frame aligned, as supplied by the user.",
                              "Head (if present) and wrists are SDK aligned on the same CFR frame grid, including held frames.",
                              "Constant inter-system offset; no clock drift correction.",
                              "Shared QR display clock; exposure and refresh limit timing accuracy."],
              "realsense_grid": sdk_grid,
              "head_alignment_method": "inherit_wrist_QR_offset_on_SDK_frame_grid" if any(v["name"] == "head" for v in sdk) else None,
              "six_camera_anchor": rig_anchor, "realsense_anchor": wrist_anchor, "cameras": cameras}
    with (report_dir / "sync_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log_progress(f"[{group.name}] RealSense starts {offset:+.6f}s relative to rig; {len(cameras)} cameras, common {common_end-common_start:.3f}s, {len(mappings)} mapped frames")
    return report


def export_synced_videos(report, output_root):
    """Export the exact contiguous frame pairs described by frame_mapping.csv."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg must be on PATH to export videos")
    cameras = report["cameras"]
    if any(abs(c["fps"] - cameras[0]["fps"]) > 1e-6 for c in cameras):
        raise RuntimeError("Video export requires equal FPS; use frame_mapping.csv for mixed FPS")
    destination = output_root / report["recording"] / "synced"
    destination.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        target = destination / f"{camera['name']}.mp4"
        start, count = camera["trim_start_frame"], report["mapped_frame_count"]
        log_progress(f"[{report['recording']}] Export {camera['name']}: frames {start}..{start+count-1}")
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", camera["path"],
            "-map", "0:v:0", "-vf", f"trim=start_frame={start}:end_frame={start+count},setpts=PTS-STARTPTS",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
            "-fps_mode", "passthrough", "-movflags", "+faststart", str(target),
        ], check=True)
        actual = probe_video(target)
        if actual["total_frames"] != count or abs(actual["fps"] - camera["fps"]) > 1e-6:
            raise RuntimeError(f"Export timing verification failed: {target}")
    sdk_camera = next(c for c in cameras if c["system"] == "realsense")
    rig_camera = next(c for c in cameras if c["system"] == "six_camera")
    export_manifest = {
        "recording": report["recording"], "frames": report["mapped_frame_count"],
        "fps": cameras[0]["fps"], "duration_seconds": report["mapped_frame_count"] / cameras[0]["fps"],
        "source_start_frames": {c["name"]: c["trim_start_frame"] for c in cameras},
        "encoding": "H.264 CRF 18; video only; source files retained",
        "pairing_error_ms": (
            report["offset_seconds"] + sdk_camera["trim_start_frame"] / sdk_camera["fps"]
            - rig_camera["trim_start_frame"] / rig_camera["fps"]
        ) * 1000,
    }
    with (destination / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(export_manifest, fh, indent=2, ensure_ascii=False)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scan-seconds", type=float, default=120.0)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--export-videos", action="store_true", help="Export all aligned MP4 files (nine when head.mp4 is present)")
    args = parser.parse_args()
    if args.scan_seconds <= 0 or args.sample_seconds <= 0:
        parser.error("scan/sample seconds must be positive")
    groups = discover_recordings(args.input_root)
    if not groups:
        parser.error(f"No group folders found in {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports, failures = [], []
    for group in groups:
        log_progress(f"[{group.name}] Starting QR synchronization")
        try:
            report = build_sync_report(group, args.output_root, args.scan_seconds, args.sample_seconds)
            reports.append(report)
            if args.export_videos:
                export_synced_videos(report, args.output_root)
        except (RuntimeError, ValueError) as exc:
            failures.append({"recording": group.name, "error": str(exc)})
            log_progress(f"[{group.name}] FAILED: {exc}")
    with (args.output_root / "sync_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"recordings": reports, "failures": failures}, fh, indent=2, ensure_ascii=False)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
