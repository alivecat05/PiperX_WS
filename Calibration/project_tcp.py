"""Overlay measured left/right TCP positions on the recorded head-camera video."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from calibrate import project


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Recorded episode directory")
    parser.add_argument("--intrinsics", type=Path, default=Path("Calibration/intrinsics.json"))
    parser.add_argument("--extrinsics", type=Path, default=Path("Calibration/extrinsics.yaml"))
    parser.add_argument("--output", type=Path, default=Path("Calibration/tcp_projection.mp4"))
    args = parser.parse_args(argv)
    if args.output.resolve() == (args.data/"head.mp4").resolve():
        raise ValueError("Output must differ from the source head.mp4")
    camera = json.loads(args.intrinsics.read_text(encoding="utf-8-sig"))["head"]
    rows = [json.loads(line) for line in (args.data/"samples.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    # Read the two plain numeric lists written by calibrate.py; no YAML dependency.
    matrices = {}
    for line in args.extrinsics.read_text(encoding="utf-8-sig").splitlines():
        if line and not line.startswith(" "):
            key = line.rstrip(":")
            matrices[key] = []
        elif line.strip():
            matrices[key].append(json.loads(line.strip().removeprefix("- ")))
    transforms = [np.asarray(matrices[key], float) for key in ("T_Head_LeftBase", "T_Head_RightBase")]
    if any(t.shape != (4, 4) or not np.isfinite(t).all() for t in transforms):
        raise ValueError("Expected two finite 4x4 transforms")
    video = cv2.VideoCapture(str(args.data/"head.mp4"))
    writer = None
    try:
        if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
            raise ValueError("head.mp4 / samples.jsonl frame count mismatch")
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = video.get(cv2.CAP_PROP_FPS)
        if size != (camera["width"], camera["height"]) or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS or RGB intrinsics resolution mismatch")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "bgr24", "-s:v", f"{size[0]}x{size[1]}", "-framerate", str(fps),
                   "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
        writer = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
        for index, row in enumerate(rows):
            ok, image = video.read()
            if not ok or row["frame_index"] != index:
                raise ValueError(f"Frame alignment error at {index}")
            for arm, (side, offset, color) in enumerate((("left", 0, (0, 255, 0)), ("right", 8, (255, 0, 255)))):
                status = "held / missing frame"
                if row.get("sample_valid", True):
                    stamps = row["source_timestamps_unix_s"]
                    times = [t for k, t in stamps["states"][side].items() if k == "tcp" or k.startswith("end_pose_")]
                    status = "no synchronized pose"
                    if times and all(np.isfinite(t) and abs(t-stamps["cameras"]["head"]) <= .02 for t in times):
                        tcp = np.asarray(row["observation.state"][offset:offset+3], float)
                        point = (transforms[arm] @ np.r_[tcp, 1])[:3]
                        status = "behind camera / invalid pose"
                        if np.isfinite(point).all() and point[2] > 0:
                            pixel = project((point[:2]/point[2])[None], camera)[0]
                            status = "outside image"
                            if np.isfinite(pixel).all() and 0 <= pixel[0] < size[0] and 0 <= pixel[1] < size[1]:
                                center = tuple(np.rint(pixel).astype(int))
                                cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 18, 2)
                                cv2.circle(image, center, 8, color, 2)
                                status = "TCP"
                cv2.putText(image, f"{side}: {status}", (12, 24+arm*28), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            writer.stdin.write(image.tobytes())
    finally:
        video.release()
        if writer is not None:
            try:
                writer.stdin.close()
            finally:
                returncode = writer.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
