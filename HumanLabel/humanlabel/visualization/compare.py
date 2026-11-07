"""Validate complete annotations and encode synchronized H.264 comparisons."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def validate(folder):
    meta = json.loads((folder / "manifest.json").read_text())
    if (
        meta["status"] != "complete"
        or meta["processed_frames"] != meta["source_frames"]
    ):
        raise ValueError(f"Incomplete model output: {folder}")
    n = meta["source_frames"]
    raw = list(csv.reader((folder / "action_raw.csv").open()))
    quality = list(csv.DictReader((folder / "quality.csv").open()))
    actions = list(csv.reader((folder / "action.csv").open()))
    mapping = list(csv.DictReader((folder / "frame_map.csv").open()))
    source = list(csv.DictReader((Path(meta["source"]) / "timestamps.csv").open()))
    if len(raw) != n + 1 or len(quality) != n or len(actions) != len(mapping) + 1:
        raise ValueError(f"CSV row counts differ: {folder}")
    if len(source) != n:
        raise ValueError("Source length changed")
    for i, (q, s, r) in enumerate(zip(quality, source, raw[1:])):
        if (
            int(q["frame_index"]) != i
            or q["grid_index"] != s["grid_index"]
            or r[0] != s["timestamp_unix_s"]
        ):
            raise ValueError(f"Source frame/time mismatch at {i}")
    last = None
    for i, (m, row) in enumerate(zip(mapping, actions[1:])):
        frame = int(m["frame_index"])
        if (
            int(m["action_row_index"]) != i
            or not np.isfinite(np.asarray(row, float)).all()
        ):
            raise ValueError("Invalid dense action mapping")
        if row != raw[1 + frame]:
            raise ValueError("Action values do not match raw frame")
        if last is not None and m["segment_id"] == last["segment_id"]:
            if (
                frame != int(last["frame_index"]) + 1
                or int(m["grid_index"]) != int(last["grid_index"]) + 1
            ):
                raise ValueError("Action segment crosses a source gap")
        for offset in (4, 12):
            if (
                abs(np.linalg.norm(np.asarray(row[offset : offset + 4], float)) - 1)
                > 1e-4
            ):
                raise ValueError("Non-unit wrist quaternion")
        last = m
    cap = cv2.VideoCapture(str(folder / "preview.mp4"))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != n:
        raise ValueError(f"Preview decoded {decoded}/{n}: {folder}")
    report = {
        "status": "passed",
        "processed_frames": n,
        "preview_decoded_frames": decoded,
        "action_rows": len(mapping),
        "timestamps_preserved": True,
        "capture_gaps_preserved": True,
    }
    if meta.get("epose_export"):
        from humanlabel.export import validate_products

        report["epose_contract"] = validate_products(folder)
    archive = folder / (
        "soma.npz" if meta.get("representation") == "SOMA" else "mano.npz"
    )
    if archive.exists():
        with np.load(archive) as d:
            joints, valid, rotations = (
                d["joints_camera_m"],
                d["model_valid"],
                d["rotmat"],
            )
            matrices = (
                d["camera_matrix_per_frame"]
                if "camera_matrix_per_frame" in d
                else np.broadcast_to(np.asarray(meta["camera_matrix"]), (n, 3, 3))
            )
            for side in range(2):
                ids = np.flatnonzero(valid[side])
                if len(ids):
                    values = np.array(
                        [r[1 + 8 * side : 4 + 8 * side] for r in raw[1:]], float
                    )
                    if not np.allclose(values[ids], joints[side, ids, 0], atol=1e-6):
                        raise ValueError(
                            "CSV wrist origin differs from archived joint 0"
                        )
                    if not np.allclose(
                        np.linalg.det(rotations[side, ids]), 1.0, atol=1e-4
                    ):
                        raise ValueError("Improper MANO rotation")
            grid = d["grid_index"]
            report["motion_statistics"] = {}
            for side, name in enumerate(("left", "right")):
                good = valid[side]
                adjacent = good[1:] & good[:-1] & (np.diff(grid) == 1)
                delta = np.linalg.norm(np.diff(joints[side, :, 0], axis=0), axis=-1)[
                    adjacent
                ]
                triple = (
                    good[2:]
                    & good[1:-1]
                    & good[:-2]
                    & (np.diff(grid)[1:] == 1)
                    & (np.diff(grid)[:-1] == 1)
                )
                acc = np.linalg.norm(np.diff(joints[side], n=2, axis=0), axis=-1)[
                    triple
                ]
                focal = np.stack((matrices[:, 0, 0], matrices[:, 1, 1]), -1)
                pixels = (
                    joints[side, :, :, :2] / joints[side, :, :, 2:] * focal[:, None]
                    + matrices[:, None, :2, 2]
                )
                pixel_delta = np.linalg.norm(np.diff(pixels[:, 0], axis=0), axis=-1)[
                    adjacent
                ]
                pixel_acc = np.linalg.norm(np.diff(pixels, n=2, axis=0), axis=-1)[
                    triple
                ]
                report["motion_statistics"][name] = {
                    "valid_frames": int(good.sum()),
                    "consecutive_wrist_step_p95_mm": float(
                        np.percentile(delta * 1000, 95)
                    )
                    if delta.size
                    else None,
                    "joint_second_difference_mean_mm": float(np.mean(acc) * 1000)
                    if acc.size
                    else None,
                    "projected_wrist_step_p95_px": float(np.percentile(pixel_delta, 95))
                    if pixel_delta.size
                    else None,
                    "projected_joint_second_difference_mean_px": float(
                        np.mean(pixel_acc)
                    )
                    if pixel_acc.size
                    else None,
                    "consecutive_pairs": int(adjacent.sum()),
                    "consecutive_triples": int(triple.sum()),
                    "note": "Continuity statistics, not physical accuracy; excludes capture discontinuities.",
                }
    (folder / "validation.json").write_text(json.dumps(report, indent=2))
    return report


def render_comparison(episode, folders, names, output, ffmpeg, tile_width=960):
    import math

    metas = [json.loads((p / "manifest.json").read_text()) for p in folders]
    counts = {x["source_frames"] for x in metas}
    if len(counts) != 1 or any(x["status"] != "complete" for x in metas):
        raise ValueError("Cannot compare incomplete/mismatched model outputs")
    n, fps = counts.pop(), metas[0]["fps"]
    if any(
        abs(x["fps"] - fps) > 0.001
        or Path(x["source"]).resolve() != Path(metas[0]["source"]).resolve()
        for x in metas
    ):
        raise ValueError("Comparison sources differ")
    tile_height = round(tile_width * metas[0]["size"][1] / metas[0]["size"][0] / 2) * 2
    cols = 2 if len(folders) >= 3 else len(folders)
    rows = math.ceil(len(folders) / cols)
    caps = [cv2.VideoCapture(str(p / "preview.mp4")) for p in folders]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{cols * tile_width}x{rows * tile_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "8",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(n):
            canvas = np.zeros((rows * tile_height, cols * tile_width, 3), np.uint8)
            for k, (cap, name) in enumerate(zip(caps, names)):
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"{names[k]} preview ended at {frame}/{n}")
                image = cv2.resize(
                    image, (tile_width, tile_height), interpolation=cv2.INTER_AREA
                )
                # Model labels stay separate from the underlying validity labels.
                cv2.rectangle(image, (0, 0), (tile_width, 32), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    f"{name} | {episode} | {frame}/{n}",
                    (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y, x = (k // cols) * tile_height, (k % cols) * tile_width
                canvas[y : y + tile_height, x : x + tile_width] = image
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("H.264 comparison encoding failed")
    finally:
        for cap in caps:
            cap.release()
        if process.poll() is None:
            process.terminate()
            process.wait()
    return {
        "file": str(output),
        "frames": n,
        "fps": fps,
        "duration_s": n / fps,
        "methods": names,
    }


def main(argv=None):
    import imageio_ffmpeg

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True, type=Path)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--output", type=Path, default=Path("result/comparison"))
    p.add_argument("--tile-width", type=int, default=960)
    args = p.parse_args(argv)
    if len(args.models) != len(args.names):
        p.error("One name is required for each model")
    episodes = sorted(x.name for x in args.models[0].glob("episode_*"))
    if not episodes:
        p.error("No episodes")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    reports, files = {}, []
    for episode in episodes:
        folders = [x / episode for x in args.models]
        reports[episode] = {
            name: validate(folder) for name, folder in zip(args.names, folders)
        }
        video = args.output / f"{episode}_comparison.mp4"
        receipt = video.with_suffix(".json")
        signature = {
            "names": args.names,
            "tile_width": args.tile_width,
            "inputs": [
                {
                    "path": str((p / "preview.mp4").resolve()),
                    "size": (p / "preview.mp4").stat().st_size,
                    "mtime_ns": (p / "preview.mp4").stat().st_mtime_ns,
                }
                for p in folders
            ],
        }
        cached = (
            json.loads(receipt.read_text())
            if receipt.exists() and video.exists()
            else {}
        )
        if cached.get("signature") == signature:
            files.append(cached["video"])
        else:
            record = render_comparison(
                episode, folders, args.names, video, ffmpeg, args.tile_width
            )
            receipt.write_text(
                json.dumps({"signature": signature, "video": record}, indent=2)
            )
            files.append(record)
        print(f"Comparison saved: {episode}", flush=True)
    concat = args.output / "concat.txt"
    concat.write_text(
        "".join(
            "file '" + str(Path(x["file"]).resolve()).replace("'", "'\\''") + "'\n"
            for x in files
        )
    )
    full = args.output / "all_episodes_comparison.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(full),
        ],
        check=True,
    )
    cap = cv2.VideoCapture(str(full))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    if decoded != sum(x["frames"] for x in files):
        raise ValueError("Merged comparison frame count differs from source")
    (args.output / "comparison.json").write_text(
        json.dumps(
            {
                "episodes": files,
                "validation": reports,
                "total_frames": sum(x["frames"] for x in files),
                "full_video": str(full),
                "full_video_decoded_frames": decoded,
                "ground_truth_available": False,
                "selection_note": "Use overlay inspection with coverage and continuity; no measured 3D accuracy ranking is asserted.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
