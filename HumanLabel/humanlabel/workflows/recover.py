"""Recover bounded EgoHand crop gaps using real capture times and current RGB.

Only crop coordinates are interpolated across acquisition gaps. Poses are
inferred from existing RGB frames; native smoothing never crosses those gaps.
Native predictions and extracted frames come from the run's temporary directory.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from humanlabel.utils.camera import captured_frames, perspective_calibration
from humanlabel.utils.episode import export, read_episode, save_npz
from humanlabel.utils.tracking import contiguous_ranges, track_boxes
from humanlabel.workflows.egohand import configure_upstream, load_predictor


def crop_candidates(
    boxes, observed, supported, grid, fps, max_seconds=3.0, edge_seconds=0.25
):
    boxes, supported = boxes.copy(), supported.copy()
    added = np.zeros_like(supported)
    for side in range(2):
        known = np.flatnonzero(observed[side])
        for i in np.flatnonzero(~supported[side]):
            pos = np.searchsorted(known, i)
            if pos == 0 or pos == len(known):
                if len(known):
                    anchor = known[0] if pos == 0 else known[-1]
                    if abs(grid[i] - grid[anchor]) / fps <= edge_seconds:
                        boxes[side, i] = boxes[side, anchor]
                        supported[side, i] = added[side, i] = True
                continue
            a, b = known[pos - 1], known[pos]
            if (grid[b] - grid[a]) / fps > max_seconds:
                continue
            t = (grid[i] - grid[a]) / (grid[b] - grid[a])
            boxes[side, i] = (1 - t) * boxes[side, a] + t * boxes[side, b]
            supported[side, i] = added[side, i] = True
    return boxes, supported, added


def run(source_dir, output_dir, episode_names=None, *, predictor=None):
    import torch

    configure_upstream()
    if predictor is None:
        torch.set_num_threads(8)
    from inference.camera.camera_models import load_camera_calibration

    summaries = []
    for folder in sorted(source_dir.glob("episode_*")):
        if episode_names and folder.name not in episode_names:
            continue
        output = output_dir / folder.name
        started = time.monotonic()
        meta = json.loads((folder / "manifest.json").read_text())
        assert meta["status"] == "complete" and meta["representation"] == "SOMA"
        intrinsics = meta.get("camera_calibration")
        manifest_path = output / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if (
            previous.get("status") == "complete"
            and previous.get("camera_calibration") == intrinsics
        ):
            summaries.append(previous)
            continue
        meta.update(
            status="processing",
            processing_chain=["EgoHand", "native smoothing", "context crops", "epose"],
            backend="NVIDIA EgoHand + context crops",
            device="cuda",
            gpu_name=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,
            ground_truth_available=False,
            crop_recovery="bounded 3 s interior crop interpolation and 0.25 s edge crop transfer in true capture grid time; current RGB inference, never pose interpolation",
            temporal_smoothing="official adaptive smoothing confined to each capture segment; inherited source poses preserved",
        )
        episode = Path(meta["source"])
        _, rows, size, fps, n = read_episode(episode)
        captured = captured_frames(rows)
        if not (folder / "tracking.npz").exists():
            cached = json.loads(
                (
                    source_dir.parent / "frames" / folder.name / "detections.json"
                ).read_text()
            )["detections"]
            grid = np.array([int(r["grid_index"]) for r in rows])
            values = track_boxes(
                [np.asarray(d, np.float32).reshape(-1, 6) for d in cached],
                grid,
                90,
                captured=captured,
            )
            save_npz(
                folder / "tracking.npz",
                dict(
                    zip(
                        ("boxes", "scores", "observed", "supported", "gap_length"),
                        values,
                    )
                )
                | {"grid": grid},
            )
        with np.load(folder / "tracking.npz") as d:
            tracking = {k: d[k] for k in d.files}
        boxes, supported, added = crop_candidates(
            tracking["boxes"],
            tracking["observed"],
            tracking["supported"],
            tracking["grid"],
            fps,
        )
        supported &= captured
        added &= captured
        with np.load(folder / "soma.npz") as d:
            if not np.array_equal(d["detected"], tracking["observed"]):
                raise ValueError("Source detector association changed")
            keys = (
                "trans",
                "rotmat",
                "identity_coeffs",
                "scale_params",
                "global_scale",
                "native_joints25_camera_m",
                "completion",
                "visible",
                "presence",
                "camera_matrix_per_frame",
            )
            pred = {k: d[k].copy() for k in keys if k in d}
            joints, vertices = (
                d["joints_camera_m"].copy(),
                d["vertices_camera_m"].copy(),
            )
            faces = {0: d["faces_left"], 1: d["faces_right"]}
        matrix = np.asarray(meta["camera_matrix"])
        source_camera = load_camera_calibration(intrinsics) if intrinsics is not None else None
        output.mkdir(parents=True, exist_ok=True)
        try:
            if added.any():
                if predictor is None:
                    predictor = load_predictor()
                camera = source_camera or load_camera_calibration(
                    perspective_calibration(size, matrix)
                )
                indices = np.flatnonzero(added.any(0))
                ids = [str(i) for i in indices]
                bboxes = {
                    str(i): {
                        name: [
                            *((boxes[s, i, :2] + boxes[s, i, 2:]) / 2).tolist(),
                            float(np.max(boxes[s, i, 2:] - boxes[s, i, :2]) * 1.4),
                        ]
                        for s, name in enumerate(("left", "right"))
                        if added[s, i]
                    }
                    for i in indices
                }
                images = [
                    cv2.imread(
                        str(
                            source_dir.parent / "frames" / folder.name / f"{i:06d}.jpg"
                        )
                    )
                    for i in indices
                ]
                times = {
                    str(i): float(rows[i]["head_timestamp_unix_s"]) for i in indices
                }
                result = predictor.run(
                    images,
                    ids,
                    bboxes,
                    ["left", "right"],
                    intrinsics_per_image={k: camera for k in ids},
                    frame_times=times,
                    defer_smoothing=True,
                )
                for a, b in contiguous_ranges(tracking["grid"], captured):
                    keys = [str(i) for i in indices if a <= i < b]
                    if not keys:
                        continue
                    part = {
                        s: {k: v for k, v in result.get(s, {}).items() if k in keys}
                        for s in ("left", "right")
                    }
                    predictor.postprocess_results(
                        part,
                        keys,
                        {k: bboxes[k] for k in keys},
                        intrinsics_per_image={k: camera for k in keys},
                        frame_times={k: times[k] for k in keys},
                    )
                    for side, name in enumerate(("left", "right")):
                        for key, r in part.get(name, {}).items():
                            i = int(key)
                            if not added[side, i]:
                                continue
                            t = np.asarray(r["transl"]).reshape(3)
                            native = (
                                np.asarray(r["pred_keypoints_3d"]).reshape(25, 3) + t
                            )
                            joints[side, i] = native[meta["soma25_to_preview21"]]
                            vertices[side, i] = (
                                np.asarray(r["pred_vertices"]).reshape(-1, 3) + t
                            )
                            for k, rk in [
                                ("rotmat", "poses"),
                                ("identity_coeffs", "identity_coeffs"),
                                ("scale_params", "scale_params"),
                                ("global_scale", "global_scale"),
                            ]:
                                pred[k][side, i] = np.asarray(r[rk]).reshape(
                                    pred[k][side, i].shape
                                )
                            pred["trans"][side, i] = t
                            pred["native_joints25_camera_m"][side, i] = native
                            pred["visible"][side, i] = pred["presence"][side, i] = 1.0
            tracking.update(
                boxes=boxes, supported=supported, crop_context_recovered=added
            )
            export(
                episode,
                output,
                rows,
                size,
                fps,
                pred,
                joints,
                vertices,
                faces,
                tracking["observed"],
                supported,
                tracking["scores"],
                tracking["gap_length"],
                matrix,
                meta,
                source_camera=source_camera,
            )
            meta.update(
                status="complete",
                added_crop_predictions=int(added.sum()),
                completion_wall_s=time.monotonic() - started,
            )
            summaries.append(meta)
            print(
                f"Context crops: {folder.name}, {int(added.sum())} predictions, {meta['both_valid_frames']}/{n} both valid",
                flush=True,
            )
        except BaseException as exc:
            meta.update(status="failed", error=str(exc))
            raise
        finally:
            (output / "manifest.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", help="Process only these episode names")
    args = parser.parse_args(argv)
    run(args.source, args.output, args.episodes)


if __name__ == "__main__":
    main()
