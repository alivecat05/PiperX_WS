"""Run public NVIDIA EgoHand/SOMA on every source head-view frame."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from humanlabel.utils.camera import camera_model
from humanlabel.utils.episode import detections, export, read_episode, save_npz
from humanlabel.utils.paths import MODEL, ROOT
from humanlabel.utils.tracking import contiguous_ranges, track_boxes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path, default=ROOT / "human_demonstration"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episodes", nargs="+", help="Process only these episode names"
    )
    args = parser.parse_args(argv)
    import torch

    repo = MODEL / "IsaacCapture/src/python/isaaccapture_egopost"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(MODEL / "dinov3"))
    torch.set_num_threads(8)
    from inference.camera.camera_models import load_camera_calibration
    from inference.pipeline.predictor import EgoHandPredictor

    predictor = EgoHandPredictor.from_checkpoint(
        MODEL / "egohand/egohand_s060000.ckpt",
        MODEL / "egohand/soma",
        batch_size=4,
        device="cuda",
        dtype=torch.bfloat16,
        smooth=True,
        use_perspective_cropping=True,
    )
    predictor.model.eval().requires_grad_(False)
    assets = np.load(MODEL / "egohand/soma/SOMAHand.npz", allow_pickle=True)
    faces = {
        0: assets["left_triangles"].astype(np.int32),
        1: assets["right_triangles"].astype(np.int32),
    }
    identity_dim = int(predictor.model.soma_hand.num_shape_components)
    scale_dim = int(predictor.model.soma_hand.num_scale_params)
    # Native SOMA-25 has one extra metacarpal per non-thumb finger.
    # Preserve native geometry in the archive as well as this OpenPose-like view.
    order = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    episodes = (
        [args.source]
        if (args.source / "head.mp4").exists()
        else sorted(args.source.glob("episode_*/"))
    )
    if args.episodes:
        episodes = [episode for episode in episodes if episode.name in args.episodes]
    summaries = []
    for episode in episodes:
        output = args.output / episode.name
        mp = output / "manifest.json"
        if mp.exists() and json.loads(mp.read_text()).get("status") == "complete":
            summaries.append(json.loads(mp.read_text()))
            continue
        started = time.monotonic()
        source, rows, size, fps, count = read_episode(episode)
        output.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": "human_label.camera_pseudo_action.v2",
            "backend": "NVIDIA EgoHand",
            "representation": "SOMA",
            "status": "processing",
            "source": str(episode.resolve()),
            "source_frames": count,
            "source_status": source["status"],
            "source_dropped_grid_ticks": source.get("dropped_grid_ticks", 0),
            "fps": fps,
            "size": size,
            "robot_base_calibrated": False,
            "camera_fixed": True,
            "left_template": "native SOMAHand left/right rig",
            "dtype": "bfloat16 model autocast; fp32 rig geometry",
            "pose_convention": "SOMA wrist root in head optical camera frame; meters, quaternion xyzw; native root axes converted by SOMA FK for epose",
            "code_revision": "d10c8818c0d556bc79d4598b343e95bf9e3ed178",
            "soma25_to_preview21": order,
            "identity_model_type": predictor.model.soma_hand.identity_model_type,
            "identity_dim": identity_dim,
            "scale_dim": scale_dim,
        }
        meta.update(
            device="cuda",
            gpu_name=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,
            ground_truth_available=False,
        )
        meta.update(
            temporal_smoothing="official motion-adaptive SOMA smoothing, split at capture gaps",
            score_semantics="visible/presence indicate bounded tracker support, not learned hand-presence probabilities",
        )
        try:
            ds, paths = detections(
                episode,
                rows,
                size,
                MODEL / "weights/external/detector.pt",
                args.output.parent / "frames" / episode.name,
            )
            grid = np.array([int(r["grid_index"]) for r in rows])
            boxes, scores, observed, supported, gap_length = track_boxes(ds, grid, 90)
            save_npz(
                output / "tracking.npz",
                {
                    "boxes": boxes,
                    "scores": scores,
                    "observed": observed,
                    "supported": supported,
                    "gap_length": gap_length,
                    "grid": grid,
                },
            )
            # Bounded tracked crops allow reconstruction of interior detector misses.
            matrix, _, basis = camera_model(*size)
            camera = load_camera_calibration(
                {
                    "model": "PINHOLE",
                    "image_width": size[0],
                    "image_height": size[1],
                    "camera_matrix": {"matrix_3x3": matrix.tolist()},
                    "distortion_coefficients": [],
                }
            )
            meta.update(camera_matrix=matrix.tolist(), intrinsics_basis=basis)
            cache = output / "prediction_raw.npz"
            if cache.exists():
                with np.load(cache) as f:
                    prediction = {k: f[k] for k in f.files}
                joints, vertices = (
                    prediction.pop("joints_camera_m"),
                    prediction.pop("vertices_camera_m"),
                )
            else:
                n = len(paths)
                vertices_count = len(assets["left_vert_ids"])
                prediction = {
                    "trans": np.full((2, n, 3), np.nan, np.float32),
                    "rotmat": np.full((2, n, 25, 3, 3), np.nan, np.float32),
                    "identity_coeffs": np.full(
                        (2, n, identity_dim), np.nan, np.float32
                    ),
                    "scale_params": np.full((2, n, scale_dim), np.nan, np.float32),
                    "global_scale": np.full((2, n), np.nan, np.float32),
                    "native_joints25_camera_m": np.full(
                        (2, n, 25, 3), np.nan, np.float32
                    ),
                    "completion": np.zeros((2, n), bool),
                    "visible": supported.astype(np.float32),
                    "presence": supported.astype(np.float32),
                }
                joints = np.full((2, n, 21, 3), np.nan, np.float32)
                vertices = np.full((2, n, vertices_count, 3), np.nan, np.float32)
                all_results = {"left": {}, "right": {}}
                all_boxes = {}
                for a in range(0, n, 16):
                    ids = [str(i) for i in range(a, min(a + 16, n))]
                    images = [cv2.imread(paths[int(i)]) for i in ids]
                    bboxes = {}
                    for key in ids:
                        i = int(key)
                        bboxes[key] = {}
                        for side, name in enumerate(("left", "right")):
                            if supported[side, i]:
                                b = boxes[side, i]
                                bboxes[key][name] = [
                                    *((b[:2] + b[2:]) / 2).tolist(),
                                    float(np.max(b[2:] - b[:2]) * 1.4),
                                ]
                    result = predictor.run(
                        images,
                        ids,
                        bboxes,
                        ["left", "right"],
                        intrinsics_per_image={k: camera for k in ids},
                        frame_times={
                            k: float(rows[int(k)]["head_timestamp_unix_s"]) for k in ids
                        },
                        defer_smoothing=True,
                    )
                    all_boxes.update(bboxes)
                    for name in all_results:
                        all_results[name].update(result.get(name, {}))
                    if a % 160 == 0:
                        print(f"  EgoHand {episode.name}: {a}/{n}", flush=True)
                for first, last in contiguous_ranges(grid):
                    ids = [str(i) for i in range(first, last)]
                    result = {
                        s: {k: all_results[s][k] for k in ids if k in all_results[s]}
                        for s in all_results
                    }
                    predictor.postprocess_results(
                        result,
                        ids,
                        {k: all_boxes[k] for k in ids},
                        intrinsics_per_image={k: camera for k in ids},
                        frame_times={
                            k: float(rows[int(k)]["head_timestamp_unix_s"]) for k in ids
                        },
                    )
                    for side, name in enumerate(("left", "right")):
                        for key, r in result.get(name, {}).items():
                            i = int(key)
                            j = np.asarray(r["pred_keypoints_3d"]).reshape(
                                25, 3
                            ) + np.asarray(r["transl"]).reshape(1, 3)
                            v = np.asarray(r["pred_vertices"]).reshape(
                                -1, 3
                            ) + np.asarray(r["transl"]).reshape(1, 3)
                            joints[side, i] = j[order]
                            vertices[side, i] = v
                            prediction["rotmat"][side, i] = r["poses"]
                            prediction["trans"][side, i] = r["transl"]
                            identity = np.asarray(r["identity_coeffs"]).reshape(-1)
                            prediction["identity_coeffs"][side, i, : len(identity)] = (
                                identity
                            )
                            if r.get("scale_params") is not None:
                                prediction["scale_params"][side, i] = np.asarray(
                                    r["scale_params"]
                                ).reshape(-1)
                            prediction["global_scale"][side, i] = r["global_scale"]
                            prediction["native_joints25_camera_m"][side, i] = j
                save_npz(
                    cache,
                    {
                        **prediction,
                        "joints_camera_m": joints,
                        "vertices_camera_m": vertices,
                    },
                )
            export(
                episode,
                output,
                rows,
                size,
                fps,
                prediction,
                joints,
                vertices,
                faces,
                observed,
                supported,
                scores,
                gap_length,
                matrix,
                meta,
            )
            meta.update(status="complete", wall_s=time.monotonic() - started)
            summaries.append(meta)
            print(f"Saved EgoHand: {episode.name}", flush=True)
        except BaseException as exc:
            meta.update(status="failed", error=str(exc))
            raise
        finally:
            mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (args.output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
