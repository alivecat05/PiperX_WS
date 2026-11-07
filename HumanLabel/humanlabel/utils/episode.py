"""Batch head-view annotation and reproducible full-length model comparison."""

import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from humanlabel.export import (
    HEADER,
    camera_wrist_rotations,
    capture_contract,
    rotation_contract,
    wrist_values,
    write_products,
)
from humanlabel.utils.camera import timestamps
from humanlabel.utils.paths import ROOT
from humanlabel.utils.tracking import contiguous_ranges


def read_episode(episode, max_frames=0):
    meta = json.loads((episode / "manifest.json").read_text(encoding="utf-8-sig"))
    cap = cv2.VideoCapture(str(episode / "head.mp4"))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {episode}/head.mp4")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(3)), int(cap.get(4)))
    cap.release()
    if count != meta["frames"] or abs(fps - meta["fps"]) > 0.01:
        raise ValueError("Source manifest/video disagreement")
    rows = timestamps(episode, count, meta, ["head"])
    n = min(count, max_frames) if max_frames else count
    return meta, rows[:n], size, fps, count


def detections(episode, rows, size, detector_path):
    """Shared detector baseline; original frame indexes are never re-timed."""
    cache = ROOT / ".cache/head" / episode.name
    cache.mkdir(parents=True, exist_ok=True)
    stat = (episode / "head.mp4").stat()
    signature = hashlib.sha256(
        (
            str(episode.resolve())
            + str((stat.st_size, stat.st_mtime_ns))
            + "yolo960_conf015_v1"
        ).encode()
    ).hexdigest()
    record = cache / "detections.json"
    if record.exists():
        d = json.loads(record.read_text())
        paths = [str(cache / f"{i:06d}.jpg") for i in range(len(rows))]
        if (
            d["signature"] == signature
            and len(d["detections"]) >= len(rows)
            and all(Path(p).exists() for p in paths)
        ):
            return [
                np.asarray(x, np.float32).reshape(-1, 6)
                for x in d["detections"][: len(rows)]
            ], paths
    import torch

    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/yolo"))
    from ultralytics import YOLO

    torch.set_num_threads(8)
    detector = YOLO(str(detector_path))
    cap = cv2.VideoCapture(str(episode / "head.mp4"))
    output, paths, images = [], [], []
    for i in range(len(rows)):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Decode failed at frame {i}")
        path = cache / f"{i:06d}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError(path)
        paths.append(str(path))
        images.append(image)
        if len(images) == 16 or i == len(rows) - 1:
            with torch.inference_mode():
                results = detector.predict(
                    images,
                    device="cuda",
                    imgsz=960,
                    conf=0.15,
                    iou=0.45,
                    verbose=False,
                    half=False,
                )
            for r in results:
                output.append(
                    np.column_stack(
                        (
                            r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy(),
                        )
                    ).astype(np.float32)
                )
            images.clear()
        if (i + 1) % 300 == 0:
            print(f"  detector {episode.name}: {i + 1}/{len(rows)}", flush=True)
    cap.release()
    record.write_text(
        json.dumps(
            {
                "signature": signature,
                "detector": str(detector_path),
                "detections": [x.tolist() for x in output],
            }
        )
    )
    del detector
    torch.cuda.empty_cache()
    return output, paths


def save_npz(path, data):
    temp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(temp, **data)
    temp.replace(path)


def infer_hawor(backend, paths, boxes, observed, supported, grid, matrix, output):
    raw_path = output / "prediction_raw.npz"
    if raw_path.exists():
        with np.load(raw_path) as d:
            raw = {k: d[k] for k in d.files}
    else:
        raw = backend.inference(paths, boxes, observed, supported, grid, matrix)
        save_npz(raw_path, raw)
    completed, fill = backend.complete(raw, observed, supported, grid)
    # Inferred crops without sufficient learned-infiller anchors remain explicit.
    return {
        **completed,
        "completion": fill,
        "visible": supported.astype(np.float32),
        "presence": supported.astype(np.float32),
    }, raw


def geometry_arrays(backend, prediction):
    import torch

    n = prediction["trans"].shape[1]
    joints = np.full((2, n, 21, 3), np.nan, np.float32)
    vertices = np.full((2, n, 778, 3), np.nan, np.float32)
    for side in range(2):
        good = np.isfinite(prediction["trans"][side]).all(-1)
        ids = np.flatnonzero(good)
        for a in range(0, len(ids), 128):
            idx = ids[a : a + 128]
            args = [
                torch.as_tensor(
                    prediction[k][side, idx], dtype=torch.float32, device="cuda"
                )
                for k in ("rotmat", "betas", "trans")
            ]
            with torch.no_grad():
                j, v = backend.geometry(*args, side)
            joints[side, idx], vertices[side, idx] = j.cpu().numpy(), v.cpu().numpy()
    return joints, vertices


def export(
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
    no_preview=False,
):
    n = len(rows)
    contract = rotation_contract(capture_contract(), meta.get("representation", "MANO"))
    meta.update(
        epose_export=contract,
        pose_convention="T_C_W: bind-wrist axes to head optical camera; origin=joint 0; joint-orient conversion in epose_export; +X right +Y down +Z forward; meters; quaternion xyzw",
        action_semantics="current-frame estimated camera wrist epose; not PiperCapture next-grid measured TCP",
    )
    quality_fields = [
        "frame_index",
        "grid_index",
        "timestamp_unix_s",
        "head_timestamp_unix_s",
        "capture_segment",
    ]
    side_fields = [
        "valid",
        "detected",
        "source",
        "needs_review",
        "detector_score",
        "model_visible",
        "model_presence",
        "missing_run_frames",
        "wrist_in_image",
        "pinch_m",
        "wrist_u_px",
        "wrist_v_px",
        "pose_valid",
        "pinch_valid",
        "pinch_source",
        "reason",
        "pinch_distance_m",
        "gripper_m",
    ]
    quality_fields += [f"{s}.{k}" for s in ("left", "right") for k in side_fields]
    valid = np.zeros((2, n), bool)
    pixels = np.full((2, n, 21, 2), np.nan, np.float32)
    matrices = prediction.get(
        "camera_matrix_per_frame", np.broadcast_to(matrix, (n, 3, 3))
    )
    for side in range(2):
        finite = np.isfinite(joints[side]).all((1, 2))
        ids = np.flatnonzero(finite)
        if len(ids):
            focal = np.stack((matrices[ids, 0, 0], matrices[ids, 1, 1]), -1)
            pixels[side, ids] = (
                joints[side, ids, :, :2] / joints[side, ids, :, 2:]
            ) * focal[:, None] + matrices[ids, None, :2, 2]
        geometry = (
            finite
            & (joints[side, :, :, 2] > 0.01).all(-1)
            & (joints[side, :, :, 2] < 3.0).all(-1)
        )
        # ACE consumes full RGB clips and predicts presence itself; detector
        # misses must not suppress its independent occlusion predictions.
        support = np.ones(n, bool) if meta["backend"] == "ace" else supported[side]
        valid[side] = geometry & support & (prediction["presence"][side] > 0.5)
    wrist_rotations = camera_wrist_rotations(
        prediction["rotmat"][:, :, 0], meta.get("representation", "MANO")
    )
    values, pinch_distances, transforms = wrist_values(
        joints, wrist_rotations, valid, contract["gripper_max_m"]
    )
    representation = meta.get("representation", "MANO")
    faces_left = faces[0] if isinstance(faces, dict) else faces[:, [0, 2, 1]]
    faces_right = faces[1] if isinstance(faces, dict) else faces
    save_npz(
        output / ("soma.npz" if representation == "SOMA" else "mano.npz"),
        {
            **prediction,
            "joints_camera_m": joints,
            "vertices_camera_m": vertices,
            "faces_left": faces_left,
            "faces_right": faces_right,
            "model_valid": valid,
            "wrist_rotation_camera": wrist_rotations,
            "detected": observed,
            "supported": supported,
            "grid_index": np.array([int(r["grid_index"]) for r in rows]),
            "camera_matrix": matrix,
            "sides": np.array(["left", "right"]),
            "axis_order": np.array("side,frame,..."),
            "local_rotation_convention": np.array(
                "native SOMA poses"
                if representation == "SOMA"
                else "absolute local rotations; mean already included"
            ),
            "left_template": np.array(
                meta.get("left_template", "mathematically reflected MANO_RIGHT")
            ),
        },
    )
    capture_segment = np.zeros(n, int)
    for k, (a, b) in enumerate(contiguous_ranges([int(r["grid_index"]) for r in rows])):
        capture_segment[a:b] = k
    preview = None
    if not no_preview:
        preview = cv2.VideoWriter(
            str(output / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not preview.isOpened():
            raise IOError("Cannot open preview encoder")
    cap = cv2.VideoCapture(str(episode / "head.mp4"))
    saved, segment, last_frame = 0, -1, -2
    try:
        with (
            (output / "action_raw.csv").open("w", newline="") as rf,
            (output / "action.csv").open("w", newline="") as af,
            (output / "frame_map.csv").open("w", newline="") as mf,
            (output / "quality.csv").open("w", newline="") as qf,
        ):
            rw, aw, mw = csv.writer(rf), csv.writer(af), csv.writer(mf)
            qw = csv.DictWriter(qf, quality_fields)
            rw.writerow(HEADER)
            aw.writerow(HEADER)
            qw.writeheader()
            mw.writerow(
                [
                    "action_row_index",
                    "frame_index",
                    "grid_index",
                    "segment_id",
                    "timestamp_unix_s",
                    "head_timestamp_unix_s",
                ]
            )
            for i, row in enumerate(rows):
                record = [row["timestamp_unix_s"], *values[i]]
                rw.writerow(record)
                qrow = {k: row[k] for k in quality_fields[:4]}
                qrow["capture_segment"] = int(capture_segment[i])
                if valid[:, i].all():
                    if (
                        i != last_frame + 1
                        or capture_segment[i] != capture_segment[last_frame]
                    ):
                        segment += 1
                    aw.writerow(record)
                    mw.writerow(
                        [
                            saved,
                            i,
                            row["grid_index"],
                            segment,
                            row["timestamp_unix_s"],
                            row["head_timestamp_unix_s"],
                        ]
                    )
                    saved, last_frame = saved + 1, i
                if preview:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot render source frame {i}")
                    cv2.rectangle(image, (0, 0), (size[0], 115), (20, 20, 20), -1)
                    cv2.putText(
                        image,
                        f"{meta['backend']} | {episode.name} | frame {i}/{n} | grid {row['grid_index']}",
                        (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                for side, name in enumerate(("left", "right")):
                    image_source = (
                        "image_single_frame"
                        if meta["backend"] == "wilor"
                        else "image_temporal"
                    )
                    source = (
                        image_source
                        if observed[side, i]
                        else "learned_completion"
                        if prediction["completion"][side, i]
                        else "tracked_crop"
                        if supported[side, i]
                        else "unsupported"
                    )
                    wrist = pixels[side, i, 0]
                    inside = bool(
                        np.isfinite(wrist).all()
                        and (wrist >= 0).all()
                        and (wrist < size).all()
                    )
                    review = (
                        not observed[side, i]
                        or not inside
                        or not valid[side, i]
                        or prediction["visible"][side, i] < 0.5
                    )
                    pinch = float(
                        np.linalg.norm(joints[side, i, 4] - joints[side, i, 8])
                    )
                    vals = [
                        int(valid[side, i]),
                        int(observed[side, i]),
                        source,
                        int(review),
                        float(scores[side, i]),
                        float(prediction["visible"][side, i]),
                        float(prediction["presence"][side, i]),
                        int(gap_length[side, i]),
                        int(inside),
                        pinch,
                        float(wrist[0]),
                        float(wrist[1]),
                        int(valid[side, i]),
                        int(valid[side, i] and np.isfinite(pinch)),
                        "head" if valid[side, i] else "none",
                        source if valid[side, i] else "unsupported_or_invalid_geometry",
                        pinch,
                        float(values[i, side * 8 + 7]),
                    ]
                    qrow.update({f"{name}.{k}": v for k, v in zip(side_fields, vals)})
                    if preview:
                        color = (80, 220, 80) if side == 0 else (0, 180, 255)
                        cv2.putText(
                            image,
                            f"{name}: {source} | {'REVIEW' if review else 'estimated'} | pinch {pinch * 1000:.0f}mm",
                            (15, 60 + 30 * side),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                        if valid[side, i]:
                            pts = pixels[side, i]
                            verts = vertices[side, i]
                            frame_matrix = matrices[i]
                            vp = (
                                verts[:, :2] / verts[:, 2:] * np.diag(frame_matrix)[:2]
                                + frame_matrix[:2, 2]
                            )
                            triangles = vp[faces_left if side == 0 else faces_right]
                            good = np.isfinite(triangles).all((1, 2)) & (
                                np.abs(triangles).max((1, 2)) < 4 * max(size)
                            )
                            overlay = image.copy()
                            cv2.polylines(
                                overlay,
                                np.rint(triangles[good]).astype(np.int32),
                                True,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.addWeighted(overlay, 0.65, image, 0.35, 0, dst=image)
                            for finger in range(5):
                                ids = [0, *range(1 + finger * 4, 5 + finger * 4)]
                                cv2.polylines(
                                    image,
                                    [np.rint(pts[ids]).astype(np.int32)],
                                    False,
                                    color,
                                    2,
                                    cv2.LINE_AA,
                                )
                            if inside:
                                cv2.drawMarker(
                                    image,
                                    tuple(np.rint(wrist).astype(int)),
                                    (255, 255, 0),
                                    cv2.MARKER_CROSS,
                                    17,
                                    2,
                                )
                qw.writerow(qrow)
                if preview:
                    preview.write(image)
        write_products(
            output,
            [r["timestamp_unix_s"] for r in rows],
            [r["grid_index"] for r in rows],
            values,
            pinch_distances,
            transforms,
            valid,
            contract,
        )
        meta.update(
            processed_frames=n,
            valid_frames_per_hand={
                s: int(valid[k].sum()) for k, s in enumerate(("left", "right"))
            },
            both_valid_frames=saved,
            contiguous_action_segments=segment + 1,
            detected_frames={
                s: int(observed[k].sum()) for k, s in enumerate(("left", "right"))
            },
            recovered_frames={
                s: int((valid[k] & ~observed[k]).sum())
                for k, s in enumerate(("left", "right"))
            },
        )
    finally:
        cap.release()
        if preview:
            preview.release()
