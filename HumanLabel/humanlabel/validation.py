"""Validate EgoHand annotations, timestamps, capture gaps and previews."""

import csv
import json
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
    archive = folder / "soma.npz"
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
                        raise ValueError("Improper SOMA rotation")
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
