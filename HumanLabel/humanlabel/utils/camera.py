"""Pinhole intrinsics and original capture timestamp validation."""

import csv
import json
from pathlib import Path

import numpy as np


def camera_model(width, height, path=None):
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if (data["width"], data["height"]) != (width, height):
            raise ValueError(
                "Intrinsics resolution must match head.mp4; do not rescale implicitly"
            )
        matrix = np.asarray(data["camera_matrix"], dtype=float)
        distortion = np.asarray(data.get("distortion_coeffs", [0.0] * 5), dtype=float)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()
        ):
            raise ValueError("Invalid OpenCV pinhole intrinsics/distortion")
        basis = "provided_opencv_pinhole"
    else:
        focal = float(max(width, height))
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        distortion, basis = (
            np.zeros(5),
            "ASSUMED_fx_fy_max_width_height_zero_distortion",
        )
    return matrix, distortion, basis


def timestamps(episode, count, manifest, cameras=("head",)):
    with (episode / "timestamps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count or manifest["frames"] != count or not rows:
        raise ValueError(
            "Video/manifest/timestamps frame counts differ or episode is empty"
        )
    for i, row in enumerate(rows):
        for key in ("timestamp_unix_s", *[f"{c}_timestamp_unix_s" for c in cameras]):
            if not np.isfinite(float(row[key])):
                raise ValueError("Non-finite source timestamp")
            if i and float(row[key]) <= float(rows[i - 1][key]):
                raise ValueError("Source timestamps must increase strictly")
        if int(row["frame_index"]) != i or (
            i and int(row["grid_index"]) <= int(rows[i - 1]["grid_index"])
        ):
            raise ValueError("Invalid source frame/grid order")
    return rows
