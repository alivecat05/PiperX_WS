"""Pinhole intrinsics and original capture timestamp validation."""

import csv
import json
from pathlib import Path

import numpy as np


def _read_head_intrinsics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Expected a head camera intrinsics object")
    head = data.get("head", data)
    if not isinstance(head, dict):
        raise ValueError("Expected an intrinsics object under 'head'")
    return head


def _pinhole_parameters(data):
    matrix = np.asarray(data["camera_matrix"], dtype=float)
    coefficients = data.get("dist_coeffs", data.get("distortion_coeffs"))
    model = data.get("distortion_model")
    if model not in (None, "brown_conrady", "none"):
        raise ValueError(f"Unsupported head distortion model: {model}")
    if model == "brown_conrady" and coefficients is None:
        raise ValueError("Missing Brown–Conrady distortion coefficients")
    distortion = (
        np.zeros(5) if coefficients is None else np.asarray(coefficients, dtype=float).ravel()
    )
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or min(matrix[0, 0], matrix[1, 1]) <= 0
        or matrix[0, 1] != 0
        or matrix[1, 0] != 0
        or not np.array_equal(matrix[2], [0, 0, 1])
        or distortion.size not in (0, 4, 5, 8, 12, 14)
        or not np.isfinite(distortion).all()
        or (model == "none" and distortion.any())
    ):
        raise ValueError("Invalid OpenCV pinhole intrinsics/distortion")
    return matrix, distortion


def camera_model(width, height, path=None):
    if path:
        data = _read_head_intrinsics(path)
        check_intrinsics_size((width, height), (data["width"], data["height"]))
        matrix, distortion = _pinhole_parameters(data)
        return matrix, distortion, "provided_opencv_pinhole"
    focal = float(max(width, height))
    matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    return matrix, np.zeros(5), "ASSUMED_fx_fy_max_width_height_zero_distortion"


def check_intrinsics_size(video_size, calibration_size):
    if video_size != calibration_size:
        raise ValueError(
            f"Intrinsics resolution {calibration_size} must match head.mp4 {video_size}"
        )


def perspective_calibration(size, matrix, distortion=()):
    """Encode source pixels for EgoHand's shared camera JSON loader."""
    # Brown–Conrady uses the first five OpenCV rational coefficient slots.
    # The remaining rational, prism and tilt terms are zero for this model.
    coefficients = (
        np.pad(distortion, (0, 14 - len(distortion))).tolist() if np.any(distortion) else []
    )
    return {
        "model": "RATIONAL_POLYNOMIAL" if coefficients else "PINHOLE",
        "image_width": size[0],
        "image_height": size[1],
        "camera_matrix": {"matrix_3x3": matrix.tolist()},
        "distortion_coefficients": coefficients,
    }


def load_head_calibration(path):
    data = _read_head_intrinsics(path)
    size = data["width"], data["height"]
    if any(type(value) is not int or value <= 0 for value in size):
        raise ValueError("Camera width and height must be positive integers")
    calibration = perspective_calibration(size, *_pinhole_parameters(data))
    if "serial" in data:
        calibration["serial"] = str(data["serial"])
    return calibration


def captured_frames(rows):
    """Older captures without a validity column contain only captured frames."""
    flags = [str(row.get("sample_valid", True)).lower() for row in rows]
    if any(flag not in ("true", "false") for flag in flags):
        raise ValueError("Invalid sample_valid flag in source timestamps")
    return np.asarray([flag == "true" for flag in flags], dtype=bool)


def timestamps(episode, count, manifest, cameras=("head",)):
    with (episode / "timestamps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count or manifest["frames"] != count or not rows:
        raise ValueError(
            "Video/manifest/timestamps frame counts differ or episode is empty"
        )
    camera_keys = tuple(f"{camera}_timestamp_unix_s" for camera in cameras)
    previous = {}
    for i, (row, captured) in enumerate(zip(rows, captured_frames(rows))):
        # Placeholder rows still have a grid time, but may repeat camera times
        # from the held image or omit them entirely for a blank frame.
        keys = ("timestamp_unix_s", *camera_keys) if captured else ("timestamp_unix_s",)
        for key in keys:
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite source timestamp {key} at frame {i}")
            if key in previous and value <= previous[key]:
                raise ValueError(
                    f"Source timestamps must increase strictly: {key} at frame {i}"
                )
            previous[key] = value
        if int(row["frame_index"]) != i or (
            i and int(row["grid_index"]) <= int(rows[i - 1]["grid_index"])
        ):
            raise ValueError("Invalid source frame/grid order")
    return rows
