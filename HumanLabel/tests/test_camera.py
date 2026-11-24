import csv

import numpy as np
import pytest

from humanlabel.utils.camera import (
    camera_model,
    captured_frames,
    load_head_calibration,
    timestamps,
)


def test_intrinsics_require_matching_resolution(tmp_path):
    import json

    p = tmp_path / "camera.json"
    p.write_text(
        json.dumps(
            {
                "width": 640,
                "height": 480,
                "camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            }
        )
    )
    k, _, basis = camera_model(640, 480, p)
    assert basis == "provided_opencv_pinhole" and k[0, 0] == 500
    with pytest.raises(ValueError, match="resolution"):
        camera_model(1280, 800, p)


def test_missing_calibration_is_explicitly_assumed():
    k, distortion, basis = camera_model(1280, 800)
    np.testing.assert_array_equal(k, [[1280, 0, 640], [0, 1280, 400], [0, 0, 1]])
    assert "ASSUMED" in basis and not distortion.any()


def test_realsense_head_intrinsics_use_egohand_rational_schema(tmp_path):
    import json

    matrix = [[645, 0, 635], [0, 644, 405], [0, 0, 1]]
    brown_coeffs = [-0.054, 0.061, -0.0004, 0.0006, -0.019]
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps({
        "head": {
            "serial": "419222301962",
            "width": 1280,
            "height": 800,
            "camera_matrix": matrix,
            "distortion_model": "brown_conrady",
            "dist_coeffs": brown_coeffs,
        },
        "left_wrist": {"distortion_model": "inverse_brown_conrady"},
    }))
    calibration = load_head_calibration(path)
    assert calibration["model"] == "RATIONAL_POLYNOMIAL"
    assert calibration["serial"] == "419222301962"
    assert (calibration["image_width"], calibration["image_height"]) == (1280, 800)
    assert calibration["camera_matrix"]["matrix_3x3"] == matrix
    assert calibration["distortion_coefficients"] == brown_coeffs + [0] * 9


def test_original_timestamp_gaps_are_retained_and_duplicates_rejected(tmp_path):
    path = tmp_path / "timestamps.csv"
    header = ["frame_index", "grid_index", "timestamp_unix_s", "head_timestamp_unix_s"]
    rows = [[0, 0, 100, 100], [1, 1, 100.033, 100.033], [2, 5, 100.166, 100.166]]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    actual = timestamps(tmp_path, 3, {"frames": 3})
    assert [r["grid_index"] for r in actual] == ["0", "1", "5"]
    rows[-1][2] = 100.033
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    with pytest.raises(ValueError, match="increase strictly"):
        timestamps(tmp_path, 3, {"frames": 3})


def test_placeholders_preserve_grid_time_without_new_camera_observations(tmp_path):
    path = tmp_path / "timestamps.csv"
    path.write_text(
        "frame_index,grid_index,timestamp_unix_s,head_timestamp_unix_s,sample_valid,frame_status\n"
        "0,0,100.000,,False,blank\n"
        "1,1,100.033,100.030,True,captured\n"
        "2,2,100.067,100.030,False,held\n"
        "3,3,100.100,100.098,True,captured\n"
    )
    rows = timestamps(tmp_path, 4, {"frames": 4})
    assert captured_frames(rows).tolist() == [False, True, False, True]
    assert [row["head_timestamp_unix_s"] for row in rows] == [
        "", "100.030", "100.030", "100.098"
    ]
    path.write_text(path.read_text().replace("100.098", "100.030"))
    with pytest.raises(ValueError, match="increase strictly: head_timestamp_unix_s at frame 3"):
        timestamps(tmp_path, 4, {"frames": 4})
