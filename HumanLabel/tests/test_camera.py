import csv

import numpy as np
import pytest

from humanlabel.utils.camera import camera_model, timestamps


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
