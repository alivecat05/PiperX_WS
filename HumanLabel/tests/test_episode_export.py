import json

import cv2
import numpy as np
import pytest


def test_export_preserves_capture_gaps_and_uses_each_frames_camera(tmp_path):
    import csv

    from humanlabel.export import write_csv
    from humanlabel.utils.episode import export
    from humanlabel.validation import validate

    n = 4
    rows = [
        {
            "frame_index": str(i),
            "grid_index": str(g),
            "timestamp_unix_s": str(100 + g / 30),
            "head_timestamp_unix_s": str(100 + g / 30),
        }
        for i, g in enumerate([0, 1, 5, 6])
    ]
    matrices = np.tile(
        np.array([[100.0, 0, 320], [0, 100.0, 240], [0, 0, 1.0]]), (n, 1, 1)
    )
    matrices[1, 0, 0] = 200
    joints = np.tile([0.1, 0, 1.0], (2, n, 21, 1)).astype(np.float32)
    vertices = np.tile([0.1, 0, 1.0], (2, n, 8, 1)).astype(np.float32)
    prediction = {
        "rotmat": np.broadcast_to(np.eye(3), (2, n, 25, 3, 3)).copy(),
        "trans": joints[:, :, 0].copy(),
        "identity_coeffs": np.zeros((2, n, 10)),
        "presence": np.ones((2, n)),
        "visible": np.ones((2, n)),
        "completion": np.zeros((2, n), bool),
        "camera_matrix_per_frame": matrices,
    }
    mask = np.ones((2, n), bool)
    meta = {
        "backend": "NVIDIA EgoHand",
        "representation": "SOMA",
        "status": "complete",
        "source": str(tmp_path),
        "source_frames": n,
        "camera_matrix": matrices[0].tolist(),
    }
    write_csv(
        tmp_path / "timestamps.csv", list(rows[0]), [list(r.values()) for r in rows]
    )
    video = cv2.VideoWriter(
        str(tmp_path / "head.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480)
    )
    assert video.isOpened()
    for _ in rows:
        video.write(np.zeros((480, 640, 3), np.uint8))
    video.release()
    export(
        tmp_path,
        tmp_path,
        rows,
        (640, 480),
        30.0,
        prediction,
        joints,
        vertices,
        np.array([[0, 1, 2]]),
        mask,
        mask,
        mask.astype(float),
        np.zeros((2, n), int),
        matrices[0],
        meta,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    q = list(csv.DictReader((tmp_path / "quality.csv").open()))
    assert float(q[0]["left.wrist_u_px"]) == pytest.approx(330.0)
    assert float(q[1]["left.wrist_u_px"]) == pytest.approx(340.0)
    assert q[0]["left.pose_valid"] == q[0]["left.pinch_valid"] == "1"
    assert q[0]["left.source"] == "image_temporal"
    mapping = list(csv.DictReader((tmp_path / "frame_map.csv").open()))
    assert [m["segment_id"] for m in mapping] == ["0", "0", "1", "1"]
    assert (tmp_path / "soma.npz").exists()
    report = validate(tmp_path)
    assert report["preview_decoded_frames"] == n
    assert report["epose_contract"]["camera_wrist_transform_verified"]
    assert report["motion_statistics"]["left"]["consecutive_pairs"] == 2
