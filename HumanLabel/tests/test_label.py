import csv
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import label


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def geometry():
    points = np.random.default_rng(41).normal(0, .025, (21, 3))
    points[0], points[9] = [0, -.04, 0], [0, .025, 0]
    points[5], points[17] = [.035, .02, 0], [-.035, .02, 0]
    points[4], points[8] = [.015, .08, .015], [-.01, .09, -.008]
    return points


def options():
    return SimpleNamespace(hand_scale=1., gripper_max_m=.1, pinch_closed_m=0., pinch_open_m=.1,
                           max_reprojection_px=12., min_handedness=.8, no_preview=True,
                           intrinsics=None, swap_handedness=False, max_wrist_disagreement_m=.025)


def test_csv_contract_matches_robot_source():
    core = Path(__file__).resolve().parents[2] / "PiperX_Teleop/piperx_teleop/core.py"
    actual = runpy.run_path(str(core))["VECTOR_NAMES"]
    assert label.HEADER == ["timestamp_unix_s", *actual]


def test_pose_axes_are_right_handed_and_covariant():
    points = geometry()
    p, q = label.wrist_pose(points)
    expected = np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
    np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(), expected, atol=1e-12)
    np.testing.assert_allclose(p, points[0])
    changed_tips = points.copy()
    changed_tips[[4, 8]] += [.05, .02, .03]
    unchanged_p, unchanged_q = label.wrist_pose(changed_tips)
    np.testing.assert_allclose(unchanged_p, p)
    np.testing.assert_allclose(unchanged_q, q)
    transform = Rotation.from_euler("xyz", [.2, -.3, .5]).as_matrix()
    translation = np.array([.2, .1, .8])
    p2, q2 = label.wrist_pose(points @ transform.T + translation)
    np.testing.assert_allclose(p2, transform @ p + translation, atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(q2).as_matrix(), transform @ expected, atol=1e-12)


def test_pnp_recovers_camera_pose_and_physical_distance():
    points = geometry()
    matrix, distortion, _ = label.camera_model(1280, 800)
    rvec, tvec = np.array([.15, -.2, .05]), np.array([.05, .04, .65])
    pixels = cv2.projectPoints(points, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    value, distance, error = label.estimate(points, pixels, matrix, distortion, options())
    camera_points = points @ cv2.Rodrigues(rvec)[0].T + tvec
    p, q = label.wrist_pose(camera_points)
    np.testing.assert_allclose(value[:3], p, atol=1e-7)
    np.testing.assert_allclose(Rotation.from_quat(value[3:7]).as_matrix(), Rotation.from_quat(q).as_matrix(), atol=1e-6)
    np.testing.assert_allclose(distance, np.linalg.norm(points[4] - points[8]), atol=1e-8)
    assert error < 1e-5


def test_wrist_pinch_fusion_requires_support_and_agreement():
    head = {"pinch_valid": True, "distance": .04}
    wrist = {"pinch_valid": True, "distance": .05, "inside": np.ones(21, bool),
             "pixels": np.tile([100., 200.], (21, 1)), "image_height": 240}
    distance, source, reason = label.fuse_pinch(head, wrist, "left", options())
    assert np.isclose(distance, .045) and source == "head+left_wrist" and reason == "two_view_agreement"
    wrist["distance"] = .10
    assert label.fuse_pinch(head, wrist, "left", options()) == (.04, "head", "wrist_disagreement")
    wrist["inside"][7] = False
    assert label.fuse_pinch(head, wrist, "left", options()) == (.04, "head", "wrist_unavailable")
    wrist["inside"][7] = True
    head.update(pinch_valid=False, distance=np.nan)
    assert label.fuse_pinch(head, wrist, "left", options()) == (.1, "left_wrist", "wrist_fallback")
    wrist["pixels"][0, 1] = 20  # Opposite hand in the far/background part of the wrist image.
    assert np.isnan(label.fuse_pinch(head, wrist, "left", options())[0])


@pytest.mark.parametrize("include_wrists", [False, True])
def test_dropout_and_capture_gap_keep_mapping_and_segments(tmp_path, monkeypatch, include_wrists):
    episode = tmp_path / "source"
    episode.mkdir()
    writer = cv2.VideoWriter(str(episode / "head.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()
    if include_wrists:
        for camera in ("left_wrist", "right_wrist"):
            writer = cv2.VideoWriter(str(episode / f"{camera}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
            assert writer.isOpened()
            for _ in range(6):
                writer.write(np.zeros((240, 320, 3), np.uint8))
            writer.release()
    (episode / "manifest.json").write_text(json.dumps({"schema": "human_capture.rgb.v1", "simulated": False,
                                                       "status": "incomplete", "task": "test", "fps": 30, "frames": 6}))
    with (episode / "timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        header = ["frame_index", "grid_index", "timestamp_unix_s", "head_timestamp_unix_s"]
        if include_wrists:
            header += ["left_wrist_timestamp_unix_s", "right_wrist_timestamp_unix_s"]
        writer.writerow(header)
        for i, grid in enumerate([0, 1, 4, 5, 6, 7]):
            stamp = 1000 + grid / 30
            writer.writerow([i, grid, stamp, stamp, *([stamp + .005, stamp + .01] if include_wrists else [])])
    points = geometry()
    matrix, distortion, _ = label.camera_model(320, 240)
    pixels = cv2.projectPoints(points, np.array([.1, .1, .05]), np.array([0., 0., .65]), matrix, distortion)[0].reshape(-1, 2)
    image_lm = [SimpleNamespace(x=x / 320, y=y / 240) for x, y in pixels]
    world_lm = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]

    class Detector:
        def __init__(self, camera):
            self.index, self.camera = 0, camera

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect_for_video(self, image, ms):
            sides = ["Left"] if self.camera == 0 and self.index == 1 else ["Left", "Right"]
            landmarks = [image_lm] * len(sides)
            world_landmarks = [world_lm] * len(sides)
            if self.camera:
                shifted = [SimpleNamespace(x=p.x, y=min(.95, p.y + .35)) for p in image_lm]
                changed = world_lm.copy()
                changed[4] = SimpleNamespace(x=world_lm[4].x + .005, y=world_lm[4].y, z=world_lm[4].z)
                landmarks, world_landmarks = [shifted] * len(sides), [changed] * len(sides)
            if self.camera == 0 and self.index == 3:
                cropped = image_lm.copy()
                cropped[0] = SimpleNamespace(x=-.01, y=.5)
                landmarks = [image_lm, cropped]
            self.index += 1
            return SimpleNamespace(handedness=[[SimpleNamespace(category_name=s, score=.99)] for s in sides],
                                   hand_landmarks=landmarks, hand_world_landmarks=world_landmarks)

    created = []

    def detector_factory(opts):
        detector = Detector(len(created))
        created.append(detector)
        return detector

    fake_mp = SimpleNamespace(tasks=SimpleNamespace(BaseOptions=lambda **kw: kw, vision=SimpleNamespace(
        HandLandmarkerOptions=lambda **kw: kw, RunningMode=SimpleNamespace(VIDEO="video"),
        HandLandmarker=SimpleNamespace(create_from_options=detector_factory))),
        Image=lambda **kw: kw, ImageFormat=SimpleNamespace(SRGB="rgb"))
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setattr(label, "version", lambda package: "test")
    output = tmp_path / "result"
    label.process(episode, output, Path("unused.task"), options())
    actions, raw = read_csv(output / "action.csv"), read_csv(output / "action_raw.csv")
    mapping, quality = read_csv(output / "frame_map.csv"), read_csv(output / "quality.csv")
    assert len(actions) == 4 and len(raw) == len(quality) == 6
    assert [int(r["frame_index"]) for r in mapping] == [0, 2, 4, 5]
    assert [int(r["segment_id"]) for r in mapping] == [0, 1, 2, 2]
    assert quality[1]["right.reason"] == "not_detected"
    assert np.isnan(float(raw[1]["right.x"]))
    assert quality[3]["right.reason"] == "landmarks_outside_image"
    assert quality[3]["right.pinch_valid"] == "1"
    assert quality[3]["right.pose_valid"] == "0"
    assert quality[3]["right.wrist_in_image"] == "0"
    assert np.isnan(float(raw[3]["right.x"])) and np.isfinite(float(raw[3]["right.gripper_m"]))
    assert all(np.isfinite(float(v)) for row in actions for v in row.values())
    assert [r["timestamp_unix_s"] for r in actions] == [r["timestamp_unix_s"] for r in mapping]
    meta = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete" and meta["source_status"] == "incomplete"
    assert meta["robot_base_calibrated"] is False
    assert meta["pipeline_revision"] == label.REVISION
    assert "WRIST landmark #0" in meta["pose_convention"]
    if include_wrists:
        assert quality[0]["left.pinch_source"] == "head+left_wrist"
        assert quality[0]["right.pinch_source"] == "head+right_wrist"
        assert quality[1]["right.pinch_source"] == "right_wrist"
        expected_p, _ = label.wrist_pose(points @ cv2.Rodrigues(np.array([.1, .1, .05]))[0].T + [0., 0., .65])
        np.testing.assert_allclose([float(raw[0]["left." + k]) for k in ("x", "y", "z")], expected_p, atol=1e-7)
        assert len(created) == 3
