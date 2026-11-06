import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import label
from mano_backend import WiLoRBackend, camera_translation, fit_perspective_camera, mano_geometry, THUMB_TIP_VERTEX, INDEX_TIP_VERTEX


def model_geometry():
    rng = np.random.default_rng(92)
    joints = rng.normal(0, .025, (21, 3))
    joints[0] = [.01, -.02, .005]  # The wrist root is deliberately not model translation.
    vertices = rng.normal(0, .03, (778, 3))
    vertices[THUMB_TIP_VERTEX] = joints[4]
    vertices[INDEX_TIP_VERTEX] = joints[8]
    return joints, vertices


@pytest.mark.parametrize("is_right", [True, False])
def test_mano_root_and_tips_use_one_mesh_and_proper_rotation(is_right):
    joints, vertices = model_geometry()
    root = np.array([.3, -.4, .2])
    translation = np.array([.1, .05, .6])
    value, distance, camera_joints, camera_vertices, _, _ = mano_geometry(
        joints, vertices, root, np.zeros((15, 3)), translation, is_right, 1.2)
    reflection = np.diag([1, 1, 1] if is_right else [-1, 1, 1])
    expected_root = (reflection @ joints[0] + translation) * 1.2
    np.testing.assert_allclose(value[:3], expected_root)
    assert not np.allclose(value[:3], translation * 1.2)
    rotation = Rotation.from_quat(value[3:7]).as_matrix()
    np.testing.assert_allclose(rotation, reflection @ Rotation.from_rotvec(root).as_matrix() @ reflection, atol=1e-12)
    assert np.linalg.det(rotation) > .999999
    np.testing.assert_allclose(distance, np.linalg.norm(vertices[THUMB_TIP_VERTEX] - vertices[INDEX_TIP_VERTEX]) * 1.2)
    np.testing.assert_allclose(camera_joints[[4, 8]], camera_vertices[[THUMB_TIP_VERTEX, INDEX_TIP_VERTEX]])


def test_mano_rejects_wrong_fingertip_layout():
    joints, vertices = model_geometry()
    vertices[THUMB_TIP_VERTEX] += .02
    with pytest.raises(ValueError, match="fingertip indexing"):
        mano_geometry(joints, vertices, [0, 0, 0], np.zeros((15, 3)), [0, 0, .5], True)


def test_camera_translation_uses_real_principal_point_and_focal():
    matrix = np.array([[800., 0, 610], [0, 1000., 390], [0, 0, 1.]])
    z = 2 * np.sqrt(800 * 1000) / (400 * 2)
    expected = np.array([(650 - 610) * z / 800 + .02, (420 - 390) * z / 1000 - .01, z])
    np.testing.assert_allclose(camera_translation([2, .02, -.01], [650, 420], 400, matrix, True), expected)
    expected[0] -= .04
    np.testing.assert_allclose(camera_translation([2, .02, -.01], [650, 420], 400, matrix, False), expected)


@pytest.mark.parametrize("is_right", [True, False])
def test_nominal_to_perspective_conversion_preserves_projection_and_wrist_pivot(is_right):
    joints, vertices = model_geometry()
    matrix = np.array([[1280., 0, 640], [0, 1280., 400], [0, 0, 1.]])
    nominal = matrix.copy()
    nominal[0, 0] = nominal[1, 1] = 25000.
    camera, center, size = [4., .01, -.02], [450, 650], 400
    translation, delta, residual = fit_perspective_camera(camera, center, size, matrix, (1280, 800), joints, is_right)
    value, distance, corrected, _, root, _ = mano_geometry(
        joints, vertices, [.2, -.1, .3], np.zeros((15, 3)), translation, is_right, camera_rotation=delta)
    reflected = joints * ([1, 1, 1] if is_right else [-1, 1, 1])
    target = cv2.projectPoints(reflected, np.zeros(3),
        camera_translation(camera, center, size, nominal, is_right), nominal, None)[0].reshape(-1, 2)
    old = cv2.projectPoints(reflected, np.zeros(3),
        camera_translation(camera, center, size, matrix, is_right), matrix, None)[0].reshape(-1, 2)
    fitted = cv2.projectPoints(corrected, np.zeros(3), np.zeros(3), matrix, None)[0].reshape(-1, 2)
    assert residual < np.sqrt(np.mean(np.sum((old - target)**2, axis=1))) / 3
    np.testing.assert_allclose(value[:3], corrected[0])
    np.testing.assert_allclose(corrected[0], reflected[0] + translation)
    np.testing.assert_allclose(distance, np.linalg.norm(joints[4] - joints[8]))
    np.testing.assert_allclose(np.sqrt(np.mean(np.sum((fitted - target)**2, axis=1))), residual)
    original_root = np.array([.2, -.1, .3]) * ([1, 1, 1] if is_right else [1, -1, -1])
    np.testing.assert_allclose(Rotation.from_rotvec(root).as_matrix(), delta @ Rotation.from_rotvec(original_root).as_matrix())


def test_mano_uses_distal_fingers_with_head_agreement_but_requires_more_for_fallback():
    head = {"pinch_valid": True, "distance": .02}
    inside = np.ones(21, bool)
    inside[[1, 2, 5, 6]] = False
    wrist = {"mano": {}, "pinch_valid": True, "inside": inside, "distance": .024,
             "pixels": np.tile([100, 230], (21, 1)), "image_height": 240}
    options = SimpleNamespace(max_wrist_disagreement_m=.025)
    distance, source, _ = label.fuse_pinch(head, wrist, "left", options)
    assert np.isclose(distance, .022) and source == "head+left_wrist"
    head.update(pinch_valid=False, distance=np.nan)
    assert np.isnan(label.fuse_pinch(head, wrist, "left", options)[0])


def test_gpu_batches_do_not_mix_camera_outputs():
    backend = WiLoRBackend.__new__(WiLoRBackend)
    backend.gpu_model = SimpleNamespace(infer=lambda patches: {"value": patches[:, 0, 0, 0]})
    backend.decode_image = lambda context, predictions, args: predictions["value"].tolist()
    prepared = {"head": {"patches": [np.full((2, 2, 3), 11)]},
                "left_wrist": {"patches": [np.full((2, 2, 3), 20), np.full((2, 2, 3), 21)]},
                "right_wrist": {"patches": []}}
    assert backend.infer_many(prepared, None) == {"head": [11.], "left_wrist": [20., 21.], "right_wrist": []}


def test_gpu_rejects_degenerate_rotation_outputs():
    from openvino_mano import decode_outputs
    outputs = [np.tile(np.eye(3), (1, 1, 1, 1)), np.tile(np.eye(3), (1, 15, 1, 1)),
               np.zeros((1, 10)), np.ones((1, 3)), np.zeros((1, 21, 3)), np.zeros((1, 778, 3))]
    assert decode_outputs(outputs)["global_orient"].shape == (1, 1, 3)
    outputs[0][..., :, 2] = 0
    with pytest.raises(ValueError, match="invalid MANO rotation"):
        decode_outputs(outputs)


def test_gpu_prefetch_preserves_every_video_frame():
    class Video:
        def __init__(self, base):
            self.base, self.index = base, 0

        def read(self):
            image = np.full((2, 2, 3), self.base + self.index)
            self.index += 1
            return True, image

    caps = {"head": Video(0), "left_wrist": Video(10), "right_wrist": Video(20)}
    backend = WiLoRBackend.__new__(WiLoRBackend)
    backend.prepare_many = lambda images, camera_models: {k: int(v[0, 0, 0]) for k, v in images.items()}
    backend.infer_many = lambda prepared, args: prepared
    actual = list(backend.iter_frames(caps, {}, None, 5))
    for i, (images, output) in enumerate(actual):
        assert output == {"head": i, "left_wrist": i + 10, "right_wrist": i + 20}
        assert int(images["head"][0, 0, 0]) == i
    assert all(video.index == 5 for video in caps.values())


def test_mano_exports_native_root_and_parameter_archive_without_mediapipe(tmp_path):
    episode = tmp_path / "source"
    episode.mkdir()
    for camera in ("head", "left_wrist", "right_wrist"):
        video = cv2.VideoWriter(str(episode / f"{camera}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        assert video.isOpened()
        for _ in range(2):
            video.write(np.zeros((240, 320, 3), np.uint8))
        video.release()
    (episode / "manifest.json").write_text(json.dumps({"schema": "human_capture.rgb.v1", "simulated": False,
        "status": "complete", "task": "test", "fps": 30, "frames": 2}), encoding="utf-8")
    with (episode / "timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "grid_index", "timestamp_unix_s", "head_timestamp_unix_s",
                         "left_wrist_timestamp_unix_s", "right_wrist_timestamp_unix_s"])
        for i in range(2):
            writer.writerow([i, i, *([1000 + i / 30] * 4)])
    args = SimpleNamespace(backend="mano", hand_scale=1., gripper_max_m=.1, pinch_closed_m=0., pinch_open_m=.1,
        max_reprojection_px=12., min_handedness=.8, no_preview=True, intrinsics=None, swap_handedness=False,
        max_wrist_disagreement_m=.025, max_frames=1)
    root_vec = np.array([.2, -.1, .4])
    root_pos = np.array([.05, .1, .6])
    vector = np.r_[root_pos, Rotation.from_rotvec(root_vec).as_quat()]

    class Predictor:
        def provenance(self):
            return {"backend": "test"}

        def detect(self, image, matrix, distortion, options):
            result = {}
            for side in ("left", "right"):
                joints = np.tile(root_pos, (21, 1))
                joints[4] += [.01, .03, 0]
                joints[8] += [.04, .03, 0]
                pixels = np.tile([150., 180.], (21, 1))
                result[side] = [{"pixels": pixels, "inside": np.ones(21, bool), "image_height": 240,
                    "world": joints, "wrist_in_image": True, "pinch_valid": True, "reason": "ok",
                    "score": .9, "handedness_score": np.nan, "pose": vector, "distance": .03, "distance_px": 15.,
                    "mano": {"global_orient": root_vec, "hand_pose": np.zeros((15, 3)), "betas": np.zeros(10),
                             "camera_translation_m": root_pos - [.01, 0, 0], "joints_camera_m": joints,
                             "is_right": side == "right"}}]
            return result

    output = tmp_path / "result"
    label.process(episode, output, Path("unused.task"), args, Predictor())
    with (output / "action.csv").open() as stream:
        actions = list(csv.DictReader(stream))
    assert len(actions) == 1
    for side in ("left", "right"):
        np.testing.assert_allclose([float(actions[0][side + "." + k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")], vector)
        assert np.isclose(float(actions[0][side + ".gripper_m"]), .03)
    archive = np.load(output / "mano.npz", allow_pickle=False)
    assert archive["joints_camera_m"].shape == (1, 3, 2, 21, 3)
    assert archive["mano_is_right"][0, 0].tolist() == [0, 1]
    np.testing.assert_allclose(archive["joints_camera_m"][0, 0, 0, 0], root_pos, atol=1e-7)
    np.testing.assert_allclose(archive["global_orient"][0, 0, 0], root_vec, atol=1e-7)
    meta = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert meta["status"] == "partial" and meta["processed_frames"] == 1 and meta["source_frames"] == 2
    assert meta["backend"] == "mano" and meta["pipeline_revision"] != label.REVISION
