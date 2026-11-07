from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanlabel.backends.hawor import HaWoRBackend
from humanlabel.utils.hand_model import prepare_mano
from humanlabel.utils.tracking import (
    contiguous_ranges,
    interpolate_parameters,
    project_rotations,
    track_boxes,
)


def boxes(left=True, right=True):
    return np.array(
        ([20, 100, 100, 200, 0.95, 0],) * int(left)
        + ([400, 100, 480, 200, 0.95, 1],) * int(right),
        dtype=np.float32,
    ).reshape(-1, 6)


def test_bounded_missing_hand_is_completed_only_between_real_anchors():
    ds = [boxes(), boxes(False), boxes(False), boxes(), boxes(False)]
    b, score, observed, supported, length = track_boxes(ds, np.arange(5), max_gap=2)
    assert supported[0].tolist() == [True, True, True, True, False]
    assert observed[0].tolist() == [True, False, False, True, False]
    np.testing.assert_allclose(b[0, 1:3], np.tile(b[0, 0], (2, 1)))
    assert length[0, 1] == 2
    assert np.isnan(b[0, -1]).all()


def test_no_completion_crosses_a_capture_discontinuity():
    ds = [boxes(), boxes(False), boxes(False), boxes()]
    _, _, observed, supported, _ = track_boxes(ds, [0, 1, 8, 9], max_gap=90)
    np.testing.assert_array_equal(supported, observed)
    assert contiguous_ranges([0, 1, 8, 9]) == [(0, 2), (2, 4)]


def test_tracker_uses_future_and_past_to_correct_one_frame_class_flip():
    ds = [boxes() for _ in range(7)]
    ds[3][:, 5] = 1 - ds[3][:, 5]
    b, _, _, _, _ = track_boxes(ds, np.arange(7))
    assert b[0, 3, 0] < 200
    assert b[1, 3, 0] > 300


def test_rotation_averages_are_proper_and_slerp_avoids_pi_wrap():
    aa = np.array([[0, 0, np.deg2rad(179)], [0, 0, np.deg2rad(-179)]])
    r = Rotation.from_rotvec(aa).as_matrix()
    m = project_rotations(r.mean(0))
    np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-12)
    assert np.linalg.det(m) > 0.9999
    data = {
        "trans": np.zeros((2, 3, 3)),
        "betas": np.zeros((2, 3, 10)),
        "rot": np.zeros((2, 3, 3)),
        "hand_pose": np.zeros((2, 3, 45)),
    }
    data["rot"][:, [0, 2]] = aa
    result = interpolate_parameters(data, np.array([[True, False, True]] * 2))
    assert abs(np.linalg.norm(result["rot"][0, 1]) - np.pi) < 1e-6


def test_infiller_is_noop_when_every_supported_hand_is_observed():
    backend = object.__new__(HaWoRBackend)
    data = {
        "trans": np.zeros((2, 3, 3)),
        "betas": np.zeros((2, 3, 10)),
        "rotmat": np.broadcast_to(np.eye(3), (2, 3, 16, 3, 3)).copy(),
    }
    observed = np.array([[True, True, False], [True, True, True]])
    result, completed = backend.complete(data, observed, observed, [0, 1, 2])
    assert not completed.any()
    for key in data:
        np.testing.assert_array_equal(result[key], data[key])


@pytest.mark.parametrize("representation", ["MANO", "SOMA"])
@pytest.mark.parametrize("backend", ["wilor", "ace"])
def test_export_preserves_capture_gaps_and_uses_each_frames_camera(
    tmp_path, representation, backend
):
    import csv

    from humanlabel.utils.episode import export

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
        "rotmat": np.broadcast_to(
            np.eye(3), (2, n, 25 if representation == "SOMA" else 16, 3, 3)
        ).copy(),
        "trans": joints[:, :, 0].copy(),
        "betas": np.zeros((2, n, 10)),
        "presence": np.ones((2, n)),
        "visible": np.ones((2, n)),
        "completion": np.zeros((2, n), bool),
        "camera_matrix_per_frame": matrices,
    }
    mask = np.ones((2, n), bool)
    if backend == "ace":
        mask[0, 0] = False
        prediction["completion"][0, 0] = True
    meta = {"backend": backend, "representation": representation}
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
        no_preview=True,
    )
    q = list(csv.DictReader((tmp_path / "quality.csv").open()))
    assert float(q[0]["left.wrist_u_px"]) == pytest.approx(330.0)
    assert float(q[1]["left.wrist_u_px"]) == pytest.approx(340.0)
    assert q[0]["left.pose_valid"] == q[0]["left.pinch_valid"] == "1"
    assert q[0]["left.source"] == (
        "image_single_frame" if backend == "wilor" else "learned_completion"
    )
    if backend == "ace":
        assert q[0]["left.detected"] == "0"
        assert q[0]["left.needs_review"] == "1"
    mapping = list(csv.DictReader((tmp_path / "frame_map.csv").open()))
    assert [m["segment_id"] for m in mapping] == ["0", "0", "1", "1"]
    assert (
        tmp_path / ("soma.npz" if representation == "SOMA" else "mano.npz")
    ).exists()


def test_context_crop_recovery_respects_elapsed_capture_time():
    from humanlabel.workflows.recover import crop_candidates

    boxes = np.zeros((2, 5, 4), np.float32)
    boxes[:, 1] = [10, 20, 30, 40]
    boxes[:, 3] = [30, 40, 50, 60]
    observed = np.zeros((2, 5), bool)
    observed[:, [1, 3]] = True
    grid = np.array([0, 1, 11, 21, 22])
    recovered, support, added = crop_candidates(
        boxes, observed, observed, grid, 30, edge_seconds=0.0
    )
    np.testing.assert_allclose(recovered[:, 2], [[20, 30, 40, 50]] * 2)
    assert added[:, 2].all() and not added[:, [0, 4]].any()
    assert not crop_candidates(
        boxes, observed, observed, grid * 10, 30, edge_seconds=0.0
    )[2].any()
    assert crop_candidates(boxes, observed, observed, grid, 30)[2][:, [0, 4]].all()


def test_temporal_objective_never_spans_missing_frames_or_capture_gaps():
    from humanlabel.utils.refinement import temporal_triples

    grid = np.array([0, 1, 2, 8, 9, 10, 11, 12])
    ids = np.array([0, 1, 2, 3, 4, 5, 7])
    np.testing.assert_array_equal(
        temporal_triples(ids, grid), [True, False, False, True, False]
    )


def test_reflected_mano_asset_preserves_articulated_geometry(tmp_path):
    torch = pytest.importorskip("torch")
    smplx = pytest.importorskip("smplx")
    source = Path(__file__).resolve().parents[1] / ".cache/mano/MANO_RIGHT.pkl"
    if not source.exists():
        pytest.skip("Optional licensed model asset not installed")
    prepare_mano(source, tmp_path)
    right = smplx.MANOLayer(str(tmp_path), is_rhand=True)
    left = smplx.MANOLayer(str(tmp_path), is_rhand=False)
    rng = np.random.default_rng(83)
    r = torch.from_numpy(
        Rotation.from_rotvec(rng.normal(0, 0.4, (32, 3))).as_matrix().astype(np.float32)
    ).reshape(2, 16, 3, 3)
    beta = torch.from_numpy(rng.normal(0, 0.8, (2, 10)).astype(np.float32))
    f = torch.tensor([-1.0, 1.0, 1.0])
    lm = r * f[:, None] * f[None, :]
    with torch.no_grad():
        rr = right(
            global_orient=r[:, :1], hand_pose=r[:, 1:], betas=beta, pose2rot=False
        )
        ll = left(
            global_orient=lm[:, :1], hand_pose=lm[:, 1:], betas=beta, pose2rot=False
        )
    np.testing.assert_allclose(
        ll.vertices.numpy(), (rr.vertices * f).numpy(), atol=1e-6
    )
    np.testing.assert_allclose(ll.joints.numpy(), (rr.joints * f).numpy(), atol=1e-6)
