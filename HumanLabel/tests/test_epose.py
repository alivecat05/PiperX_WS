import csv
import importlib.util
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from humanlabel.export import (
    HEADER,
    camera_wrist_rotations,
    capture_contract,
    update_episode,
    validate_products,
    wrist_values,
    write_csv,
)
from humanlabel.utils.paths import ROOT


def geometry():
    joints = np.zeros((2, 2, 21, 3))
    joints[:] = [0.15, -0.2, 1.2]
    joints[0, :, 4, 0] += 0.12
    joints[1, :, 4, 0] += 0.03
    r = Rotation.from_euler("xyz", [0.3, -0.4, 0.2]).as_matrix()
    roots = np.broadcast_to(r, (2, 2, 3, 3)).copy()
    valid = np.ones((2, 2), bool)
    return joints, roots, valid


def test_actual_pipercapture_columns_and_xyzw_convention():
    spec = importlib.util.spec_from_file_location(
        "piper_contract_test_core", ROOT.parent / "PiperCapture/piper_capture/core.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert HEADER == ["timestamp_unix_s", *module.VECTOR_NAMES]
        joints, roots, valid = geometry()
        values, _, _ = wrist_values(joints, roots, valid, {"left": 0.07, "right": 0.07})
        official = module.pose_to_vector([0.15, -0.2, 1.2, 0.3, -0.4, 0.2], 0.07)
        np.testing.assert_allclose(values[0, :8], official, atol=1e-12)
    finally:
        sys.modules.pop(spec.name, None)


def test_width_clamps_without_rescaling_and_retains_raw_distance(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "arms": {
                    "left": {"gripper_max_m": 0.04},
                    "right": {"gripper_max_m": 0.07},
                }
            }
        )
    )
    contract = capture_contract(path)
    joints, roots, valid = geometry()
    values, distance, transforms = wrist_values(
        joints, roots, valid, contract["gripper_max_m"]
    )
    np.testing.assert_allclose(distance[:, 0], [0.12, 0.03])
    np.testing.assert_allclose(values[0, [7, 15]], [0.04, 0.03])
    np.testing.assert_allclose(transforms[:, :, :3, 3], joints[:, :, 0])
    np.testing.assert_allclose(transforms[:, :, :3, :3], roots)


def test_soma_camera_rotation_matches_official_fk_and_camera_covariance():
    reference = ROOT / "tests/fixtures/soma_wrist_fk.json"
    if not reference.exists():
        pytest.skip("Official SOMA FK verification artifact not installed")
    evidence = json.loads(reference.read_text())
    native = np.asarray([evidence[s]["native_input"] for s in ("left", "right")])[
        :, None
    ]
    expected = np.asarray([evidence[s]["fk_root"] for s in ("left", "right")])[:, None]
    np.testing.assert_allclose(
        camera_wrist_rotations(native, "SOMA"), expected, atol=2e-7
    )
    axes = json.loads((ROOT / "configs/soma_wrist_axes.json").read_text())["A"]
    a = np.asarray([axes[s] for s in ("left", "right")])[:, None]
    camera_rotation = Rotation.from_euler("xyz", [-0.6, 0.2, 0.8]).as_matrix()
    restored = a @ camera_rotation @ a.swapaxes(-1, -2) @ native
    np.testing.assert_allclose(
        camera_wrist_rotations(restored, "SOMA"), camera_rotation @ expected, atol=3e-7
    )
    np.testing.assert_array_equal(camera_wrist_rotations(native, "MANO"), native)


def test_saved_epose_preserves_frame_alignment_and_uses_fk_not_raw_soma_rotation(
    tmp_path,
):
    source, output, backup = [tmp_path / s for s in ("source", "output", "backup")]
    source.mkdir()
    output.mkdir()
    stamps = ["100.000000001", "100.100000001"]
    write_csv(
        source / "timestamps.csv",
        ["timestamp_unix_s", "grid_index"],
        zip(stamps, [0, 3]),
    )
    joints, roots, valid = geometry()
    rotations = np.broadcast_to(np.eye(3), (2, 2, 25, 3, 3)).copy()
    rotations[:, :, 0] = roots
    np.savez(
        output / "soma.npz", joints_camera_m=joints, rotmat=rotations, model_valid=valid
    )
    old, _, _ = wrist_values(joints, roots, valid, {"left": 0.1, "right": 0.1})
    records = [[t, *v] for t, v in zip(stamps, old)]
    for name in ("action.csv", "action_raw.csv"):
        write_csv(output / name, HEADER, records)
    write_csv(output / "frame_map.csv", ["frame_index", "segment_id"], [[0, 0], [1, 1]])
    write_csv(
        output / "quality.csv", ["left.gripper_m", "right.gripper_m"], old[:, [7, 15]]
    )
    meta = {
        "status": "complete",
        "source": str(source),
        "processed_frames": 2,
        "source_frames": 2,
        "representation": "SOMA",
    }
    (output / "manifest.json").write_text(json.dumps(meta))
    update_episode(output, capture_contract(), backup)
    assert validate_products(output)["status"] == "passed"
    assert list(csv.reader((output / "epose.csv").open())) == list(
        csv.reader((output / "action_raw.csv").open())
    )
    prior = np.asarray(list(csv.reader((backup / "action.csv").open()))[1:])
    assert float(prior[0, 8]) == pytest.approx(0.1)
    current = list(csv.reader((output / "epose.csv").open()))
    assert current[1][0] == stamps[0] and float(current[1][8]) == pytest.approx(0.07)
    # Rerunning must accept already-converted camera wrist quaternions.
    update_episode(output, capture_contract(), backup)
    assert validate_products(output)["status"] == "passed"
