"""Every clutch defines a fresh controller/TCP alignment without moving the TCP."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piperx_teleop.core import PoseMapper
from piperx_teleop.app import load_config
from test_pipeline import cfg, packet, make_controller, step


def mapper_and_state(cfg):
    arm = dict(cfg["arms"]["left"], pose_smoothing_s=0., tremor_smoothing_s=0.,
               rotation_deadband_rad=0., position_deadband_m=0., max_rotation_rad_s=100.)
    mapper = PoseMapper(arm)
    tcp_r = Rotation.from_euler("xyz", [25, -30, 70], degrees=True)
    state = [.3, 0., .3, *tcp_r.as_quat(), .04]
    hand = packet()["left"]
    hand.update(grip=True, quaternion=Rotation.from_euler("xyz", [-40, 15, 120], degrees=True).as_quat())
    return mapper, hand, state


def test_first_press_aligns_arbitrary_hand_and_tcp_in_place(cfg):
    mapper, hand, state = mapper_and_state(cfg)
    target = mapper.update(hand, state, .02)
    assert target[:7] == pytest.approx(state[:7])
    calibration = mapper.calibration_status()
    assert calibration["calibrated"] and calibration["id"] == 1
    assert calibration["zone_center_xr"] == hand["position"]
    basis = Rotation.from_matrix(mapper.basis)
    corrected = (basis*Rotation.from_quat(hand["quaternion"])*basis.inv()
                 * Rotation.from_quat(calibration["orientation_offset_quaternion"]))
    assert (corrected*Rotation.from_quat(state[3:7]).inv()).magnitude() < 1e-12


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_controller_rotation_uses_the_calibrated_reference(cfg, axis):
    mapper, hand, state = mapper_and_state(cfg)
    mapper.update(hand, state, .02)
    delta = Rotation.from_euler(axis, 18, degrees=True)
    hand["quaternion"] = (delta*Rotation.from_quat(hand["quaternion"])).as_quat()
    target = mapper.update(hand, state, .02)
    basis = Rotation.from_matrix(mapper.basis)
    expected = basis*delta*basis.inv()*Rotation.from_quat(state[3:7])
    assert (Rotation.from_quat(target[3:7])*expected.inv()).magnitude() < 1e-12


def test_reclutch_replaces_calibration_at_new_hand_and_robot_pose(cfg):
    mapper, hand, state = mapper_and_state(cfg)
    mapper.update(hand, state, .02)
    previous = mapper.calibration_status()
    hand["grip"] = False
    mapper.update(hand, state, .02)
    assert not mapper.calibration_status()["calibrated"]
    hand.update(grip=True, position=[.7, 1.3, -.4], quaternion=[0., 0., 0., 1.])
    state[3:7] = Rotation.from_euler("xyz", [-10, 20, -35], degrees=True).as_quat()
    assert mapper.update(hand, state, .02)[:7] == pytest.approx(state[:7])
    current = mapper.calibration_status()
    assert current["id"] == previous["id"]+1
    assert current["zone_center_xr"] == hand["position"]
    assert current["orientation_offset_quaternion"] != previous["orientation_offset_quaternion"]


def test_control_press_calibrates_without_sending_cartesian_motion(cfg, monkeypatch):
    control = make_controller(cfg)
    sent = []
    monkeypatch.setattr(control.robots["right"], "send", lambda target: sent.append(target))
    data = packet()
    data["right"]["grip"] = True
    step(control, data)
    assert sent == []
    assert control.status()["calibration"]["right"]["calibrated"]
    assert not control.status()["calibration"]["left"]["calibrated"]
    data["right"]["grip"] = False
    step(control, data)
    assert not control.status()["calibration"]["right"]["calibrated"]


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("xr_axis,base_axis", [([0., 0., -1.], [1., 0., 0.]),
                                                ([-1., 0., 0.], [0., 1., 0.]),
                                                ([0., 1., 0.], [0., 0., 1.])])
def test_x_aligned_start_preserves_directions_and_one_to_one_scale(cfg, side, xr_axis, base_axis):
    arm = dict(cfg["arms"][side], position_scale=1., pose_smoothing_s=0., tremor_smoothing_s=0.,
               position_deadband_m=0., rotation_deadband_rad=0., max_speed_m_s=10., max_rotation_rad_s=100.)
    mapper = PoseMapper(arm)
    tcp_r = Rotation.from_euler("xyz", [25, -30, 70], degrees=True)
    hand_r = Rotation.from_euler("xyz", [-40, 15, 120], degrees=True)
    state = [.3, 0., .3, *tcp_r.as_quat(), .04]
    hand = packet()[side]
    hand.update(grip=True, quaternion=hand_r.as_quat())
    assert mapper.update(hand, state, .02)[:7] == pytest.approx(state[:7])
    hand["position"] = np.array(hand["position"])+.02*np.array(xr_axis)
    hand["quaternion"] = (Rotation.from_rotvec(.1*np.array(xr_axis))*hand_r).as_quat()
    target = mapper.update(hand, state, .02)
    np.testing.assert_allclose(np.subtract(target[:3], state[:3]), .02*np.array(base_axis), atol=1e-12)
    delta = Rotation.from_quat(target[3:7])*tcp_r.inv()
    np.testing.assert_allclose(delta.as_rotvec(), .1*np.array(base_axis), atol=1e-12)


def test_default_config_enables_tremor_filter_without_fixed_smoothing(cfg, tmp_path):
    for arm in cfg["arms"].values():
        for key in ("pose_smoothing_s", "tremor_smoothing_s", "joint_guard_enabled"):
            arm.pop(key, None)
    path = tmp_path/"defaults.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    for arm in load_config(path)["arms"].values():
        assert arm["pose_smoothing_s"] == 0.
        assert arm["tremor_smoothing_s"] == .04
        assert not arm["joint_guard_enabled"]


def test_episode_snapshots_configured_axes_and_current_clutch_reference(cfg):
    control = make_controller(cfg)
    try:
        data = packet()
        data["right"]["grip"] = True
        step(control, data)
        data["buttons"]["record_start"] = True
        step(control, data)
        assert control.recorder.active
        reference = control.recorder.config_snapshot["teleop_coordinate_calibration"]["right"]
        assert reference["calibrated"]
        np.testing.assert_array_equal(reference["xr_to_base"], cfg["arms"]["right"]["xr_to_base"])
        assert reference["zone_center_xr"] == data["right"]["position"]
    finally:
        control.close()
