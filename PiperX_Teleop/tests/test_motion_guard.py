"""Directional soft limits allow retreat and do not end acquisition."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piperx_teleop.motion_guard import MotionGuard
from piperx_teleop.robot import RecoverableRobotError
from test_motion_recovery import cfg, controller, neutral
from test_pipeline import step


def simple_fk(q):
    return [*(np.asarray(q[:3])*.25), *Rotation.from_euler("xyz", q[3:]).as_quat(), .04]


def guard_and_state(first_joint):
    guard = MotionGuard(simple_fk, {f"joint{i}": [-2., 2.] for i in range(1, 7)}, {})
    q = np.array([first_joint, 0., 0., 0., 0., 0.])
    return guard, {"joints": q.tolist(), "vector": simple_fk(q)}


def test_near_upper_limit_slows_outward_but_allows_retreat():
    guard, state = guard_and_state(1.96)
    q = np.asarray(state["joints"])
    outward, info = guard.apply(simple_fk(q+np.array([.02, 0, 0, 0, 0, 0])), state, .02)
    assert info["limited"] and "J1" in info["joints"]
    assert 0 < outward[0]-state["vector"][0] < .25*.02
    inward = simple_fk(q-np.array([.02, 0, 0, 0, 0, 0]))
    result, info = guard.apply(inward, state, .02)
    assert not info["limited"] and result == pytest.approx(inward)


def test_on_boundary_blocks_outward_without_blocking_inward():
    guard, state = guard_and_state(2.)
    q = np.asarray(state["joints"])
    target = simple_fk(q+np.array([.03, 0, 0, 0, 0, 0]))
    result, info = guard.apply(target, state, .02)
    assert info["limited"] and result == pytest.approx(state["vector"])
    inward = simple_fk(q-np.array([.03, 0, 0, 0, 0, 0]))
    result, info = guard.apply(inward, state, .02)
    assert not info["limited"] and result == pytest.approx(inward)


def test_far_from_limits_preserves_requested_pose_and_scale():
    guard, state = guard_and_state(0.)
    target = simple_fk([.005, -.004, .003, .01, -.007, .008])
    result, info = guard.apply(target, state, .02)
    assert not info["limited"] and result == pytest.approx(target)


def test_regular_workspace_does_not_add_an_extra_joint_speed_gate():
    guard, state = guard_and_state(0.)
    target = simple_fk([.12, 0., 0., 0., 0., 0.])
    result, info = guard.apply(target, state, .01)
    assert not info["limited"]
    assert result == pytest.approx(target)


def test_singularity_projects_unreachable_rotation_without_huge_joint_motion():
    def singular_fk(q):
        return [*(np.asarray(q[:3])*.25), *Rotation.from_rotvec([q[3], q[4], 0.]).as_quat(), .04]
    guard = MotionGuard(singular_fk, {f"joint{i}": [-2., 2.] for i in range(1, 7)}, {})
    state = {"joints": [0.]*6, "vector": singular_fk([0.]*6)}
    desired = [*state["vector"][:3], *Rotation.from_euler("z", .2).as_quat(), .04]
    result, info = guard.apply(desired, state, .02)
    assert info["limited"] and info["reason"] == "singularity"
    assert np.isfinite(result).all()
    assert max(np.abs(info["predicted_joint_delta_rad"])) <= .04
    assert Rotation.from_quat(result[3:7]).magnitude() < 1e-8


def test_official_piperx_model_reduces_outward_wrist_motion():
    from pyAgxArm.utiles.mdh_kinematics import get_mdh, fk_from_mdh
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET
    from piperx_teleop.core import pose_to_vector
    mdh = get_mdh("piper_x")
    fk = lambda q: pose_to_vector(fk_from_mdh(mdh, list(q)), .04)
    bounds = ROBOT_JOINT_LIMIT_PRESET["piper_x"]
    q = np.array([.2, .8, -.9, bounds["joint4"][1]-.06, .5, .3])
    guard = MotionGuard(fk, bounds, {})
    state = {"joints": q.tolist(), "vector": fk(q)}
    desired_q = q.copy()
    desired_q[3] += .02
    target, info = guard.apply(fk(desired_q), state, .02)
    assert info["limited"] and "J4" in info["joints"]
    assert 0 < info["predicted_joint_delta_rad"][3] < .02
    assert np.isfinite(target).all()
    assert np.linalg.norm(target[:3]-np.asarray(state["vector"][:3])) < .02


def test_leading_target_is_bounded_while_recording_continues(cfg, monkeypatch):
    control = controller(cfg)
    sent = []
    monkeypatch.setattr(control.robots["left"], "send", lambda target: sent.append(list(target)))
    data = neutral()
    data["left"]["grip"] = True
    data["buttons"]["record_start"] = True
    step(control, data)
    data["buttons"]["record_start"] = False
    try:
        for _ in range(20):
            data["left"]["position"][2] -= .01
            step(control, data)
        assert control.mode == "teleop" and control.recorder.active
        assert control.target_lead_limited["left"]
        assert sent and max(np.linalg.norm(np.asarray(t[:3])-control.robots["left"].vector[:3]) for t in sent) <= cfg["max_position_tracking_error_m"]+1e-9
        assert not any(c[0] == "stop" for r in control.robots.values() for c in r.calls)
    finally:
        control.recorder.stop()
        assert control.recorder.wait(3)


def test_controller_target_rejection_is_local_and_does_not_end_episode(cfg, monkeypatch):
    control = controller(cfg)
    data = neutral()
    data["buttons"]["record_start"] = True
    step(control, data)
    right = control.robots["right"]
    original = right.read
    def read():
        if right.code == 4:
            raise RecoverableRobotError("target rejected", kind="target_limit")
        return original()
    monkeypatch.setattr(right, "read", read)
    right.code = 4
    try:
        step(control, neutral())
        assert control.local_limit_latched["right"] and control.recorder.active
        assert control.mode == "idle"
        holds = len([c for c in right.calls if c[0] == "hold"])
        for _ in range(3):
            data = neutral()
            data["left"]["grip"] = True
            step(control, data)
        assert len([c for c in right.calls if c[0] == "hold"]) == holds
        assert control.recorder.active and control.mode == "teleop"
        right.code = 0
        step(control, neutral())
        assert not control.local_limit_latched["right"]
        data = neutral()
        data["right"]["grip"] = True
        step(control, data)
        assert control.status()["calibration"]["right"]["calibrated"]
    finally:
        control.recorder.stop()
        assert control.recorder.wait(3)
