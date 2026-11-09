"""Recoverable motion errors keep torque and permit deliberate recovery after teaching."""
from types import SimpleNamespace as NS
import json
import time

import numpy as np
import pytest

from piperx_teleop.robot import MockPiperX, PiperX, RecoverableRobotError
from test_pipeline import cfg, make_controller, packet, step


class RecoveryRobot(MockPiperX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.code, self.errors, self.teaching = 0, 0, False
        self.power, self.mode = True, 1
        self.calls = []
        self.issue = ""

    def read(self):
        if self.errors or self.code not in (0, 4, 11):
            raise RuntimeError("hard robot error")
        if self.code == 4 or self.teaching:
            raise RecoverableRobotError("target limit or teaching")
        return self.read_recovery()

    def read_recovery(self):
        if self.errors or self.code not in (0, 4, 11):
            raise RuntimeError("hard robot error")
        stamp, state = MockPiperX.read(self)
        state.update(arm_status=self.code, err_code=self.errors, teaching=self.teaching,
                     ctrl_mode=2 if self.teaching else 1, arm_enabled=self.power, mode_feedback=self.mode)
        return stamp, state

    def hold(self, joints):
        self.calls.append(("hold", list(joints)))
        self.mode = 1

    def send(self, target):
        self.calls.append(("send", list(target)))
        self.vector = list(target)
        self.mode = 0

    def send_gripper(self, width):
        self.calls.append(("gripper", width))
        self.vector[7] = width

    def enable(self):
        self.calls.append(("enable",))
        self.power = True
        return True

    def stop(self):
        self.calls.append(("stop",))

    def close(self):
        self.calls.append(("close",))

    def cartesian_start_issue(self, joints):
        return self.issue


def controller(cfg):
    control = make_controller(cfg)
    control.robots = {s: RecoveryRobot(arm) for s, arm in cfg["arms"].items()}
    return control


def neutral():
    data = packet()
    for side in ("left", "right"):
        data[side]["trigger"] = 0
    return data


def enter_limit_recovery(control):
    step(control, neutral())
    for robot in control.robots.values():
        robot.calls.clear()
    control.robots["right"].code = 4
    step(control, neutral())
    assert control.mode == "recovery"


def arm_recovery(control):
    step(control, neutral())
    control.recovery_neutral_since = time.perf_counter()-.31
    step(control, neutral())
    assert control.recovery_y_armed


def test_target_limit_keeps_both_arms_energized_and_continues_feedback(cfg):
    control = controller(cfg)
    control.robots["right"].joints = [.1]*6
    enter_limit_recovery(control)
    for side, robot in control.robots.items():
        assert [c[0] for c in robot.calls] == ["hold"]
        assert robot.power and not control.ready
    assert control.robots["right"].calls[0][1] == [.1]*6
    old = control.states["right"].latest().t
    step(control, neutral())
    assert control.states["right"].latest().t > old
    assert "NOT confirmed" in control.status()["recovery_hold"]["right"]


def test_tracking_error_limits_target_without_global_recovery(cfg):
    control = controller(cfg)
    data = neutral()
    data["left"]["grip"] = True
    step(control, data)
    control.robots["left"].vector[0] += .2
    step(control, data)
    assert control.mode == "teleop" and control.target_lead_limited["left"]
    assert all(not any(c[0] == "stop" for c in robot.calls) for robot in control.robots.values())


def test_joint_proximity_does_not_interrupt_motion_after_valid_clutch(cfg):
    control = controller(cfg)
    data = neutral()
    data["left"]["grip"] = True
    step(control, data)
    control.robots["left"].issue = "J3 near upper limit"
    data["left"]["position"][2] -= .03
    step(control, data)
    assert control.mode == "teleop"
    assert any(c[0] == "send" for c in control.robots["left"].calls)
    assert not any(c[0] == "stop" for c in control.robots["left"].calls)


def test_teaching_suspends_all_commands_and_held_buttons_do_not_resume(cfg):
    control = controller(cfg)
    enter_limit_recovery(control)
    right = control.robots["right"]
    right.teaching, right.code = True, 11
    calls = {s: list(r.calls) for s, r in control.robots.items()}
    data = packet(record_start=True, reset=True)
    data["left"]["grip"] = data["right"]["grip"] = True
    for _ in range(5):
        control.request_recovery()
        step(control, data)
    assert control.mode == "recovery" and control.recorder.path is None
    assert "示教" in control.message and "松开" in control.message
    assert {s: r.calls for s, r in control.robots.items()} == calls


def test_operator_teaching_in_normal_mode_is_not_overridden_with_hold(cfg):
    control = controller(cfg)
    step(control, neutral())
    left = control.robots["left"]
    left.calls.clear()
    left.code, left.teaching = 11, True
    step(control, neutral())
    assert control.mode == "recovery" and left.calls == []


def test_exit_teaching_requires_explicit_request_and_fresh_enable_before_reclutch(cfg):
    control = controller(cfg)
    enter_limit_recovery(control)
    right = control.robots["right"]
    right.teaching, right.code = True, 11
    step(control, neutral())
    right.teaching, right.code, right.power = False, 0, False
    right.joints = [.15]*6
    arm_recovery(control)
    assert control.mode == "recovery"
    assert not any(c[0] == "enable" for r in control.robots.values() for c in r.calls)
    control.request_recovery()
    step(control, neutral())
    assert control.mode == "recovery"  # Wait for feedback newer than the explicit enable request.
    for robot in control.robots.values():
        assert [c[0] for c in robot.calls][-2:] == ["hold", "enable"]
    step(control, neutral())
    assert control.mode == "paused" and not control.ready
    step(control, neutral())
    control.resume_since = time.perf_counter()-.31
    step(control, neutral())
    assert control.mode == "idle"
    step(control, neutral())
    data = neutral()
    data["right"]["position"] = [.5, .5, -.5]
    data["right"]["grip"] = True
    step(control, data)
    assert not any(c[0] == "send" for r in control.robots.values() for c in r.calls)
    np.testing.assert_allclose(control.mappers["right"].anchor[2][:7], right.vector[:7])


def test_y_recovery_requires_a_new_long_press_after_neutral_release(cfg):
    control = controller(cfg)
    enter_limit_recovery(control)
    control.robots["right"].code = 0
    data = neutral()
    data["buttons"]["reset"] = True
    control.hold_since["reset"] = time.perf_counter()-2
    step(control, data)
    assert not control.recovery_enabling
    arm_recovery(control)
    step(control, data)
    control.hold_since["reset"] = time.perf_counter()-2
    step(control, data)
    assert control.recovery_enabling
    step(control, data)
    assert control.mode == "paused"  # Releasing Y is still required before ordinary control.


@pytest.mark.parametrize("cause", ["limit", "pose", "buttons", "input"])
def test_resume_is_rejected_when_not_ready(cfg, cause):
    control = controller(cfg)
    enter_limit_recovery(control)
    control.robots["right"].code = 0
    arm_recovery(control)
    data = neutral()
    if cause == "limit": control.robots["right"].code = 4
    if cause == "pose": control.robots["right"].issue = "J3 too close to limit"
    if cause == "buttons": data["left"]["grip"] = True
    if cause == "input": data["active"] = False
    control.request_recovery()
    step(control, data)
    assert control.mode == "recovery" and not control.recovery_enabling
    assert not any(c[0] == "enable" for r in control.robots.values() for c in r.calls)


def test_close_during_recovery_does_not_turn_hold_into_emergency_stop(cfg):
    control = controller(cfg)
    enter_limit_recovery(control)
    control.close()
    for robot in control.robots.values():
        assert [c[0] for c in robot.calls] == ["hold", "close"]


def test_motion_rejection_ends_recording_as_incomplete(cfg):
    control = controller(cfg)
    data = neutral()
    data["buttons"]["record_start"] = True
    step(control, data)
    control.robots["right"].code = 4
    step(control, neutral())
    assert control.mode == "recovery" and control.recorder.wait(3)
    meta = json.loads((control.recorder.path/"manifest.json").read_text())
    assert meta["status"] == "incomplete"


def sdk_adapter(code, error=0, teaching=False):
    robot = PiperX.__new__(PiperX)
    robot.clock_offset = time.perf_counter()-time.time()
    stamp = time.time()
    msg = lambda value: NS(timestamp=stamp, msg=value)
    status = NS(arm_status=NS(value=code), err_code=error, ctrl_mode=2 if teaching else 1,
                teach_status=1 if teaching else 0, mode_feedback=1)
    parser = NS(**{key: msg(None) for key in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz",
                                           "joint_12", "joint_34", "joint_56")})
    robot.robot = NS(get_tcp_pose=lambda: msg([.3, 0, .3, 0, 0, 0]),
                    get_joint_angles=lambda: msg([0.]*6), get_arm_status=lambda: msg(status),
                    get_joints_enable_status_list=lambda: [True]*6, _parser=parser)
    # Bind the installed SDK implementation, preserving its required joint_index argument.
    from pyAgxArm.protocols.can_protocol.drivers.piper.default.driver import Driver
    robot.robot.get_joint_enable_status = Driver.get_joint_enable_status.__get__(robot.robot)
    robot.gripper = NS(get_gripper_status=lambda: msg(NS(mode="width", value=.04,
                        foc_status=NS(driver_enable_status=True), status_code=0)))
    return robot


@pytest.mark.parametrize("code,teaching", [(4, False), (11, True), (0, True)])
def test_sdk_recovery_read_preserves_target_limit_and_teaching_feedback(code, teaching):
    robot = sdk_adapter(code, teaching=teaching)
    with pytest.raises(RecoverableRobotError): robot.read()
    _, state = robot.read_recovery()
    assert state["arm_status"] == code and state["teaching"] == teaching
    assert state["arm_enabled"] is True


@pytest.mark.parametrize("code,error", [(4, 1), (1, 0), (5, 0), (7, 0), (14, 0)])
def test_sdk_hard_faults_are_not_allowed_as_recoverable_feedback(code, error):
    robot = sdk_adapter(code, error)
    with pytest.raises(RuntimeError) as result: robot.read_recovery()
    assert not isinstance(result.value, RecoverableRobotError)
