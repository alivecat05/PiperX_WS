"""Explicit seamless exit uses the installed SDK protocol, never motor-power reset."""
from types import SimpleNamespace as NS
import time

import pytest

from piperx_teleop.robot import PiperX
from piperx_teleop.operator_guidance import recovery_guidance
from piperx_teleop.preview import banner_style
from test_motion_recovery import cfg, controller, enter_limit_recovery, neutral, sdk_adapter
from test_pipeline import step


def test_real_sdk_exit_packet_has_no_emergency_stop_or_reset_and_holds_current_joints():
    from pyAgxArm.protocols.can_protocol.drivers.piper.default.parser import Parser
    adapter = PiperX.__new__(PiperX)
    adapter.cfg = {"firmware": "v189"}
    sent = []
    adapter.robot = NS(get_joint_limits_enabled=lambda: False,
        _send_msg=lambda msg: sent.append(Parser(None).pack(msg)),
        move_j=lambda joints: sent.append(list(joints)))
    joints = [.1, .4, -.4, .1, .2, .1]
    adapter.exit_teaching(joints)
    assert sent[0].arbitration_id == 0x150
    assert bytes(sent[0].data) == bytes([0, 6, 2, 0, 0, 0, 0, 0])
    assert sent[1] == joints


@pytest.mark.parametrize("firmware,joints", [("v183", [0]*6), ("v188", [0]*6),
                                            ("v189", [0]*5), ("v189", [float("nan")]*6)])
def test_unsupported_or_invalid_exit_sends_nothing(firmware, joints):
    adapter = PiperX.__new__(PiperX)
    adapter.cfg = {"firmware": firmware}
    adapter.robot = NS(get_joint_limits_enabled=lambda: False,
                       _send_msg=lambda _: pytest.fail("Invalid exit packet sent"),
                       move_j=lambda _: pytest.fail("Invalid hold sent"))
    with pytest.raises(ValueError): adapter.exit_teaching(joints)


@pytest.mark.parametrize("mode", [0, 1])
def test_old_teach_status_does_not_override_fresh_normal_mode(mode):
    adapter = sdk_adapter(0)
    status = adapter.robot.get_arm_status().msg
    status.teach_status = 1
    status.ctrl_mode = mode
    _, state = adapter.read()
    assert state["teaching"] is False and state["teach_status"] == 1


def test_manual_teach_exit_with_power_restored_returns_to_ready_at_new_pose(cfg):
    control = teaching_controller(cfg)
    for robot in control.robots.values():
        robot.joints = [.15]*6
        robot.teaching, robot.code = False, 0
        robot.calls.clear()
    step(control, neutral())
    assert control.recovery_after_teach_exit
    assert all(target == [.15]*6 for target in control.recovery_hold_targets.values())
    control.recovery_neutral_since = time.perf_counter()-.31
    step(control, neutral())
    step(control, neutral())
    assert control.mode == "paused"
    step(control, neutral())
    control.resume_since = time.perf_counter()-.31
    step(control, neutral())
    assert control.mode == "idle" and control.ready
    assert banner_style(control.status())[0] == "#168a46"
    for robot in control.robots.values():
        assert all(c[1] == [.15]*6 for c in robot.calls if c[0] == "hold")
        assert not any(c[0] in ("stop", "send", "exit_teach") for c in robot.calls)


def test_manual_exit_can_start_a_new_episode_after_teaching_interrupted_capture(cfg):
    control = controller(cfg)
    control.begin_input()
    data = neutral()
    data["buttons"]["record_start"] = True
    step(control, data)
    previous_path = control.recorder.path
    assert control.recorder.active
    control.robots["left"].code, control.robots["left"].teaching = 11, True
    step(control, neutral())
    assert control.mode == "recovery" and control.recorder.wait(3)
    control.robots["left"].code, control.robots["left"].teaching = 0, False
    control.robots["left"].joints = [.2]*6
    step(control, neutral())
    held = neutral()
    held["left"]["grip"] = True
    step(control, held)
    assert control.mode == "recovery" and not control.recovery_enabling
    step(control, neutral())
    control.recovery_neutral_since = time.perf_counter()-.31
    step(control, neutral())
    step(control, neutral())
    assert control.mode == "paused"
    step(control, neutral())
    control.resume_since = time.perf_counter()-.31
    step(control, neutral())
    assert banner_style(control.status())[0] == "#168a46"
    step(control, data)
    try:
        assert control.recorder.active and control.recorder.path != previous_path
        assert banner_style(control.status())[0] == "#cf233a"
    finally:
        control.recorder.stop()
        assert control.recorder.wait(3)


def teaching_controller(cfg, acknowledge=True):
    control = controller(cfg)
    control.begin_input()
    enter_limit_recovery(control)
    for robot in control.robots.values():
        robot.code, robot.teaching = 11, True
        def exit_teach(joints, robot=robot):
            robot.calls.append(("exit_teach", list(joints)))
            robot.hold(joints)
            if acknowledge:
                robot.code, robot.teaching = 0, False
        robot.exit_teaching = exit_teach
    step(control, neutral())
    return control


def test_one_click_exits_then_waits_for_fresh_normal_feedback_and_neutral(cfg):
    control = teaching_controller(cfg)
    assert recovery_guidance(control.status())["can_recover"]
    control.request_recovery()
    step(control, neutral())
    assert control.teach_exit_pending and control.mode == "recovery"
    assert not any(c[0] == "enable" for r in control.robots.values() for c in r.calls)
    step(control, neutral())
    assert not control.teach_exit_pending and not control.recovery_enabling
    control.recovery_neutral_since = time.perf_counter()-.31
    step(control, neutral())
    assert control.recovery_enabling  # The same explicit click continues after safe confirmation.
    step(control, neutral())
    assert control.mode == "paused"
    for robot in control.robots.values():
        assert len([c for c in robot.calls if c[0] == "exit_teach"]) == 1
        assert not any(c[0] in ("stop", "send") for c in robot.calls)


def test_exit_timeout_does_not_repeat_reset_or_enable(cfg):
    control = teaching_controller(cfg, acknowledge=False)
    control.request_recovery()
    step(control, neutral())
    for _ in range(5): step(control, neutral())
    for side in control.teach_exit_pending:
        control.teach_exit_pending[side] = time.perf_counter()-3.1
    step(control, neutral())
    assert control.mode == "recovery" and control.teach_exit_error and not control.teach_exit_pending
    for robot in control.robots.values():
        assert len([c for c in robot.calls if c[0] == "exit_teach"]) == 1
        assert not any(c[0] in ("stop", "enable", "send") for c in robot.calls)


def test_exit_without_quest_does_not_automatically_resume_control(cfg):
    control = teaching_controller(cfg)
    control.disconnect()
    control.request_recovery()
    control.tick(time.perf_counter(), .02)
    control.tick(time.perf_counter(), .02)
    assert control.mode == "recovery" and not control.recovery_after_teach_exit
    assert not any(c[0] == "enable" for r in control.robots.values() for c in r.calls)


def test_pressed_middle_button_blocks_explicit_teach_exit(cfg):
    control = teaching_controller(cfg)
    data = neutral()
    data["left"]["grip"] = True
    control.request_recovery()
    step(control, data)
    assert not control.teach_exit_pending
    assert not any(c[0] == "exit_teach" for r in control.robots.values() for c in r.calls)
