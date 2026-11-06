"""RGB failures must not request the motor SDK's damped, power-losing stop."""
import json
import time

import pytest

from piperx_teleop.core import CAMERAS, History
from piperx_teleop.operator_guidance import recovery_guidance
from test_motion_recovery import cfg, controller, neutral
from test_pipeline import step


@pytest.mark.parametrize("camera", CAMERAS)
@pytest.mark.parametrize("mode", ["idle", "recording", "resetting"])
def test_rgb_worker_error_holds_both_arms_without_emergency_stop(cfg, camera, mode):
    control = controller(cfg)
    control.begin_input()
    step(control, neutral())
    if mode == "recording":
        data = neutral()
        data["buttons"]["record_start"] = True
        step(control, data)
    elif mode == "resetting":
        for arm in cfg["arms"].values(): arm["home_waypoints"] = [[.2]*6]
        control._begin_reset(time.perf_counter())
    control.cameras.errors[camera] = "USB connection lost"
    step(control, neutral())
    assert control.mode == "recovery" and f"RGB {camera}" in control.recovery_reason
    for robot in control.robots.values():
        assert robot.power
        assert robot.calls[-1][0] == "hold"
        assert not any(call[0] == "stop" for call in robot.calls)
    step(control, neutral())
    guide = recovery_guidance(control.status())
    assert not guide["can_recover"] and "相机" in guide["title"]
    assert "重新启动程序" in guide["steps"][1]
    control.cameras.errors.clear()
    step(control, neutral())
    assert control.mode == "recovery"  # No automatic replay of pre-failure input.
    if mode == "recording":
        assert control.recorder.wait(3)
        meta = json.loads((control.recorder.path/"manifest.json").read_text())
        assert meta["status"] == "incomplete"
    if mode == "resetting":
        assert control.reset_jobs == [] and control.reset_active is None


def test_camera_failure_with_robot_feedback_loss_does_not_use_stale_hold(cfg, monkeypatch):
    control = controller(cfg)
    step(control, neutral())
    control.cameras.errors["left_wrist"] = "USB connection lost"
    control.states["right"] = History()
    monkeypatch.setattr(control.robots["right"], "read", lambda: None)
    step(control, neutral())
    assert control.mode == "fault" and "right feedback" in control.message
    assert all(any(call[0] == "stop" for call in robot.calls) for robot in control.robots.values())
