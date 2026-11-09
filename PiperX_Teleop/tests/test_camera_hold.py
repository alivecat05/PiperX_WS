"""RGB quality must never trigger motor recovery or end an active episode."""
import json
import time

import pytest

from piperx_teleop.core import CAMERAS, History
from piperx_teleop.operator_guidance import recovery_guidance
from test_motion_recovery import cfg, controller, neutral
from test_pipeline import step


@pytest.mark.parametrize("camera", CAMERAS)
@pytest.mark.parametrize("mode", ["idle", "teleop", "recording", "resetting"])
def test_rgb_worker_error_only_warns_and_control_continues(cfg, camera, mode):
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
    elif mode == "teleop":
        data = neutral()
        data["right"]["grip"] = True
        step(control, data)
    before = {side: len(robot.calls) for side, robot in control.robots.items()}
    control.cameras.errors[camera] = "USB connection lost"
    data = neutral()
    if mode == "teleop":
        data["right"]["grip"] = True
        data["right"]["position"][2] -= .02
    step(control, data)
    assert control.mode not in ("recovery", "fault")
    assert camera in control.status()["camera_warnings"]
    for side, robot in control.robots.items():
        assert robot.power
        assert not any(call[0] == "stop" for call in robot.calls[before[side]:])
        if mode != "resetting":
            assert not any(call[0] == "hold" for call in robot.calls[before[side]:])
    if mode == "teleop":
        assert any(call[0] == "send" for call in control.robots["right"].calls[before["right"]:])
    if mode == "recording":
        assert control.recorder.active
    control.cameras.errors.clear()
    step(control, neutral())
    assert control.mode not in ("recovery", "fault")
    if mode == "recording":
        control.recorder.stop(complete=False)
        assert control.recorder.wait(3)
        meta = json.loads((control.recorder.path/"manifest.json").read_text())
        assert meta["status"] == "incomplete"


def test_camera_can_be_missing_at_record_start_without_blocking_motor_readiness(cfg):
    control = controller(cfg)
    step(control, neutral(), camera_frames=False)
    assert control.ready and control.mode == "idle"
    data = neutral()
    data["buttons"]["record_start"] = True
    step(control, data, camera_frames=False)
    assert control.recorder.active and control.status()["camera_warnings"]
    control.recorder.stop(complete=False)
    assert control.recorder.wait(3)


def test_camera_failure_with_robot_feedback_loss_does_not_use_stale_hold(cfg, monkeypatch):
    control = controller(cfg)
    step(control, neutral())
    control.cameras.errors["left_wrist"] = "USB connection lost"
    control.states["right"] = History()
    monkeypatch.setattr(control.robots["right"], "read", lambda: None)
    step(control, neutral())
    assert control.mode == "fault" and "right feedback" in control.message
    assert all(any(call[0] == "stop" for call in robot.calls) for robot in control.robots.values())
