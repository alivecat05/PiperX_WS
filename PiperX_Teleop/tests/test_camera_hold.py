"""RGB quality must never trigger motor recovery or end an active episode."""
import ctypes
import json
import os
import time

import numpy as np
import pytest

from piperx_teleop import cameras
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


def _capture_during_reconnect(cfg, _settings, stop, buffer, available, sender):
    pixels = np.frombuffer(buffer, dtype=np.uint8)
    stats = cameras._diagnostics()
    with sender:
        for sequence in (1, 2):
            if cfg["serial"] == "left_wrist" and sequence == 2:
                stats["reconnects"] += 1
                sender.send((None, None, stats, "Simulated camera reconnect"))
                if os.name == "nt":
                    ctypes.PyDLL("kernel32").Sleep(1600)  # Hold the worker's GIL like pipeline.start().
                else:
                    time.sleep(1.6)
            if not available.acquire(timeout=4):
                return
            pixels.fill(sequence)
            stamp = time.perf_counter()
            stats["received_frames"] += 1
            sender.send((stamp, {"sequence": sequence, "receive_s": stamp,
                                 "exposure_settings": []}, stats, None))
        stop.wait(10)


def _wait_for(predicate, timeout=8):
    deadline = time.perf_counter() + timeout
    while not predicate() and time.perf_counter() < deadline:
        time.sleep(.01)
    assert predicate()


def test_camera_reconnect_isolated_from_control_and_preserves_rgb_frames(cfg, monkeypatch):
    for name, camera in cfg["cameras"].items():
        camera["serial"] = name
    monkeypatch.setattr(cameras, "_capture_camera", _capture_during_reconnect)
    control = controller(cfg)
    control.cameras.mock = False
    try:
        control.cameras.start()
        _wait_for(lambda: "left_wrist" in control.cameras.errors)
        history = control.cameras.histories["left_wrist"]
        first = history.latest()
        assert first.value["sequence"] == 1
        data = neutral()
        data["right"]["grip"] = True
        step(control, data)
        data["right"]["position"][2] -= .02
        step(control, data)
        assert control.mode == "teleop"
        assert any(call[0] == "send" for call in control.robots["right"].calls)
        assert "left_wrist" in control.cameras.errors
        _wait_for(lambda: control.cameras.diagnostics["left_wrist"]["received_frames"] == 2)
        _wait_for(lambda: all(h.latest() and h.latest().value["sequence"] == 2
                              for h in control.cameras.histories.values()))
        assert "left_wrist" not in control.cameras.errors
        assert np.all(first.value["bgr"] == 1) and np.all(history.latest().value["bgr"] == 2)
        assert history.latest().t == history.latest().value["receive_s"]
    finally:
        control.cameras.close()
    assert all(not thread.is_alive() for thread in control.cameras.threads)
