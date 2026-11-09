"""Release freezes the last dispatched filtered pose without replacing it from feedback."""
import copy

import numpy as np
import pytest

from test_pipeline import cfg, make_controller, packet, step


@pytest.mark.parametrize("side", ["left", "right"])
def test_release_freezes_last_filtered_command_without_mode_or_speed_change(cfg, monkeypatch, side):
    control = make_controller(cfg)
    robot = control.robots[side]
    held, sent, speeds = [], [], []
    monkeypatch.setattr(robot, "hold", lambda joints: held.append(list(joints)))
    monkeypatch.setattr(robot, "set_motion_speed", lambda percent: speeds.append(percent))
    # Keep feedback behind the target, as on a real arm while it is still settling.
    monkeypatch.setattr(robot, "send", lambda target: sent.append(list(target)))
    data = packet()
    data[side].update(grip=True, trigger=.2)
    step(control, data)
    data[side]["position"][2] -= .03
    step(control, data)
    assert len(held) == len(sent) == 1
    target = list(sent[-1])
    assert np.linalg.norm(np.asarray(target[:3])-robot.vector[:3]) > .001
    command = copy.deepcopy(control.last_commands[side])
    data[side].update(grip=False, trigger=.8)
    data[side]["position"][2] -= .2  # Motion on the release packet is not a new target.
    step(control, data)
    assert len(held) == 1 and speeds == []
    # Neither changed feedback nor the discarded filter residual may move the target.
    robot.vector[:3] = [.32, .01, .31]
    robot.joints[0] += .03
    for _ in range(10):
        data[side]["position"][2] -= .01
        step(control, data)
        action = control.actions[side].latest().value
        np.testing.assert_allclose(action[:7], target[:7])
        assert action[7] == pytest.approx(.014)
    assert len(held) == len(sent) == 1 and speeds == []
    assert control.last_commands[side] == command
    assert control.mode == "idle" and control.enabled[side]
    assert robot.vector[7] == pytest.approx(.014)  # Trigger remains independent of the clutch.
    # Re-clutch uses actual feedback and discards all hand motion made while released.
    data[side]["grip"] = True
    step(control, data)
    assert len(held) == 2 and len(sent) == 1
    np.testing.assert_allclose(control.mappers[side].anchor[2][:7], robot.vector[:7])


def test_release_keeps_recording_and_other_arm_active(cfg, monkeypatch):
    control = make_controller(cfg)
    data = packet(record_start=True)
    data["left"]["grip"] = data["right"]["grip"] = True
    sent = []
    monkeypatch.setattr(control.robots["right"], "send", lambda target: sent.append(target))
    try:
        step(control, data)
        data["buttons"]["record_start"] = False
        data["left"]["position"][2] -= .02
        data["right"]["position"][2] -= .02
        step(control, data)
        assert control.last_commands["left"]["kind"] == "cartesian"
        held = list(control.last_commands["left"]["target"])
        data["left"]["grip"] = False
        data["right"]["position"][2] -= .02
        step(control, data)
        assert control.recorder.active and control.mode == "teleop"
        np.testing.assert_allclose(control.actions["left"].latest().value[:7], held[:7])
        assert len(sent) == 2 and control.enabled["right"]
    finally:
        control.close()


def test_release_during_filter_catchup_discards_residual(cfg):
    from piperx_teleop.core import PoseMapper
    mapper = PoseMapper(dict(cfg["arms"]["left"], position_scale=1.))
    state = [.3, 0., .3, 0., 0., 0., 1., .04]
    data = packet()["left"]
    data["grip"] = True
    mapper.update(data, state, 1/60)
    data["position"][2] = -.1
    target = mapper.update(data, state, 1/60)
    assert .3 < target[0] < .4
    data["grip"] = False
    data["position"][2] = -.5
    measured = [.305, .01, .31, 0., 0., 0., 1., .04]
    for _ in range(120):
        np.testing.assert_array_equal(mapper.update(data, measured, 1/60)[:7], target[:7])
    data["grip"] = True
    np.testing.assert_allclose(mapper.update(data, measured, 1/60)[:7], measured[:7])
