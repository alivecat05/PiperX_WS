"""A normal clutch release preserves the dispatched pose without a P/J mode change."""
import copy

import numpy as np
import pytest

from test_pipeline import cfg, make_controller, packet, step


@pytest.mark.parametrize("side", ["left", "right"])
def test_release_keeps_last_command_despite_lagging_feedback_and_moving_hand(cfg, monkeypatch, side):
    control = make_controller(cfg)
    robot = control.robots[side]
    held, sent = [], []
    monkeypatch.setattr(robot, "hold", lambda joints: held.append(list(joints)))
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
    for _ in range(10):
        data[side]["position"][2] -= .01
        step(control, data)
        action = control.actions[side].latest().value
        np.testing.assert_allclose(action[:7], target[:7])
        assert action[7] == pytest.approx(.014)
    assert len(held) == len(sent) == 1
    assert control.last_commands[side] == command
    assert control.mode == "idle" and control.enabled[side]
    assert robot.vector[7] == pytest.approx(.014)  # Trigger remains independent of the clutch.
    # Re-clutch uses actual feedback and discards all hand motion made while released.
    data[side]["grip"] = True
    step(control, data)
    assert len(held) == 2 and len(sent) == 1
    np.testing.assert_allclose(control.mappers[side].anchor[2][:7], robot.vector[:7])
