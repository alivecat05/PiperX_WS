"""Small oscillations are damped without changing scale or accumulating motion lag."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piperx_teleop.app import load_config
from piperx_teleop.core import PoseMapper


def arm_config():
    return load_config(Path(__file__).resolve().parents[1]/"config.json")["arms"]["left"]


def hand():
    return {"grip": True, "trigger": 0., "position": [0., 1., 0.],
            "quaternion": [0., 0., 0., 1.]}


def settled_mapper(arm):
    mapper, controller = PoseMapper(arm), hand()
    state = [.3, 0., .3, 0., 0., 0., 1., .04]
    mapper.update(controller, state, 1/60)
    controller["position"][2] = -.03
    for _ in range(180):
        state = mapper.update(controller, state, 1/60)
    return mapper, controller, state


@pytest.mark.parametrize("frequency", [8, 10, 12])
def test_tremor_is_attenuated_after_deliberate_motion(frequency):
    measurements = {}
    for enabled in (False, True):
        arm = arm_config()
        assert arm["tremor_smoothing_s"] == .04
        if not enabled:
            arm["tremor_smoothing_s"] = 0.
        mapper, controller, state = settled_mapper(arm)
        positions, rotations = [], []
        for i in range(600):
            wave = np.sin(2*np.pi*frequency*i/60)
            controller["position"][2] = -.03+.0015*wave
            controller["quaternion"] = Rotation.from_euler("z", .7*wave, degrees=True).as_quat()
            controller["trigger"] = (i % 11)/10
            state = mapper.update(controller, state, 1/60)
            assert state[7] == pytest.approx((1-controller["trigger"])*arm["gripper_max_m"])
            if i >= 120:
                positions.append(state[:3])
                rotations.append(Rotation.from_quat(state[3:7]).as_rotvec())
        measurements[enabled] = (np.linalg.norm(np.std(positions, axis=0)),
                                 np.linalg.norm(np.std(rotations, axis=0)))
    raw_position, raw_rotation = measurements[False]
    filtered_position, filtered_rotation = measurements[True]
    assert raw_position > .0001 and raw_rotation > np.deg2rad(.01)
    assert filtered_position < raw_position*.5
    assert filtered_rotation < raw_rotation*.5
    assert filtered_position < .00016
    assert filtered_rotation < np.deg2rad(.075)


def test_one_to_one_motion_has_bounded_lag_and_no_overshoot():
    arm = arm_config()
    assert arm["position_scale"] == 1.
    mapper, controller = PoseMapper(arm), hand()
    state = [.3, 0., .3, 0., 0., 0., 1., .04]
    mapper.update(controller, state, 1/60)
    for i in range(1, 61):
        controller["position"][2] = -.2*i/60
        desired = .3+.2*i/60
        previous = state[0]
        state = mapper.update(controller, state, 1/60)
        assert previous <= state[0] <= desired
        assert desired-state[0] < .003
        assert state[0]-previous <= arm["max_speed_m_s"]/60+1e-9
    stopped = state[0]
    for _ in range(60):
        state = mapper.update(controller, state, 1/60)
        assert stopped <= state[0] <= .5
        assert state[0]-stopped < .002
    assert .5-state[0] <= arm["position_deadband_m"]+1e-6


def test_releasing_and_reclutching_does_not_replay_filtered_motion():
    arm = arm_config()
    mapper, controller, state = settled_mapper(arm)
    controller["position"][2] -= .02
    last_target = mapper.update(controller, state, 1/60)
    measured = [.31, .02, .3, 0., 0., 0., 1., .04]
    controller["grip"] = False
    assert mapper.update(controller, measured, 1/60)[:7] == last_target[:7]
    controller["position"][2] = -1.
    controller["grip"] = True
    assert mapper.update(controller, measured, 1/60)[:7] == measured[:7]


@pytest.mark.parametrize("value", [-.01, .21, float("nan"), float("inf")])
def test_invalid_tremor_filter_configuration_is_rejected(tmp_path, value):
    cfg = json.loads((Path(__file__).resolve().parents[1]/"config.json").read_text())
    cfg["arms"]["left"]["tremor_smoothing_s"] = value
    path = tmp_path/"config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="tremor_smoothing_s"):
        load_config(path)
