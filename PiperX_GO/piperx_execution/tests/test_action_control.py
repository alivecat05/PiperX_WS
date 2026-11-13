"""Offline command-contract regressions; no SDK or physical robot connections."""
import asyncio
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control import (ActionTimeline, CartesianFilter, blend_chunk_start,
                     blend_pose, fresh_actions, smooth_chunk)
from hardware import MockHardware
from client import control_loop


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'config.pc.json').read_text(encoding='utf-8-sig'))


def pose(x=.3):
    return np.tile([x, 0., .3, 0., 0., 0., 1., .04], 2).astype(np.float32)


def rotation_delta(first, second):
    return (Rotation.from_quat(second)*Rotation.from_quat(first).inv()).as_rotvec()


def test_chunk_fir_reduces_alternation_and_preserves_quaternion_and_gripper():
    raw = np.tile(pose(), (30, 1))
    raw[:, 0] += .01*(-1.)**np.arange(30)
    raw[:, 3:7] = Rotation.from_rotvec(np.c_[np.zeros(30), np.zeros(30), .1*(-1.)**np.arange(30)]).as_quat()
    raw[::2, 3:7] *= -1
    raw[:, 7] = np.linspace(0, .1, 30)
    result = smooth_chunk(raw, 5)
    assert np.std(result[3:-3, 0]) < .2*np.std(raw[3:-3, 0])
    assert np.max(np.linalg.norm(Rotation.from_quat(result[3:-3, 3:7]).as_rotvec(), axis=1)) < .02
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(result[:, 7], raw[:, 7])
    np.testing.assert_allclose(raw[:, 0], .3+.01*(-1.)**np.arange(30), atol=1e-7)


def test_quaternion_sign_and_pi_crossing_use_shortest_path():
    first, second = pose(), pose()
    first[3:7] = Rotation.from_euler('z', 179, degrees=True).as_quat()
    second[3:7] = Rotation.from_euler('z', -179, degrees=True).as_quat()
    blended = blend_pose(first, second, .5)
    assert np.linalg.norm(rotation_delta(first[3:7], blended[3:7])) == pytest.approx(np.deg2rad(1), abs=1e-6)
    second[3:7] *= -1
    np.testing.assert_allclose(blend_pose(first, second, .5), blended, atol=1e-6)


def test_expired_samples_are_trimmed_and_boundary_is_ramped():
    raw = np.tile(pose(), (30, 1))
    raw[:, 0] += np.arange(30)*.001
    fresh, skipped = fresh_actions(raw, 10, 18)
    assert skipped == 8
    assert fresh[0, 0] == raw[8, 0]
    blended = blend_chunk_start(fresh, pose(), 5)
    assert 0 < blended[0, 0]-.3 < .001
    np.testing.assert_allclose(blended[4], fresh[4])
    with pytest.raises(TimeoutError):
        fresh_actions(raw, 10, 40)


def test_rtc_same_time_overlap_and_no_replay():
    old, new = np.tile(pose(.32), (30, 1)), np.tile(pose(.34), (30, 1))
    timeline = ActionTimeline(5)
    timeline.install(old, 0, 3, pose())
    timeline.take(3)
    timeline.install(new, 2, 4, pose())
    assert .30 < timeline.take(4)[0] < .34
    assert timeline.take(4) is None
    with pytest.raises(ValueError):
        timeline.install(new, 2, 5)
    with pytest.raises(TimeoutError):
        timeline.install(new, 3, 33)


def test_cartesian_command_speed_and_acceleration_survive_abrupt_reversals():
    cfg = config()
    limiter = CartesianFilter(cfg, 30)
    sent = pose()
    previous_linear = np.zeros(3)
    previous_angular = np.zeros(3)
    previous_gripper = 0.
    for index in range(240):
        desired = pose()
        sign = 1 if (index//24) % 2 == 0 else -1
        desired[:3] += sign*np.array([.03, .025, .02])
        desired[3:7] = Rotation.from_rotvec(sign*np.array([.15, .12, .08])).as_quat()
        desired[7] = .07 if sign > 0 else .01
        next_sent, _ = limiter.apply(desired, sent)
        linear = (next_sent[:3]-sent[:3])*30
        angular = rotation_delta(sent[3:7], next_sent[3:7])*30
        gripper = float(next_sent[7]-sent[7])*30
        assert np.linalg.norm(linear) <= .080002
        assert np.linalg.norm(angular) <= .50001
        assert abs(gripper) <= cfg['inference']['max_gripper_speed_m_s']+2e-6
        assert np.linalg.norm(linear-previous_linear)*30 <= .2501
        assert np.linalg.norm(angular-previous_angular)*30 <= 1.5001
        assert abs(gripper-previous_gripper)*30 <= cfg['inference']['max_gripper_acceleration_m_s2']+1e-4
        np.testing.assert_allclose(np.linalg.norm(next_sent[3:7]), 1, atol=1e-6)
        limiter.commit(next_sent)
        sent = next_sent
        previous_linear, previous_angular, previous_gripper = linear, angular, gripper
    assert limiter.diagnostics['acceleration_limited'] > 0
    assert limiter.diagnostics['reversals_blocked'] > 0


def test_reversal_debounce_survives_pause_but_accepts_sustained_reverse():
    limiter = CartesianFilter(config(), 30)
    current = pose()
    forward = pose(.33)
    for _ in range(8):
        current, _ = limiter.apply(forward, current)
        limiter.commit(current)
    limiter.reset(current, preserve_direction=True)
    backward = pose(.27)
    for _ in range(2):
        candidate, _ = limiter.apply(backward, current)
        assert candidate[0] == current[0]
        limiter.commit(candidate)
    candidate, _ = limiter.apply(backward, current)
    assert candidate[0] < current[0]


def test_raw_invalid_chunk_and_emitted_tracking_limits_still_reject():
    limiter = CartesianFilter(config(), 30)
    raw = np.tile(pose(), (30, 1))
    raw[10, 0] = 2
    with pytest.raises(ValueError, match='workspace'):
        limiter.validate_chunk(raw)
    limiter.reset(pose(.45))
    with pytest.raises(ValueError, match='commanded TCP.*feedback'):
        limiter.apply(pose(.45), pose())
    with pytest.raises(ValueError, match='workspace'):
        limiter.apply(pose(2), pose())


def test_right_model_goal_103mm_ahead_is_limited_without_aborting():
    limiter = CartesianFilter(config(), 30)
    feedback, desired = pose(), pose()
    desired[8] += .103
    sent, limited = limiter.apply(desired, feedback)
    assert limited
    assert limiter.diagnostics['tracking_limited'] == 1
    assert feedback[8] < sent[8] < feedback[8]+.08/30
    np.testing.assert_array_equal(sent[:8], feedback[:8])
    assert desired[8] == pytest.approx(.403)
    limiter.commit(sent)


@pytest.mark.parametrize('limit', [.02, .1])
def test_model_goal_can_stay_far_ahead_but_sent_tcp_never_exceeds_lead(limit):
    cfg = config()
    cfg['max_position_tracking_error_m'] = limit
    limiter = CartesianFilter(cfg, 30)
    feedback, desired = pose(), pose(.55)
    previous = feedback.copy()
    previous_velocity = np.zeros(3)
    for _ in range(400):
        sent, limited = limiter.apply(desired, feedback)
        distance = np.linalg.norm(sent[:3]-feedback[:3])
        assert distance <= limit
        velocity = (sent[:3]-previous[:3])*30
        assert np.linalg.norm(velocity) <= .080002
        assert np.linalg.norm(velocity-previous_velocity)*30 <= .2501
        limiter.commit(sent)
        previous, previous_velocity = sent.copy(), velocity
    assert distance == pytest.approx(limit, abs=1e-5)
    assert limited


@pytest.mark.parametrize('limit', [0, -1, float('nan'), float('inf')])
def test_invalid_tracking_limit_rejected(limit):
    cfg = config()
    cfg['max_position_tracking_error_m'] = limit
    with pytest.raises(ValueError, match='max_position_tracking_error_m'):
        CartesianFilter(cfg, 30)


@pytest.mark.parametrize('key,value', [('position_filter_tau_s', -1), ('max_acceleration_m_s2', 0),
    ('reverse_confirm_steps', 0), ('chunk_smoothing_window', 4), ('max_speed_m_s', float('nan'))])
def test_bad_filter_configuration_rejected(key, value):
    cfg = config()
    cfg['inference'][key] = value
    with pytest.raises(ValueError):
        CartesianFilter(cfg, 30)


@pytest.mark.parametrize('rtc', [False, True])
def test_mock_loop_chunk_transition_and_wait_do_not_repeat_hold(rtc):
    cfg = config()

    class Hardware(MockHardware):
        def __init__(self):
            super().__init__(cfg)
            self.holds = 0
            self.targets = []

        def hold(self):
            self.holds += 1

        def send(self, target):
            self.targets.append(target.copy())
            self.state = target.copy()

    class Policy:
        n_obs_steps = 1

        async def infer(self, observation, anchor_tick):
            await asyncio.sleep(.075)
            plan = np.tile(observation['state'], (30, 1))
            plan[:, 0] += .025
            return dict(actions=plan, observation_tick=anchor_tick, roundtrip_s=.075,
                        request_id=anchor_tick+1)

    hardware = Hardware()
    args = SimpleNamespace(control_hz=30, blend_steps=5, max_steps=12,
        rtc_lookahead_steps=3, rtc=rtc, action_steps=4, log_every=30)
    result = asyncio.run(control_loop(hardware, Policy(), args, cfg))
    assert result['control_steps'] == 12
    assert result['discarded_actions'] >= 2
    assert result['inferences'] >= 2
    assert hardware.holds < result['hold_ticks']
    steps = np.diff(np.array(hardware.targets)[:, :3], axis=0)
    assert np.max(np.linalg.norm(steps, axis=1)) < .08/30+1e-6


@pytest.mark.parametrize('rtc', [False, True])
def test_expired_responses_hold_retry_and_resume_on_fresh_chunk(rtc, capsys):
    cfg = config()

    class Policy:
        n_obs_steps = 1
        calls = 0
        fresh_returned = False

        async def infer(self, observation, anchor_tick):
            self.calls += 1
            request_id = self.calls
            await asyncio.sleep(.01)
            expired = request_id <= 2
            self.fresh_returned |= not expired
            plan = np.tile(observation['state'], (30, 1))
            plan[:, 0] += .01
            return dict(actions=plan, observation_tick=anchor_tick-(30 if expired else 0),
                        roundtrip_s=.01, request_id=request_id)

    policy = Policy()

    class Hardware(MockHardware):
        holds = 0

        def send(self, target):
            assert policy.fresh_returned, 'An expired response reached hardware.send'
            super().send(target)

        def hold(self):
            self.holds += 1

    hardware = Hardware(cfg)
    args = SimpleNamespace(control_hz=30, blend_steps=5, max_steps=6,
        rtc_lookahead_steps=3, rtc=rtc, action_steps=4, log_every=30)
    result = asyncio.run(control_loop(hardware, policy, args, cfg))
    assert result['expired_chunks'] == 2
    assert result['discarded_actions'] >= 60
    assert result['control_steps'] == hardware.sent == 6
    assert hardware.holds >= 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    expired = [event for event in events if event['event'] == 'action_chunk_expired']
    assert [event['consecutive_expired_chunks'] for event in expired] == [1, 2]
    assert all(event['observation_age_ms'] >= event['horizon_ms'] == 1000 for event in expired)
    assert all(event['accepted'] is False for event in events if event['event'] == 'inference' and event['request_id'] <= 2)


@pytest.mark.parametrize('rtc', [False, True])
def test_hardware_watchdog_still_terminates_during_repeated_expiration(rtc):
    cfg = config()

    class Hardware(MockHardware):
        reads = 0

        def read(self):
            self.reads += 1
            if self.reads >= 8:
                raise TimeoutError('CAN feedback watchdog expired')
            return super().read()

        def send(self, target):
            raise AssertionError('Expired action must never be sent')

    class Policy:
        n_obs_steps = 1

        async def infer(self, observation, anchor_tick):
            await asyncio.sleep(.01)
            return dict(actions=np.tile(observation['state'], (30, 1)),
                        observation_tick=anchor_tick-30, roundtrip_s=.01, request_id=anchor_tick+31)

    args = SimpleNamespace(control_hz=30, blend_steps=5, max_steps=6,
        rtc_lookahead_steps=3, rtc=rtc, action_steps=4, log_every=30)
    with pytest.raises(TimeoutError, match='CAN feedback watchdog'):
        asyncio.run(control_loop(Hardware(cfg), Policy(), args, cfg))
