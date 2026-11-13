"""Offline home-sequence and startup watchdog regressions; no SDK motion."""
import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hardware
from homing import HomeSequence, load_home_config
from piperx_teleop.core import History


ARMS = ('left', 'right')


def state(joints=None):
    return dict(joints=list(joints or [0.]*6), arm_status=0, err_code=0,
                arm_enabled=True, teaching=False, vector=[.3, 0, .3, 0, 0, 0, 1, .04],
                source_timestamps={'tcp': 0., 'status': 0.}, gripper_enabled=False)


def homes():
    return {s: dict(home_waypoints=[[v]*6 for v in ((.1, .2) if s == 'left' else (.3,))],
                    reset_speed_percent=10) for s in ARMS}


def test_both_arms_start_together_and_each_waypoint_requires_continuous_settling():
    sequence = HomeSequence(homes(), 5)
    states = {s: state() for s in ARMS}
    jobs = sequence.step(states, 0.)
    assert [s for s, _, _ in jobs] == ['left', 'right']
    states = {s: state(homes()[s]['home_waypoints'][0]) for s in ARMS}
    assert sequence.step(states, .1) == []
    states['left']['joints'] = [0.]*6  # Losing the target resets left's settle timer.
    assert sequence.step(states, .3) == []
    states['left']['joints'] = [.1]*6
    assert sequence.step(states, .41) == []
    assert sequence.step(states, .6) == []
    assert sequence.step(states, .72) == [('left', 1, [.2]*6)]
    states['left']['joints'] = [.2]*6
    assert sequence.step(states, .8) == []
    assert sequence.step(states, 1.11) == []
    assert sequence.done


@pytest.mark.parametrize('fault', ['disabled', 'fault', 'teaching', 'stale_goal'])
def test_home_never_advances_on_invalid_state_or_waypoint_timeout(fault):
    sequence = HomeSequence(homes(), 1)
    states = {s: state() for s in ARMS}
    sequence.step(states, 0.)
    if fault == 'disabled': states['right']['arm_enabled'] = False
    if fault == 'fault': states['right']['arm_status'] = 1
    if fault == 'teaching': states['right']['teaching'] = True
    with pytest.raises((RuntimeError, TimeoutError)):
        sequence.step(states, 1.1)


def make_hardware(tmp_path, monkeypatch):
    class Clock:
        now = 0.
        async def sleep(self, duration): self.now += duration
    clock = Clock()
    monkeypatch.setattr(hardware.time, 'perf_counter', lambda: clock.now)
    monkeypatch.setattr(hardware.asyncio, 'sleep', clock.sleep)
    calls = []
    class Robot:
        def __init__(self, side):
            self.side = side
            self.cfg = dict(reset_speed_percent=99)
            self.joints = [0.]*6
        def cartesian_start_issue(self, joints): return ''
        def move_home(self, joints):
            calls.append((self.side, 'home', list(joints), self.cfg['reset_speed_percent'], clock.now))
            self.joints = list(joints)
        def hold(self, joints): calls.append((self.side, 'hold', list(joints)))
        def set_motion_speed(self, speed): calls.append((self.side, 'speed', speed))
        def stop(self): calls.append((self.side, 'stop'))
    bridge = hardware.PiperHardware.__new__(hardware.PiperHardware)
    bridge.execute = bridge.enabled = bridge.command_started = True
    bridge.teleop_root = tmp_path
    bridge.cfg = {'arms': {s: dict(interface='agx_cando', channel=str(i)) for i, s in enumerate(ARMS)}}
    bridge.sdk_speed_percent = dict.fromkeys(ARMS, 20)
    bridge.robots = {s: Robot(s) for s in ARMS}
    bridge.histories = {s: History() for s in ARMS}
    bridge.cameras = type('Cameras', (), {'histories': {s: History() for s in ('head', 'left', 'right')}})()
    bridge._observation_after = float('-inf')
    bridge.current = {}
    def read():
        bridge.current = {s: state(r.joints) for s, r in bridge.robots.items()}
        for s in ARMS:
            bridge.current[s]['source_timestamps'] = dict(tcp=clock.now, status=clock.now)
            bridge.histories[s].add(clock.now, bridge.current[s])
        return np.tile(bridge.current['left']['vector'], 2)
    bridge.read = read
    bridge.check_cameras = lambda: None
    bridge.observation = lambda: ({'observed_at': clock.now} if clock.now > bridge._observation_after else None)
    for history in [*bridge.histories.values(), *bridge.cameras.histories.values()]:
        history.add(-1, {'old': True})
    saved = {'arms': {s: dict(bridge.cfg['arms'][s], **arm) for s, arm in homes().items()}, 'reset_timeout_s': 2}
    (tmp_path/'config.json').write_text(json.dumps(saved), encoding='utf-8')
    return bridge, clock, calls, saved


def test_home_uses_live_teleop_file_then_restores_speed_and_discards_old_history(tmp_path, monkeypatch):
    bridge, clock, calls, _ = make_hardware(tmp_path, monkeypatch)
    asyncio.run(bridge.go_home())
    motions = [c for c in calls if c[1] == 'home']
    assert [c[:4] for c in motions] == [('left', 'home', [.1]*6, 10),
                                        ('right', 'home', [.3]*6, 10),
                                        ('left', 'home', [.2]*6, 10)]
    assert motions[0][-1] == motions[1][-1]
    assert all(bridge.robots[s].cfg['reset_speed_percent'] == 99 for s in ARMS)
    assert all((s, 'speed', 20) in calls for s in ARMS)
    assert all(all(sample.t >= bridge._observation_after for sample in h.snapshot())
               for h in [*bridge.histories.values(), *bridge.cameras.histories.values()])
    assert clock.now > bridge._observation_after


@pytest.mark.parametrize('invalid', ['mapping', 'nonfinite', 'empty', 'software_limit'])
def test_home_preflight_validates_both_arms_and_all_waypoints_before_any_command(tmp_path, monkeypatch, invalid):
    bridge, _, calls, saved = make_hardware(tmp_path, monkeypatch)
    if invalid == 'mapping': saved['arms']['right']['channel'] = '99'
    if invalid == 'nonfinite': saved['arms']['right']['home_waypoints'][0][0] = float('nan')
    if invalid == 'empty': saved['arms']['right']['home_waypoints'] = []
    if invalid == 'software_limit':
        bridge.robots['right'].cartesian_start_issue = lambda joints: 'outside limit' if joints[0] else ''
    (tmp_path/'config.json').write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError): asyncio.run(bridge.go_home())
    assert calls == []


def test_readonly_home_call_does_not_read_config_or_send_commands(tmp_path, monkeypatch):
    bridge, _, calls, _ = make_hardware(tmp_path, monkeypatch)
    bridge.execute = False
    (tmp_path/'config.json').unlink()
    asyncio.run(bridge.go_home())
    assert calls == []


def test_home_watchdog_failure_holds_and_propagates_instead_of_continuing(tmp_path, monkeypatch):
    bridge, clock, calls, _ = make_hardware(tmp_path, monkeypatch)
    def watchdog():
        if clock.now > .1: raise TimeoutError('head camera stale')
    bridge.check_cameras = watchdog
    with pytest.raises(TimeoutError, match='head camera stale'):
        asyncio.run(bridge.go_home())
    assert len([c for c in calls if c[1] == 'home']) == 2
    assert all(bridge.robots[s].cfg['reset_speed_percent'] == 99 for s in ARMS)
    assert [c[1] for c in calls[-2:]] == ['hold', 'hold']


def test_feedback_watchdog_still_rejects_one_stale_fragment(monkeypatch):
    monkeypatch.setattr(hardware.time, 'perf_counter', lambda: 10.)
    bridge = hardware.PiperHardware.__new__(hardware.PiperHardware)
    bridge.cfg = {'feedback_timeout_s': .15}
    bridge.histories = {s: History() for s in ARMS}
    samples = {s: state() for s in ARMS}
    for item in samples.values(): item['source_timestamps'] = dict(tcp=9.99, joint_12=9.99)
    samples['right']['source_timestamps']['joint_12'] = 9.8
    bridge.robots = {s: type('Robot', (), {'read': lambda self, s=s: (9.99, samples[s])})() for s in ARMS}
    with pytest.raises(TimeoutError, match='right arm feedback is stale'):
        bridge.read()


def test_enable_waits_for_fresh_normal_enabled_feedback_after_ack(monkeypatch):
    monkeypatch.setattr(hardware.time, 'perf_counter', lambda: 10.)
    bridge = hardware.PiperHardware.__new__(hardware.PiperHardware)
    bridge.execute, bridge.command_started, bridge.enabled = True, False, False
    bridge.current = {s: state() for s in ARMS}
    for item in bridge.current.values(): item['source_timestamps'] = dict(status=9.99)
    bridge.hold = lambda: None
    bridge._enable_ack = dict.fromkeys(ARMS, False)
    bridge._enable_started_at = None
    bridge.robots = {s: type('Robot', (), {'cartesian_start_issue': lambda *_: '', 'enable': lambda *_: True})() for s in ARMS}
    assert not bridge.enable()  # ACK alone is insufficient.
    for item in bridge.current.values(): item['source_timestamps']['status'] = 10.01
    bridge.current['right']['arm_enabled'] = False
    assert not bridge.enable()
    bridge.current['right']['arm_enabled'] = True
    assert bridge.enable()
