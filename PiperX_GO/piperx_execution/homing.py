"""SDK-independent execution of Teleop's saved, sequential home waypoints."""
import json
from pathlib import Path

import numpy as np

from protocol import ARMS


def load_home_config(path, execution_cfg):
    """Read the current Teleop file, never the exported training-time home."""
    path = Path(path)
    saved = json.loads(path.read_text(encoding='utf-8-sig'))
    timeout = float(saved.get('reset_timeout_s', 20))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('Teleop reset_timeout_s must be finite and positive')
    arms = {}
    for side in ARMS:
        arm = saved['arms'][side]
        actual = execution_cfg['arms'][side]
        for key in ('interface', 'channel'):
            if str(arm[key]) != str(actual[key]):
                raise ValueError(f'{side} Teleop {key} differs from the inference CAN mapping')
        waypoints = np.asarray(arm.get('home_waypoints', []), dtype=float)
        if (waypoints.ndim != 2 or waypoints.shape[1] != 6 or len(waypoints) == 0
                or not np.isfinite(waypoints).all()):
            raise ValueError(f'{side} Teleop home_waypoints must contain finite six-joint poses')
        speed = float(arm.get('reset_speed_percent', 10))
        if not np.isfinite(speed) or not 0 < speed <= 100 or int(speed) != speed:
            raise ValueError(f'{side} Teleop reset_speed_percent must be an integer in [1,100]')
        arms[side] = dict(home_waypoints=waypoints.tolist(), reset_speed_percent=int(speed))
    return arms, timeout


class HomeSequence:
    """Match Teleop: each arm advances after <0.04 rad error for 0.3 seconds."""
    def __init__(self, arms, timeout):
        self.paths = {side: arms[side]['home_waypoints'] for side in ARMS}
        self.timeout = timeout
        self.indices = dict.fromkeys(ARMS, 0)
        self.active = {}

    @property
    def done(self):
        return not self.active and all(self.indices[s] == len(self.paths[s]) for s in ARMS)

    def step(self, states, now):
        # Validate both arms before returning either arm's next waypoint.
        for side in ARMS:
            state = states[side]
            joints = np.asarray(state['joints'], dtype=float)
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError(f'{side} home requires six finite measured joint angles')
            if (state.get('arm_status') != 0 or state.get('err_code') != 0
                    or state.get('teaching', False) or state.get('arm_enabled') is not True):
                raise RuntimeError(f'{side} home requires enabled NORMAL feedback without teaching/errors')
        jobs = []
        for side in ARMS:
            active = self.active.get(side)
            if active is not None:
                goal, sent, settled = active
                if now-sent > self.timeout:
                    raise TimeoutError(f'Home waypoint timeout: {side}')
                reached = np.max(np.abs(np.asarray(states[side]['joints'])-goal)) < .04
                settled = (now if settled is None else settled) if reached else None
                self.active[side] = goal, sent, settled
                if settled is None or now-settled < .3:
                    continue
                del self.active[side]
            index = self.indices[side]
            if index < len(self.paths[side]):
                goal = self.paths[side][index]
                jobs.append((side, index, list(goal)))
                self.active[side] = goal, now, None
                self.indices[side] += 1
        return jobs
