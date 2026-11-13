"""Chunk conditioning and bounded Cartesian commands, independent of SDK/network."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from protocol import ARMS, actions, vector


def _unit_quaternion(value):
    value = np.asarray(value, dtype=float)
    return value / np.linalg.norm(value)


def _rotation_error(first, second):
    """Shortest rotation taking first to second, expressed in the arm's base."""
    return (Rotation.from_quat(second) * Rotation.from_quat(first).inv()).as_rotvec()


def _clip_norm(value, maximum):
    norm = np.linalg.norm(value)
    return value * min(1., maximum / max(norm, 1e-12))


def _ease(value):
    """Quintic transition with zero first/second derivatives at the endpoints."""
    return value**3 * (10. + value * (-15. + 6. * value))


def blend_pose(first, second, weight):
    """Linear xyz/gripper interpolation and shortest-path quaternion SLERP."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Pose blend weight must be in [0,1]')
    first, second = vector(first), vector(second)
    result = (1-weight)*first+weight*second
    for start in (0, 8):
        a = _unit_quaternion(first[start+3:start+7])
        b = _unit_quaternion(second[start+3:start+7])
        q = (Rotation.from_rotvec(weight*_rotation_error(a, b)) * Rotation.from_quat(a)).as_quat()
        if np.dot(q, a) < 0:
            q = -q
        result[start+3:start+7] = q
    return result


def smooth_chunk(predicted, window=5):
    """Triangular FIR on xyz/SO(3); leave gripper timing intact.

    Only predictions are padded/looked ahead, never real observations.
    Quaternion averaging uses local rotation vectors, so q/-q are equivalent.
    """
    if type(window) is not int or window < 1 or window > 31 or window % 2 == 0:
        raise ValueError('chunk_smoothing_window must be an odd integer in [1,31]')
    predicted = actions(predicted)
    result = predicted.copy()
    radius = window//2
    weights = radius+1-np.abs(np.arange(-radius, radius+1))
    weights = weights/weights.sum()
    for start in (0, 8):
        for index in range(len(predicted)):
            indices = np.clip(np.arange(index-radius, index+radius+1), 0, len(predicted)-1)
            result[index, start:start+3] = weights @ predicted[indices, start:start+3]
            reference = Rotation.from_quat(predicted[index, start+3:start+7])
            rotations = Rotation.from_quat(predicted[indices, start+3:start+7])
            tangent = (rotations * reference.inv()).as_rotvec()
            q = (Rotation.from_rotvec(weights @ tangent) * reference).as_quat()
            if np.dot(q, predicted[index, start+3:start+7]) < 0:
                q = -q
            result[index, start+3:start+7] = q
    return result


class StaleActionChunk(TimeoutError):
    """A valid response whose entire execution window has already elapsed."""
    def __init__(self, horizon, anchor_tick, current_tick):
        self.horizon = horizon
        self.age_steps = max(0, current_tick-anchor_tick)
        super().__init__('Inference result is older than its entire action horizon')


def fresh_actions(predicted, anchor_tick, current_tick):
    """Trim by observation time in BOTH modes; never restart an old chunk."""
    predicted = actions(predicted)
    skipped = max(0, current_tick-anchor_tick)
    if skipped >= len(predicted):
        raise StaleActionChunk(len(predicted), anchor_tick, current_tick)
    return predicted[skipped:].copy(), skipped


def blend_chunk_start(predicted, reference, blend_steps):
    result = actions(predicted).copy()
    if reference is not None and blend_steps > 0:
        # Retain ramp duration even for a shorter executable prefix.
        for index in range(min(blend_steps, len(result))):
            result[index] = blend_pose(reference, result[index], _ease((index+1)/blend_steps))
    return result


class ActionTimeline:
    """Overlap fusion only combines predictions for the SAME PC action tick."""
    def __init__(self, blend_steps=5):
        if type(blend_steps) is not int or blend_steps < 0:
            raise ValueError('blend_steps must be a non-negative integer')
        self.plans = {}
        self.blend_steps = blend_steps
        self.last_anchor = None
        self.last_taken = None

    def install(self, predicted, anchor_tick, current_tick, reference=None):
        if self.last_anchor is not None and anchor_tick <= self.last_anchor:
            raise ValueError('Action chunk observation time did not advance')
        if self.last_taken is not None and current_tick <= self.last_taken:
            raise ValueError('Action timeline cannot replay an already consumed tick')
        candidate, skipped = fresh_actions(predicted, anchor_tick, current_tick)
        previous = self.plans
        self.plans = {}
        for offset, value in enumerate(candidate):
            tick = max(anchor_tick, current_tick)+offset
            if offset < self.blend_steps:
                first = previous.get(tick, reference)
                if first is not None:
                    value = blend_pose(first, value, _ease((offset+1)/self.blend_steps))
            self.plans[tick] = value.copy()
        self.last_anchor = anchor_tick
        return skipped

    def take(self, tick):
        if self.last_taken is not None and tick <= self.last_taken:
            return None
        for old in [key for key in self.plans if key < tick]:
            del self.plans[old]
        self.last_taken = tick
        return self.plans.pop(tick, None)

    def remaining(self, tick):
        return max(self.plans, default=tick-1)-tick+1


class CartesianFilter:
    """Persistent low-pass, reversal debounce and TCP command v/a bounds.

    These are command limits, not joint/physical acceleration guarantees.
    Feedback/workspace watchdogs remain independent hard checks.
    """
    DEFAULTS = dict(
        max_speed_m_s=.08, max_rotation_rad_s=.5, max_gripper_speed_m_s=.08,
        max_acceleration_m_s2=.25, max_angular_acceleration_rad_s2=1.5,
        max_gripper_acceleration_m_s2=.50,
        position_filter_tau_s=.12, rotation_filter_tau_s=.15,
        gripper_filter_tau_s=.04,
        reverse_confirm_steps=3, reverse_threshold_m=.002,
        reverse_threshold_rad=.02,
        chunk_smoothing_window=5, sync_action_steps=4, sdk_speed_percent=20,
    )

    def __init__(self, cfg, hz):
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError('Control frequency must be positive')
        self.cfg, self.dt, self.last = cfg, 1/hz, None
        self.tracking_limit = float(cfg.get('max_position_tracking_error_m', .1))
        if not math.isfinite(self.tracking_limit) or self.tracking_limit <= 0:
            raise ValueError('max_position_tracking_error_m must be finite and positive')
        self.limits = dict(self.DEFAULTS, **cfg.get('inference', {}))
        for name in self.DEFAULTS:
            value = self.limits[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid inference.{name}')
            if name.endswith('_tau_s'):
                valid = value >= 0
            elif name in ('reverse_confirm_steps', 'sync_action_steps', 'sdk_speed_percent'):
                valid = type(value) is int and value >= 1
            else:
                valid = value > 0
            if not valid:
                raise ValueError(f'Invalid inference.{name}')
        if not 1 <= self.limits['sdk_speed_percent'] <= 100:
            raise ValueError('inference.sdk_speed_percent must be in [1,100]')
        window = self.limits['chunk_smoothing_window']
        if type(window) is not int or window > 31 or window % 2 == 0:
            raise ValueError('inference.chunk_smoothing_window must be odd and <=31')
        for side in ARMS:
            arm = cfg['arms'][side]
            for name in ('max_speed_m_s', 'max_rotation_rad_s', 'gripper_max_m'):
                if not math.isfinite(arm[name]) or arm[name] <= 0:
                    raise ValueError(f'Invalid arms.{side}.{name}')
            low, high = np.asarray(arm['workspace_min']), np.asarray(arm['workspace_max'])
            if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
                raise ValueError(f'Invalid {side} workspace')
        self.velocity = np.zeros((2, 7))  # xyz, base-frame angular velocity, gripper
        self.direction = np.zeros((2, 6), dtype=np.int8)
        self.reverse_counts = np.zeros((2, 6), dtype=np.int32)
        self._pending_velocity = None
        self.diagnostics = dict(reversals_blocked=0, acceleration_limited=0,
                                speed_limited=0, tracking_limited=0)

    def reset(self, feedback, preserve_direction=False):
        self.last = vector(feedback).copy()
        for start in (0, 8):
            self.last[start+3:start+7] = _unit_quaternion(self.last[start+3:start+7])
        self.velocity.fill(0)
        self._pending_velocity = None
        self.reverse_counts.fill(0)
        if not preserve_direction:
            self.direction.fill(0)

    def _validate(self, value, feedback, *, check_tracking=True):
        for side, start in zip(ARMS, (0, 8)):
            arm = self.cfg['arms'][side]
            target = value[start:start+8]
            if np.any(target[:3] < arm['workspace_min']) or np.any(target[:3] > arm['workspace_max']):
                raise ValueError(f'{side} predicted TCP is outside workspace')
            error = np.linalg.norm(target[:3]-feedback[start:start+3])
            if check_tracking and error > self.tracking_limit:
                raise ValueError(f'{side} commanded TCP is {error:.3f}m from feedback '
                                 f'(limit {self.tracking_limit:.3f}m)')

    def validate_chunk(self, predicted):
        """Check raw predictions BEFORE smoothing can hide an invalid point."""
        predicted = actions(predicted)
        for side, start in zip(ARMS, (0, 8)):
            arm = self.cfg['arms'][side]
            if np.any(predicted[:, start:start+3] < arm['workspace_min']) or np.any(predicted[:, start:start+3] > arm['workspace_max']):
                raise ValueError(f'{side} predicted TCP is outside workspace')
        return predicted

    def _reverse_gate(self, error, arm_index):
        thresholds = np.array([self.limits['reverse_threshold_m']]*3 +
                              [self.limits['reverse_threshold_rad']]*3)
        moving = np.abs(error) >= thresholds
        requested = np.sign(error).astype(np.int8)
        previous = self.direction[arm_index]
        reversing = moving & (previous != 0) & (requested != previous)
        counts = self.reverse_counts[arm_index]
        counts[reversing] += 1
        counts[~reversing] = 0
        blocked = reversing & (counts < self.limits['reverse_confirm_steps'])
        confirmed = moving & ~blocked
        previous[confirmed] = requested[confirmed]
        counts[confirmed] = 0
        filtered = error.copy()
        filtered[blocked] = 0
        self.diagnostics['reversals_blocked'] += int(np.count_nonzero(blocked))
        return filtered

    def _step(self, error, velocity, tau, speed, acceleration):
        alpha = 1. if tau == 0 else -math.expm1(-self.dt/tau)
        desired_velocity = alpha*error/self.dt
        # Exact discrete stopping distance: after sending velocity u, commands
        # can decelerate by a*dt each tick. A continuous sqrt(2*a*d) bound can
        # still cross the goal by a fraction of a step at the final brake.
        distance = np.linalg.norm(error)
        braking_ticks = max(1, math.ceil((math.sqrt(
            1+8*distance/(acceleration*self.dt**2))-1)/2))
        braking_speed = distance/(braking_ticks*self.dt) + acceleration*self.dt*(braking_ticks-1)/2
        requested = _clip_norm(desired_velocity, min(speed, braking_speed))
        self.diagnostics['speed_limited'] += int(np.linalg.norm(desired_velocity-requested) > 1e-9)
        change = requested-velocity
        bounded_change = _clip_norm(change, acceleration*self.dt)
        self.diagnostics['acceleration_limited'] += int(np.linalg.norm(change-bounded_change) > 1e-9)
        return _clip_norm(velocity+bounded_change, speed)

    def apply(self, desired, feedback):
        desired, feedback = vector(desired).copy(), vector(feedback)
        self._validate(desired, feedback, check_tracking=False)
        tracking_limited = False
        # A model goal is not the next command. Bound its lead using current
        # measured TCP, as Teleop does, then let the persistent v/a filter
        # approach it. Keep a tiny margin for float32 boundary rounding.
        radius = self.tracking_limit-min(1e-6, self.tracking_limit*.001)
        for start in (0, 8):
            delta = desired[start:start+3]-feedback[start:start+3]
            distance = np.linalg.norm(delta)
            if distance > radius:
                desired[start:start+3] = feedback[start:start+3]+delta*(radius/distance)
                tracking_limited = True
                self.diagnostics['tracking_limited'] += 1
        if self.last is None:
            self.reset(feedback)
        result = desired.copy()
        velocities = np.zeros_like(self.velocity)
        limits = self.limits
        for arm_index, (side, start) in enumerate(zip(ARMS, (0, 8))):
            arm = self.cfg['arms'][side]
            last, target = self.last[start:start+8], desired[start:start+8]
            error = np.r_[target[:3]-last[:3], _rotation_error(last[3:7], target[3:7])]
            error = self._reverse_gate(error, arm_index)
            velocity = self.velocity[arm_index]
            velocities[arm_index, :3] = self._step(error[:3], velocity[:3],
                limits['position_filter_tau_s'], min(arm['max_speed_m_s'], limits['max_speed_m_s']),
                limits['max_acceleration_m_s2'])
            result[start:start+3] = last[:3]+velocities[arm_index, :3]*self.dt
            velocities[arm_index, 3:6] = self._step(error[3:], velocity[3:6],
                limits['rotation_filter_tau_s'], min(arm['max_rotation_rad_s'], limits['max_rotation_rad_s']),
                limits['max_angular_acceleration_rad_s2'])
            q = (Rotation.from_rotvec(velocities[arm_index, 3:6]*self.dt) * Rotation.from_quat(last[3:7])).as_quat()
            if np.dot(q, last[3:7]) < 0:
                q = -q
            result[start+3:start+7] = q
            width = np.clip(target[7], 0., arm['gripper_max_m'])
            velocities[arm_index, 6:] = self._step(np.array([width-last[7]]), velocity[6:],
                limits['gripper_filter_tau_s'], limits['max_gripper_speed_m_s'],
                limits['max_gripper_acceleration_m_s2'])
            result[start+7] = np.clip(last[7]+velocities[arm_index, 6]*self.dt, 0., arm['gripper_max_m'])
        # Acceleration memory must not carry commands past independent hard limits.
        self._validate(result, feedback)
        self._pending_velocity = velocities
        return result, tracking_limited or not np.allclose(result, desired, atol=1e-7, rtol=0)

    def commit(self, sent):
        if self._pending_velocity is None:
            raise RuntimeError('No filtered command to commit')
        self.last = vector(sent).copy()
        self.velocity = self._pending_velocity
        self._pending_velocity = None
