"""Grip controls motion; B/A control recording. These lifecycles are independent."""
import json
from collections import deque
import logging
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .core import ARMS, History, PoseMapper, Synchronizer
from .recording import Recorder
from .capture_session import CaptureSelection
from .operator_controls import OperatorControls
from .feedback import Feedback
from .robot import RecoverableRobotError, validate_vector

LOG = logging.getLogger(__name__)
RUNTIME_REVISION = "2026-11-09-freeze-filtered-target"


class Controller:
    def __init__(self, cfg, robots, cameras, mock=False, config_path=None):
        self.cfg, self.robots, self.cameras = cfg, robots, cameras
        self.mock, self.config_path = mock, Path(config_path) if config_path else None
        self.states = {s: History() for s in ARMS}
        self.actions = {s: History() for s in ARMS}
        self.mappers = {s: PoseMapper(cfg["arms"][s]) for s in ARMS}
        self.enabled = dict.fromkeys(ARMS, False)
        self.motion_active = dict.fromkeys(ARMS, False)
        self.cartesian_blocked = dict.fromkeys(ARMS, "")
        self.local_limit_latched = dict.fromkeys(ARMS, False)
        self.target_lead_limited = dict.fromkeys(ARMS, False)
        self.joint_guard_status = {s: {"limited": False} for s in ARMS}
        self.resume_since = None
        self.resume_hands = set()
        self.hand_rearm = dict.fromkeys(ARMS, False)
        self.gripper_rearm = dict.fromkeys(ARMS, False)
        self.paused_width = {}
        self.last_commands = {}
        self.enable_since = dict.fromkeys(ARMS, None)
        self.sync = Synchronizer(cameras.histories, self.states, self.actions, cfg["sync"], 1/cfg["record_hz"])
        self.recorder = Recorder(cfg["output"], cfg, mock)
        self.capture = CaptureSelection(self.recorder, cfg["task"])
        self.operator = OperatorControls(self)
        self.record_end_t = None
        self.ready, self.feedback_missing = False, list(ARMS)
        self.mode, self.message = "waiting", "Waiting for both arms and Quest"
        self.packet = None
        self.input_frames = History()
        self.socket_connected = False
        self.button_counts = {s: {"middle": 0, "trigger": 0} for s in ARMS}
        self.input_lock = threading.Lock()
        self.input_events = deque()
        self.received_buttons = {}
        self.last_event = self.event_ack = 0
        self.feedback = Feedback(cfg.get("feedback"))
        self.stop_event = threading.Event()
        self.previous_hands = {}
        self.hold_since, self.hold_latched = {}, set()
        self.connected = False
        self.commands_started = False
        self.recovery_request = False
        self.recovery_hold_targets = {}
        self.recovery_hold_status = {}
        self.recovery_reason = ""
        self.recovery_pose_issues = {}
        self.recovery_hardware_missing = []
        self.teach_exit_pending = {}
        self.teach_exit_error = ""
        self.recovery_after_teach_exit = False
        self.recovery_teaching = set()

    def accept(self, packet, received=None):
        received = time.perf_counter() if received is None else received
        with self.input_lock:
            events = packet.get("events")
            if events is None:  # Also support local callers sending button states.
                changes = [(n, down) for n, down in packet["buttons"].items()
                           if down != self.received_buttons.get(n, False)]
                events = [{"id": self.last_event+i+1, "name": n, "down": down, "age_ms": 0}
                          for i, (n, down) in enumerate(changes)]
            new = [event for event in events if event["id"] > self.last_event]
            if [event["id"] for event in new] != list(range(self.last_event+1, self.last_event+len(new)+1)):
                raise ValueError("Button event sequence gap")
            if len(self.input_events) + len(new) > 64:
                raise ValueError("Button event queue full")
            self.input_events.extend({**event, "t": received-event["age_ms"]/1000} for event in new)
            if new:
                self.last_event = new[-1]["id"]
            self.received_buttons = dict(packet["buttons"])
            previous = self.packet[1] if self.packet else {}
            for side in ARMS:
                old, hand = previous.get(side, {}), packet[side]
                self.button_counts[side]["middle"] += int(hand["grip"] and not old.get("grip", False))
                self.button_counts[side]["trigger"] += int(hand["trigger"] > .7 and old.get("trigger", 0) <= .7)
            self.packet = (received, packet)
            self.input_frames.add(received, None)

    def begin_input(self):
        with self.input_lock:
            self.packet = None
            self.socket_connected = True
            self.input_frames = History()
            self.button_counts = {s: {"middle": 0, "trigger": 0} for s in ARMS}
            self.input_events.clear()
            self.received_buttons.clear()
            self.last_event = self.event_ack = 0

    def disconnect(self):
        with self.input_lock:
            self.socket_connected = False
            self.packet = None
            self.input_events.clear()

    def start(self):
        self.feedback.start()
        self.thread = threading.Thread(target=self._run, daemon=True, name="robot-control")
        self.thread.start()

    def status(self):
        now = time.perf_counter()
        recording = self.recorder.active and self.record_end_t is None
        return {"runtime_revision": RUNTIME_REVISION,
                "mode": self.mode, "ready": self.ready, "recording": recording,
                "capture": self.capture.snapshot(), "replay": self.operator.snapshot(),
                "position_scale": {s: self.cfg["arms"][s]["position_scale"] for s in ARMS},
                "motion_profile": {s: {key: self.cfg["arms"][s].get(key) for key in
                    ("joint_guard_enabled", "pose_smoothing_s", "tremor_smoothing_s", "position_deadband_m",
                     "rotation_deadband_rad", "position_scale", "max_speed_m_s", "max_rotation_rad_s")}
                    for s in ARMS},
                "follow": self.follow_status(now),
                "calibration": {s: self.mappers[s].calibration_status() for s in ARMS},
                "local_limits": dict(self.local_limit_latched),
                "quest": self.input_status(),
                "grippers": self.gripper_status(),
                "event_ack": self.event_ack, "notifications": self.feedback.snapshot(),
                "haptics": self.feedback.haptics,
                "saving": self.recorder.busy and not recording,
                "frames": self.recorder.count, "dropped": self.recorder.dropped,
                "valid_frames": self.recorder.valid_count, "placeholder_frames": self.recorder.placeholder_count,
                "recording_result": self.recorder.result,
                "message": self.message, "camera_errors": dict(self.cameras.errors),
                "camera_timestamps": {n: (h.latest().value.get("timestamp_basis") if h.latest() else None)
                                      for n, h in self.cameras.histories.items()},
                "camera_settings": dict(self.cameras.exposure_settings),
                "camera_warnings": self.camera_warnings(now),
                "camera_diagnostics": {n: dict(values) for n, values in self.cameras.diagnostics.items()},
                "recording_drops": dict(self.recorder.drop_reasons),
                "recovery_hold": dict(self.recovery_hold_status),
                "robot_status": {s: ({k: h.latest().value.get(k) for k in
                    ("arm_status", "err_code", "teaching", "teach_status", "ctrl_mode", "arm_enabled", "teach_exit_supported", "joints")}
                    if h.latest() else {}) for s, h in self.states.items()},
                "recovery": {"reason": self.recovery_reason,
                             "armed": getattr(self, "recovery_y_armed", False),
                             "enabling": getattr(self, "recovery_enabling", False),
                             "exiting_teaching": bool(self.teach_exit_pending), "teach_exit_error": self.teach_exit_error,
                             "pose_issues": dict(self.recovery_pose_issues),
                             "hardware_missing": list(self.recovery_hardware_missing)},
                "fps": {"cameras": {n: h.fps(now) for n, h in self.cameras.histories.items()},
                        "action": {n: h.fps(now) for n, h in self.actions.items()},
                        "state": {n: h.fps(now) for n, h in self.states.items()},
                        "recording": self.recorder.writes.fps(now) if self.recorder.busy else 0.}}

    def follow_status(self, now):
        result = {}
        for side in ARMS:
            state, action = self.states[side].latest(), self.actions[side].latest()
            result[side] = {
                "position_error_mm": (round(float(np.linalg.norm(np.asarray(action.value[:3])-
                    state.value["vector"][:3]))*1000, 1) if state and action else None),
                "command_age_ms": (round((now-action.t)*1000, 1)
                                   if action and self.mode in ("idle", "teleop") else None),
                "mapping_error_mm": (round(float(np.linalg.norm(np.asarray(
                    self.mappers[side].requested_target[:3])-action.value[:3]))*1000, 1)
                    if action and self.mappers[side].anchor is not None else None),
                "target_position": list(action.value[:3]) if action else None,
                "actual_position": list(state.value["vector"][:3]) if state else None,
                "rate_limited": bool(self.mappers[side].rate_limited),
                "lead_limited": bool(self.target_lead_limited[side]),
                "joint_guard": dict(self.joint_guard_status[side]),
            }
        return result

    def gripper_status(self):
        result = {}
        for side in ARMS:
            state, action = self.states[side].latest(), self.actions[side].latest()
            result[side] = {
                "actual_mm": round(state.value["vector"][7]*1000, 1) if state else None,
                "target_mm": round(action.value[7]*1000, 1) if action else None,
                "enabled": state.value.get("gripper_enabled") if state else None,
                "status_code": state.value.get("gripper_status") if state else None,
                "feedback_age_ms": round((time.perf_counter()-state.t)*1000, 1) if state else None,
            }
        return result

    def input_status(self):
        with self.input_lock:
            packet = self.packet
            now = time.perf_counter()
            data = packet[1] if packet else {}
            return {"connected": self.socket_connected,
                    "age_ms": round(max(0, now-packet[0])*1000, 1) if packet else None,
                    "hz": round(self.input_frames.fps(now), 1), "active": data.get("active", False),
                    "reason": data.get("inactive_reason", ""),
                      "buttons": dict(data.get("buttons", {})),
                    **{s: {"tracked": data.get(s, {}).get("tracked", False),
                           "middle": data.get(s, {}).get("grip", False),
                           "trigger": data.get(s, {}).get("trigger", 0),
                           "presses": dict(self.button_counts[s]),
                           "reason": data.get(s, {}).get("tracking_reason", "")} for s in ARMS}}

    def _notify(self, kind, message):
        self.message = message
        self.feedback.emit(kind, message)

    def _fault(self, message):
        self.operator.abort()
        LOG.info("Last commands before fault: %s", json.dumps(self.last_commands))
        LOG.info("Quest at fault: %s", json.dumps(self.input_status()))
        for robot in self.robots.values() if self.commands_started else ():
            try:
                robot.stop()
            except Exception:
                LOG.exception("Electronic stop failed")
        self.mode = "fault"
        self._notify("fault", message)
        self.ready = False
        self.recorder.stop(message, complete=False, end_time=self.record_end_t or time.perf_counter())
        with self.input_lock:
            self.input_events.clear()
            self.event_ack = self.last_event
        LOG.error(message)

    def _read(self, now):
        for side, robot in self.robots.items():
            try:
                result = robot.read()
            except RecoverableRobotError as error:
                if error.kind != "target_limit":
                    raise RecoverableRobotError(f"{side}: {error}", kind=error.kind) from error
                result = robot.read_recovery()
                if result is None or any(not -.05 <= time.perf_counter()-stamp <= self.cfg["feedback_timeout_s"]
                                         for stamp in result[1]["source_timestamps"].values()):
                    raise RuntimeError(f"{side}: stale feedback after target rejection") from error
                if not self.local_limit_latched[side]:
                    state = result[1]
                    robot.hold(list(state["joints"]))
                    self.commands_started = True
                    self.local_limit_latched[side] = True
                    self.mappers[side].release(state["vector"])
                    self.motion_active[side] = self.enabled[side] = False
                    self.enable_since[side] = None
                    self.previous_hands.pop(side, None)
                    self.last_commands[side] = {"kind": "joint_hold", "joints": list(state["joints"]),
                                                "state": list(state["vector"])}
                    self.actions[side].add(time.perf_counter(), list(state["vector"]))
                    LOG.warning("%s target rejected; local hold only; release middle to recalibrate", side)
            except Exception as error:
                raise RuntimeError(f"{side}: {error}") from error
            if result:
                self.states[side].add(*result)
        latest = {side: history.latest() for side, history in self.states.items()}
        self.feedback_missing = [side for side, state in latest.items()
            if state is None or any(now - stamp > self.cfg["feedback_timeout_s"] or stamp > now + .05
                                   for stamp in state.value["source_timestamps"].values())]
        if self.feedback_missing:
            return None
        return {side: sample.value for side, sample in latest.items()}

    def request_recovery(self):
        """Queue an explicit operator request; the control thread validates and executes it."""
        with self.input_lock:
            if self.mode == "recovery":
                self.recovery_request = True

    def _read_recovery_states(self):
        states = {}
        for side, robot in self.robots.items():
            sample = robot.read_recovery()
            if sample is None:
                raise RuntimeError(f"{side}: no feedback for recovery hold")
            _, state = sample
            now = time.perf_counter()
            stamps = state["source_timestamps"]
            if not stamps or any(not -.05 <= now-stamp <= self.cfg["feedback_timeout_s"]
                                 for stamp in stamps.values()):
                raise RuntimeError(f"{side}: stale feedback for recovery hold")
            self.states[side].add(*sample)
            states[side] = state
        self.feedback_missing = []
        return states

    def _enter_recovery(self, reason):
        """Preserve motor power for rejected targets; never reset or disable automatically."""
        self.operator.abort()
        self.recovery_reason = reason
        self.teach_exit_pending = {}
        self.teach_exit_error = ""
        self.recovery_after_teach_exit = False
        self.recovery_teaching = set()
        self.recovery_pose_issues = {}
        self.recovery_hardware_missing = []
        self.ready = False
        self.recorder.stop(reason, complete=False, end_time=self.record_end_t or time.perf_counter())
        self.record_end_t = None
        self.reset_jobs, self.reset_active = [], None
        self.recovery_hold_targets = {}
        self.recovery_hold_status = {}
        self.recovery_started = time.perf_counter()
        self.recovery_settled_since = dict.fromkeys(ARMS, None)
        self.recovery_last_joints = {}
        self.recovery_neutral_since = None
        self.recovery_y_armed = False
        self.recovery_enabling = False
        self.recovery_enable_ok = dict.fromkeys(ARMS, False)
        self.hold_since.clear()
        self.hold_latched.clear()
        self.previous_hands.clear()
        with self.input_lock:
            self.recovery_request = False
            self.input_events.clear()
            self.event_ack = self.last_event
        try:
            states = self._read_recovery_states()
            for side, state in states.items():
                self.recovery_last_joints[side] = (time.perf_counter(), np.asarray(state["joints"]).copy())
                self.motion_active[side] = self.enabled[side] = False
                self.enable_since[side] = None
                self.cartesian_blocked[side] = ""
                self.hand_rearm[side] = self.gripper_rearm[side] = True
                self.mappers[side].release(state["vector"])
                if state.get("teaching", False):
                    self.recovery_teaching.add(side)
                    self.recovery_hold_status[side] = "teaching - no commands"
                    continue
                self.commands_started = True
                self.robots[side].hold(list(state["joints"]))
                self.recovery_hold_targets[side] = list(state["joints"])
                self.recovery_hold_status[side] = "requested - awaiting feedback"
                if state.get("gripper_enabled", False):
                    self.robots[side].send_gripper(state["vector"][7])
                self.actions[side].add(time.perf_counter(), list(state["vector"]))
        except Exception as error:
            self._fault(f"Recovery hold unavailable: {error}")
            return
        self.mode = "recovery"
        self._notify("paused", reason + "; hold requested; teach manually, exit teaching, release controls, then hold Y or press R")
        LOG.warning("Recoverable motion rejection; holding without emergency stop/reset: %s", reason)

    def _recovery_tick(self, now, events):
        with self.input_lock:
            packet = self.packet
            requested, self.recovery_request = self.recovery_request, False
        states = self._read_recovery_states()
        for side, sent in list(self.teach_exit_pending.items()):
            state = states[side]
            if (not state.get("teaching") and state.get("ctrl_mode") == 1
                    and state.get("arm_status", 0) == 0 and not state.get("err_code", 0)
                    and state["source_timestamps"].get("status", 0) > sent):
                del self.teach_exit_pending[side]
            elif time.perf_counter()-sent > 3:
                self.teach_exit_pending.clear()
                self.recovery_after_teach_exit = False
                self.teach_exit_error = "退出示教未收到新的 CAN 正常状态确认，请扶稳机械臂并检查本体状态；未执行 reset 或失能"
                self._notify("rejected", self.teach_exit_error)
                break
        self._reject_events(events, "Recovery; teleoperation and recording are suspended")
        teaching = []
        for side, state in states.items():
            if side in self.recovery_teaching and not state.get("teaching", False):
                # A manually taught pose replaces every pre-teaching hold target.
                # Never pull the arm back to where it was before the operator moved it.
                if state.get("arm_status", 0) == 0 and not state.get("err_code", 0):
                    self.robots[side].hold(list(state["joints"]))
                    self.recovery_hold_targets[side] = list(state["joints"])
                    self.recovery_settled_since[side] = None
                    self.recovery_teaching.discard(side)
                    if all(s.get("arm_enabled") is True and s.get("arm_status", 0) == 0
                           and not s.get("teaching", False) and not s.get("err_code", 0)
                           for s in states.values()):
                        self.recovery_after_teach_exit = True
                        self.teach_exit_deadline = time.perf_counter()+5
            target = self.recovery_hold_targets.get(side)
            checked_at = time.perf_counter()
            last_t, last_joints = self.recovery_last_joints[side]
            joints = np.asarray(state["joints"])
            quiet = np.max(np.abs(joints-last_joints))/max(checked_at-last_t, 1e-6) < .05
            self.recovery_last_joints[side] = checked_at, joints.copy()
            if state.get("teaching", False):
                self.recovery_teaching.add(side)
                self.recovery_hold_targets.pop(side, None)
                teaching.append(side)
                self.recovery_hold_status[side] = "teaching - no commands"
                self.recovery_settled_since[side] = None
            elif (target is not None and state.get("arm_status", 0) == 0
                  and state.get("arm_enabled") is True and state.get("mode_feedback") == 1
                  and quiet and np.max(np.abs(joints-target)) < .01):
                since = self.recovery_settled_since[side]
                self.recovery_settled_since[side] = checked_at if since is None else since
                self.recovery_hold_status[side] = ("joint hold confirmed" if
                    checked_at-self.recovery_settled_since[side] >= .2 else "joint hold settling")
            else:
                self.recovery_settled_since[side] = None
                self.recovery_hold_status[side] = "hold NOT confirmed - support arm"
        if teaching:
            self.recovery_enabling = self.recovery_y_armed = False
            self.recovery_neutral_since = None
            self.hold_since.clear()
            if requested and not self.teach_exit_pending:
                held_input = (packet and 0 <= time.perf_counter()-packet[0] <= self.cfg["input_timeout_s"]
                    and (any(packet[1][s]["grip"] or packet[1][s]["trigger"] > .05 for s in ARMS)
                         or any(v for k, v in packet[1]["buttons"].items() if k != "reset")))
                if held_input:
                    self.message = "请先松开手柄的中指键、食指扳机和面键，再点击恢复遥操退出示教"
                    return
                self.teach_exit_error = ""
                try:
                    for side in teaching:
                        if not states[side].get("teach_exit_supported"):
                            raise ValueError(f"{side}: 当前固件配置不支持软件无断电退出，请使用本体退出示教")
                    for side in teaching:
                        joints = list(states[side]["joints"])
                        self.robots[side].exit_teaching(joints)
                        self.teach_exit_pending[side] = time.perf_counter()
                        self.recovery_hold_targets[side] = joints
                    self.recovery_after_teach_exit = True
                    self.teach_exit_deadline = time.perf_counter()+5
                except ValueError as error:
                    self.recovery_after_teach_exit = False
                    self.teach_exit_error = str(error)
                    self._notify("rejected", str(error))
                    return
            if self.teach_exit_pending:
                self.message = "正在主动退出示教并保持当前关节位置，等待控制器反馈确认；未执行 reset 或失能"
                return
            self.message = "Teaching: " + ", ".join(teaching) + "; no motion commands; exit teaching before recovery"
            return
        if self.teach_exit_pending:
            self.message = "已发送退出示教，等待新的 CAN 正常状态确认；保持暂停"
            return
        self.recovery_pose_issues = {}
        for side in ARMS:
            issue = self.robots[side].cartesian_start_issue(states[side]["joints"])
            if issue:
                self.recovery_pose_issues[side] = issue
        self.recovery_hardware_missing = self._waiting_for(now, [])
        data = packet[1] if packet and 0 <= time.perf_counter()-packet[0] <= self.cfg["input_timeout_s"] else None
        valid_input = data and data["active"] and all(data[s]["tracked"] for s in ARMS)
        hands_neutral = valid_input and all(not data[s]["grip"] and data[s]["trigger"] <= .05 for s in ARMS)
        neutral = hands_neutral and not any(data["buttons"].values())
        if neutral:
            self.recovery_neutral_since = now if self.recovery_neutral_since is None else self.recovery_neutral_since
            if now-self.recovery_neutral_since >= .3:
                self.recovery_y_armed = True
        elif not hands_neutral:
            self.recovery_neutral_since = None
            self.recovery_y_armed = False
        y_only = hands_neutral and not any(value for name, value in data["buttons"].items() if name != "reset")
        if self.recovery_after_teach_exit and not self.teach_exit_pending:
            if not valid_input or time.perf_counter() > self.teach_exit_deadline:
                self.recovery_after_teach_exit = False
            elif self.recovery_y_armed and y_only:
                requested = True
                self.recovery_after_teach_exit = False
        if self.recovery_y_armed and self._held("reset", bool(y_only and data["buttons"]["reset"]), now, 1.5):
            requested = True
        self.message = ("Recovery: check hold confirmation; teach manually if needed, exit teaching; "
                        "release controls for 0.3s, then hold Y 1.5s or press R")
        if self.recovery_enabling and not hands_neutral:
            self.recovery_enabling = False
        if requested and not self.recovery_enabling:
            if not self.recovery_y_armed or not y_only:
                self.message = "Recovery rejected: restore tracking and release middle buttons/triggers first"
                return
            if any(state.get("arm_status", 0) != 0 or state.get("err_code", 0) for state in states.values()):
                self.message = "Recovery rejected: controller still reports a fault; no reset or disable was sent"
                return
            missing = self._waiting_for(now, [])
            issues = []
            for side in ARMS:
                issue = self.robots[side].cartesian_start_issue(states[side]["joints"])
                if issue:
                    issues.append(f"{side}: {issue}")
            if missing or issues:
                self.message = "Recovery rejected: " + "; ".join(missing+issues)
                return
            for side, state in states.items():
                self.robots[side].hold(list(state["joints"]))  # Replace all old targets BEFORE enabling.
                self.recovery_hold_targets[side] = list(state["joints"])
                self.recovery_settled_since[side] = None
            self.recovery_enable_ok = dict.fromkeys(ARMS, False)
            self.recovery_enable_start = time.perf_counter()
            self.recovery_enabling = True
        if not self.recovery_enabling:
            return
        if now-self.recovery_enable_start > 5:
            self.recovery_enabling = False
            self.message = "Recovery enable timeout; remain held; check controller status"
            return
        for side in ARMS:
            if not self.recovery_enable_ok[side]:
                self.recovery_enable_ok[side] = bool(self.robots[side].enable())
        if not all(self.recovery_enable_ok.values()) or any(
                state.get("arm_enabled") is not True or state.get("arm_status", 0) != 0
                or state["source_timestamps"].get("status", 0) <= self.recovery_enable_start
                for state in states.values()):
            self.message = "Recovery: waiting for fresh enabled NORMAL feedback; no teleoperation targets"
            return
        for side, state in states.items():
            self.mappers[side].release(state["vector"])
            self.paused_width[side] = state["vector"][7]
            self.motion_active[side] = self.enabled[side] = False
            self.hand_rearm[side] = self.gripper_rearm[side] = True
        self.previous_hands.clear()
        self.resume_hands = set(ARMS)
        self.resume_since = None
        self.recovery_enabling = False
        self.connected = True
        self.mode = "paused"
        self._notify("paused", "Recovery complete; release all controls, then press middle again to take control")

    def _pause_input(self, states, reason, required_hands=()):
        self.resume_since = None
        self.ready = False
        if self.mode != "paused":
            self.resume_hands = set(required_hands)
            self.recorder.stop(reason, complete=False, end_time=self.record_end_t or time.perf_counter())
            self.record_end_t = None
            for side, robot in self.robots.items():
                # Fresh robot feedback is required by the caller. Failure falls back to _fault.
                if self.enabled[side] or self.enable_since[side] is not None or self.mode == "resetting":
                    robot.hold(states[side]["joints"])
                self.paused_width[side] = states[side]["vector"][7]
                if self.commands_started:
                    robot.send_gripper(self.paused_width[side])
                    self.actions[side].add(time.perf_counter(), list(states[side]["vector"]))
                self.mappers[side].release(states[side]["vector"])
                self.motion_active[side] = self.enabled[side] = False
                self.enable_since[side] = None
                self.cartesian_blocked[side] = ""
                self.gripper_rearm[side] = True
                self.hand_rearm[side] = True
            self.reset_jobs, self.reset_active = [], None
            self.previous_hands.clear()
            self.hold_since.clear()
            self.hold_latched.clear()
            self.mode = "paused"
            LOG.warning("Input paused; holding robot positions: %s", reason)
            self._notify("paused", reason)
        self.message = reason + "; restore tracking, then release all buttons/triggers"

    def _waiting_for(self, now, input_missing):
        # RGB quality does not govern motor control or stop an episode. The
        # synchronizer marks missing observations and preserves their timeline.
        missing = [f"{side} feedback" for side in self.feedback_missing]
        missing.extend(input_missing)
        return missing

    def camera_warnings(self, now):
        warnings = {}
        for name, history in self.cameras.histories.items():
            frame = history.latest()
            error = self.cameras.errors.get(name)
            if error:
                warnings[name] = str(error)
            elif frame is None:
                warnings[name] = "相机尚无画面，正在等待或预热"
            elif not -.05 <= now-frame.value.get("receive_s", frame.t) <= self.cfg["camera_timeout_s"]:
                warnings[name] = "画面暂时未更新，等待自动恢复"
            elif frame.value.get("timestamp_basis") == "host_receive":
                warnings[name] = "SDK 时间戳未同步，缺失同步样本将明确标记"
            elif history.fps(now) < self.cfg["cameras"][name]["fps"]*.8:
                warnings[name] = f"实际 {history.fps(now):.1f} FPS，低于配置 {self.cfg['cameras'][name]['fps']} FPS"
        return warnings

    def _held(self, name, down, now, duration):
        if not down:
            self.hold_since.pop(name, None)
            self.hold_latched.discard(name)
            return False
        self.hold_since.setdefault(name, now)
        if now - self.hold_since[name] < duration or name in self.hold_latched:
            return False
        self.hold_latched.add(name)
        return True

    def _save_home(self, states):
        if self.recorder.busy or self.mode != "idle":
            self._notify("rejected", "Release both middle buttons and finish saving before saving home")
            return
        if self.mock or self.config_path is None:
            self._notify("rejected", "Home is saved only in real robot mode with a config file")
            return
        for side in ARMS:
            issue = self.robots[side].cartesian_start_issue(states[side]["joints"])
            if issue:
                self._notify("rejected", f"Home not saved: {side}: {issue}")
                return
        saved = json.loads(self.config_path.read_text(encoding="utf-8"))
        for side in ARMS:
            path = saved["arms"][side]["home_waypoints"]
            joints = list(states[side]["joints"])
            saved["arms"][side]["home_waypoints"] = [*path[:-1], joints] if path else [joints]
        temp = self.config_path.with_suffix(".tmp")
        temp.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.config_path)
        for side in ARMS:
            self.cfg["arms"][side]["home_waypoints"] = saved["arms"][side]["home_waypoints"]
        self._notify("home_saved", "Current joint poses saved as home")

    def _begin_reset(self, now):
        if self.recorder.busy:
            self._notify("rejected", "Press A to end recording; wait for saving before resetting")
            return
        if any(not self.cfg["arms"][s]["home_waypoints"] for s in ARMS):
            self._notify("rejected", "Save home with X first, or configure home_waypoints")
            return
        for side in ARMS:
            issue = self.robots[side].cartesian_start_issue(self.cfg["arms"][side]["home_waypoints"][-1])
            if issue:
                self._notify("rejected", f"Home needs a working pose: {side}: {issue}")
                return
        self.mode = "resetting"
        self._notify("home_started", "Going home; release Y and press Y again to cancel")
        self.reset_jobs = [(s, joints) for s in ARMS for joints in self.cfg["arms"][s]["home_waypoints"]]
        self.reset_active = {}
        self.reset_prepared = False
        self.reset_start = now

    def _cancel_reset(self, states):
        # Replace joint trajectories with measured joint targets; keep motors enabled.
        for side in ARMS:
            self.robots[side].hold(states[side]["joints"])
            self.mappers[side].release(states[side]["vector"])
            self.motion_active[side] = False
            self.enabled[side] = False  # Restore normal speed on the next grip.
        self.reset_jobs.clear()
        self.reset_active = None
        self.previous_hands.clear()
        self.hold_latched.add("reset")  # Holding the cancel press must not start another reset.
        self.mode = "idle"
        self._notify("home_cancelled", "Home cancelled")

    def _reset_tick(self, states, now):
        self.commands_started = True
        if not self.reset_prepared:
            for side in ARMS:
                self.robots[side].hold(states[side]["joints"])
            self.reset_prepared = True
        # Wait for enable acknowledgements without blocking input/feedback watchdogs.
        for side in ARMS:
            if not self.enabled[side]:
                self.enabled[side] = bool(self.robots[side].enable())
        if not all(self.enabled.values()):
            if now - self.reset_start > 5:
                raise TimeoutError("Home enable timeout")
            return
        # Both arms start in this tick and advance their own waypoints independently.
        for side in ARMS:
            active = self.reset_active.get(side)
            if active is not None:
                joints, sent, settled = active
                if now - sent > self.cfg["reset_timeout_s"]:
                    raise TimeoutError(f"Home waypoint timeout: {side}")
                reached = np.max(np.abs(np.array(states[side]["joints"]) - joints)) < .04
                settled = (now if settled is None else settled) if reached else None
                self.reset_active[side] = joints, sent, settled
                if settled is None or now - settled < .3:
                    continue
                del self.reset_active[side]
            job = next((i for i, (arm, _) in enumerate(self.reset_jobs) if arm == side), None)
            if job is not None:
                _, joints = self.reset_jobs.pop(job)
                self.robots[side].move_home(joints)
                self.reset_active[side] = joints, now, None
        if not self.reset_jobs and not self.reset_active:
            self.mode = "idle"
            self._notify("home_done", "Home complete")
            for side in ARMS:
                self.mappers[side].release(states[side]["vector"])
                self.motion_active[side] = False
                self.enabled[side] = False  # Restore normal speed on next grip press.
            return

    def _finish_record(self):
        if not self.recorder.busy and self.record_end_t is not None:
            self.record_end_t = None
            if self.recorder.result == "complete":
                self._notify("saved", f"Saved {self.recorder.path}")
            else:
                self._notify("rejected", f"Episode incomplete: {self.recorder.dropped} missing grid ticks; "
                             f"{dict(self.recorder.drop_reasons)}; see manifest.json")

    def _record_tick(self, now):
        if self.recorder.error:
            raise RuntimeError(self.recorder.error)
        self._finish_record()
        if not self.recorder.active:
            return
        limit = now - self.cfg["sync"]["delay_ms"] / 1000
        if self.record_end_t is not None:
            limit = min(limit, self.record_end_t)
        while self.next_record_t <= limit:
            if self.record_end_t is not None and self.next_record_t >= self.record_end_t:
                break
            sample = self.sync.sample(self.next_record_t)
            if sample:
                self.recorder.add(sample, self.grid_index)
            else:
                self.recorder.drop(self.sync.rejection or "synchronization_missing")
            self.grid_index += 1
            self.next_record_t = self.recorder.start_time + self.grid_index / self.cfg["record_hz"]
        if self.record_end_t is not None and self.next_record_t >= self.record_end_t:
            self.recorder.stop(end_time=self.record_end_t)

    def _reject_events(self, events, message):
        for event in events:
            if event["down"]:
                self._notify("rejected", message)
            self.event_ack = event["id"]

    def _button_event(self, event, states):
        name, down = event["name"], event["down"]
        if name in ("reset", "set_home"):
            if down:
                self.hold_since[name] = event["t"]
            else:
                self.hold_since.pop(name, None)
                self.hold_latched.discard(name)
        if not down:
            return
        if self.mode == "resetting":
            if name == "reset":
                self._cancel_reset(states)
            else:
                self._notify("rejected", "Going home; press Y again to cancel first")
            return
        if name == "record_start":
            self.cfg["teleop_coordinate_calibration"] = {
                side: mapper.calibration_status() for side, mapper in self.mappers.items()}
            if self.capture.error:
                self._notify("rejected", self.capture.error)
                return
            unmapped = [n for n, h in self.cameras.histories.items() if h.latest()
                        and h.latest().value.get("timestamp_basis") == "host_receive"]
            if self.cfg["sync"].get("require_sdk_timestamps", True) and unmapped:
                LOG.warning("Recording with unsynchronized RGB; gaps will be marked: %s", unmapped)
            if not all(self.input_status()[s]["tracked"] for s in ARMS):
                self._notify("rejected", "Recording needs both controllers tracked")
                return
            if self.recorder.busy:
                self._notify("rejected", "Already recording or saving; wait before starting another episode")
                return
            try:
                self.capture.start(event["t"])
            except (ValueError, OSError) as error:
                self.capture.error = str(error)
                self._notify("rejected", str(error))
                return
            self.record_end_t = None
            self.next_record_t, self.grid_index = event["t"], 0
            self.sync.last_frames = {}
            self._notify("record_started", "Recording; A: stop and save")
        elif name == "record_stop":
            if self.recorder.active and self.record_end_t is None:
                self.record_end_t = max(self.recorder.start_time, event["t"])
                self._notify("saving", "Saving; teleoperation remains available")
            else:
                self._notify("rejected", "Already saving" if self.recorder.busy else "Not recording")

    def tick(self, now, dt):
        self.capture.refresh()
        self.operator.apply_requests(now)
        with self.input_lock:
            events = list(self.input_events)
            self.input_events.clear()
        if self.mode == "replaying":
            if any(event["name"] == "reset" and event["down"] for event in events):
                self.operator.cancel_requested = True
            self._reject_events(events, "Replay running; teleoperation and recording are suspended")
            self.operator.tick(now)
            return
        if self.mode == "fault":
            self._reject_events(events, "Fault; restart the program before operating")
            return
        if self.mode == "recovery":
            self._recovery_tick(now, events)
            return
        if self.recorder.error:
            raise RuntimeError(self.recorder.error)
        try:
            states = self._read(now)
        except RecoverableRobotError as error:
            self._enter_recovery(str(error))
            return
        with self.input_lock:
            packet = self.packet
            input_age = time.perf_counter() - packet[0] if packet else float("inf")
        data = packet[1] if packet and 0 <= input_age <= self.cfg["input_timeout_s"] else None
        required_hands = set()
        if self.mode == "paused":
            required_hands = self.resume_hands
        elif self.mode == "resetting" or self.recorder.active:
            required_hands = set(ARMS)
        else:
            for side in ARMS:
                previous = self.previous_hands.get(side, {})
                action = self.actions[side].latest()
                gripper_moving = states is not None and action is not None and abs(
                    action.value[7]-states[side]["vector"][7]) > .002
                if previous.get("tracked") and (previous.get("grip") or
                        previous.get("trigger", 0) > .05 or gripper_moving):
                    required_hands.add(side)
        input_missing = []
        if packet is None:
            input_missing.append("Quest no XR frames" if self.socket_connected else "Quest disconnected")
        elif data is None:
            input_missing.append(f"Quest input timeout ({input_age*1000:.0f} ms)")
        else:
            if not data["active"]:
                input_missing.append("Quest " + (data.get("inactive_reason") or "session inactive"))
            for side in ARMS:
                if side in required_hands and not data[side]["tracked"]:
                    input_missing.append(f"Quest {side}: " + (data[side].get("tracking_reason") or "tracking lost"))
        hardware_missing = self._waiting_for(now, [])
        missing = hardware_missing + input_missing
        self.ready = not missing
        if missing:
            if self.connected:
                if hardware_missing:
                    self._fault("Lost: " + "; ".join(missing))
                else:
                    self._pause_input(states, "Lost: " + "; ".join(input_missing), required_hands)
                    self._reject_events(events, self.message)
            else:
                self.mode, self.message = "waiting", "Waiting: " + "; ".join(missing)
                # A button held during startup must not become a queued command.
                self._reject_events(events, self.message)
            return
        if self.mode == "paused":
            self.ready = False
            neutral = (not any(data["buttons"].values()) and
                       all(not data[s]["grip"] and data[s]["trigger"] <= .05 for s in ARMS))
            self.resume_since = (self.resume_since if self.resume_since is not None else now) if neutral else None
            self.message = "Tracking restored; release all buttons/triggers for 0.3s, then press again"
            self._reject_events(events, self.message)
            if self.resume_since is not None and now-self.resume_since >= .3:
                self.mode, self.ready = "idle", True
                self.message = "Ready: press middle to take control; squeeze trigger to resume gripper"
            return
        if not self.connected:
            self.message = "Ready | Middle button: arm | Index trigger: gripper | B: record | A: save"
        self.connected = True
        buttons = data["buttons"]
        released = not any(data[s]["grip"] and data[s]["tracked"] for s in ARMS)
        self._finish_record()
        was_resetting = self.mode == "resetting"
        if self.mode != "resetting":
            self.mode = "idle" if released else "teleop"
        for event in events:
            if now - event["t"] > self.cfg["input_timeout_s"]:
                self._fault("Button event timed out")
                return
            self._button_event(event, states)
            self.event_ack = event["id"]
        if was_resetting and self.mode != "resetting":
            return
        if self.mode == "resetting":
            self._reset_tick(states, now)
            return
        self.operator.result = ""
        self.commands_started = True
        for side in ARMS:
            hand, previous = data[side], self.previous_hands.get(side)
            if self.local_limit_latched[side]:
                if states[side].get("arm_status", 0) == 0 and not hand["grip"]:
                    self.local_limit_latched[side] = False
                    self.previous_hands.pop(side, None)
                    previous = None
                    self.mappers[side].release(states[side]["vector"])
                else:
                    width = (1-hand["trigger"])*self.cfg["arms"][side]["gripper_max_m"]
                    self.robots[side].send_gripper(width)
                    self.actions[side].add(time.perf_counter(), [*states[side]["vector"][:7], width])
                    continue
            if not hand["tracked"]:
                self.hand_rearm[side] = True
                self.previous_hands.pop(side, None)
                continue  # An idle missing controller must never block the other hand or X/A/B.
            if self.hand_rearm[side]:
                if hand["grip"] or hand["trigger"] > .05:
                    self.message = f"{side}: release middle/trigger once before taking control"
                    continue
                self.hand_rearm[side] = False
            if previous and hand["grip"] and previous["grip"]:
                if np.linalg.norm(np.array(hand["position"]) - previous["position"]) > .15:
                    raise RuntimeError(f"{side}: controller tracking jumped")
            self.previous_hands[side] = {**hand, "position": list(hand["position"])}
            effective = dict(hand)
            pressed = hand["grip"] and not (previous and previous["grip"])
            released_hand = not hand["grip"] and previous and previous["grip"]
            if pressed:
                self.cartesian_blocked[side] = self.robots[side].cartesian_start_issue(states[side]["joints"])
                if not self.cartesian_blocked[side]:
                    try:
                        validate_vector(states[side]["vector"], self.cfg["arms"][side])
                    except ValueError:
                        self.cartesian_blocked[side] = "TCP outside configured workspace; adjust working pose before taking control"
                if self.cartesian_blocked[side]:
                    self._notify("rejected", f"{side}: {self.cartesian_blocked[side]}")
                else:
                    self.message = f"{side}: hold; move controller to follow"
            if pressed and not self.cartesian_blocked[side]:
                # Replace any old trajectory BEFORE enabling. No Cartesian IK on the clutch edge.
                joints = list(states[side]["joints"])
                self.last_commands[side] = {"kind": "joint_hold", "joints": joints,
                                            "state": list(states[side]["vector"])}
                try:
                    self.robots[side].hold(joints)
                except Exception as error:
                    raise RuntimeError(f"{side}: {error}") from error
                self.motion_active[side] = False
                self.mappers[side].release(states[side]["vector"])
            if released_hand:
                # Freeze exactly the last dispatched filtered pose. Neither the
                # release-frame hand pose nor lagging feedback may replace it.
                # Clearing the anchor discards the filter's remaining catch-up.
                command = self.last_commands.get(side, {})
                held = (command["target"] if command.get("kind") == "cartesian" else
                        command.get("state", states[side]["vector"]))
                self.mappers[side].release(held)
                self.motion_active[side] = False
            if hand["grip"] and not self.cartesian_blocked[side] and not self.enabled[side]:
                self.enable_since[side] = self.enable_since[side] or now
                self.enabled[side] = bool(self.robots[side].enable())
                if not self.enabled[side]:
                    effective["grip"] = False
                    if now - self.enable_since[side] > 5:
                        raise TimeoutError(f"{side}: enable timeout")
            if not hand["grip"]:
                self.enable_since[side] = None
                self.motion_active[side] = False
                self.cartesian_blocked[side] = ""
                self.joint_guard_status[side] = {"limited": False}
            if self.cartesian_blocked[side]:
                effective["grip"] = False
            if self.gripper_rearm[side]:
                if hand["trigger"] > .05:
                    self.gripper_rearm[side] = False
                else:
                    effective["trigger"] = 1-self.paused_width[side]/self.cfg["arms"][side]["gripper_max_m"]
            calibration_id = self.mappers[side].calibration_id
            target = self.mappers[side].update(effective, states[side]["vector"], min(dt, .05))
            if self.mappers[side].calibration_id != calibration_id:
                LOG.info("%s controller/TCP calibrated in place: %s", side,
                         json.dumps(self.mappers[side].calibration_status()))
            self.target_lead_limited[side] = False
            if effective["grip"]:
                delta = np.asarray(target[:3])-states[side]["vector"][:3]
                distance = np.linalg.norm(delta)
                lead = self.cfg["max_position_tracking_error_m"]
                if distance > lead:
                    target[:3] = np.asarray(states[side]["vector"][:3])+delta*lead/distance
                    self.target_lead_limited[side] = True
                self.mappers[side].target = list(target)
            anchor = self.mappers[side].anchor
            if effective["grip"] and anchor is not None:
                origin_p, origin_r, _ = anchor
                # Require deliberate motion before switching from joint hold to Cartesian IK.
                hand_rotation = origin_r.inv() * Rotation.from_quat(hand["quaternion"])
                arm_cfg = self.cfg["arms"][side]
                moved = (np.linalg.norm(np.array(hand["position"])-origin_p)*arm_cfg["position_scale"]
                         > max(1e-6, arm_cfg.get("position_deadband_m", .001))
                         or hand_rotation.magnitude() > max(1e-6, arm_cfg.get("rotation_deadband_rad", np.deg2rad(.5))))
                self.motion_active[side] = self.motion_active[side] or moved
            validate_vector(target, self.cfg["arms"][side],
                            check_workspace=bool(effective["grip"] and self.motion_active[side]))
            if effective["grip"] and self.motion_active[side]:
                target, self.joint_guard_status[side] = self.robots[side].constrain_target(
                    target, states[side], min(dt, .05))
                target[:3] = np.clip(target[:3], self.cfg["arms"][side]["workspace_min"],
                                     self.cfg["arms"][side]["workspace_max"])
                self.mappers[side].target = list(target)
                validate_vector(target, self.cfg["arms"][side])
                # Current joint proximity does not predict whether the next TCP
                # step moves towards or away from a limit. Firmware rejects
                # actual invalid targets via RecoverableRobotError in _read().
                self.robots[side].send(target)
                self.last_commands[side] = {"kind": "cartesian", "target": list(target),
                    "state": list(states[side]["vector"]), "joints": list(states[side]["joints"])}
            else:
                # Gripper remains independent of Grip; no move_p while idle or awaiting enable.
                if anchor is not None:
                    target[:7] = anchor[2][:7]
                    self.mappers[side].target[:7] = target[:7]
                self.robots[side].send_gripper(target[7])
            self.actions[side].add(time.perf_counter(), target)
        self._record_tick(now)
        if self._held("set_home", buttons["set_home"] and released, now, 2.):
            self._save_home(states)
        if self._held("reset", buttons["reset"] and released, now, 1.5):
            self._begin_reset(now)

    def _run(self):
        last = time.perf_counter()
        deadline = last
        while not self.stop_event.is_set():
            now = time.perf_counter()
            try:
                self.tick(now, now-last)
            except Exception as error:
                LOG.info("Control fault details", exc_info=True)
                self._fault(str(error))
            last = now
            deadline += 1 / self.cfg["control_hz"]
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                deadline = time.perf_counter()  # Never burst old motion commands to catch up.
            else:
                time.sleep(remaining)  # Python's high-resolution Windows timer, unlike Event.wait.

    def close(self):
        self.stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=8)
        # Cancel outstanding trajectories without cutting motor power on a normal exit.
        # Fault handling already stopped the robots; never replace its emergency stop.
        for robot in self.robots.values():
            try:
                if self.commands_started and self.mode not in ("fault", "recovery"):
                    if hasattr(self, "thread") and self.thread.is_alive():
                        raise RuntimeError("Control thread did not stop before shutdown hold")
                    sample = robot.read()
                    if sample is None:
                        raise RuntimeError("No robot feedback for shutdown hold")
                    _, state = sample
                    now = time.perf_counter()
                    stamps = state["source_timestamps"]
                    if not stamps or any(not -.05 <= now-stamp <= self.cfg["feedback_timeout_s"]
                                         for stamp in stamps.values()):
                        raise RuntimeError("Stale robot feedback for shutdown hold")
                    robot.hold(list(state["joints"]))
                    # The SDK gripper command enables its motor; preserve a disabled gripper.
                    if state.get("gripper_enabled", False):
                        robot.send_gripper(state["vector"][7])
            except Exception:
                LOG.exception("Shutdown hold failed; requesting emergency stop")
                try:
                    robot.stop()
                except Exception:
                    LOG.exception("Shutdown emergency stop failed")
            try:
                robot.close()
            except Exception:
                LOG.exception("Robot disconnect failed")
        self.recorder.stop("shutdown", complete=False, end_time=self.record_end_t or time.perf_counter())
        if not self.recorder.wait(timeout=5):
            LOG.error("Episode writer still busy at shutdown; do not use an unfinished episode")
        self.feedback.close()
