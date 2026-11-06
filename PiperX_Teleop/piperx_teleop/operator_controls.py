"""GUI requests run on the control thread; CSV parsing runs on a separate worker."""
import copy
from queue import Empty, SimpleQueue
from threading import Thread
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .core import ARMS
from .replay import ReplayTrajectory, interpolate_action
from .robot import RecoverableRobotError


class OperatorControls:
    def __init__(self, controller):
        self.control = controller
        self.requests = SimpleQueue()
        self.trajectory = None
        self.loading = False
        self.error = ""
        self.job = None
        self.last_targets = {}
        self.progress = 0.
        self.cancel_requested = False
        self.result = ""

    def request(self, operation, **arguments):
        self.requests.put((operation, arguments))

    def snapshot(self):
        trajectory = self.trajectory
        return {"loaded": trajectory is not None, "loading": self.loading,
                "path": str(trajectory.path) if trajectory else "",
                "duration_s": trajectory.duration if trajectory else 0.,
                "frames": len(trajectory.times) if trajectory else 0,
                "phase": self.job["phase"] if self.job else "idle",
                "active": self.job is not None, "progress": self.progress, "error": self.error, "result": self.result}

    def abort(self):
        self.job = None
        self.cancel_requested = False

    def apply_requests(self, now):
        control = self.control
        while True:
            try:
                operation, args = self.requests.get_nowait()
            except Empty:
                return
            try:
                if operation in ("capture_settings", "suggest_index"):
                    if self.job is not None:
                        raise ValueError("回放期间不能修改采集设置")
                    if operation == "capture_settings":
                        control.capture.apply(**args)
                    else:
                        control.capture.suggest(**args)
                elif operation == "record_start":
                    if self.job is not None or not control.ready or control.mode not in ("idle", "teleop"):
                        raise ValueError("当前未就绪，不能开始采集")
                    packet = control.packet
                    if not packet or time.perf_counter()-packet[0] > control.cfg["input_timeout_s"]:
                        raise ValueError("Quest 输入已超时，不能开始采集")
                    if "selection" in args:
                        try:
                            control.capture.apply(**args["selection"])
                        except (ValueError, OSError) as error:
                            control.capture.error = str(error)
                            raise
                    control._button_event({"name": operation, "down": True, "t": now}, None)
                elif operation == "record_stop":
                    control._button_event({"name": operation, "down": True, "t": now}, None)
                elif operation == "load_replay":
                    if self.loading or self.job is not None or control.recorder.busy:
                        raise ValueError("请先完成采集保存、等待加载结束或停止回放")
                    self.loading, self.error = True, ""
                    cfg = copy.deepcopy(control.cfg)
                    def load(path=args["path"], cfg=cfg):
                        try:
                            result = ReplayTrajectory.load(path, cfg)
                            self.request("replay_loaded", trajectory=result, error="")
                        except Exception as error:
                            self.request("replay_loaded", trajectory=None, error=str(error))
                    Thread(target=load, daemon=True, name="replay-csv-loader").start()
                elif operation == "replay_loaded":
                    self.loading = False
                    self.trajectory, self.error = args["trajectory"], args["error"]
                    self.progress = 0.
                    self.result = ""
                    if self.error:
                        control._notify("rejected", self.error)
                elif operation == "replay_start":
                    self.start(now, float(args.get("speed", 1.)))
                elif operation == "replay_stop":
                    self.cancel_requested = True
            except (ValueError, OSError) as error:
                if operation in ("capture_settings", "suggest_index"):
                    control.capture.error = str(error)
                elif operation not in ("record_start", "record_stop"):
                    self.error = str(error)
                control._notify("rejected", str(error))
            except RecoverableRobotError as error:
                control._enter_recovery(str(error))

    def start(self, now, speed):
        control, trajectory = self.control, self.trajectory
        if trajectory is None or self.loading:
            raise ValueError("请先加载并校验 action.csv")
        if self.job is not None or control.recorder.busy or control.mode in ("fault", "recovery", "resetting"):
            raise ValueError("请先停止采集/复位并完成恢复，再启动回放")
        trajectory.check_speed(control.cfg, speed)
        packet = control.packet
        if packet and 0 <= time.perf_counter()-packet[0] <= control.cfg["input_timeout_s"]:
            data = packet[1]
            if any(data["buttons"].values()) or any(data[s]["grip"] or data[s]["trigger"] > .05 for s in ARMS):
                raise ValueError("启动回放前请松开全部手柄按键和扳机")
        states = control._read(now)
        if states is None:
            raise ValueError("双臂反馈未就绪，不能启动回放")
        for side, offset in zip(ARMS, (0, 8)):
            state = states[side]
            issue = control.robots[side].cartesian_start_issue(state["joints"])
            if issue:
                raise ValueError(f"{side}: {issue}")
            arm = control.cfg["arms"][side]
            angle = (Rotation.from_quat(trajectory.actions[0, offset+3:offset+7])*
                     Rotation.from_quat(state["vector"][3:7]).inv()).magnitude()
            if arm["max_rotation_rad_s"] == 0 and angle > 1e-6:
                raise ValueError(f"{side}: 起点需要旋转，但旋转速度上限为 0")
            if (np.any(np.asarray(state["vector"][:3]) < arm["workspace_min"])
                    or np.any(np.asarray(state["vector"][:3]) > arm["workspace_max"])):
                raise ValueError(f"{side}: 当前位姿在工作空间之外，请先调整到有效工作空间")
        control.commands_started = True
        self.last_targets = {}
        for side, state in states.items():
            control.robots[side].hold(list(state["joints"]))
            self.last_targets[side] = list(state["vector"])
        self.job = {"trajectory": trajectory, "speed": speed, "prepared": time.perf_counter(),
                    "start": None, "phase": "enabling_approach", "enabled": dict.fromkeys(ARMS, False)}
        self.progress = 0.
        self.result = ""
        self.cancel_requested = False
        self.error = ""
        control.ready = False
        control.mode = "replaying"
        control.previous_hands.clear()
        control._notify("home_started", "Replay: waiting for enabled feedback; Y / desktop Stop cancels")

    def finish(self, cancelled):
        control = self.control
        for side, target in self.last_targets.items():
            control.mappers[side].release(target)
            control.paused_width[side] = float(np.clip(target[7], 0, control.cfg["arms"][side]["gripper_max_m"]))
            control.motion_active[side] = control.enabled[side] = False
            control.hand_rearm[side] = control.gripper_rearm[side] = True
        self.abort()
        self.result = "stopped" if cancelled else "complete"
        control.resume_hands = set(ARMS)
        control.resume_since = None
        control.previous_hands.clear()
        control.hold_since.clear()
        control.hold_latched.clear()
        control.connected = True
        control.mode, control.ready = "paused", False
        control._notify("home_cancelled" if cancelled else "home_done",
                        "Replay stopped; last target retained" if cancelled else
                        "Replay complete; final target retained; release controls before teleoperation")

    def dispatch(self, action, states, pose_only=False):
        control = self.control
        for side, offset in zip(ARMS, (0, 8)):
            target = action[offset:offset+8]
            issue = control.robots[side].cartesian_start_issue(states[side]["joints"])
            if issue or np.linalg.norm(target[:3]-states[side]["vector"][:3]) > control.cfg["max_position_tracking_error_m"]:
                control._enter_recovery(f"{side}: replay cannot follow target; {issue}")
                return False
        for side, offset in zip(ARMS, (0, 8)):
            target = action[offset:offset+8].tolist()
            control.last_commands[side] = {"kind": "cartesian", "target": list(target),
                "state": list(states[side]["vector"]), "joints": list(states[side]["joints"])}
            if pose_only:
                control.robots[side].send_pose(target)
            else:
                control.robots[side].send(target)
            self.last_targets[side] = target
            control.actions[side].add(time.perf_counter(), list(target))
        return True

    def tick(self, now):
        control, job = self.control, self.job
        if job is None:
            return
        if self.cancel_requested:
            self.finish(cancelled=True)
            return
        try:
            states = control._read(now)
        except RecoverableRobotError as error:
            control._enter_recovery(str(error))
            return
        if states is None:
            raise RuntimeError("Replay lost fresh robot feedback")
        if job["phase"].startswith("enabling"):
            if time.perf_counter()-job["prepared"] > 5:
                control._enter_recovery("Replay enable timeout")
                return
            for side in ARMS:
                if not job["enabled"][side]:
                    job["enabled"][side] = bool(control.robots[side].enable())
            if not all(job["enabled"].values()) or any(
                    state.get("arm_enabled") is not True
                    or state["source_timestamps"].get("status", 0) <= job["prepared"]
                    for state in states.values()):
                return
            if job["phase"] == "enabling_approach":
                initial = np.concatenate([states[side]["vector"] for side in ARMS])
                target = job["trajectory"].actions[0].copy()
                duration = 0.
                for side, offset in zip(ARMS, (0, 8)):
                    control.robots[side].set_motion_speed(10)
                    arm = control.cfg["arms"][side]
                    distance = np.linalg.norm(target[offset:offset+3]-initial[offset:offset+3])
                    angle = (Rotation.from_quat(target[offset+3:offset+7])*
                             Rotation.from_quat(initial[offset+3:offset+7]).inv()).magnitude()
                    duration = max(duration, distance/min(.05, arm["max_speed_m_s"]),
                                   angle/max(1e-9, min(.35, arm["max_rotation_rad_s"])))
                    target[offset+7] = initial[offset+7]  # Preserve gripper during positioning.
                job.update(phase="approaching", approach_start=time.perf_counter(), approach_duration=duration,
                           initial=initial, approach_target=target, settled=None)
            else:
                job["start"] = time.perf_counter()
                job["phase"] = "playing"
        if job["phase"] == "approaching":
            elapsed = time.perf_counter()-job["approach_start"]
            if elapsed > job["approach_duration"]+10:
                control._enter_recovery("Replay positioning timeout")
                return
            alpha = min(1., elapsed/max(job["approach_duration"], 1e-6))
            action = interpolate_action(job["initial"], job["approach_target"], alpha)
            if not self.dispatch(action, states, pose_only=True):
                return
            reached = alpha >= 1.
            for side, offset in zip(ARMS, (0, 8)):
                target, state = job["approach_target"][offset:offset+8], states[side]["vector"]
                reached &= (np.linalg.norm(target[:3]-state[:3]) <= .003 and
                            (Rotation.from_quat(target[3:7])*Rotation.from_quat(state[3:7]).inv()).magnitude()
                            <= np.deg2rad(2))
            job["settled"] = (now if job["settled"] is None else job["settled"]) if reached else None
            control.message = f"Positioning both arms to replay start: {alpha*100:.0f}% | 0.05 m/s | Stop cancels"
            if job["settled"] is not None and now-job["settled"] >= .3:
                job.update(phase="enabling_replay", enabled=dict.fromkeys(ARMS, False), prepared=time.perf_counter())
            return
        elapsed = max(0., time.perf_counter()-job["start"])*job["speed"]
        action = job["trajectory"].sample(elapsed)
        if not self.dispatch(action, states):
            return
        self.progress = min(1., elapsed/job["trajectory"].duration)
        control.message = f"Replay {self.progress*100:.1f}% | {job['speed']:.2f}x | Y / desktop Stop cancels"
        if elapsed >= job["trajectory"].duration:
            self.finish(cancelled=False)
