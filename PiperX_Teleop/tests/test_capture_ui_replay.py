"""Capture labels and replay preflight, positioning, timing, cancellation and isolation."""
import csv
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piperx_teleop.capture_session import CaptureSelection
from piperx_teleop.core import CAMERAS, VECTOR_NAMES
from piperx_teleop.preview import banner_style
from piperx_teleop.recording import Recorder
from piperx_teleop.replay import ReplayTrajectory
from piperx_teleop.robot import MockPiperX
from test_pipeline import cfg, make_controller, packet, step, synced_sample


class SpyRobot(MockPiperX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = []
    def hold(self, joints):
        self.calls.append(("hold", list(joints)))
        super().hold(joints)
    def enable(self):
        self.calls.append(("enable",))
        return True
    def set_motion_speed(self, percent):
        self.calls.append(("speed", percent))
    def send_pose(self, target):
        self.calls.append(("pose", list(target)))
        super().send_pose(target)
    def send(self, target):
        self.calls.append(("send", list(target)))
        super().send(target)
    def stop(self):
        self.calls.append(("stop",))


def csv_file(tmp_path, positions=(.4, .401, .402), dt=1/30, angle=20):
    path = tmp_path/"action.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_unix_s", *VECTOR_NAMES])
        for i, x in enumerate(positions):
            q = Rotation.from_euler("z", angle, degrees=True).as_quat().tolist()
            writer.writerow([1_700_000_000+i*dt, *([x, 0, .3, *q, .02]*2)])
    return path


def setup_replay(cfg, tmp_path, monkeypatch, **kwargs):
    clock = [100.]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    control = make_controller(cfg)
    control.robots = {s: SpyRobot(arm) for s, arm in cfg["arms"].items()}
    control.operator.trajectory = ReplayTrajectory.load(str(csv_file(tmp_path, **kwargs)), cfg)
    return control, clock


def tick(control, clock, delta=.02):
    clock[0] += delta
    for history in control.cameras.histories.values():
        history.add(clock[0], {"frame_number": round(clock[0]*1000),
                              "bgr": np.zeros((48, 64, 3), np.uint8)})
    control.tick(clock[0], delta)


def test_custom_task_index_and_auto_increment_skip_existing_directories(cfg):
    writer = Recorder(cfg["output"], cfg, mock=True)
    selection = CaptureSelection(writer, cfg["task"])
    selection.apply("把杯子放进托盘", "42", True)
    (writer.task_directory(selection.task)/"episode_043").mkdir(parents=True)
    selection.start(10.)
    assert writer.path.parts[-2:] == ("把杯子放进托盘", "episode_042")
    sync, _, _, _ = synced_sample(cfg)
    writer.add(sync.sample(10.), 0)
    writer.stop()
    assert writer.wait(3)
    selection.refresh()
    assert selection.index == 44 and selection.active is None
    with pytest.raises(ValueError, match="已存在"):
        selection.apply(selection.task, 42, True)


def test_manual_numbering_consumes_index_without_overwriting(cfg):
    writer = Recorder(cfg["output"], cfg, mock=True)
    selection = CaptureSelection(writer, cfg["task"])
    selection.apply("manual", 7, False)
    selection.start(10.)
    with pytest.raises(ValueError, match="期间"):
        selection.apply("other", 8, True)
    writer.stop(complete=False)
    assert writer.wait(3)
    selection.refresh()
    assert selection.index == 7 and selection.error
    with pytest.raises(ValueError, match="已存在"):
        selection.start(11.)
    selection.suggest("manual", True)
    assert selection.index == 8


@pytest.mark.parametrize("task,index", [("../outside", 1), ("CON", 1), ("valid", 0), ("valid", "abc")])
def test_invalid_capture_settings_are_rejected_without_motion(cfg, task, index):
    control = make_controller(cfg)
    control.robots = {s: SpyRobot(arm) for s, arm in cfg["arms"].items()}
    control.operator.request("capture_settings", task=task, index=index, auto_increment=True)
    control.operator.apply_requests(time.perf_counter())
    assert control.capture.error and control.recorder.path is None
    assert all(not robot.calls for robot in control.robots.values())


def test_invalid_inline_settings_do_not_start_old_capture_or_fault_robot(cfg):
    control = make_controller(cfg)
    data = packet()
    for side in ("left", "right"): data[side]["trigger"] = 0
    step(control, data)
    control.operator.request("record_start", selection=dict(task="bad/name", index="1", auto_increment=True))
    step(control, data)
    assert control.mode == "idle" and control.recorder.path is None and control.capture.error


def test_recording_red_ready_green_and_saving_distinct(cfg):
    status = make_controller(cfg).status()
    status.update(mode="idle", ready=True)
    assert banner_style(status) == ("#168a46", "● 就绪 · 未采集")
    status["recording"] = True
    assert banner_style(status) == ("#cf233a", "● 采集中")
    status.update(recording=False, saving=True)
    assert banner_style(status)[0] not in ("#168a46", "#cf233a")


def test_csv_interpolation_uses_original_timestamps_and_shortest_quaternion_path(cfg, tmp_path):
    path = csv_file(tmp_path, positions=(.3, .34), dt=1., angle=0)
    rows = list(csv.reader(path.open(newline="")))
    q = Rotation.from_euler("z", 60, degrees=True).as_quat()
    for offset in (0, 8):
        for j, value in enumerate(-q): rows[2][1+offset+3+j] = str(value)
    with path.open("w", newline="") as stream: csv.writer(stream).writerows(rows)
    trajectory = ReplayTrajectory.load(str(path), cfg)
    trajectory.check_speed(cfg, 1.)
    halfway = trajectory.sample(.5)
    assert halfway[0] == pytest.approx(.32)
    assert Rotation.from_quat(halfway[3:7]).magnitude() == pytest.approx(np.deg2rad(30))
    assert trajectory.duration == 1.


@pytest.mark.parametrize("bad", ["columns", "time", "nan", "workspace", "quaternion", "gripper"])
def test_bad_csv_is_rejected_before_replay(cfg, tmp_path, bad):
    path = csv_file(tmp_path)
    with path.open(newline="") as stream: rows = list(csv.reader(stream))
    if bad == "columns": rows[0][0] = "wrong"
    if bad == "time": rows[2][0] = rows[1][0]
    if bad == "nan": rows[1][1] = "nan"
    if bad == "workspace": rows[1][1] = "9"
    if bad == "quaternion": rows[1][4:8] = ["0"]*4
    if bad == "gripper": rows[1][8] = "9"
    with path.open("w", newline="") as stream: csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError): ReplayTrajectory.load(str(path), cfg)


def test_loading_csv_does_not_start_or_enable_arms(cfg, tmp_path):
    control = make_controller(cfg)
    control.robots = {s: SpyRobot(arm) for s, arm in cfg["arms"].items()}
    control.operator.request("load_replay", path=str(csv_file(tmp_path)))
    control.operator.apply_requests(time.perf_counter())
    deadline = time.perf_counter()+3
    while control.operator.loading and time.perf_counter() < deadline:
        time.sleep(.01)
        control.operator.apply_requests(time.perf_counter())
    assert control.operator.trajectory and not control.operator.loading
    assert control.operator.job is None and control.mode == "waiting"
    assert all(not robot.calls for robot in control.robots.values())


def test_replay_first_positions_both_arms_then_starts_csv_clock(cfg, tmp_path, monkeypatch):
    control, clock = setup_replay(cfg, tmp_path, monkeypatch)
    control.operator.request("replay_start", speed=1.)
    tick(control, clock)
    assert control.mode == "replaying"
    for _ in range(250):
        job = control.operator.job
        if job and job["phase"] == "playing": break
        tick(control, clock)
        if control.operator.job and control.operator.job["phase"] == "approaching":
            assert control.operator.job["start"] is None and control.operator.progress == 0
            assert all(not any(c[0] == "send" for c in r.calls) for r in control.robots.values())
    assert control.operator.job["phase"] == "playing"
    for robot in control.robots.values():
        assert [c[0] for c in robot.calls][:2] == ["hold", "enable"]
        assert ("speed", 10) in robot.calls
        poses = [c[1] for c in robot.calls if c[0] == "pose"]
        assert poses and poses[-1][0] == pytest.approx(.4)
        assert all(p[7] == pytest.approx(.04) for p in poses)  # Positioning never changes the gripper.
        assert all(np.linalg.norm(np.asarray(b[:3])-a[:3]) <= .05*.02+1e-8 for a, b in zip(poses, poses[1:]))
    for _ in range(10):
        tick(control, clock)
        if control.operator.job is None: break
    assert control.operator.job is None and control.mode == "paused"
    assert control.operator.progress == 1.
    for robot in control.robots.values():
        assert robot.vector[0] == pytest.approx(.402)
        assert robot.vector[7] == pytest.approx(.02)
        assert not any(c[0] == "stop" for c in robot.calls)


@pytest.mark.parametrize("phase", ["approaching", "playing"])
def test_stop_cancels_positioning_or_replay_without_mode_switch_or_extra_target(cfg, tmp_path, monkeypatch, phase):
    control, clock = setup_replay(cfg, tmp_path, monkeypatch, dt=.5)
    control.operator.request("replay_start", speed=.5)
    for _ in range(250):
        tick(control, clock)
        if control.operator.job and control.operator.job["phase"] == phase: break
    assert control.operator.job["phase"] == phase
    calls = {s: len(r.calls) for s, r in control.robots.items()}
    targets = {s: list(r.vector) for s, r in control.robots.items()}
    control.operator.request("replay_stop")
    tick(control, clock)
    assert control.operator.job is None and control.mode == "paused"
    for side, robot in control.robots.items():
        assert len(robot.calls) == calls[side]
        assert robot.vector == targets[side]


def test_excessive_replay_speed_is_rejected_before_any_command(cfg, tmp_path, monkeypatch):
    control, clock = setup_replay(cfg, tmp_path, monkeypatch, positions=(.3, .33), angle=0)
    control.operator.request("replay_start", speed=1.)
    tick(control, clock)
    assert control.operator.job is None and control.operator.error
    assert all(not r.calls for r in control.robots.values())
    control.operator.request("replay_start", speed=.5)
    tick(control, clock)
    assert control.operator.job is not None


def test_replay_tracking_failure_enters_power_preserving_recovery(cfg, tmp_path, monkeypatch):
    control, clock = setup_replay(cfg, tmp_path, monkeypatch)
    control.operator.request("replay_start", speed=1.)
    for _ in range(5): tick(control, clock)
    control.robots["left"].vector[0] += .2
    tick(control, clock)
    assert control.mode == "recovery" and control.operator.job is None
    assert all(not any(c[0] == "stop" for c in r.calls) for r in control.robots.values())


def test_recording_cannot_start_during_replay(cfg, tmp_path, monkeypatch):
    control, clock = setup_replay(cfg, tmp_path, monkeypatch)
    control.operator.request("replay_start", speed=1.)
    tick(control, clock)
    control.operator.request("record_start")
    tick(control, clock)
    assert control.mode == "replaying" and control.recorder.path is None


def test_zero_rotation_limit_rejects_required_approach_rotation_before_motion(cfg, tmp_path, monkeypatch):
    control, clock = setup_replay(cfg, tmp_path, monkeypatch, angle=20)
    for arm in cfg["arms"].values(): arm["max_rotation_rad_s"] = 0
    control.operator.request("replay_start", speed=1.)
    tick(control, clock)
    assert control.operator.job is None and control.operator.error
    assert all(not robot.calls for robot in control.robots.values())


def test_zero_rotation_limit_can_replay_without_required_rotation(cfg, tmp_path, monkeypatch):
    control, clock = setup_replay(cfg, tmp_path, monkeypatch, positions=(.3, .301), angle=0)
    for arm in cfg["arms"].values(): arm["max_rotation_rad_s"] = 0
    control.operator.request("replay_start", speed=1.)
    for _ in range(60):
        tick(control, clock)
        if control.operator.progress == 1.: break
    assert control.operator.progress == 1. and control.operator.job is None
