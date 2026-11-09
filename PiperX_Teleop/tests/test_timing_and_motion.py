"""Real frame cadence, timestamp bounds, MP4/CSV alignment, and motion continuity."""
import csv
import json
import time

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from piperx_teleop.app import load_config
from piperx_teleop.cameras import frame_timestamp
from piperx_teleop.core import ARMS, CAMERAS, VECTOR_NAMES, History, PoseMapper, Synchronizer
from piperx_teleop.export import EpisodeFrames, read_segments
from piperx_teleop.recording import Recorder
from test_pipeline import cfg, make_controller, packet, synced_sample


def streams(cfg, camera_fps=30, count=300):
    cameras = {n: History(count+10) for n in CAMERAS}
    states = {n: History(count*2+10) for n in ARMS}
    actions = {n: History(count*2+10) for n in ARMS}
    vector = [.3, 0, .3, 0, 0, 0, 1, .04]
    for name, phase in zip(CAMERAS, (-.015, .002, .015)):
        for i in range(int(count*camera_fps/30)+3):
            t = 10+i/camera_fps+phase+.0005*np.sin(i*.73+phase*100)
            cameras[name].add(t, {"bgr": np.zeros((48, 64, 3), np.uint8),
                "frame_number": i, "timestamp_basis": "global_time"})
    for side in ARMS:
        for i in range(count*2+3):
            t = 10+i/60-.002
            states[side].add(t, {"vector": vector, "joints": [0]*6,
                "source_timestamps": {"tcp": t, "gripper": t-.001}})
            actions[side].add(t, vector)
    return Synchronizer(cameras, states, actions, cfg["sync"])


def test_thirty_hz_phase_shifted_cameras_have_no_artificial_drops(cfg):
    sync = streams(cfg, count=600)
    ids = {name: [] for name in CAMERAS}
    for i in range(600):
        t = 10+i/30
        row = sync.sample(t)
        assert row is not None, (i, sync.rejection)
        stamps = list(row["source_timestamps"]["cameras"].values())
        stamps += list(row["source_timestamps"]["actions"].values())
        stamps += [v for arm in row["source_timestamps"]["states"].values() for v in arm.values()]
        assert max(stamps)-min(stamps) <= 1/30+1e-9
        for name in CAMERAS:
            ids[name].append(row["camera_metadata"][name]["frame_number"])
    for frames in ids.values():
        assert len(set(frames)) == 600
        assert all(b == a+1 for a, b in zip(frames, frames[1:]))


@pytest.mark.parametrize("fps", [20, 25])
def test_true_slow_cameras_cannot_be_padded_to_thirty(cfg, fps):
    sync = streams(cfg, camera_fps=fps)
    rows = [sync.sample(10+i/30) for i in range(300)]
    assert sum(row is not None for row in rows) < 270
    for name in CAMERAS:
        ids = [row["camera_metadata"][name]["frame_number"] for row in rows if row]
        assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("domain", ["global_time", "system_time"])
def test_camera_sdk_timestamp_preserves_exposure_time(domain):
    stamp, basis = frame_timestamp(1_700_000_000_000, "timestamp_domain."+domain,
                                  100.07, 100-1_700_000_000)
    assert stamp == 100 and basis == domain


def test_raw_device_timestamp_is_identified_and_cannot_claim_sync(cfg):
    stamp, basis = frame_timestamp(1234, "hardware_clock", 100, 100-1_700_000_000)
    assert stamp == 100 and basis == "host_receive"
    sync, cameras, _, _ = synced_sample(cfg)
    for history in cameras.values():
        history.latest().value["timestamp_basis"] = "host_receive"
    assert sync.sample(10) is None


def test_mp4_csv_and_jsonl_rows_share_unix_timestamps_and_frame_order(cfg):
    for camera in cfg["cameras"].values():
        camera.update(width=64, height=48)
    sync, _, _, _ = synced_sample(cfg)
    row = sync.sample(10.)
    writer = Recorder(cfg["output"], cfg, mock=True)
    writer.start(10.)
    for i in range(12):
        image = np.full((48, 64, 3), (20+i*5, 70, 180), dtype=np.uint8)
        sample = dict(row, timestamp_monotonic_s=10+i/30,
                      images={name: image for name in CAMERAS})
        writer.add(sample, i)
    writer.stop(end_time=10.4)
    assert writer.wait(5) and writer.result == "complete", writer.error
    meta = json.loads((writer.path/"manifest.json").read_text())
    rows = [json.loads(line) for line in (writer.path/"samples.jsonl").read_text().splitlines()]
    with (writer.path/"action.csv").open(newline="") as stream:
        records = list(csv.reader(stream))
    assert records[0] == ["timestamp_unix_s", *VECTOR_NAMES]
    assert len(records) == 13 and meta["effective_fps"] == pytest.approx(30)
    assert not list(writer.path.rglob("*.jpg"))
    with EpisodeFrames(writer.path, meta) as frames:
        for i, (row, csv_row) in enumerate(zip(rows, records[1:])):
            assert row["frame_index"] == row["grid_index"] == i
            assert abs(float(csv_row[0])-row["timestamp_unix_s"]) < 1e-6
            assert [float(x) for x in csv_row[1:]] == row["action"]
            for image in frames.read(row).values():
                assert abs(image[:, :, 0].mean()-(20+i*5)) < 10
    assert len(read_segments(cfg["output"], allow_mock=True)[0][2]) == 12
    with (writer.path/"action.csv").open("a") as stream:
        stream.write("0,"+",".join(["0"]*16)+"\n")
    with pytest.raises(ValueError, match="extra rows"):
        read_segments(cfg["output"], allow_mock=True)


def test_missing_grid_never_marks_lower_rate_video_complete(cfg):
    sync, _, _, _ = synced_sample(cfg)
    row = sync.sample(10.)
    writer = Recorder(cfg["output"], cfg, mock=True)
    writer.start(10.)
    for i in (0, 1, 3, 4):
        writer.add(dict(row, timestamp_monotonic_s=10+i/30), i)
    writer.stop(end_time=10.2)
    assert writer.wait(5) and writer.result == "incomplete"
    meta = json.loads((writer.path/"manifest.json").read_text())
    assert meta["dropped_grid_ticks"] == 2 and meta["effective_fps"] < 30
    assert meta["frames"] == 6 and meta["valid_frames"] == 4
    assert meta["video_duration_s"] == pytest.approx(.2)


def test_encoder_failure_cannot_commit_complete_episode(cfg, monkeypatch):
    class FailedWriter:
        def isOpened(self): return False
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoWriter", lambda *_: FailedWriter())
    writer = Recorder(cfg["output"], cfg, mock=True)
    writer.start(10.)
    assert writer.wait(5)
    assert writer.result == "incomplete" and "Could not open" in writer.error


def test_mp4_export_adapter_passes_decoded_frames_and_task_to_dataset(cfg, tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from piperx_teleop import export
    for camera in cfg["cameras"].values():
        camera.update(width=64, height=48)
    sync, _, _, _ = synced_sample(cfg)
    row = sync.sample(10.)
    recorder = Recorder(cfg["output"], cfg, mock=True)
    recorder.start(10.)
    for i in range(3):
        recorder.add(dict(row, timestamp_monotonic_s=10+i/30), i)
    recorder.stop()
    assert recorder.wait(5) and recorder.result == "complete"
    frames, created, saved = [], [], []
    class Dataset:
        @classmethod
        def create(cls, **kwargs):
            created.append(kwargs)
            (kwargs["root"]/"meta").mkdir(parents=True)
            return cls()
        def add_frame(self, frame): frames.append(frame)
        def save_episode(self): saved.append(True)
        def finalize(self): pass
    for name in ("av", "lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = Dataset
    monkeypatch.setattr(export, "version", lambda _: "0.4.2")
    target = tmp_path/"dataset"
    export.convert(cfg["output"], target, "local/test", allow_mock=True)
    assert len(frames) == 3 and saved == [True] and created[0]["fps"] == 30
    assert frames[0]["task"] == cfg["task"]
    for name in CAMERAS:
        assert frames[0][f"observation.images.{name}"].shape == (48, 64, 3)
    assert (target/"meta"/"piperx_sources.json").exists()


def test_old_jpeg_episodes_remain_exportable_with_gaps(cfg, tmp_path):
    sync, _, _, _ = synced_sample(cfg)
    sample = sync.sample(10.)
    path = tmp_path/"legacy"
    path.mkdir()
    rows = []
    for index, grid in enumerate((0, 1, 4, 5)):
        row = {k: v for k, v in sample.items() if k != "images"}
        row.update(grid_index=grid, frame_index=index, timestamp_monotonic_s=10+grid/30, images={})
        for name, bgr in sample["images"].items():
            filename = f"{name}_{index}.jpg"
            assert cv2.imwrite(str(path/filename), bgr)
            row["images"][name] = filename
        rows.append(row)
    meta = {"schema": "piperx.rgb_action.v1", "status": "complete", "simulated": True,
            "frames": 4, "fps": 30, "task": "legacy", "vector_names": VECTOR_NAMES, "config": cfg}
    (path/"manifest.json").write_text(json.dumps(meta))
    (path/"samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    assert [len(rows) for _, _, rows in read_segments(path, allow_mock=True)] == [2, 2]


def test_pose_noise_is_filtered_after_motion_has_started(cfg):
    mapper = PoseMapper(cfg["arms"]["left"])
    state = [.3, 0, .3, 0, 0, 0, 1, .04]
    hand = packet()["left"]
    hand.update(grip=True, trigger=0.)
    mapper.update(hand, state, 1/60)
    hand["position"][2] -= .03
    for _ in range(150):
        state = mapper.update(hand, state, 1/60)
    rng = np.random.default_rng(7)
    positions, rotations, widths = [], [], []
    for i in range(300):
        hand["position"][2] = -.03+rng.normal(0, .001)
        hand["quaternion"] = Rotation.from_euler("z", rng.normal(0, .25), degrees=True).as_quat()
        hand["trigger"] = (i % 11)/10
        state = mapper.update(hand, state, 1/60)
        positions.append(state[0])
        rotations.append(Rotation.from_quat(state[3:7]).magnitude())
        widths.append(state[7])
        assert widths[-1] == pytest.approx((1-hand["trigger"])*.07)
    assert np.std(positions) < .0004
    assert np.std(rotations) < np.deg2rad(.1)


@pytest.mark.parametrize("scale", [.5, 1., 2.])
def test_position_scale_multiplies_hand_displacement(cfg, scale):
    arm = dict(cfg["arms"]["left"], position_scale=scale, pose_smoothing_s=0,
               tremor_smoothing_s=0, position_deadband_m=0, max_speed_m_s=10)
    mapper = PoseMapper(arm)
    state = [.3, 0, .3, 0, 0, 0, 1, .04]
    hand = packet()["left"]
    hand["grip"] = True
    mapper.update(hand, state, .02)
    hand["position"][2] -= .04
    target = mapper.update(hand, state, .02)
    assert target[0] == pytest.approx(.3+scale*.04)


def test_default_follow_does_not_accumulate_catchup_at_normal_hand_speed(cfg):
    arm = cfg["arms"]["left"]
    assert arm["pose_smoothing_s"] == 0
    mapper = PoseMapper(arm)
    state = [.3, 0, .3, 0, 0, 0, 1, .04]
    hand = packet()["left"]
    hand["grip"] = True
    mapper.update(hand, state, 1/60)
    for i in range(1, 61):
        hand["position"][2] = -.2*i/60  # 20 cm/s hand motion for one second.
        target = mapper.update(hand, state, 1/60)
        desired = .3+arm["position_scale"]*.2*i/60
        assert abs(target[0]-desired) <= .003  # Small bounded filter lag, not growing catch-up.
        assert not mapper.rate_limited
        state = target
    stopped = list(target)
    for _ in range(60):
        target = mapper.update(hand, state, 1/60)
        assert np.linalg.norm(np.asarray(target[:3])-stopped[:3]) <= .002
    assert abs(target[0]-desired) <= arm["position_deadband_m"]+1e-6


def test_rate_limit_is_reported_when_hand_exceeds_follow_speed(cfg):
    mapper = PoseMapper(cfg["arms"]["left"])
    state = [.3, 0, .3, 0, 0, 0, 1, .04]
    hand = packet()["left"]
    hand["grip"] = True
    mapper.update(hand, state, 1/60)
    hand["position"][2] = -.1
    target = mapper.update(hand, state, 1/60)
    assert mapper.rate_limited
    assert target[0]-.3 <= cfg["arms"]["left"]["max_speed_m_s"]/60+1e-9


def test_parallel_home_progresses_right_while_left_has_not_settled(cfg, monkeypatch):
    control = make_controller(cfg)
    calls = []
    for side in ARMS:
        cfg["arms"][side]["home_waypoints"] = [[.1]*6, [.2]*6]
        monkeypatch.setattr(control.robots[side], "move_home",
                            lambda joints, side=side: calls.append((side, list(joints))))
    control._begin_reset(10.)
    states = {side: control.robots[side].read()[1] for side in ARMS}
    control._reset_tick(states, 10.)
    assert calls == [("left", [.1]*6), ("right", [.1]*6)]
    states["right"]["joints"] = [.1]*6
    control._reset_tick(states, 10.1)
    control._reset_tick(states, 10.41)
    assert calls[-1] == ("right", [.2]*6)
    assert len([side for side, _ in calls if side == "left"]) == 1
    assert control.mode == "resetting"


def test_parallel_home_waypoint_timeout_identifies_stalled_arm(cfg):
    control = make_controller(cfg)
    for side in ARMS:
        cfg["arms"][side]["home_waypoints"] = [[.1]*6]
    control._begin_reset(10.)
    states = {side: control.robots[side].read()[1] for side in ARMS}
    control._reset_tick(states, 10.)
    with pytest.raises(TimeoutError, match="left"):
        control._reset_tick(states, 10.+cfg["reset_timeout_s"]+.1)
