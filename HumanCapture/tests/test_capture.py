import csv
import json
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import cv2
import numpy as np
import pytest

from human_capture.app import load_config
from human_capture.capture_session import CaptureSelection
from human_capture.controller import Controller
from human_capture.core import CAMERAS, History, Synchronizer
from human_capture.recording import Recorder


def frame(t, index, cfg, name, basis="global_time"):
    c = cfg["cameras"][name]
    return {"bgr": np.full((c["height"], c["width"], 3), index % 200, np.uint8),
            "frame_number": index, "timestamp_basis": basis, "clock": basis,
            "receive_s": t, "device_ms": t*1000}


def cameras():
    return SimpleNamespace(histories={n: History(900) for n in CAMERAS}, errors={})


def add_group(cams, cfg, t, i, basis="global_time"):
    for n, phase in zip(CAMERAS, (0, .004, .011)):
        cams.histories[n].add(t+phase, frame(t+phase, i, cfg, n, basis))


def test_twenty_seconds_of_30hz_phased_jittered_frames_without_drops(cfg):
    cams = cameras()
    for i in range(603):
        add_group(cams, cfg, 100+i/30+.0005*np.sin(i*.3), i)
    sync = Synchronizer(cams.histories, cfg["sync"])
    ids = {n: [] for n in CAMERAS}
    for i in range(600):
        sample = sync.sample(100+i/30)
        assert sample is not None, sync.rejection
        assert sample["synchronization_span_s"] <= 1/30+1e-9
        for n in CAMERAS:
            ids[n].append(sample["camera_metadata"][n]["frame_number"])
    assert all(len(set(seq)) == 600 and seq == list(range(600)) for seq in ids.values())


def test_reused_frame_is_rejected(cfg):
    cams = cameras()
    add_group(cams, cfg, 100, 1)
    sync = Synchronizer(cams.histories, cfg["sync"])
    assert sync.sample(100)
    assert sync.sample(100.01) is None
    assert "reused" in sync.rejection


def test_clock_and_excess_span_are_rejected(cfg):
    cams = cameras()
    add_group(cams, cfg, 100, 1, "host_receive")
    sync = Synchronizer(cams.histories, cfg["sync"])
    assert sync.sample(100) is None
    assert "clock_not_synchronized" in sync.rejection
    cams = cameras()
    for n, offset in zip(CAMERAS, (-.025, 0, .025)):
        cams.histories[n].add(100+offset, frame(100+offset, 1, cfg, n))
    sync = Synchronizer(cams.histories, cfg["sync"])
    assert sync.sample(100) is None
    assert sync.rejection == "timestamp_span_exceeds_one_frame"


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "CON", "bad.", "a:b", " task"])
def test_invalid_task_names_are_rejected(cfg, bad):
    with pytest.raises(ValueError):
        Recorder(cfg["output"], cfg).task_directory(bad)


def test_video_csv_alignment_and_auto_increment(cfg):
    cams = cameras()
    control = Controller(cfg, cams, mock=True)
    for i in range(22):
        add_group(cams, cfg, 100+i/30, i)
    selection = {"task": "人类拿杯子", "index": "42", "auto_increment": True}
    control.request("record_start", selection=selection)
    control.tick(100)
    control.request("record_stop")
    control.tick(100.5)
    assert control.recording  # Preserve stop boundary while buffering the tail.
    control.tick(100.5+cfg["sync"]["delay_ms"]/1000+.01)
    assert not control.recording
    assert control.recorder.wait(5)
    control.tick(100.5+cfg["sync"]["delay_ms"]/1000+.02)
    path = Path(cfg["output"])/"人类拿杯子"/"episode_042"
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["frames"] == 15
    assert manifest["dropped_grid_ticks"] == 0 and manifest["simulated"]
    assert abs(manifest["effective_fps"]-30) < 1e-8
    assert control.capture.index == 43
    rows = list(csv.DictReader((path/"timestamps.csv").open(encoding="utf-8")))
    samples = [json.loads(x) for x in (path/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(samples) == 15
    assert [int(x["grid_index"]) for x in rows] == list(range(15))
    for name in CAMERAS:
        video = cv2.VideoCapture(str(path/f"{name}.mp4"))
        try:
            assert video.get(cv2.CAP_PROP_FPS) == 30
            assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 15
            for row, sample in zip(rows, samples):
                ok, image = video.read()
                assert ok and image.shape == (48, 64, 3)
                assert abs(float(row[f"{name}_timestamp_unix_s"])-sample["source_timestamps_unix_s"][name]) < 1e-6
        finally:
            video.release()
    assert not (path/"action.csv").exists()
    with pytest.raises(ValueError, match="存在"):
        control.capture.apply("人类拿杯子", 42, True)


@pytest.mark.parametrize("fps", [20, 25])
def test_slow_cameras_cannot_be_disguised_as_complete_30hz(cfg, fps):
    cams = cameras()
    for i in range(fps+2):
        add_group(cams, cfg, 100+i/fps, i)
    sync = Synchronizer(cams.histories, cfg["sync"])
    rec = Recorder(cfg["output"], cfg, mock=True)
    rec.start(100)
    for i in range(30):
        sample = sync.sample(100+i/30)
        if sample is None:
            rec.drop(sync.rejection)
        else:
            rec.add(sample, i)
    rec.stop(end_time=101)
    assert rec.wait(5)
    assert rec.result == "incomplete" and rec.dropped >= 3
    assert rec.valid_count < 30 and rec.count == 30


def test_camera_loss_and_staleness_do_not_stop_recording_and_new_frames_resume(cfg):
    cams = cameras()
    add_group(cams, cfg, 100, 0)
    control = Controller(cfg, cams, mock=True)
    control.request("record_start", selection={"task": "task", "index": 1, "auto_increment": True})
    control.tick(100)
    cams.errors["left_wrist"] = "RGB stream lost"
    control.tick(100.8)
    assert control.recording and control.recorder.active and control.recorder.dropped
    cams.errors.clear()
    for i in range(10):
        add_group(cams, cfg, 100.9+i/30, i+1)
    control.tick(101.2)
    assert control.recording
    control.request("record_stop")
    control.tick(101.2)
    control.tick(101.4)
    assert not control.recording
    assert control.recorder.wait(5)
    assert control.recorder.result == "incomplete"
    assert control.recorder.reason == "operator_stop"
    rows = [json.loads(row) for row in (control.recorder.path/"samples.jsonl").read_text().splitlines()]
    assert any(row["sample_valid"] and row["grid_index"] >= 27 for row in rows)


def test_reconnecting_camera_frame_number_reset_does_not_block_matching(cfg):
    cams = cameras()
    add_group(cams, cfg, 100, 1000)
    for history in cams.histories.values():
        history.latest().value["sequence"] = 10
    sync = Synchronizer(cams.histories, cfg["sync"])
    assert sync.sample(100)
    add_group(cams, cfg, 100+1/30, 1)
    for history in cams.histories.values():
        history.latest().value["sequence"] = 11
    assert sync.sample(100+1/30)


def test_saturated_writer_drops_one_group_without_stopping_and_can_resume(cfg, monkeypatch):
    rec = Recorder(cfg["output"], cfg, mock=True)
    entered, release = Event(), Event()
    write = rec._write
    def blocked_writer():
        entered.set()
        assert release.wait(5)
        write()
    monkeypatch.setattr(rec, "_write", blocked_writer)
    cams = cameras()
    add_group(cams, cfg, 100, 0)
    sample = Synchronizer(cams.histories, cfg["sync"]).sample(100)
    try:
        rec.start(100)
        assert entered.wait(1)
        for i in range(rec.queue.maxsize):
            assert rec.add(dict(sample, timestamp_monotonic_s=100+i/30), i)
        assert rec.add(sample, i+1) is False
        assert rec.active and rec.error is None and rec.dropped == 1
        # Allow a subsequent group in this episode after backpressure clears.
        rec.queue.get_nowait()
        rec.drop("test_removed_group")
        assert rec.add(dict(sample, timestamp_monotonic_s=100+(i+2)/30), i+2)
        rec.stop(end_time=103)
    finally:
        release.set()
        assert rec.wait(5)
    assert rec.result == "incomplete"
    assert rec.drop_reasons["writer_queue_full"] == 1


def test_no_auto_increment_requires_new_index(cfg):
    rec = Recorder(cfg["output"], cfg, mock=True)
    selection = CaptureSelection(rec, "task")
    selection.apply("task", 9, False)
    selection.start(100)
    rec.stop(complete=False)
    assert rec.wait(5)
    selection.refresh()
    assert selection.index == 9 and selection.error


def test_timestamp_mapping_matches_teleop():
    from human_capture.cameras import frame_timestamp
    assert frame_timestamp(1000000, "global_time", 100, -900) == (100, "global_time")
    assert frame_timestamp(99, "hardware_clock", 100, -900) == (100, "host_receive")
    with pytest.raises(RuntimeError):
        frame_timestamp(100, "system_time", 100, 0)


def test_config_rejects_lower_frame_rate_and_unsynced_timestamps(cfg, tmp_path):
    path = tmp_path/"config.json"
    cfg["record_hz"] = 25
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="30"):
        load_config(path)
    cfg["record_hz"] = 30
    cfg["sync"]["require_sdk_timestamps"] = False
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="SDK"):
        load_config(path)
