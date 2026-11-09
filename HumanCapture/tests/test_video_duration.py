"""Loss affects freshness, never the saved video's elapsed time."""
import json
import time

import cv2
import pytest

from human_capture.core import Synchronizer
from human_capture.recording import Recorder
from test_capture import cameras, add_group


def make_sample(cfg):
    cams = cameras()
    add_group(cams, cfg, 100, 0)
    return Synchronizer(cams.histories, cfg["sync"]).sample(100)

def enqueue(recorder, sample, grid):
    deadline = time.perf_counter()+5
    while recorder.queue.full() and time.perf_counter() < deadline:
        assert recorder.error is None
        time.sleep(.001)
    assert recorder.add(sample, grid)


def shift_stamps(value, offset):
    return {k: shift_stamps(v, offset) for k, v in value.items()} if isinstance(value, dict) else value+offset


def test_thirty_seconds_with_only_five_source_fps_still_exports_thirty_seconds(cfg):
    sample = make_sample(cfg)
    rec = Recorder(cfg["output"], cfg, mock=True)
    rec.start(100.)
    try:
        for i in range(150):
            stamp = 100+i/5
            frame = dict(sample, timestamp_monotonic_s=stamp,
                         source_timestamps=shift_stamps(sample["source_timestamps"], i/5),
                         camera_metadata={n: dict(m, frame_number=i, receive_s=stamp)
                                          for n, m in sample["camera_metadata"].items()})
            enqueue(rec, frame, i*6)
        rec.stop(end_time=130.)
    finally:
        if rec.active:
            rec.stop(end_time=130.)
        assert rec.wait(8), rec.error
    assert rec.error is None
    meta = json.loads((rec.path/"manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(row) for row in (rec.path/"samples.jsonl").read_text().splitlines()]
    assert meta["status"] == "incomplete"
    assert meta["frames"] == len(rows) == 900
    assert meta["valid_frames"] == 150 and meta["placeholder_frames"] == 750
    assert meta["dropped_grid_ticks"] == 750
    assert meta["video_duration_s"] == meta["capture_duration_s"] == 30
    assert meta["effective_fps"] == pytest.approx(5)
    assert [row["grid_index"] for row in rows] == list(range(900))
    assert sum(row["sample_valid"] for row in rows) == 150
    assert rows[1]["frame_status"] == "held" and rows[1]["held_from_grid"] == 0
    assert rows[1]["source_timestamps_unix_s"] == rows[0]["source_timestamps_unix_s"]
    assert rows[-1]["frame_status"] == "held"  # Missing tail must also preserve time.
    assert rows[-1]["timestamp_unix_s"]-rows[0]["timestamp_unix_s"] == pytest.approx(899/30, abs=1e-6)
    for name in cfg["cameras"]:
        video = cv2.VideoCapture(str(rec.path/(name+".mp4")))
        try:
            assert video.isOpened()
            assert video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS) == pytest.approx(30.)
        finally:
            video.release()
    


@pytest.mark.parametrize("first_grid", [None, 30])
def test_leading_or_total_camera_loss_is_black_and_retains_full_duration(cfg, first_grid):
    sample = make_sample(cfg)
    rec = Recorder(cfg["output"], cfg, mock=True)
    rec.start(100.)
    if first_grid is not None:
        enqueue(rec, dict(sample, timestamp_monotonic_s=101.), first_grid)
    rec.stop(end_time=103.)
    assert rec.wait(5) and rec.error is None
    rows = [json.loads(row) for row in (rec.path/"samples.jsonl").read_text().splitlines()]
    assert len(rows) == rec.count == 90
    assert rows[0]["frame_status"] == "blank" and not rows[0]["sample_valid"]
    assert rows[0]["held_from_grid"] is None
    assert rows[-1]["frame_status"] == ("blank" if first_grid is None else "held")
    assert rec.valid_count == (0 if first_grid is None else 1)
    assert rec.placeholder_count == 90-rec.valid_count
    video = cv2.VideoCapture(str(rec.path/"head.mp4"))
    try:
        ok, image = video.read()
        assert ok and image.max() <= 3
        assert video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS) == pytest.approx(3.)
    finally:
        video.release()
