"""Exercise the actual encoder, including non-contiguous frames and failures."""
import cv2
import numpy as np
import pytest

from piperx_teleop.ffmpeg_writer import FFmpegWriter
from piperx_teleop.recording import Recorder
from test_pipeline import cfg, synced_sample


def test_h264_round_trip_with_unicode_path_and_strided_frames(tmp_path):
    path = tmp_path / "采集 视频.mp4"
    writer = FFmpegWriter(path, 30, (64, 48))
    try:
        for i in range(30):
            frame = np.full((48, 128, 3), (20+i*5, 70, 180), np.uint8)[:, ::2]
            assert not frame.flags.c_contiguous
            writer.write(frame)
    finally:
        writer.release()
    writer.release()  # Cleanup remains safe after successful completion.
    video = cv2.VideoCapture(str(path))
    try:
        assert video.isOpened()
        assert video.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
        assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 30
        assert int(video.get(cv2.CAP_PROP_FOURCC)) in {
            cv2.VideoWriter_fourcc(*tag) for tag in ("avc1", "H264", "h264")}
        for i in range(30):
            ok, frame = video.read()
            assert ok and frame.shape == (48, 64, 3)
            assert np.max(np.abs(frame.mean(axis=(0, 1)) - (20+i*5, 70, 180))) < 8
        assert not video.read()[0]
    finally:
        video.release()


def test_terminated_encoder_reports_failure_and_reaps_process(tmp_path):
    writer = FFmpegWriter(tmp_path / "broken.mp4", 30, (64, 48))
    writer.process.kill()
    writer.process.wait(timeout=5)
    with pytest.raises(OSError, match="encoder exited"):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    with pytest.raises(OSError, match="encoder exited"):
        writer.release()
    assert writer.log.closed and writer.process.stdin.closed
    writer.release()


@pytest.mark.parametrize("stage", ["write", "release"])
def test_recording_cannot_commit_after_encoder_failure(cfg, monkeypatch, stage):
    class FailingWriter:
        def __init__(self, *args): pass
        def write(self, frame):
            if stage == "write":
                raise OSError("encoder write failure")
        def release(self):
            if stage == "release":
                raise OSError("encoder release failure")

    monkeypatch.setattr("piperx_teleop.recording.FFmpegWriter", FailingWriter)
    recorder = Recorder(cfg["output"], cfg, mock=True)
    recorder.start(10.)
    recorder.add(synced_sample(cfg)[0].sample(10.), 0)
    recorder.stop(end_time=10.+1/30)
    assert recorder.wait(5)
    assert recorder.result == "incomplete"
    assert f"encoder {stage} failure" in recorder.error
