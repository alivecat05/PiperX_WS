"""Exercise SDK buffering and diagnostics without RealSense hardware."""
from collections import deque
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from human_capture import app
from human_capture.cameras import Cameras
from human_capture.controller import Controller
from human_capture.core import CAMERAS, History


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    cfg = json.loads((ROOT/"config.example.json").read_text(encoding="utf-8"))
    for c in cfg["cameras"].values():
        c.update(width=64, height=48)
    return cfg


class StreamHealthTests(unittest.TestCase):
    def test_short_consumer_stalls_retain_intermediate_sdk_frames(self):
        camera, queue, pipe = self.run_stream([100+i/30 for i in range(9)])
        numbers = [x.value["frame_number"] for x in camera.histories["head"].snapshot()]
        self.assertEqual(numbers, list(range(1, 10)))
        self.assertEqual(camera.diagnostics["head"]["source_gaps"], 0)
        self.assertEqual(camera.diagnostics["head"]["sdk_queue_peak"], 2)
        self.assertTrue(queue.keep_frames)
        self.assertTrue(pipe.stopped)

    def test_timestamp_regression_is_counted_while_receiving_continues(self):
        camera, _, _ = self.run_stream([100., 100.03, 100.02, 100.1])
        self.assertEqual(camera.diagnostics["head"]["received_frames"], 4)
        self.assertEqual(camera.diagnostics["head"]["timestamp_regressions"], 1)
        self.assertEqual([x.value["frame_number"] for x in camera.histories["head"].snapshot()], [1, 2, 4])
        self.assertNotIn("head", camera.errors)

    def test_failed_timestamp_mapping_is_distinct_from_missing_source_frames(self):
        camera, _, _ = self.run_stream([100., 200., 100.1])
        self.assertEqual(camera.diagnostics["head"]["timestamp_rejections"], 1)
        self.assertEqual(camera.diagnostics["head"]["source_gaps"], 0)
        self.assertEqual(camera.diagnostics["head"]["received_frames"], 3)

    def run_stream(self, timestamps):
        cfg = configuration()
        camera = Cameras(cfg)
        camera.exposure_settings["head"] = []  # Exercise the reconnect path, without AE warmup.
        clock = [100.]
        sensor_profile = NS(get_device=lambda: NS(get_info=lambda _: "D555", query_sensors=lambda: []))
        frames = []
        for number, stamp in enumerate(timestamps, 1):
            frame = NS(get_frame_number=lambda n=number: n,
                       get_frame_timestamp_domain=lambda: "global_time",
                       get_timestamp=lambda t=stamp: t*1000,
                       get_data=lambda: np.zeros((48, 64, 3), np.uint8))
            frames.append(NS(as_frameset=lambda f=frame: NS(get_color_frame=lambda: f)))

        class Queue:
            def __init__(self, capacity, keep_frames=False):
                self.frames = deque(maxlen=capacity)
                self.keep_frames = keep_frames
                self.consumed = self.produced = 0

            def size(self):
                return len(self.frames)

            def try_wait_for_frame(self, timeout):
                if not self.frames:
                    # A stalled consumer resumes after three camera frames have
                    # arrived. A latest-only queue would discard the first two.
                    end = min(self.produced+3, len(frames))
                    self.frames.extend(frames[self.produced:end])
                    self.produced = end
                    clock[0] += .1
                self.consumed += 1
                clock[0] += .0001
                if self.consumed == len(frames):
                    camera.stop_event.set()
                return True, self.frames.popleft()

        class Pipeline:
            stopped = False

            def start(self, config, *, queue):
                self.queue = queue
                return sensor_profile

            def try_wait_for_frames(self, timeout):
                raise AssertionError("The latest-only pipeline queue must not be used")

            def stop(self):
                self.stopped = True

        pipe = Pipeline()
        sdk = NS(pipeline=lambda _: pipe, config=lambda: object(), frame_queue=Queue,
                 camera_info=NS(name="name"), option=NS(global_time_enabled="global"))
        with (patch.dict(sys.modules, pyrealsense2=sdk),
              patch("human_capture.cameras.prepare_rgb"),
              patch("human_capture.cameras.exposure_sensors", return_value=[]),
              patch("human_capture.cameras.lock_exposure"),
              patch("human_capture.cameras.time", NS(perf_counter=lambda: clock[0], time=lambda: clock[0]))):
            camera._stream("head")
        return camera, pipe.queue, pipe

    def test_host_arrival_fps_remains_visible_when_no_sdk_timestamp_is_accepted(self):
        cfg = configuration()
        with TemporaryDirectory() as folder:
            cfg["output"] = folder
            cameras = Cameras(cfg)
            for i in range(61):
                for name in CAMERAS:
                    cameras._note_received(name, 100+i/30, i)
            control = Controller(cfg, cameras)
            control._publish(102.)
            state = control.status()
            self.assertFalse(state["ready"])
            for name in CAMERAS:
                self.assertAlmostEqual(state["cameras"][name]["receive_fps"], 30.)
                self.assertEqual(state["cameras"][name]["fps"], 0.)
            with self.assertLogs("human_capture.controller", level="INFO") as logs:
                control._log_diagnostics(102.)
                control._log_diagnostics(103.)
                control._log_diagnostics(104.)
            self.assertEqual(len(logs.output), 2)
            self.assertIn('"receive_fps": 30.0', logs.output[0])

    def test_camera_counter_reset_does_not_invent_a_source_gap(self):
        cameras = Cameras(configuration())
        cameras._note_received("head", 100., 1000)
        cameras._note_received("head", 100.1, 1)
        cameras._note_received("head", 100.2, 4)
        self.assertEqual(cameras.diagnostics["head"]["source_gaps"], 2)

    def test_diagnostics_run_without_tk_or_recording(self):
        original_threads = cv2.getNumThreads()
        try:
            with TemporaryDirectory() as folder:
                root = Path(folder)
                cfg = configuration()
                cfg["output"] = str(root/"episodes")
                config_path = root/"config.json"
                config_path.write_text(json.dumps(cfg), encoding="utf-8")
                output = io.StringIO()
                with (patch.object(app, "ROOT", root),
                      patch.object(sys, "argv", ["capture", "--config", str(config_path),
                                                 "--mock", "--diagnose-seconds", ".3"]),
                      patch("human_capture.app.logging.basicConfig"),
                      redirect_stdout(output)):
                    # No record command is ever issued by this entry point.
                    self.assertEqual(app.main(), 0)
                state = json.loads(output.getvalue())
                self.assertTrue(state["ready"])
                self.assertFalse(state["recording"])
                self.assertEqual(state["frames"], 0)
                self.assertFalse((root/"episodes").exists())
                self.assertEqual(cv2.getNumThreads(), 1)
        finally:
            cv2.setNumThreads(original_threads)

    def test_history_reports_rejected_timestamps(self):
        history = History()
        self.assertTrue(history.add(100., 1))
        self.assertFalse(history.add(100., 2))
        self.assertFalse(history.add(99., 3))
        self.assertEqual(history.latest().value, 1)


if __name__ == "__main__":
    unittest.main()
