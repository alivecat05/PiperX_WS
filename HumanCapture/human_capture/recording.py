"""Bounded MP4 writer; one timestamp row represents the same frame in all videos."""
from collections import Counter
import copy
import csv
import json
import os
from pathlib import Path, PureWindowsPath
from queue import Empty, Full, Queue
import re
from threading import Event, Thread
import time

import cv2
import numpy as np

from .core import CAMERAS, History
from .video_timeline import VideoTimeline, grid_count


class Recorder:
    def __init__(self, root, cfg, mock=False):
        self.root, self.cfg, self.mock = Path(root), cfg, mock
        self.thread = None
        self.finished = Event()
        self.path = self.result = self.error = None
        self.count = self.dropped = self.valid_count = self.placeholder_count = 0
        self.drop_reasons = Counter()
        self.writes = History()
        self.camera_diagnostics = {}

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    @property
    def active(self):
        return self.busy and not self.finished.is_set()

    def task_directory(self, task):
        if (not isinstance(task, str) or not task.strip() or task != task.strip()
                or re.search(r'[<>:"/\\|?*\x00-\x1f]', task) or task.endswith('.')
                or PureWindowsPath(task).is_reserved()):
            raise ValueError("任务名不能为空，也不能包含路径或 Windows 文件名禁用字符")
        path = self.root/task
        if path.resolve().parent != self.root.resolve():
            raise ValueError("任务目录必须位于采集输出目录内")
        return path

    def next_index(self, task):
        numbers = [int(match[1]) for p in self.task_directory(task).glob("episode_*")
                   if (match := re.fullmatch(r"episode_(\d+)", p.name))]
        return max(numbers, default=0)+1

    def start(self, start_time, *, task=None, capture_index=None):
        if self.busy:
            raise ValueError("正在录制或保存，请等待")
        task = self.cfg["task"] if task is None else task
        capture_index = self.next_index(task) if capture_index is None else capture_index
        if type(capture_index) is not int or capture_index < 1:
            raise ValueError("采集序号必须是正整数")
        path = self.task_directory(task)/f"episode_{capture_index:03d}"
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            raise ValueError(f"序号已存在，禁止覆盖：{path}") from None
        self.path, self.task, self.capture_index = path, task, capture_index
        self.config_snapshot = copy.deepcopy(self.cfg)
        self.config_snapshot["task"] = task
        self.count = self.dropped = self.valid_count = self.placeholder_count = 0
        self.drop_reasons = Counter()
        self.writes = History()
        self.first_t = self.last_t = None
        self.last_grid = -1
        self.reported_missing = 0
        self.max_span = 0.
        self.queue_high_watermark = 0
        self.encode_total_s = self.max_encode_s = 0.
        self.error = self.result = None
        self.start_time = start_time
        before = time.perf_counter()
        wall = time.time()
        self.unix_offset = wall-(before+time.perf_counter())/2
        self.end_time = None
        self.complete, self.reason = True, "operator_stop"
        # Bound retained full-resolution RGB images by bytes, rather than 60 groups
        # (~516 MB with the deployed profiles), in addition to camera histories.
        group_bytes = sum(c["width"]*c["height"]*3 for c in self.cfg["cameras"].values())
        self.queue = Queue(maxsize=max(2, min(60, 128*1024*1024//group_bytes)))
        self.finished = Event()
        # A reserved folder remains identifiable even after an abrupt process termination.
        self._manifest("recording")
        self.thread = Thread(target=self._write, name="human-video-writer", daemon=True)
        self.thread.start()

    def drop(self, reason):
        self.dropped += 1
        self.reported_missing += 1
        self.drop_reasons[reason] += 1

    def add(self, sample, grid_index):
        if self.error:
            raise RuntimeError(self.error)
        if not self.active:
            raise RuntimeError("视频写入线程已停止")
        if type(grid_index) is not int or grid_index <= self.last_grid:
            raise ValueError("录制帧索引必须递增")
        gap = max(0, grid_index-self.last_grid-1-self.reported_missing)
        if gap:
            self.dropped += gap
            self.drop_reasons["grid_gap"] += gap
        try:
            self.queue.put_nowait((sample, grid_index))
        except Full:
            self.drop("writer_queue_full")
            self.last_grid, self.reported_missing = grid_index, 0
            return False
        self.last_grid, self.reported_missing = grid_index, 0
        self.queue_high_watermark = max(self.queue_high_watermark, self.queue.qsize())
        return True

    def stop(self, reason="operator_stop", complete=True, end_time=None):
        self.reason, self.complete, self.end_time = reason, complete, end_time
        self.finished.set()

    def wait(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)
        return not self.busy

    def _write(self):
        videos = {}
        try:
            for name in CAMERAS:
                c = self.config_snapshot["cameras"][name]
                writer = cv2.VideoWriter(str(self.path/f"{name}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), self.cfg["record_hz"], (c["width"], c["height"]))
                videos[name] = writer
                if not writer.isOpened():
                    raise OSError(f"无法创建视频：{name}.mp4")
            with ((self.path/"samples.jsonl").open("w", encoding="utf-8") as samples,
                  (self.path/"timestamps.csv").open("w", encoding="utf-8", newline="") as stamps):
                csv_writer = csv.writer(stamps)
                csv_writer.writerow(["frame_index", "grid_index", "timestamp_unix_s",
                    *[f"{n}_timestamp_unix_s" for n in CAMERAS],
                    *[f"{n}_frame_number" for n in CAMERAS], "synchronization_span_s", "sample_valid", "frame_status"])
                timeline = VideoTimeline(self.config_snapshot, self.start_time)
                while not self.finished.is_set() or not self.queue.empty():
                    try:
                        sample, grid = self.queue.get(timeout=.02)
                    except Empty:
                        continue
                    for frame, index in timeline.accept(sample, grid):
                        self._write_frame(frame, index, videos, samples, csv_writer)
                count = (grid_count(self.start_time, self.end_time, self.cfg["record_hz"])
                         if self.end_time is not None else self.last_grid+1+self.reported_missing)
                for frame, index in timeline.finish(count):
                    self._write_frame(frame, index, videos, samples, csv_writer)
                unreported = max(0, self.placeholder_count-self.dropped)
                self.dropped += unreported
                if unreported:
                    self.drop_reasons["tail_gap"] += unreported
                samples.flush()
                stamps.flush()
                os.fsync(samples.fileno())
                os.fsync(stamps.fileno())
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            for writer in videos.values():
                try:
                    writer.release()
                except Exception as error:
                    self.error = str(error)
        try:
            if not self.error:
                for name in CAMERAS:
                    video = cv2.VideoCapture(str(self.path/f"{name}.mp4"))
                    try:
                        if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != self.count:
                            raise OSError(f"视频帧数校验失败：{name}")
                    finally:
                        video.release()
            # A missing tail grid is also data loss, even if no later frame was accepted.
            valid = self.complete and not self.error and self.valid_count > 0 and not self.dropped
            self._manifest("complete" if valid else "incomplete")
            self.result = "complete" if valid else "incomplete"
        except Exception as error:
            self.error = f"保存校验失败：{error}"
            self.result = "incomplete"
            try:
                self._manifest("incomplete")
            except OSError:
                pass

    def _write_frame(self, sample, grid, videos, samples, csv_writer):
        unix = lambda stamp: stamp+self.unix_offset if stamp is not None else None
        number = lambda value: f"{value:.9f}" if value is not None else ""
        source = {n: unix(sample["source_timestamps"][n]) for n in CAMERAS}
        metadata = {}
        for n in CAMERAS:
            metadata[n] = {k: v for k, v in sample["camera_metadata"][n].items()
                           if k not in ("receive_s", "device_ms")}
            metadata[n]["receive_unix_s"] = unix(sample["camera_metadata"][n].get("receive_s"))
        t = sample["timestamp_monotonic_s"]
        row = {"frame_index": self.count, "grid_index": grid, "timestamp_unix_s": unix(t),
               "source_timestamps_unix_s": source, "camera_metadata": metadata,
               "synchronization_span_s": sample["synchronization_span_s"],
               **{k: sample[k] for k in ("sample_valid", "frame_status", "held_from_grid")}}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n"
        encode_start = time.perf_counter()
        for n in CAMERAS:
            c, image = self.config_snapshot["cameras"][n], sample["images"][n]
            if image.shape != (c["height"], c["width"], 3) or image.dtype != np.uint8:
                raise ValueError(f"图像尺寸或格式无效：{n}")
            videos[n].write(image)
        elapsed = time.perf_counter()-encode_start
        self.encode_total_s += elapsed
        self.max_encode_s = max(self.max_encode_s, elapsed)
        samples.write(encoded)
        csv_writer.writerow([self.count, grid, number(unix(t)), *[number(source[n]) for n in CAMERAS],
            *[metadata[n]["frame_number"] for n in CAMERAS], number(sample["synchronization_span_s"]),
            sample["sample_valid"], sample["frame_status"]])
        self.count += 1
        if sample["sample_valid"]:
            self.valid_count += 1
            self.first_t = t if self.first_t is None else self.first_t
            self.last_t = t
            self.max_span = max(self.max_span, sample["synchronization_span_s"])
        else:
            self.placeholder_count += 1
        self.writes.add(time.perf_counter(), self.count)

    def _manifest(self, status):
        manifest = {"schema": "human_capture.rgb.v1", "status": status,
            "task": self.task, "capture_index": self.capture_index, "reason": self.reason,
            "simulated": self.mock, "fps": self.cfg["record_hz"], "frames": self.count,
            "valid_frames": self.valid_count, "placeholder_frames": self.placeholder_count,
            "video_duration_s": self.count/self.cfg["record_hz"],
            "capture_duration_s": self.end_time-self.start_time if self.end_time is not None else None,
            "dropped_grid_ticks": self.dropped, "drop_reasons": dict(self.drop_reasons),
            "camera_diagnostics": self.camera_diagnostics,
            "writer_diagnostics": {"queue_capacity": self.queue.maxsize,
                "queue_high_watermark": self.queue_high_watermark,
                "mean_encode_ms": self.encode_total_s/max(1, self.count)*1000,
                "max_encode_ms": self.max_encode_s*1000},
            "effective_fps": ((self.valid_count-1)/(self.last_t-self.first_t)
                              if self.valid_count > 1 and self.last_t > self.first_t else 0.),
            "synchronization_max_span_s": 1/self.cfg["record_hz"],
            "observed_max_synchronization_span_s": self.max_span,
            "start_unix_s": self.start_time+self.unix_offset,
            "end_unix_s": self.end_time+self.unix_offset if self.end_time is not None else None,
            "videos": {n: f"{n}.mp4" for n in CAMERAS}, "timestamp_file": "timestamps.csv",
            "frame_alignment": "MP4 frame N = timestamps.csv data row N = samples.jsonl row N (zero-based)",
            "video_timeline": "wall-time CFR grid; gaps hold previous images (black before first frame); sample_valid=false; source timestamps unchanged",
            "synchronization": "SDK global/system time mapped to a common monotonic clock; software alignment",
            "config": self.config_snapshot, "error": self.error}
        temp = self.path/"manifest.tmp"
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path/"manifest.json")
