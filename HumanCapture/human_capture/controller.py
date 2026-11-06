"""Acquisition runs independently of Tk rendering and MP4 encoding."""
import copy
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time

from .capture_session import CaptureSelection
from .core import CAMERAS, Synchronizer
from .recording import Recorder


class Controller:
    def __init__(self, cfg, cameras, mock=False):
        self.cfg, self.cameras = cfg, cameras
        self.recorder = Recorder(cfg["output"], cfg, mock)
        self.capture = CaptureSelection(self.recorder, cfg["task"])
        self.commands = Queue()
        self.stop_event = Event()
        self.lock = Lock()
        self.snapshot = {}
        self.thread = None
        self.recording = False
        self.end_time = None
        self.message = "等待三路相机就绪"
        self.last_result = None
        self._publish(time.perf_counter())

    def request(self, operation, **arguments):
        self.commands.put((operation, arguments))

    def start(self):
        self.thread = Thread(target=self._run, name="human-capture-sync", daemon=True)
        self.thread.start()

    def health(self, now):
        problems = []
        for n in CAMERAS:
            frame = self.cameras.histories[n].latest()
            if n in self.cameras.errors:
                problems.append(f"{n}：{self.cameras.errors[n]}")
            elif frame is None or now-frame.value["receive_s"] > self.cfg["camera_timeout_s"]:
                problems.append(f"等待 / 丢失相机：{n}")
            elif self.cfg["sync"]["require_sdk_timestamps"] and frame.value["timestamp_basis"] == "host_receive":
                problems.append(f"{n}：SDK 时间戳尚未同步（不能开始录制）")
        return problems

    def _commands(self, now):
        for _ in range(200):
            try:
                operation, args = self.commands.get_nowait()
            except Empty:
                break
            try:
                if operation == "record_toggle":
                    if self.recording and self.end_time is None:
                        operation = "record_stop"
                    elif self.recording or self.recorder.busy:
                        self.message = "正在保存，本次踩踏已忽略；保存完成后再踩开始下一段"
                        continue
                    else:
                        operation = "record_start"
                if operation in ("capture_settings", "next_index"):
                    if operation == "next_index":
                        self.capture.suggest(args["task"], args["auto_increment"])
                    else:
                        self.capture.apply(**args)
                    self.message = "任务和序号已同步"
                elif operation == "record_start":
                    if self.recording or self.recorder.busy:
                        raise ValueError("正在录制或保存，请等待")
                    # Desktop start always applies the actual form contents atomically.
                    self.capture.apply(**args["selection"])
                    problems = self.health(now)
                    if problems:
                        raise ValueError("；".join(problems))
                    self.capture.start(now)
                    self.sync = Synchronizer(self.cameras.histories, self.cfg["sync"])
                    self.start_time, self.grid_index = now, 0
                    self.end_time = None
                    self.recording = True
                    self.last_result = None
                    self.message = "正在录制三路同步视频"
                elif operation == "record_stop" and self.recording:
                    self.end_time = now if self.end_time is None else self.end_time
                    self.message = "正在对齐末尾帧并保存，请等待"
            except (ValueError, KeyError) as error:
                self.message = str(error)
                if operation in ("capture_settings", "next_index", "record_start"):
                    self.capture.error = self.message

    def tick(self, now):
        self._commands(now)
        if self.recording:
            problems = self.health(now)
            if problems or self.recorder.error or not self.recorder.active:
                self.message = "录制已停止："+("；".join(problems) or self.recorder.error or "写入线程停止")
                self.recorder.stop(self.message, complete=False, end_time=now)
                self.recording = False
            else:
                cutoff = now-self.cfg["sync"]["delay_ms"]/1000
                while self.start_time+self.grid_index/30 <= cutoff+1e-9:
                    t = self.start_time+self.grid_index/30
                    if self.end_time is not None and t >= self.end_time:
                        break
                    sample = self.sync.sample(t)
                    if sample is None:
                        self.recorder.drop(self.sync.rejection)
                    else:
                        self.recorder.add(sample, self.grid_index)
                    self.grid_index += 1
                if (self.end_time is not None and cutoff >= self.end_time
                        and self.start_time+self.grid_index/30 >= self.end_time):
                    self.recorder.stop(end_time=self.end_time)
                    self.recording = False
        self._check_result()
        self.capture.refresh()
        self._publish(now)

    def _check_result(self):
        if self.recorder.result is not None and self.recorder.result != self.last_result:
            self.last_result = self.recorder.result
            self.message = (f"已保存 {self.recorder.count} 帧：{self.recorder.path}"
                            if self.last_result == "complete" else
                            f"本段不完整（丢帧 {self.recorder.dropped}）：{self.recorder.error or self.recorder.reason}；{self.recorder.path}")

    def _publish(self, now):
        problems = self.health(now)
        state = {"capture": self.capture.snapshot(), "recording": self.recording and self.end_time is None,
                 "saving": (self.recording and self.end_time is not None) or (self.recorder.busy and not self.recording),
                 "busy": self.recording or self.recorder.busy,
                 "ready": not problems and not self.capture.error,
                 "problems": problems, "message": self.message, "frames": self.recorder.count,
                 "dropped": self.recorder.dropped, "drop_reasons": dict(self.recorder.drop_reasons),
                 "write_fps": self.recorder.writes.fps(now), "result": self.recorder.result,
                 "path": str(self.recorder.path or ""), "cameras": {}}
        for n in CAMERAS:
            frame = self.cameras.histories[n].latest()
            state["cameras"][n] = {"fps": self.cameras.histories[n].fps(now),
                "frame_number": frame.value["frame_number"] if frame else None,
                "clock": frame.value["timestamp_basis"] if frame else "等待相机"}
        with self.lock:
            self.snapshot = state

    def status(self):
        with self.lock:
            return copy.deepcopy(self.snapshot)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.tick(time.perf_counter())
            except Exception as error:
                self.message = f"采集错误：{type(error).__name__}: {error}"
                self.recording = False
                if self.recorder.busy:
                    self.recorder.stop(self.message, complete=False, end_time=time.perf_counter())
                self._publish(time.perf_counter())
            self.stop_event.wait(.005)

    def close(self):
        # Let the normal stop path align the final 80 ms before stopping the producer.
        self.request("record_stop")
        deadline = time.perf_counter()+2
        while self.thread and self.thread.is_alive() and self.recording and time.perf_counter() < deadline:
            time.sleep(.01)
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.recorder.busy and not self.recorder.finished.is_set():
            self.recorder.stop("application_exit", complete=False, end_time=time.perf_counter())
        self.recorder.wait()
        self._check_result()
        self.capture.refresh()
        self._publish(time.perf_counter())
