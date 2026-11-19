"""One worker per RealSense; D555 DDS and D405 USB share the same context."""
import json
import logging
import math
import threading
import time

import cv2
import numpy as np

from .core import CAMERAS, History
from .rgb_controls import exposure_sensors, lock_exposure, prepare_rgb


def frame_timestamp(timestamp_ms, domain, receive, wall_offset, latency_ms=0):
    """Use SDK global/system timestamps; a raw hardware clock is not host Unix time."""
    domain = str(domain).rsplit(".", 1)[-1]
    if domain in ("global_time", "system_time"):
        stamp, basis = timestamp_ms / 1000 + wall_offset, domain
    else:
        stamp, basis = receive, "host_receive"
    stamp -= latency_ms / 1000
    if not np.isfinite(stamp) or abs(stamp-receive) > 1:
        raise RuntimeError("Camera timestamp mapping changed by >1 second or is invalid")
    return stamp, basis


def rs_context(settings):
    import pyrealsense2 as rs
    return rs.context(json.dumps(settings))


def discover(settings):
    import pyrealsense2 as rs
    context = rs_context(settings)
    time.sleep(4)  # Allow network/DDS discovery.
    result = []
    for device in context.query_devices():
        profiles = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color:
                    video = profile.as_video_stream_profile()
                    profiles.append(f"{video.width()}x{video.height()} {profile.fps()}fps {profile.format()}")
        result.append({"name": device.get_info(rs.camera_info.name),
                       "serial": device.get_info(rs.camera_info.serial_number),
                       "color_profiles": sorted(set(profiles))})
    return result


class Cameras:
    def __init__(self, cfg, mock=False):
        self.cfg, self.mock = cfg, mock
        # Keep enough history for delayed synchronization plus scheduler jitter.
        delay = cfg["sync"]["delay_ms"]/1000
        self.histories = {name: History(max(12, math.ceil((delay+.5)*cfg["cameras"][name]["fps"])))
                          for name in CAMERAS}
        # Host arrival rate must remain observable even if SDK timestamps fail.
        self.received = {name: History(120) for name in CAMERAS}
        self.errors = {}
        self.diagnostics = {n: dict(received_frames=0, source_gaps=0, timeouts=0,
                                    reconnects=0, timestamp_rejections=0, timestamp_regressions=0,
                                    max_receive_gap_ms=0., sdk_queue_depth=0, sdk_queue_peak=0,
                                    processing_ms=0., max_processing_ms=0.)
                            for n in CAMERAS}
        self.sequences = dict.fromkeys(CAMERAS, 0)
        self.exposure_settings = {}
        self.stop_event = threading.Event()
        self.threads = []
        self.context = None

    def start(self):
        if not self.mock:
            serials = [c["serial"] for c in self.cfg["cameras"].values()]
            if not all(serials) or len(set(serials)) != 3:
                raise ValueError("Assign three distinct RealSense serial numbers in config.json")
            self.context = rs_context(self.cfg["realsense"])
            time.sleep(4)
        for name in CAMERAS:
            thread = threading.Thread(target=self._worker, args=(name,), daemon=True, name=name)
            self.threads.append(thread)
            thread.start()

    def _worker(self, name):
        while not self.stop_event.is_set():
            try:
                self._stream(name)
                return
            except Exception as error:
                self.errors[name] = f"{type(error).__name__}: {error}"
                logging.getLogger(__name__).warning("%s stream interrupted; retrying: %s", name, error)
                self.diagnostics[name]["reconnects"] += 1
                if self.stop_event.wait(1.):
                    return

    def _note_received(self, name, receive, number):
        stats = self.diagnostics[name]
        previous = self.received[name].latest()
        stats["received_frames"] += 1
        if previous is not None:
            stats["source_gaps"] += max(0, number-previous.value-1)
            stats["max_receive_gap_ms"] = round(max(stats["max_receive_gap_ms"],
                                                      (receive-previous.t)*1000), 1)
        self.received[name].add(receive, number)

    def _stream(self, name):
        pipe = None
        try:
            cfg = self.cfg["cameras"][name]
            # Device counters can restart at an unrelated number on reconnect.
            # Compare numbers within this connection, preserving cumulative gaps.
            self.received[name] = History(120)
            if self.mock:
                index = 0
                deadline = time.perf_counter()
                while not self.stop_event.is_set():
                    t = time.perf_counter()
                    image = np.zeros((cfg["height"], cfg["width"], 3), np.uint8)
                    cv2.putText(image, f"MOCK {name} {index}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                .8, (40, 210, 110), 2)
                    self._note_received(name, t, index)
                    self.histories[name].add(t, {"bgr": image, "frame_number": index,
                                                "device_ms": t * 1000, "clock": "mock",
                                                "timestamp_basis": "host_monotonic", "receive_s": t})
                    index += 1
                    deadline += 1 / cfg["fps"]
                    if deadline < time.perf_counter():
                        deadline = time.perf_counter()
                    time.sleep(max(0, deadline-time.perf_counter()))
                return
            import pyrealsense2 as rs
            pipe = rs.pipeline(self.context)
            rs_cfg = rs.config()
            prepare_rgb(rs_cfg, rs, cfg)
            # The pipeline's default wait_for_frames queue holds only the latest
            # frameset. Buffer in the SDK (without a Python/GIL callback) so short
            # scheduling stalls do not discard every intermediate frame. Keep SDK
            # buffers alive only until the image is copied into our own history.
            capacity = max(4, min(16, math.ceil(self.cfg["sync"]["delay_ms"]*cfg["fps"]/1000)+2))
            frame_queue = rs.frame_queue(capacity, keep_frames=True)
            # The keyword selects the native queue overload, rather than the
            # earlier Python-callable overload (frame_queue is also callable).
            profile = pipe.start(rs_cfg, queue=frame_queue)
            self.diagnostics[name]["sdk_queue_capacity"] = capacity
            actual = profile.get_device().get_info(rs.camera_info.name)
            if cfg["model"] not in actual:
                raise ValueError(f"{name}: expected {cfg['model']}, found {actual}")
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            sensors = exposure_sensors(profile, rs)
            if name in self.exposure_settings:
                lock_exposure(sensors, rs, self.exposure_settings[name])
            else:
                for sensor in sensors:
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
            warmup_until = time.perf_counter()+2
            wall_offset = time.perf_counter() - time.time()
            consecutive_timeouts = 0
            while not self.stop_event.is_set():
                arrived, queued_frame = frame_queue.try_wait_for_frame(500)
                if not arrived:
                    self.diagnostics[name]["timeouts"] += 1
                    self.diagnostics[name]["sdk_queue_depth"] = 0
                    consecutive_timeouts += 1
                    self.errors[name] = "等待相机帧，采集将自动恢复"
                    if consecutive_timeouts >= 4:
                        raise RuntimeError("No camera frames for 2 seconds")
                    continue
                consecutive_timeouts = 0
                frames = queued_frame.as_frameset()
                frame = frames.get_color_frame()
                if not frame:
                    continue
                receive = time.perf_counter()
                if name not in self.exposure_settings:
                    if receive < warmup_until:
                        continue
                    self.exposure_settings[name] = lock_exposure(sensors, rs, frame=frame, fps=cfg["fps"])
                    logging.getLogger(__name__).info("%s RGB only; locked exposure/gain: %s",
                                                     name, self.exposure_settings[name])
                number = frame.get_frame_number()
                stats = self.diagnostics[name]
                self._note_received(name, receive, number)
                stats["sdk_queue_depth"] = frame_queue.size()
                stats["sdk_queue_peak"] = max(stats["sdk_queue_peak"], stats["sdk_queue_depth"])
                domain = frame.get_frame_timestamp_domain()
                device_ms = frame.get_timestamp()
                try:
                    stamp, basis = frame_timestamp(device_ms, domain, receive, wall_offset, cfg["latency_ms"])
                except RuntimeError as error:
                    stats["timestamp_rejections"] += 1
                    self.errors[name] = str(error)
                    continue
                self.sequences[name] += 1  # Hardware frame numbers may reset after reconnecting.
                accepted = self.histories[name].add(stamp, {"bgr": np.asanyarray(frame.get_data()).copy(),
                    "frame_number": number, "sequence": self.sequences[name], "device_ms": device_ms,
                    "exposure_locked": True, "depth_enabled": False,
                    "exposure_settings": self.exposure_settings[name],
                    "clock": str(domain), "timestamp_basis": basis, "receive_s": receive})
                stats["processing_ms"] = round((time.perf_counter()-receive)*1000, 2)
                stats["max_processing_ms"] = max(stats["max_processing_ms"], stats["processing_ms"])
                if not accepted:
                    stats["timestamp_regressions"] += 1
                    self.errors[name] = "相机时间戳未递增，帧已拒绝；详情见诊断日志"
                    continue
                self.errors.pop(name, None)
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except RuntimeError:
                    pass

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=4)
