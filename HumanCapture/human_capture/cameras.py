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
        self.errors = {}
        self.diagnostics = {n: dict(received_frames=0, source_gaps=0, timeouts=0,
                                    reconnects=0, timestamp_rejections=0, max_receive_gap_ms=0.)
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

    def _stream(self, name):
        pipe = None
        try:
            cfg = self.cfg["cameras"][name]
            if self.mock:
                index = 0
                deadline = time.perf_counter()
                while not self.stop_event.is_set():
                    t = time.perf_counter()
                    image = np.zeros((cfg["height"], cfg["width"], 3), np.uint8)
                    cv2.putText(image, f"MOCK {name} {index}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                .8, (40, 210, 110), 2)
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
            profile = pipe.start(rs_cfg)  # Fail clearly if the selected profile isn't supported.
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
            previous_number = previous_receive = None
            consecutive_timeouts = 0
            while not self.stop_event.is_set():
                arrived, frames = pipe.try_wait_for_frames(500)
                if not arrived:
                    self.diagnostics[name]["timeouts"] += 1
                    consecutive_timeouts += 1
                    self.errors[name] = "等待相机帧，采集将自动恢复"
                    if consecutive_timeouts >= 4:
                        raise RuntimeError("No camera frames for 2 seconds")
                    continue
                consecutive_timeouts = 0
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
                stats["received_frames"] += 1
                if previous_number is not None:
                    stats["source_gaps"] += max(0, number-previous_number-1)
                    stats["max_receive_gap_ms"] = round(max(stats["max_receive_gap_ms"],
                                                            (receive-previous_receive)*1000), 1)
                previous_number, previous_receive = number, receive
                domain = frame.get_frame_timestamp_domain()
                device_ms = frame.get_timestamp()
                try:
                    stamp, basis = frame_timestamp(device_ms, domain, receive, wall_offset, cfg["latency_ms"])
                except RuntimeError as error:
                    stats["timestamp_rejections"] += 1
                    self.errors[name] = str(error)
                    continue
                self.sequences[name] += 1  # Hardware frame numbers may reset after reconnecting.
                self.histories[name].add(stamp, {"bgr": np.asanyarray(frame.get_data()).copy(),
                    "frame_number": number, "sequence": self.sequences[name], "device_ms": device_ms,
                    "exposure_locked": True, "depth_enabled": False,
                    "exposure_settings": self.exposure_settings[name],
                    "clock": str(domain), "timestamp_basis": basis, "receive_s": receive})
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
