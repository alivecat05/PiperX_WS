"""RGB capture with each RealSense SDK isolated from robot control and Quest input."""
import json
import logging
import math
import multiprocessing
import signal
import threading
import time

import cv2
import numpy as np

from .core import CAMERAS, History
from .rgb_controls import exposure_sensors, lock_exposure, prepare_rgb

LOG = logging.getLogger(__name__)


def _diagnostics():
    return dict(received_frames=0, source_gaps=0, timeouts=0, reconnects=0,
                timestamp_rejections=0, max_receive_gap_ms=0., delivery_drops=0)


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
        self.diagnostics = {n: _diagnostics() for n in CAMERAS}
        self.exposure_settings = {}
        self.stop_event = threading.Event()
        self.threads = []
        self.processes = []
        self.capture_stop = None

    def start(self):
        if self.mock:
            for name in CAMERAS:
                self._start_thread(self._mock_stream, name, name)
            return
        serials = [c["serial"] for c in self.cfg["cameras"].values()]
        if not all(serials) or len(set(serials)) != 3:
            raise ValueError("Assign three distinct RealSense serial numbers in config.json")
        ctx = multiprocessing.get_context("spawn")
        self.capture_stop = ctx.Event()
        for name in CAMERAS:
            cfg = self.cfg["cameras"][name]
            shape = (cfg["height"], cfg["width"], 3)
            buffer = ctx.RawArray("B", math.prod(shape))
            available = ctx.Semaphore(1)
            receiver, sender = ctx.Pipe(duplex=False)
            process = ctx.Process(target=_capture_camera, name=f"rgb-{name}", daemon=True,
                                  args=(cfg, self.cfg["realsense"], self.capture_stop,
                                        buffer, available, sender))
            try:
                process.start()
            except Exception:
                receiver.close()
                raise
            finally:
                sender.close()
            self.processes.append(process)
            try:
                self._start_thread(self._receive, name, name, receiver, buffer, shape, available)
            except Exception:
                receiver.close()
                raise

    def _start_thread(self, target, name, *args):
        thread = threading.Thread(target=target, args=args, daemon=True, name=name)
        thread.start()
        self.threads.append(thread)

    def _receive(self, name, receiver, buffer, shape, available):
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(shape)
        settings = None
        try:
            with receiver:
                while True:
                    stamp, frame, stats, error = receiver.recv()
                    if frame is not None:
                        try:
                            frame["bgr"] = image.copy()
                        finally:
                            available.release()
                        self.histories[name].add(stamp, frame)
                        if settings is None:
                            settings = self.exposure_settings[name] = frame["exposure_settings"]
                            LOG.info("%s RGB only; locked exposure/gain: %s", name, settings)
                    previous_reconnects = self.diagnostics[name]["reconnects"]
                    self.diagnostics[name] = stats
                    if error:
                        self.errors[name] = error
                        if stats["reconnects"] > previous_reconnects:
                            LOG.warning("%s stream interrupted; retrying: %s", name, error)
                    else:
                        self.errors.pop(name, None)
        except (EOFError, OSError):
            if not self.stop_event.is_set():
                self.errors[name] = "相机采集进程已退出"
                LOG.warning("%s capture process exited", name)

    def _mock_stream(self, name):
        cfg = self.cfg["cameras"][name]
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
            deadline = max(deadline + 1/cfg["fps"], time.perf_counter())
            self.stop_event.wait(max(0, deadline-time.perf_counter()))

    def close(self):
        self.stop_event.set()
        if self.capture_stop is not None:
            self.capture_stop.set()
        deadline = time.perf_counter() + 4
        for process in self.processes:
            process.join(timeout=max(0, deadline-time.perf_counter()))
        for process in self.processes:
            if process.is_alive():
                LOG.warning("%s did not exit; terminating camera process", process.name)
                process.terminate()
        for process in self.processes:
            process.join(timeout=1)
            if not process.is_alive():
                process.close()
        self.processes.clear()
        for thread in self.threads:
            thread.join(timeout=1)


def _capture_camera(cfg, settings, stop, buffer, available, sender):
    # SDK pipeline.start() can hold the GIL throughout a device reconnect.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    camera = _RealSenseCapture(cfg, stop, buffer, available, sender)
    with sender:
        context = None
        while not stop.is_set():
            try:
                if context is None:
                    context = rs_context(settings)
                    if cfg["model"] == "D555" and settings.get("dds", {}).get("enabled") and stop.wait(4):
                        return  # Allow DDS discovery in the camera process.
                if camera.is_d405 and camera.stats["reconnects"] and not camera.device_present(context):
                    stop.wait(.5)
                    continue  # Avoid the SDK's blocking device lookup while unplugged.
                camera.stream(context)
                return
            except (BrokenPipeError, EOFError):
                return
            except Exception as error:
                camera.stats["reconnects"] += 1
                camera.report(f"{type(error).__name__}: {error}")
                if stop.wait(1):
                    return


class _RealSenseCapture:
    def __init__(self, cfg, stop, buffer, available, sender):
        self.cfg, self.stop = cfg, stop
        self.is_d405 = cfg["model"] == "D405"
        self.available, self.sender = available, sender
        self.image = np.frombuffer(buffer, dtype=np.uint8).reshape(cfg["height"], cfg["width"], 3)
        self.sequence = 0
        self.exposure_settings = None
        self.stats = _diagnostics()

    def report(self, error):
        self.sender.send((None, None, self.stats, error))

    def device_present(self, context):
        import pyrealsense2 as rs
        serial = rs.camera_info.serial_number
        return any(device.supports(serial) and device.get_info(serial) == self.cfg["serial"]
                   for device in context.query_devices())

    def wait_for_frames(self, pipe):
        if not self.is_d405:
            return pipe.try_wait_for_frames(500)
        # try_wait_for_frames can spend much longer than its timeout reopening a lost USB device.
        deadline = time.perf_counter() + .5
        while not self.stop.is_set():
            frames = pipe.poll_for_frames()
            if frames:
                return True, frames
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            self.stop.wait(min(.004, remaining))
        return False, None

    def stream(self, context):
        pipe = None
        cfg = self.cfg
        try:
            import pyrealsense2 as rs
            pipe = rs.pipeline(context)
            rs_cfg = rs.config()
            prepare_rgb(rs_cfg, rs, cfg)
            profile = pipe.start(rs_cfg)  # Fail clearly if the selected profile isn't supported.
            actual = profile.get_device().get_info(rs.camera_info.name)
            if cfg["model"] not in actual:
                raise ValueError(f"expected {cfg['model']}, found {actual}")
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            sensors = exposure_sensors(profile, rs)
            if self.exposure_settings is not None:
                lock_exposure(sensors, rs, self.exposure_settings)
            else:
                for sensor in sensors:
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
            warmup_until = time.perf_counter()+2
            wall_offset = time.perf_counter() - time.time()
            previous_number = previous_receive = None
            consecutive_timeouts = 0
            stats = self.stats
            while not self.stop.is_set():
                arrived, frames = self.wait_for_frames(pipe)
                if not arrived:
                    if self.stop.is_set():
                        return
                    stats["timeouts"] += 1
                    consecutive_timeouts += 1
                    self.report("等待相机帧，采集将自动恢复")
                    if consecutive_timeouts >= (2 if self.is_d405 else 4):
                        raise RuntimeError(f"No camera frames for {consecutive_timeouts/2:g} seconds")
                    continue
                consecutive_timeouts = 0
                frame = frames.get_color_frame()
                if not frame:
                    continue
                receive = time.perf_counter()
                if self.exposure_settings is None:
                    if receive < warmup_until:
                        continue
                    self.exposure_settings = lock_exposure(sensors, rs, frame=frame, fps=cfg["fps"])
                number = frame.get_frame_number()
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
                    self.report(str(error))
                    continue
                self.sequence += 1  # Hardware frame numbers may reset after reconnecting.
                # Only overwrite the shared slot after the reader has copied it.
                if not self.available.acquire(block=False):
                    stats["delivery_drops"] += 1
                    continue
                try:
                    np.copyto(self.image, np.asanyarray(frame.get_data()))
                except Exception:
                    self.available.release()
                    raise
                self.sender.send((stamp, {
                    "frame_number": number, "sequence": self.sequence, "device_ms": device_ms,
                    "exposure_locked": True, "depth_enabled": False,
                    "exposure_settings": self.exposure_settings,
                    "clock": str(domain), "timestamp_basis": basis, "receive_s": receive}, stats, None))
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except RuntimeError:
                    pass
