"""Timestamp histories and unique-frame matching, following PiperX_Teleop rules."""
from collections import deque
from dataclasses import dataclass
from itertools import product
from threading import Lock

CAMERAS = ("head", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class Sample:
    t: float
    value: object


class History:
    def __init__(self, capacity=240):
        self.items = deque(maxlen=capacity)
        self.lock = Lock()

    def add(self, t, value):
        with self.lock:
            if not self.items or t > self.items[-1].t:
                self.items.append(Sample(t, value))

    def snapshot(self):
        with self.lock:
            return list(self.items)

    def latest(self):
        items = self.snapshot()
        return items[-1] if items else None

    def fps(self, now, window=2.):
        items = [x for x in self.snapshot() if now-window <= x.t <= now]
        if len(items) < 2 or now-items[-1].t > .5:
            return 0.
        return (len(items)-1)/(items[-1].t-items[0].t)


class Synchronizer:
    def __init__(self, histories, cfg, period=1/30):
        self.histories, self.cfg, self.period = histories, cfg, period
        self.last_frames = {}
        self.rejection = ""

    def sample(self, t):
        self.rejection = ""
        tolerance = min(self.period, self.cfg["camera_tolerance_ms"]/1000)
        candidates = {}
        for name in CAMERAS:
            near = [x for x in self.histories[name].snapshot()
                    if abs(x.t-t) <= tolerance+1e-9
                    and x.value["frame_number"] > self.last_frames.get(name, -1)]
            if self.cfg["require_sdk_timestamps"]:
                valid = [x for x in near if x.value["timestamp_basis"] != "host_receive"]
                if near and not valid:
                    self.rejection = f"camera_clock_not_synchronized:{name}"
                    return None
                near = valid
            if not near:
                self.rejection = f"camera_missing_or_reused:{name}"
                return None
            candidates[name] = near

        def score(group):
            stamps = [x.t for x in group]
            return (sum(stamps) if self.last_frames else sum(abs(x.t-t) for x in group),
                    max(stamps)-min(stamps))

        limit = min(self.period, self.cfg["camera_span_ms"]/1000)
        for group in sorted(product(*(candidates[n] for n in CAMERAS)), key=score):
            stamps = [x.t for x in group]
            if max(stamps)-min(stamps) > limit+1e-9:
                continue
            frames = dict(zip(CAMERAS, group))
            self.last_frames = {n: x.value["frame_number"] for n, x in frames.items()}
            return {"timestamp_monotonic_s": t, "synchronization_span_s": max(stamps)-min(stamps),
                    "source_timestamps": {n: x.t for n, x in frames.items()},
                    "camera_metadata": {n: {k: v for k, v in x.value.items() if k != "bgr"}
                                        for n, x in frames.items()},
                    "images": {n: x.value["bgr"] for n, x in frames.items()}}
        self.rejection = "timestamp_span_exceeds_one_frame"
        return None
