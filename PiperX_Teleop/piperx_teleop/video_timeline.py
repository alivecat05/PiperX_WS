"""Keep CFR videos on the recording clock; placeholders never claim fresh observations."""
import math

import numpy as np


def grid_count(start, end, fps):
    return max(0, math.ceil((end-start)*fps-1e-6))


class VideoTimeline:
    def __init__(self, cfg, start, *, robot=False):
        self.cfg, self.start, self.robot = cfg, start, robot
        self.next_grid = 0
        self.previous = None
        self.source_grid = None
        self.blank_images = None

    def _gap(self):
        t = self.start+self.next_grid/self.cfg["record_hz"]
        if self.previous is None:
            if self.blank_images is None:
                self.blank_images = {n: np.zeros((c["height"], c["width"], 3), np.uint8)
                                     for n, c in self.cfg["cameras"].items()}
            sample = {
                "images": self.blank_images,
                "source_timestamps": ({"cameras": {}, "states": {}, "actions": {}} if self.robot
                                      else {n: None for n in self.cfg["cameras"]}),
                "camera_metadata": {n: {"frame_number": None, "timestamp_basis": "missing",
                                        "receive_s": None} for n in self.cfg["cameras"]},
                "synchronization_span_s": None,
            }
            if self.robot:
                sample.update(action=[None]*16, action_timestamp_monotonic_s=None,
                              **{"observation.state": [None]*16, "joints": {}})
        else:
            sample = dict(self.previous)
            if self.robot:
                sample.setdefault("action_timestamp_monotonic_s", self.previous["timestamp_monotonic_s"])
        sample.update(timestamp_monotonic_s=t, sample_valid=False,
                      frame_status="held" if self.previous is not None else "blank",
                      held_from_grid=self.source_grid)
        grid = self.next_grid
        self.next_grid += 1
        return sample, grid

    def accept(self, sample, grid):
        if grid < self.next_grid:
            raise ValueError("Video grid indexes must increase")
        while self.next_grid < grid:
            yield self._gap()
        self.previous, self.source_grid = sample, grid
        self.next_grid = grid+1
        yield dict(sample, sample_valid=True, frame_status="captured", held_from_grid=None), grid

    def finish(self, count):
        while self.next_grid < count:
            yield self._gap()
