"""Sensor setup must exclude depth and freeze measured AE values, including reconnects."""
from types import SimpleNamespace as NS

import pytest

from human_capture.rgb_controls import exposure_sensors, lock_exposure, prepare_rgb


RS = NS(stream=NS(color="color", depth="depth"), format=NS(bgr8="bgr8"),
        option=NS(enable_auto_exposure="auto", exposure="exposure", gain="gain",
                  auto_exposure_priority="priority"))


class Sensor:
    def __init__(self, stream="color"):
        self.stream = stream
        self.values = {"auto": 1, "exposure": 1234., "gain": 17., "priority": 1}
        self.calls = []

    def get_stream_profiles(self):
        return [NS(stream_type=lambda: self.stream)]

    def supports(self, option):
        return option in self.values

    def get_option(self, option):
        return self.values[option]

    def set_option(self, option, value):
        self.calls.append((option, value))
        self.values[option] = value
        if option == "auto" and value == 0:
            # Model sensors that restore an old manual setting on AE disable.
            self.values.update(exposure=999., gain=9.)


def profile(sensors, stream="color"):
    return NS(get_streams=lambda: [NS(stream_type=lambda: stream)],
              get_device=lambda: NS(query_sensors=lambda: sensors))


def test_rgb_config_removes_all_default_streams_before_enabling_color():
    calls = []
    config = NS(disable_all_streams=lambda: calls.append(("disable_all",)),
                enable_device=lambda serial: calls.append(("device", serial)),
                enable_stream=lambda *args: calls.append(("stream", *args)))
    prepare_rgb(config, RS, dict(serial="camera", width=640, height=480, fps=30))
    assert calls == [("disable_all",), ("device", "camera"),
                     ("stream", "color", 640, 480, "bgr8", 30)]


def test_lock_reads_auto_values_before_disable_and_never_changes_depth_sensor():
    color, depth = Sensor(), Sensor("depth")
    sensors = exposure_sensors(profile([depth, color]), RS)
    assert sensors == [color] and depth.calls == []
    assert color.values["priority"] == 0
    values = lock_exposure(sensors, RS)
    assert values == [{"exposure": 1234., "gain": 17.}]
    assert color.values["auto"] == 0
    assert color.values["exposure"] == 1234. and color.values["gain"] == 17.
    # A reconnect retains the values captured before this episode.
    replacement = Sensor()
    replacement.values.update(exposure=4321., gain=21.)
    lock_exposure([replacement], RS, values)
    assert replacement.values["exposure"] == 1234. and replacement.values["gain"] == 17.
    assert ("auto", 1) not in replacement.calls


def test_depth_profile_and_unsupported_gain_cannot_silently_record():
    with pytest.raises(RuntimeError, match="RGB-only"):
        exposure_sensors(profile([Sensor()], stream="depth"), RS)
    sensor = Sensor()
    del sensor.values["gain"]
    with pytest.raises(RuntimeError, match="gain"):
        exposure_sensors(profile([sensor]), RS)


def test_failure_to_disable_auto_exposure_is_reported():
    sensor = Sensor()
    set_option = sensor.set_option
    def ignore_auto(option, value):
        if option != "auto":
            set_option(option, value)
    sensor.set_option = ignore_auto
    with pytest.raises(RuntimeError, match="auto exposure"):
        lock_exposure([sensor], RS)


def test_d405_lock_uses_applied_frame_values_and_leaves_readout_time():
    sensor = Sensor()
    sensor.values.update(exposure=33000., gain=16.)
    sensor.get_option_description = lambda _: "Depth Exposure (usec)"
    metadata = {"actual_exposure": 31979, "gain_level": 80}
    frame = NS(supports_frame_metadata=lambda key: key in metadata,
               get_frame_metadata=lambda key: metadata[key])
    rs = NS(**vars(RS), frame_metadata_value=NS(actual_exposure="actual_exposure", gain_level="gain_level"))
    values = lock_exposure([sensor], rs, frame=frame, fps=30)
    assert values == [{"exposure": 30000., "gain": 80.}]
    assert sensor.values["exposure"] == 30000. and sensor.values["gain"] == 80.
    assert sensor.values["auto"] == 0


def test_rgb_hundred_microsecond_units_are_not_confused_with_frame_microseconds():
    sensor = Sensor()
    sensor.values.update(exposure=156., gain=16.)
    sensor.get_option_description = lambda _: "Color exposure in 100 microseconds"
    frame = NS(supports_frame_metadata=lambda _: True, get_frame_metadata=lambda _: 31979)
    rs = NS(**vars(RS), frame_metadata_value=NS(actual_exposure="actual_exposure", gain_level="gain_level"))
    assert lock_exposure([sensor], rs, frame=frame, fps=30) == [{"exposure": 156., "gain": 16.}]


def test_d405_without_metadata_still_caps_default_exposure_for_requested_fps():
    sensor = Sensor()
    sensor.values.update(exposure=33000., gain=16.)
    sensor.get_option_description = lambda _: "Depth Exposure (usec)"
    frame = NS(supports_frame_metadata=lambda _: False)
    assert lock_exposure([sensor], RS, frame=frame, fps=30) == [{"exposure": 30000., "gain": 16.}]

def test_camera_worker_retries_instead_of_permanently_exiting(cfg, monkeypatch):
    from human_capture.cameras import Cameras
    cameras = Cameras(cfg)
    attempts = []
    def stream(name):
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("temporary SDK disconnect")
    monkeypatch.setattr(cameras, "_stream", stream)
    monkeypatch.setattr(cameras.stop_event, "wait", lambda _: False)
    cameras._worker("head")
    assert attempts == ["head", "head"]
    assert cameras.diagnostics["head"]["reconnects"] == 1


def test_camera_stream_locks_only_after_warmup_and_restores_on_reconnect(cfg, monkeypatch):
    import sys
    import numpy as np
    import human_capture.cameras as module
    color, depth = Sensor(), Sensor("depth")
    clock = [100.]
    config_calls = []
    stream_profile = profile([color, depth])
    device = NS(query_sensors=lambda: [color, depth], get_info=lambda _: "D555")
    stream_profile.get_device = lambda: device
    original_start = color.values.copy()
    camera = module.Cameras(cfg)
    name = "head"
    def run_stream():
        index = [0]
        def next_frame(_):
            index[0] += 1
            clock[0] += .5
            frame = NS(get_frame_number=lambda: index[0],
                get_frame_timestamp_domain=lambda: "global_time",
                get_timestamp=lambda: clock[0]*1000,
                get_data=lambda: np.zeros((48, 64, 3), np.uint8))
            if index[0] >= 5:
                camera.stop_event.set()
            return True, NS(get_color_frame=lambda: frame)
        pipe = NS(start=lambda _: stream_profile, stop=lambda: None,
                  try_wait_for_frames=next_frame)
        config = NS(disable_all_streams=lambda: config_calls.append("disable_all"),
                    enable_device=lambda _: None,
                    enable_stream=lambda stream, *args: config_calls.append(stream))
        fake_rs = NS(stream=RS.stream, format=RS.format,
            option=NS(**vars(RS.option), global_time_enabled="global"),
            camera_info=NS(name="name"), pipeline=lambda _: pipe, config=lambda: config)
        monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
        monkeypatch.setattr(module, "time", NS(perf_counter=lambda: clock[0], time=lambda: clock[0]))
        camera.stop_event.clear()
        camera._stream(name)
    run_stream()
    assert camera.exposure_settings[name] == [{"exposure": original_start["exposure"], "gain": original_start["gain"]}]
    assert camera.histories[name].latest().value["exposure_locked"]
    assert camera.histories[name].latest().value["depth_enabled"] is False
    assert camera.diagnostics[name]["received_frames"] == 2  # First 2 seconds are warmup.
    color.calls.clear()
    color.values.update(exposure=444., gain=5.)
    run_stream()
    assert ("auto", 1) not in color.calls
    assert color.values["exposure"] == original_start["exposure"]
    assert color.values["gain"] == original_start["gain"]
    assert config_calls == ["disable_all", "color", "disable_all", "color"]
    assert depth.calls == []
