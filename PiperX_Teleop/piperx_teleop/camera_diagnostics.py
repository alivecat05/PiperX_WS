"""Measure real RGB delivery before/after exposure lock; never connects to a robot."""
import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np

from .cameras import rs_context
from .rgb_controls import exposure_sensors, lock_exposure, prepare_rgb


def measure(pipe, duration):
    received, stamps, numbers, metadata = [], [], [], {}
    deadline = time.perf_counter()+duration
    while time.perf_counter() < deadline:
        arrived, frames = pipe.try_wait_for_frames(500)
        if not arrived:
            continue
        frame = frames.get_color_frame()
        if not frame:
            continue
        received.append(time.perf_counter())
        stamps.append(frame.get_timestamp())
        numbers.append(frame.get_frame_number())
        import pyrealsense2 as rs
        for key in ("actual_exposure", "gain_level", "actual_fps", "frame_timestamp", "sensor_timestamp"):
            option = getattr(rs.frame_metadata_value, key, None)
            if option is not None and frame.supports_frame_metadata(option):
                metadata[key] = frame.get_frame_metadata(option)
    gaps = np.diff(received)*1000
    return {"frames": len(received),
            "host_fps": (len(received)-1)/(received[-1]-received[0]) if len(received)>1 else 0.,
            "device_timestamp_fps": ((len(stamps)-1)*1000/(stamps[-1]-stamps[0])
                                     if len(stamps)>1 and stamps[-1]>stamps[0] else 0.),
            "frame_number_gaps": sum(max(0, b-a-1) for a, b in zip(numbers, numbers[1:])),
            "receive_gap_p95_ms": float(np.percentile(gaps, 95)) if len(gaps) else None,
            "receive_gap_max_ms": float(max(gaps)) if len(gaps) else None,
            "last_metadata_raw": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]/"config.json"))
    parser.add_argument("--camera", choices=("left_wrist", "right_wrist"), required=True)
    parser.add_argument("--seconds", type=float, default=3.)
    parser.add_argument("--exposures", type=float, nargs="+", default=[33000., 20000., 10000.])
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 10:
        parser.error("--seconds must be in [1, 10]")
    try:
        connection = socket.create_connection(("127.0.0.1", args.port), timeout=.3)
    except OSError:
        pass
    else:
        connection.close()
        parser.error("遥操服务仍在运行，请先停止保存并正常关闭窗口，再做相机单测")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    camera = cfg["cameras"][args.camera]
    if "D405" not in camera["model"]:
        parser.error("Exposure values in this diagnostic are for D405 microsecond units")
    import pyrealsense2 as rs
    context = rs_context(cfg["realsense"])
    time.sleep(4)
    pipe = rs.pipeline(context)
    config = rs.config()
    prepare_rgb(config, rs, camera)
    profile = pipe.start(config)
    original, sensors = [], []
    try:
        sensors = exposure_sensors(profile, rs)
        original = [{"auto": s.get_option(rs.option.enable_auto_exposure),
                     "exposure": s.get_option(rs.option.exposure), "gain": s.get_option(rs.option.gain)}
                    for s in sensors]
        result = {"camera": args.camera, "serial": camera["serial"],
                  "active_profiles": [str(p) for p in profile.get_streams()], "cases": []}
        for sensor in sensors:
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        measure(pipe, 2.)
        result["cases"].append(dict(mode="auto", **measure(pipe, args.seconds)))
        for exposure in args.exposures:
            values = [{"exposure": exposure, "gain": s.get_option(rs.option.gain)} for s in sensors]
            lock_exposure(sensors, rs, values)
            measure(pipe, 1.)
            result["cases"].append(dict(mode="manual", settings=values, **measure(pipe, args.seconds)))
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded, encoding="utf-8")
        print(encoded)
    finally:
        try:
            for sensor, values in zip(sensors, original):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, values["exposure"])
                sensor.set_option(rs.option.gain, values["gain"])
                sensor.set_option(rs.option.enable_auto_exposure, values["auto"])
        finally:
            pipe.stop()


if __name__ == "__main__":
    main()
