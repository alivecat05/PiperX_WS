"""Standalone RealSense human demonstration recorder."""
import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time

from .cameras import Cameras, discover
from .controller import Controller
from .core import CAMERAS

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).resolve()
    source = path if path.exists() else ROOT/"config.example.json"
    if not path.exists() and path.name != "config.json":
        raise ValueError(f"配置文件不存在：{path}")
    cfg = json.loads(source.read_text(encoding="utf-8-sig"))
    if cfg["record_hz"] != 30:
        raise ValueError("当前采集频率固定为 30 Hz")
    if set(cfg["cameras"]) != set(CAMERAS):
        raise ValueError("需配置 head、left_wrist、right_wrist 三路相机")
    for name, camera in cfg["cameras"].items():
        if camera["fps"] != 30:
            raise ValueError(f"{name} 必须配置为 30 FPS")
        if any(type(camera[k]) is not int or camera[k] <= 0 or camera[k] % 2 for k in ("width", "height")):
            raise ValueError(f"{name} 的 MP4 分辨率必须为正偶数")
        if not camera["model"] or not isinstance(camera["serial"], str):
            raise ValueError(f"{name} 的型号或序列号格式无效")
    sync = cfg["sync"]
    if not (0 < sync["camera_tolerance_ms"] <= 1000/30+1e-8
            and 0 < sync["camera_span_ms"] <= 1000/30+1e-8
            and 1000/30 <= sync["delay_ms"] <= 500):
        raise ValueError("同步容差及跨度应在一帧内，延迟应为 33.33–500 ms")
    if sync["require_sdk_timestamps"] is not True:
        raise ValueError("真实录制必须使用 SDK global/system 时间戳")
    if not .1 <= cfg["camera_timeout_s"] <= 2:
        raise ValueError("相机超时应在 0.1–2 秒内")
    cfg = copy.deepcopy(cfg)
    pedal = cfg.setdefault("pedal", {})
    pedal.setdefault("key", None)
    pedal.setdefault("debounce_ms", 250)
    if (pedal["key"] is not None and (not isinstance(pedal["key"], str)
            or not pedal["key"].strip() or len(pedal["key"]) > 64)):
        raise ValueError("踏板 key 应为键名或 null，请使用界面识别踏板")
    if type(pedal["debounce_ms"]) not in (int, float) or not 30 <= pedal["debounce_ms"] <= 2000:
        raise ValueError("踏板 debounce_ms 应在 30–2000 毫秒内")
    cfg["output"] = str((path.parent/Path(cfg["output"])).resolve())
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "discover"), default="run")
    parser.add_argument("--config", default=str(ROOT/"config.json"))
    parser.add_argument("--mock", action="store_true", help="Use synthetic cameras; no hardware access")
    parser.add_argument("--smoke-seconds", type=float, help="Headless mock recording check; requires --mock")
    args = parser.parse_args()
    if args.smoke_seconds is not None and (not args.mock or not 0 < args.smoke_seconds <= 60):
        parser.error("--smoke-seconds requires --mock and a duration in (0, 60]")
    cameras = control = view = None
    try:
        cfg = load_config(args.config)
        if args.command == "discover":
            print(json.dumps(discover(cfg["realsense"]), ensure_ascii=False, indent=2))
            return 0
        log = ROOT/"outputs/logs/capture.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log, encoding="utf-8", level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s", force=True)
        cameras = Cameras(cfg, args.mock)
        cameras.start()
        control = Controller(cfg, cameras, args.mock)
        control.start()
        if args.smoke_seconds is not None:
            deadline = time.perf_counter()+5
            while not control.status()["ready"] and time.perf_counter() < deadline:
                time.sleep(.02)
            if not control.status()["ready"]:
                raise RuntimeError(str(control.status()["problems"]))
            control.request("record_start", selection={"task": cfg["task"],
                            "index": control.capture.index, "auto_increment": True})
            time.sleep(args.smoke_seconds)
            control.request("record_stop")
            time.sleep(.15)
            control.close()
            status = control.status()
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0 if control.recorder.result == "complete" else 1
        from .preview import Preview
        view = Preview(control, cameras, args.mock, config_path=args.config)
        view.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        logging.exception("HumanCapture failed")
        print(f"无法运行：{error}\n请检查 config.json、相机连接和 outputs/logs/capture.log", file=sys.stderr)
        return 1
    finally:
        if control:
            control.close()
        if cameras:
            cameras.close()
        if view:
            view.destroy()


if __name__ == "__main__":
    sys.exit(main())
