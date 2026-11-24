"""Read the three RealSense RGB intrinsics using an existing config/manifest."""
import argparse
import json
from pathlib import Path
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Capture config.json or episode manifest.json")
    parser.add_argument("--output", type=Path, default=Path("calibration/intrinsics.json"))
    args = parser.parse_args(argv)
    document = json.loads(args.config.read_text(encoding="utf-8-sig"))
    config = document["config"] if "schema" in document else document
    names = ("head", "left_wrist", "right_wrist")
    serials = [config["cameras"][name]["serial"] for name in names]
    if not all(serials) or len(set(serials)) != 3:
        raise ValueError("Configure three distinct camera serial numbers")
    import pyrealsense2 as rs
    context = rs.context(json.dumps(config["realsense"]))
    if config["realsense"].get("dds", {}).get("enabled"):
        time.sleep(4)  # Same discovery interval as the existing capture code.
    result = {}
    for name in names:
        camera = config["cameras"][name]
        pipeline, settings = rs.pipeline(context), rs.config()
        settings.disable_all_streams()
        settings.enable_device(camera["serial"])
        settings.enable_stream(rs.stream.color, camera["width"], camera["height"], rs.format.bgr8, camera["fps"])
        profile = pipeline.start(settings)
        try:
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            result[name] = {
                "serial": profile.get_device().get_info(rs.camera_info.serial_number),
                "width": intr.width, "height": intr.height,
                "camera_matrix": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                "dist_coeffs": list(intr.coeffs),
                "distortion_model": str(intr.model).split(".")[-1],
            }
            print(f"{name}: {intr.width}x{intr.height}, {intr.model}")
        finally:
            pipeline.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
