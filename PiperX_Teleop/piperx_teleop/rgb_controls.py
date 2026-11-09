"""RGB stream and exposure setup; values remain in each sensor's native SDK units."""
import math


def prepare_rgb(config, rs, cfg):
    config.disable_all_streams()
    config.enable_device(cfg["serial"])
    config.enable_stream(rs.stream.color, cfg["width"], cfg["height"], rs.format.bgr8, cfg["fps"])


def exposure_sensors(profile, rs):
    if any(p.stream_type() != rs.stream.color for p in profile.get_streams()):
        raise RuntimeError("Expected RGB-only profile; depth/other streams must be disabled")
    sensors = [s for s in profile.get_device().query_sensors()
               if any(p.stream_type() == rs.stream.color for p in s.get_stream_profiles())]
    if not sensors:
        raise RuntimeError("No RGB exposure sensor found")
    for sensor in sensors:
        for option in (rs.option.enable_auto_exposure, rs.option.exposure, rs.option.gain):
            if not sensor.supports(option):
                raise RuntimeError(f"RGB sensor cannot lock {option}")
        if sensor.supports(rs.option.auto_exposure_priority):
            sensor.set_option(rs.option.auto_exposure_priority, 0)
    return sensors


def exposure_snapshot(sensors, rs, frame=None, fps=None):
    values = []
    for sensor in sensors:
        fixed = {"exposure": float(sensor.get_option(rs.option.exposure)),
                 "gain": float(sensor.get_option(rs.option.gain))}
        description = (sensor.get_option_description(rs.option.exposure).lower()
                       if hasattr(sensor, "get_option_description") else "")
        microseconds = (("usec" in description or "microsecond" in description)
                        and "100" not in description and "hundred" not in description)
        # D405 get_option(exposure) can retain the manual default (33000) while
        # AE is actually using a different value. Frame metadata reports AE's
        # applied value. Only use it when the sensor option has matching units.
        if microseconds and frame is not None and len(sensors) == 1:
            for key, metadata in (("exposure", "actual_exposure"), ("gain", "gain_level")):
                enum = getattr(getattr(rs, "frame_metadata_value", None), metadata, None)
                if enum is not None and frame.supports_frame_metadata(enum):
                    measured = float(frame.get_frame_metadata(enum))
                    if math.isfinite(measured) and measured > 0:
                        fixed[key] = measured
        if microseconds and fps is not None:
            # Leave 10% of the requested frame period for sensor readout. On
            # the connected D405, manual 33000 us halved a 30 Hz stream to 15 Hz.
            maximum = .9*1_000_000/fps
            if hasattr(sensor, "get_option_range"):
                option_range = sensor.get_option_range(rs.option.exposure)
                maximum = min(option_range.max, max(option_range.min, maximum))
                if option_range.step > 0:
                    maximum = option_range.min+math.floor((maximum-option_range.min)/option_range.step)*option_range.step
            fixed["exposure"] = min(fixed["exposure"], maximum)
        values.append(fixed)
    return values


def lock_exposure(sensors, rs, values=None, *, frame=None, fps=None):
    # Read before disabling auto exposure; some sensors restore old manual values
    # when auto exposure is disabled. Reapply both measured values explicitly.
    if values is None:
        values = exposure_snapshot(sensors, rs, frame, fps)
    if len(values) != len(sensors):
        raise RuntimeError("RGB sensor layout changed; cannot restore locked exposure")
    for sensor, fixed in zip(sensors, values):
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, fixed["exposure"])
        sensor.set_option(rs.option.gain, fixed["gain"])
        if sensor.get_option(rs.option.enable_auto_exposure) != 0:
            raise RuntimeError("RGB auto exposure did not turn off")
        for key, option in (("exposure", rs.option.exposure), ("gain", rs.option.gain)):
            actual = float(sensor.get_option(option))
            if abs(actual-fixed[key]) > max(1e-6, abs(fixed[key])*1e-5):
                raise RuntimeError(f"RGB {key} did not lock: requested {fixed[key]}, got {actual}")
    return values
