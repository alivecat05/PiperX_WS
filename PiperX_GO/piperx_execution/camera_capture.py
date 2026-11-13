"""Measured RealSense acquisition using PiperX_Teleop's camera configuration.

Acquisition/exposure/timestamp behavior follows PiperX_WS commit
10a79ddac294ed5bc586c5e0c7fc8078094dcff5 (Apache-2.0).
Modified here to attach SDK wait/extraction timings to each individual frame.
"""
import logging
import time

import numpy as np


def create_cameras(cfg):
    from piperx_teleop.cameras import Cameras, frame_timestamp
    from piperx_teleop.rgb_controls import exposure_sensors, lock_exposure, prepare_rgb

    class MeasuredCameras(Cameras):
        def _stream(self, name):
            import pyrealsense2 as rs
            pipe = rs.pipeline(self.context)
            try:
                config = self.cfg['cameras'][name]
                stream_config = rs.config()
                prepare_rgb(stream_config,rs,config)
                profile = pipe.start(stream_config)
                actual = profile.get_device().get_info(rs.camera_info.name)
                if config['model'] not in actual:
                    raise ValueError(f"{name}: expected {config['model']}, found {actual}")
                for sensor in profile.get_device().query_sensors():
                    if sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled,1)
                sensors = exposure_sensors(profile,rs)
                if name in self.exposure_settings:
                    lock_exposure(sensors,rs,self.exposure_settings[name])
                else:
                    for sensor in sensors:
                        sensor.set_option(rs.option.enable_auto_exposure,1)
                warmup_until = time.perf_counter()+2
                wall_offset = time.perf_counter()-time.time()
                previous_number = previous_receive = None
                timeouts = 0
                while not self.stop_event.is_set():
                    begin = time.perf_counter()
                    arrived,frames = pipe.try_wait_for_frames(500)
                    waited = time.perf_counter()
                    if not arrived:
                        self.diagnostics[name]['timeouts'] += 1
                        timeouts += 1
                        self.errors[name] = 'Waiting for camera frame'
                        if timeouts >= 4:
                            raise RuntimeError('No camera frames for 2 seconds')
                        continue
                    timeouts = 0
                    frame = frames.get_color_frame()
                    if not frame:
                        continue
                    receive = time.perf_counter()
                    if name not in self.exposure_settings:
                        if receive < warmup_until:
                            continue
                        self.exposure_settings[name] = lock_exposure(sensors,rs,frame=frame,fps=config['fps'])
                        logging.info('%s RGB exposure/gain locked: %s',name,self.exposure_settings[name])
                    copied_at = time.perf_counter()
                    bgr = np.asanyarray(frame.get_data()).copy()
                    extracted = time.perf_counter()
                    number = frame.get_frame_number()
                    stats = self.diagnostics[name]
                    stats['received_frames'] += 1
                    if previous_number is not None:
                        stats['source_gaps'] += max(0,number-previous_number-1)
                        stats['max_receive_gap_ms'] = max(stats['max_receive_gap_ms'],(receive-previous_receive)*1000)
                    previous_number,previous_receive = number,receive
                    domain,device_ms = frame.get_frame_timestamp_domain(),frame.get_timestamp()
                    try:
                        stamp,basis = frame_timestamp(device_ms,domain,receive,wall_offset,config['latency_ms'])
                    except RuntimeError as error:
                        stats['timestamp_rejections'] += 1
                        self.errors[name] = str(error)
                        continue
                    self.sequences[name] += 1
                    wait_ms = (waited-begin)*1000
                    extract_ms = ((receive-waited)+(extracted-copied_at))*1000
                    self.histories[name].add(stamp,dict(bgr=bgr,frame_number=number,
                        sequence=self.sequences[name],device_ms=device_ms,
                        exposure_locked=True,depth_enabled=False,exposure_settings=self.exposure_settings[name],
                        clock=str(domain),timestamp_basis=basis,receive_s=receive,
                        sdk_wait_ms=wait_ms,sdk_extract_ms=extract_ms,sdk_acquire_ms=wait_ms+extract_ms))
                    self.errors.pop(name,None)
            finally:
                try:
                    pipe.stop()
                except RuntimeError:
                    pass

    return MeasuredCameras(cfg)
