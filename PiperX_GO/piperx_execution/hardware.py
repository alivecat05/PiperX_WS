"""PC-side bridge to the user's PiperX_Teleop SDK/camera adapters."""
from itertools import product
import asyncio
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from protocol import ARMS, CAMERAS, vector
from camera_capture import create_cameras
from homing import HomeSequence, load_home_config


class PiperHardware:
    def __init__(self, cfg, teleop_root, execute=False):
        if teleop_root:
            sys.path.insert(0,str(Path(teleop_root).resolve()))
        try:
            from piperx_teleop import robot as robot_module
            from piperx_teleop.robot import PiperX
            from piperx_teleop.core import History
        except ImportError as error:
            raise RuntimeError('Install PiperX_Teleop on the PC or pass --teleop-root') from error
        self.cfg, self.execute = cfg, execute
        self.teleop_root = (Path(teleop_root).resolve() if teleop_root else
                            Path(robot_module.__file__).resolve().parents[1])
        speed_cap = float(cfg.get('inference', {}).get('sdk_speed_percent', 20))
        if not np.isfinite(speed_cap) or not 0 < speed_cap <= 100 or int(speed_cap) != speed_cap:
            raise ValueError('inference.sdk_speed_percent must be an integer in [1,100]')
        self.sdk_speed_percent = {s: min(cfg['arms'][s]['speed_percent'], int(speed_cap)) for s in ARMS}
        self.robots = {}
        self.enabled = False
        self.command_started = False
        self.current = {}
        self._enable_started_at = None
        self._enable_ack = dict.fromkeys(ARMS, False)
        self._observation_after = float('-inf')
        self.histories = {s:History() for s in ARMS}
        self.cameras = create_cameras(cfg)
        try:
            for side in ARMS:
                arm_cfg = dict(cfg['arms'][side], speed_percent=self.sdk_speed_percent[side])
                self.robots[side] = PiperX(arm_cfg)
            self.cameras.start()
            print(json.dumps(dict(event='hardware_limits', sdk_speed_percent=self.sdk_speed_percent)), flush=True)
        except BaseException:
            self.close()
            raise

    def read(self):
        now = time.perf_counter()
        current = {}
        for side,robot in self.robots.items():
            try:
                sample = robot.read()
            except TimeoutError as error:
                raise TimeoutError(f'{side}: {error}') from error
            if sample is None:
                raise TimeoutError(f'{side} arm has no feedback')
            stamp,state = sample
            fields = state['source_timestamps']
            if not fields or any(not -.05 <= now-t <= self.cfg['feedback_timeout_s'] for t in fields.values()):
                raise TimeoutError(f'{side} arm feedback is stale')
            vector(np.tile(state['vector'],2))
            self.histories[side].add(stamp,state)
            current[side] = state
        self.current = current
        return vector(np.concatenate([current[s]['vector'] for s in ARMS]))

    def check_cameras(self):
        now = time.perf_counter()
        for name,history in self.cameras.histories.items():
            frame = history.latest()
            if frame is None or not -.05 <= now-frame.t <= self.cfg['camera_timeout_s']:
                raise TimeoutError(f'{name} camera is missing or stale: {self.cameras.errors.get(name, "")}')

    def observation(self):
        """Newest aligned RGB/state snapshot within one 30Hz training interval."""
        select_started = time.perf_counter()
        self.check_cameras()
        now = time.perf_counter()
        delay = self.cfg.get('sync',{}).get('delay_ms',80)/1000
        target = now-delay
        tolerance = min(1/30,self.cfg['sync']['camera_tolerance_ms']/1000)
        candidates = [[s for s in self.cameras.histories[n].snapshot()
                       if s.t >= self._observation_after and abs(s.t-target)<=tolerance]
                      for n in CAMERAS]
        if any(not values for values in candidates):
            return None
        groups = sorted(product(*candidates),key=lambda g:sum(abs(s.t-target) for s in g))
        for frames in groups:
            stamps = [s.t for s in frames]
            anchor = max(stamps)
            if max(stamps)-min(stamps) > tolerance+1e-6:
                continue
            states = [self.histories[s].select(anchor,tolerance) for s in ARMS]
            if any(s is None for s in states):
                continue
            if any(t < self._observation_after for s in states for t in s.value['source_timestamps'].values()):
                continue
            stamps += [t for s in states for t in s.value['source_timestamps'].values()]
            if max(stamps)-min(stamps)>1/30+1e-6:
                continue
            images = {name:cv2.cvtColor(s.value['bgr'],cv2.COLOR_BGR2RGB)
                      for name,s in zip(CAMERAS,frames)}
            timing = dict(camera_sdk_acquire_ms={n:s.value.get('sdk_acquire_ms') for n,s in zip(CAMERAS,frames)},
                camera_sdk_wait_ms={n:s.value.get('sdk_wait_ms') for n,s in zip(CAMERAS,frames)},
                camera_sdk_extract_ms={n:s.value.get('sdk_extract_ms') for n,s in zip(CAMERAS,frames)},
                camera_frame_age_ms={n:(time.perf_counter()-s.t)*1000 for n,s in zip(CAMERAS,frames)},
                camera_select_ms=(time.perf_counter()-select_started)*1000,camera_timing_source='realsense_sdk')
            return dict(observed_at=anchor,
                state=vector(np.concatenate([s.value['vector'] for s in states])),
                images=images,camera_timing=timing)
        return None

    def enable(self):
        if not self.execute:
            self.enabled = True
            return True
        if not self.command_started:
            self.command_started = True
            # Replace any previous motion target before motor enable.
            self.hold()
            self._enable_started_at = time.perf_counter()
        enabled = []
        for side in ARMS:
            issue = self.robots[side].cartesian_start_issue(self.current[side]['joints'])
            if issue:
                raise RuntimeError(f'{side}: {issue}')
            if not self._enable_ack[side]:
                self._enable_ack[side] = bool(self.robots[side].enable())
            state = self.current[side]
            enabled.append(self._enable_ack[side] and state.get('arm_enabled') is True
                           and state.get('arm_status') == 0 and state.get('err_code') == 0
                           and state['source_timestamps'].get('status', 0) > self._enable_started_at)
        self.enabled = all(enabled)
        if not self.enabled and time.perf_counter()-self._enable_started_at > 5:
            raise TimeoutError('PC arm enable did not produce fresh enabled NORMAL feedback within 5 seconds')
        return self.enabled

    async def go_home(self):
        """Execute the home saved by Teleop's X/Y flow before policy control."""
        if not self.execute:
            return
        arms, timeout = load_home_config(self.teleop_root/'config.json', self.cfg)
        self.read()
        self.check_cameras()
        sequence = HomeSequence(arms, timeout)
        for side in ARMS:
            robot = self.robots[side]
            for joints in [self.current[side]['joints'], *arms[side]['home_waypoints']]:
                issue = robot.cartesian_start_issue(joints)
                if issue:
                    raise ValueError(f'{side} home rejected before motion: {issue}')
        # Validate current states before issuing even the first hold/waypoint.
        # The temporary sequence is discarded so the real sequence starts below.
        HomeSequence(arms, timeout).step(self.current, time.perf_counter())
        if not self.enabled:
            raise RuntimeError('PC hardware must be enabled and ready before going home')
        previous_reset_speeds = {s: self.robots[s].cfg['reset_speed_percent'] for s in ARMS}
        print(json.dumps(dict(event='home_started', config=str(self.teleop_root/'config.json'),
                              reset_speed_percent={s: arms[s]['reset_speed_percent'] for s in ARMS})), flush=True)
        try:
            for side in ARMS:
                self.robots[side].cfg['reset_speed_percent'] = arms[side]['reset_speed_percent']
            self.hold()
            while not sequence.done:
                self.read()
                self.check_cameras()
                now = time.perf_counter()
                for side, index, joints in sequence.step(self.current, now):
                    self.robots[side].move_home(joints)
                    print(json.dumps(dict(event='home_waypoint', arm=side, index=index,
                                          joints=joints)), flush=True)
                if not sequence.done:
                    await asyncio.sleep(1/30)
            for side in ARMS:
                self.robots[side].set_motion_speed(self.sdk_speed_percent[side])
            # No pre-home camera/state sample may seed the first policy request.
            self._observation_after = time.perf_counter()
            for history in [*self.histories.values(), *self.cameras.histories.values()]:
                with history.lock:
                    history.items.clear()
            deadline = time.perf_counter()+timeout
            last_error = None
            while True:
                try:
                    self.read()
                    self.check_cameras()
                    if self.observation() is not None:
                        break
                except TimeoutError as error:
                    last_error = error
                if time.perf_counter() > deadline:
                    raise TimeoutError(f'Home finished but fresh synchronized observation is unavailable: {last_error}')
                await asyncio.sleep(1/30)
            print(json.dumps(dict(event='home_done', sdk_speed_percent=self.sdk_speed_percent)), flush=True)
        except BaseException:
            try:
                self.hold()
            except Exception:
                for robot in self.robots.values():
                    try:
                        robot.stop()
                    except Exception:
                        logging.exception('PiperX emergency stop after home failure failed')
            raise
        finally:
            for side in ARMS:
                self.robots[side].cfg['reset_speed_percent'] = previous_reset_speeds[side]

    def send(self, target):
        target = vector(target)
        if self.execute:
            if not self.enabled:
                raise RuntimeError('Arm enable acknowledgement is missing')
            self.command_started = True
            for side,start in zip(ARMS,(0,8)):
                self.robots[side].send(target[start:start+8].tolist())

    def hold(self):
        if not self.execute or not self.command_started:
            return
        self.read()
        for side,robot in self.robots.items():
            state = self.current[side]
            robot.hold(state['joints'])
            if state.get('gripper_enabled',False):
                robot.send_gripper(state['vector'][7])

    def close(self):
        # Normal exit/transport loss holds fresh measured joints without homing.
        if self.command_started:
            try:
                self.hold()
            except Exception:
                for robot in self.robots.values():
                    try:
                        robot.stop()
                    except Exception:
                        logging.exception('PiperX emergency stop failed')
        for robot in self.robots.values():
            try:
                robot.close()
            except Exception:
                logging.exception('PiperX disconnect failed')
        try:
            self.cameras.close()
        except Exception:
            logging.exception('Camera shutdown failed')


class MockHardware:
    """No SDK/RealSense imports; used for end-to-end network checks."""
    def __init__(self, cfg, state=None):
        self.cfg = cfg
        self.state = vector(state if state is not None else [.3,0,.3,0,0,0,1,.04]*2)
        self.sent = 0

    def read(self):
        return self.state.copy()

    def check_cameras(self):
        pass

    def observation(self):
        images = {}
        for index,name in enumerate(CAMERAS):
            config = self.cfg['cameras'][name]
            image = np.zeros((config['height'],config['width'],3),np.uint8)
            image[:,:,index] = 70
            images[name] = image
        return dict(observed_at=time.perf_counter(),state=self.read(),images=images,
                    camera_timing={'camera_sdk_acquire_ms':dict.fromkeys(CAMERAS,None),'camera_timing_source':'mock'})

    def enable(self):
        return True

    def send(self, target):
        self.state = vector(target).copy()
        self.sent += 1

    def hold(self):
        pass

    def close(self):
        pass
