"""Client transport: image encoding, observation history, handshake and inference."""
import asyncio
from collections import deque
import hashlib
from pathlib import Path
import sys
import time

import av
import cv2
import numpy as np

from protocol import ARMS, CAMERAS, DIM, SEMANTICS, VERSION, actions, pack, stamp, unpack, vector
from timing import ClockEstimate

for project_root in Path(__file__).resolve().parents:
    if (project_root/'ace_training/preprocessing.py').is_file():
        sys.path.insert(0,str(project_root))
        break
from ace_training import preprocessing as shared_preprocessing


def encode_observation(observation, codec, quality, image_size=None):
    images = {}
    preprocess_ms = encode_ms = 0.
    for name in CAMERAS:
        rgb = observation['images'][name]
        if image_size is not None:
            begin = time.perf_counter()
            if isinstance(rgb,dict):
                frame = av.VideoFrame.from_ndarray(rgb['data'],format='yuv420p')
            else:
                frame = av.VideoFrame.from_ndarray(rgb,format='rgb24')
            rgb = shared_preprocessing.letterbox_rgb(frame,image_size)
            preprocess_ms += (time.perf_counter()-begin)*1000
        begin = time.perf_counter()
        if isinstance(rgb,dict):
            images[name] = rgb  # Lossless native YUV420p replay/consistency checks.
        elif codec == 'raw':
            images[name] = rgb
        else:
            suffix = '.jpg' if codec == 'jpeg' else '.png'
            options = ([cv2.IMWRITE_JPEG_QUALITY,quality] if codec == 'jpeg'
                       else [cv2.IMWRITE_PNG_COMPRESSION,1])
            okay,image = cv2.imencode(suffix,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),options)
            if not okay:
                raise RuntimeError(f'Could not encode {name} image')
            images[name] = image.tobytes()
        encode_ms += (time.perf_counter()-begin)*1000
    return images,preprocess_ms,encode_ms


class RemotePolicy:
    def __init__(self, websocket, metadata, timeout=5, codec='png', quality=90, allow_lossy=False):
        self.websocket,self.metadata = websocket,metadata
        self.timeout,self.codec,self.quality = timeout,codec,quality
        self.counter = 0
        self.clock = ClockEstimate()
        self.allow_lossy = allow_lossy
        self.n_obs_steps = metadata.get('n_obs_steps', 1)
        self.history = deque(maxlen=self.n_obs_steps)
        self.prepared = 'input_contract' in metadata
        if self.prepared:
            expected_hash = hashlib.sha256(Path(shared_preprocessing.__file__).read_bytes()).hexdigest()
            if metadata['input_contract'].get('preprocess_sha256') != expected_hash:
                raise ValueError('PC and server preprocessing source hashes differ; update the client bundle')
            if metadata['input_contract'].get('pyav_version') != av.__version__:
                raise ValueError('PC and server must use the same PyAV version for preprocessing')

    def observe(self, observation):
        """Collect camera/state snapshots at control ticks, independently of inference requests."""
        if observation is None:
            return None
        if self.history:
            gap = observation['observed_at'] - self.history[-1]['observed_at']
            if gap <= 0:
                return None
            if gap > 2 / self.metadata['control_hz']:
                self.history.clear()
        self.history.append(observation)
        if len(self.history) < self.n_obs_steps:
            return None
        return dict(observation, observation_history=list(self.history))

    async def synchronize_clock(self, samples=7):
        estimate = ClockEstimate()
        for _ in range(samples):
            self.counter += 1
            body = pack(dict(type='clock_sync',protocol_version=VERSION,request_id=self.counter))
            sent = time.perf_counter()
            await self.websocket.send(stamp(body,sent))
            data = await asyncio.wait_for(self.websocket.recv(),self.timeout)
            received = time.perf_counter()
            response = unpack(data)
            if response.get('type') != 'clock_sync' or response.get('request_id') != self.counter:
                raise RuntimeError('Clock synchronization response does not match request')
            estimate.update(sent,response['server_receive_s'],response['_sent_at'],received)
        self.clock = estimate

    async def infer(self, observation, anchor_tick):
        if self.codec=='jpeg' and not self.allow_lossy:
            raise ValueError('JPEG is lossy; use PNG/raw for training-consistent input')
        if self.clock.offset is None or time.perf_counter()-self.clock.updated_at>30:
            await self.synchronize_clock()
        self.counter += 1
        request_id = self.counter
        started = time.perf_counter()

        async def exchange():
            encode_start = time.perf_counter()
            def encode():
                samples = observation.get('observation_history') if self.n_obs_steps > 1 else [observation]
                if not isinstance(samples, list) or len(samples) != self.n_obs_steps:
                    raise ValueError(f'Expected {self.n_obs_steps} consecutive observations')
                history = []
                preprocess_ms = encode_ms = 0.
                for sample in samples:
                    images, pre_ms, enc_ms = encode_observation(sample, self.codec, self.quality,
                        self.metadata['image_size'] if self.prepared else None)
                    history.append(dict(state=vector(sample['state']), images=images))
                    preprocess_ms += pre_ms
                    encode_ms += enc_ms
                payload = history[-1] if self.n_obs_steps == 1 else dict(observation_history=history)
                return payload, preprocess_ms, encode_ms
            payload,preprocess_ms,encode_ms = await asyncio.to_thread(encode)
            encoded = time.perf_counter()
            body = pack(dict(type='infer',protocol_version=VERSION,
                request_id=request_id,observation_tick=anchor_tick,
                **payload,allow_lossy_images=self.allow_lossy,
                images_preprocessed=self.prepared,input_contract=self.metadata.get('input_contract')))
            sent = time.perf_counter()
            await self.websocket.send(stamp(body,sent))
            data = await self.websocket.recv()
            received = time.perf_counter()
            result = unpack(data)
            decoded = time.perf_counter()
            if result.get('type') == 'error':
                raise RuntimeError(result['error'])
            if result.get('type') != 'action' or result.get('request_id') != request_id:
                raise RuntimeError('Unexpected or stale inference response')
            if result.get('observation_tick') != anchor_tick:
                raise RuntimeError('Inference observation tick does not match request')
            result['actions'] = actions(result['actions'],self.metadata['horizon'])
            result['roundtrip_s'] = time.perf_counter()-started
            result['timing'] = dict(self.clock.estimate(sent,result['server_receive_s'],result['_sent_at'],received),
                client_preprocess_ms=preprocess_ms,image_encode_ms=encode_ms,
                client_prepare_wall_ms=(encoded-encode_start)*1000,request_pack_ms=(sent-encoded)*1000,
                response_unpack_ms=(decoded-received)*1000,request_bytes=len(body)+12,response_bytes=len(data),
                **observation.get('camera_timing',{}),**result.get('server_timing',{}))
            return result

        return await asyncio.wait_for(exchange(),timeout=self.timeout)


def validate_metadata(metadata, cfg, hz):
    expected = dict(type='hello',protocol_version=VERSION,action_dim=DIM,
                    action_semantics=SEMANTICS,camera_names=list(CAMERAS),control_hz=hz)
    for name,value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f'Server {name} mismatch: {metadata.get(name)!r} != {value!r}')
    if type(metadata.get('horizon')) is not int or not 1 <= metadata['horizon'] <= 1024:
        raise ValueError('Invalid server action horizon')
    n_obs_steps = metadata.get('n_obs_steps', 1)
    if type(n_obs_steps) is not int or not 1 <= n_obs_steps <= 16:
        raise ValueError('Invalid server observation history length')
    action_steps = metadata.get('action_steps', min(15, metadata['horizon']))
    if type(action_steps) is not int or not 1 <= action_steps <= metadata['horizon']:
        raise ValueError('Invalid server action execution steps')
    for side in ARMS:
        if not np.allclose(cfg['arms'][side]['tcp_offset'],metadata['tcp_offsets'][side],atol=1e-7):
            raise ValueError(f'{side} TCP offset differs from the training data')
    if 'input_contract' in metadata:
        contract = metadata['input_contract']
        if contract.get('preprocess') != 'yuv420p_resize_rgb24_letterbox_v1':
            raise ValueError('Server preprocessing differs from the trained input contract')
        for name in CAMERAS:
            shape = [cfg['cameras'][name]['height'],cfg['cameras'][name]['width'],3]
            if metadata['camera_shapes'][name] != shape:
                raise ValueError(f'{name} capture resolution differs from training')
