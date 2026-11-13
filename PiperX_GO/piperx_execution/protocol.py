"""Binary WebSocket messages; NumPy representation compatible with openpi-client."""
import math
import struct
import time

import msgpack
import numpy as np

VERSION = 2
STAMP = b'PX02'
CAMERAS = ('head', 'left_wrist', 'right_wrist')
ARMS = ('left', 'right')
DIM = 16
MAX_BYTES = 16 * 1024 * 1024
SEMANTICS = 'absolute_tcp_xyz_quaternion_xyzw_gripper_m_each_arm_base'


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise ValueError('Only numeric arrays are supported')
        return {b'__ndarray__': True, b'data': value.tobytes(),
                b'dtype': value.dtype.str, b'shape': value.shape}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported message type: {type(value)}')


def _decode(value):
    if b'__ndarray__' not in value:
        return value
    dtype = np.dtype(value[b'dtype'])
    shape = value[b'shape']
    if dtype.kind not in 'biuf' or not isinstance(shape, (list, tuple)) or len(shape) > 5:
        raise ValueError('Invalid array dtype or dimensions')
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError('Invalid array shape')
    data = value[b'data']
    if math.prod(shape) * dtype.itemsize != len(data):
        raise ValueError('Array byte length does not match shape')
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def pack(message):
    data = msgpack.packb(message, default=_encode, use_bin_type=True)
    if len(data) > MAX_BYTES:
        raise ValueError('Message is too large')
    return data


def unpack(data):
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError('Expected a bounded binary WebSocket message')
    sent_at = None
    if data.startswith(STAMP):
        if len(data)<12:
            raise ValueError('Truncated timing envelope')
        sent_at = struct.unpack('!d',data[4:12])[0]
        if not math.isfinite(sent_at):
            raise ValueError('Invalid send timestamp')
        data = data[12:]
    value = msgpack.unpackb(data, object_hook=_decode, raw=False, strict_map_key=False)
    if not isinstance(value, dict):
        raise ValueError('Expected a message dictionary')
    if sent_at is not None:
        value['_sent_at'] = sent_at
    return value


def stamp(body, sent_at=None):
    if len(body)+12 > MAX_BYTES:
        raise ValueError('Message is too large')
    return STAMP+struct.pack('!d',time.perf_counter() if sent_at is None else sent_at)+body


def vector(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (DIM,) or not np.isfinite(value).all():
        raise ValueError('Expected a finite 16D dual-arm TCP vector')
    for start in (0, 8):
        if np.linalg.norm(value[start+3:start+7]) < 1e-6:
            raise ValueError('Zero quaternion')
    return value


def actions(value, horizon=None):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != DIM or not 1 <= len(value) <= 1024:
        raise ValueError('Expected action shape [horizon,16]')
    if horizon is not None and len(value) != horizon:
        raise ValueError('Action horizon differs from server metadata')
    if not np.isfinite(value).all():
        raise ValueError('Non-finite action')
    for start in (0, 8):
        if np.any(np.linalg.norm(value[:, start+3:start+7], axis=1) < 1e-6):
            raise ValueError('Zero action quaternion')
    return value
