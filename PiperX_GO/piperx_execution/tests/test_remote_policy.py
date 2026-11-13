"""Exercise the client/server module boundary over real local WebSocket frames."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import numpy as np
import pytest
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client import RemotePolicy, control_loop, encode_observation, run, validate_metadata
from hardware import MockHardware
from image_inputs import ImageInputs
from protocol import CAMERAS, DIM, SEMANTICS, VERSION, unpack
from ace_training import preprocessing as shared_preprocessing
from server import PolicyServer


def config():
    path = Path(__file__).resolve().parents[1] / 'config.pc.json'
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    for camera in cfg['cameras'].values():
        camera.update(height=24, width=32)
    return cfg


class EchoBackend(ImageInputs):
    """Decode the real image contract and return deterministic, valid TCP targets."""
    def __init__(self, cfg, n_obs_steps):
        self.image_size = (12, 16)
        self.resize = shared_preprocessing.letterbox_rgb
        self.requests = []
        self.metadata = dict(
            type='hello', protocol_version=VERSION, action_dim=DIM,
            action_semantics=SEMANTICS, control_hz=30, camera_names=list(CAMERAS),
            horizon=30, n_obs_steps=n_obs_steps, action_steps=4,
            image_size=list(self.image_size), checkpoint_step=0,
            tcp_offsets={side: arm['tcp_offset'] for side, arm in cfg['arms'].items()},
            camera_shapes={name: [24, 32, 3] for name in CAMERAS},
            input_contract=dict(
                preprocess=shared_preprocessing.IMAGE_PREPROCESS,
                preprocess_sha256=hashlib.sha256(Path(shared_preprocessing.__file__).read_bytes()).hexdigest(),
                pyav_version=av.__version__),
        )

    def infer(self, request):
        self.requests.append(request)
        assert request['images_preprocessed'] is True
        assert request['input_contract'] == self.metadata['input_contract']
        history = request.get('observation_history', [request])
        assert len(history) == self.metadata['n_obs_steps']
        for sample in history:
            for name in CAMERAS:
                decoded = self._image(sample['images'][name], expected_shape=[12, 16, 3], prepared=True)
                assert decoded.dtype == np.uint8
        return np.tile(history[-1]['state'], (30, 1))


@pytest.mark.parametrize('rtc', [False, True])
@pytest.mark.parametrize('n_obs_steps', [1, 2])
@pytest.mark.parametrize('codec', ['png', 'raw'])
def test_websocket_handshake_clock_history_and_execution(rtc, n_obs_steps, codec):
    async def scenario():
        cfg = config()
        backend = EchoBackend(cfg, n_obs_steps)
        service = PolicyServer(backend, token='local-test')
        try:
            async with websockets.serve(service.handler, '127.0.0.1', 0,
                                        process_request=service.process_request, compression=None) as listener:
                port = listener.sockets[0].getsockname()[1]
                async with websockets.connect(f'ws://127.0.0.1:{port}', compression=None,
                                              extra_headers={'Authorization': 'Bearer local-test'}) as socket:
                    metadata = unpack(await socket.recv())
                    validate_metadata(metadata, cfg, 30)
                    policy = RemotePolicy(socket, metadata, codec=codec)
                    await policy.synchronize_clock()
                    assert policy.clock.offset is not None
                    hardware = MockHardware(cfg)
                    args = SimpleNamespace(control_hz=30, blend_steps=5, max_steps=9,
                                           rtc_lookahead_steps=3, rtc=rtc, action_steps=4, log_every=30)
                    result = await control_loop(hardware, policy, args, cfg)
                    assert result['control_steps'] == hardware.sent == 9
                    assert result['mode'] == ('rtc' if rtc else 'no_rtc')
                    assert result['inferences'] >= 2
                    assert result['expired_chunks'] == 0
                    requests = backend.requests
                    assert all(request['protocol_version'] == VERSION for request in requests)
                    assert all(request['_sent_at'] > 0 for request in requests)
                    assert all(a['request_id'] < b['request_id'] and
                               a['observation_tick'] < b['observation_tick']
                               for a, b in zip(requests, requests[1:]))
                    if n_obs_steps > 1:
                        assert all('state' not in request and 'images' not in request for request in requests)
        finally:
            service.executor.shutdown(wait=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_lossless_codecs_preserve_preprocessed_pixels():
    cfg = config()
    backend = EchoBackend(cfg, 1)
    observation = MockHardware(cfg).observation()
    rng = np.random.default_rng(42)
    observation['images'] = {name: rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for name in CAMERAS}
    png, _, _ = encode_observation(observation, 'png', 90, backend.image_size)
    raw, _, _ = encode_observation(observation, 'raw', 90, backend.image_size)
    for name in CAMERAS:
        np.testing.assert_array_equal(backend._image(png[name], prepared=True), raw[name])


def test_history_ignores_duplicates_and_restarts_after_a_gap():
    backend = EchoBackend(config(), 2)
    policy = RemotePolicy(None, backend.metadata)
    first = dict(observed_at=1.)
    assert policy.observe(first) is None
    assert policy.observe(copy.copy(first)) is None
    second = dict(observed_at=1. + 1 / 30)
    assert policy.observe(second)['observation_history'] == [first, second]
    assert policy.observe(dict(observed_at=2.)) is None
    assert len(policy.history) == 1


def test_handshake_rejects_mismatched_preprocessing_before_inference():
    metadata = EchoBackend(config(), 1).metadata
    metadata['input_contract']['preprocess_sha256'] = 'different-source'
    with pytest.raises(ValueError, match='hashes differ'):
        RemotePolicy(None, metadata)


@pytest.mark.parametrize('rtc', [False, True])
def test_mock_client_lifecycle_and_metadata_action_steps(tmp_path, monkeypatch, rtc):
    cfg = config()
    cfg['inference']['sync_action_steps'] = 2
    config_path = tmp_path / 'config.pc.json'
    config_path.write_text(json.dumps(cfg), encoding='utf-8')
    monkeypatch.setenv('PIPERX_TEST_TOKEN', 'local-test')

    async def scenario():
        backend = EchoBackend(cfg, 2)
        service = PolicyServer(backend, token='local-test')
        try:
            async with websockets.serve(service.handler, '127.0.0.1', 0,
                                        process_request=service.process_request) as listener:
                port = listener.sockets[0].getsockname()[1]
                args = SimpleNamespace(
                    config=config_path, control_hz=30, action_steps=None, rtc=rtc,
                    rtc_lookahead_steps=3, blend_steps=5, max_steps=6, log_every=30,
                    inference_timeout=5., jpeg_quality=90, token_env='PIPERX_TEST_TOKEN',
                    server=f'ws://127.0.0.1:{port}', execute=False, mock=True,
                    image_codec='png', allow_lossy_images=False, startup_timeout=5.,
                )
                result = await run(args)
                assert result['control_steps'] == 6
                assert args.action_steps == (4 if rtc else 2)
                assert result['expired_chunks'] == 0
        finally:
            service.executor.shutdown(wait=True)

    asyncio.run(asyncio.wait_for(scenario(), timeout=10))
