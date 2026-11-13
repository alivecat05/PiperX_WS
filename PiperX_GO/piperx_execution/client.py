"""PC-side WebSocket inference client for dual PiperX, with optional RTC."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import websockets

from control import CartesianFilter
from execution import control_loop
from hardware import MockHardware, PiperHardware
from protocol import MAX_BYTES, unpack
# Keep the original client imports available to scripts and offline checks.
from remote_policy import RemotePolicy, encode_observation, validate_metadata


async def wait_ready(hardware, timeout):
    until = time.perf_counter()+timeout
    last_error = None
    while True:
        try:
            hardware.read()
            hardware.check_cameras()
            if hardware.observation() is not None and hardware.enable():
                return
        except TimeoutError as error:
            last_error = error
        if time.perf_counter()>until:
            raise TimeoutError(f'PC hardware startup timeout: {last_error or "no synchronized observation/enable acknowledgement"}') from last_error
        await asyncio.sleep(1/30)


async def run(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    filter_config = CartesianFilter(cfg,args.control_hz).limits
    if (args.action_steps is not None and args.action_steps < 1) or args.control_hz != 30 or args.rtc_lookahead_steps < 1:
        raise ValueError('Use action_steps>=1, lookahead>=1 and the trained 30Hz control frequency')
    if args.blend_steps < 0 or args.max_steps < 0 or args.log_every < 1:
        raise ValueError('Invalid blend_steps, max_steps or log_every')
    if args.inference_timeout <= 0 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError('Invalid timeout or JPEG quality')
    token = os.environ.get(args.token_env)
    headers = {'Authorization':'Bearer '+token} if token else None
    hardware = None
    async with websockets.connect(args.server,compression=None,max_size=MAX_BYTES,
        max_queue=2,extra_headers=headers,open_timeout=args.inference_timeout) as websocket:
        metadata = unpack(await asyncio.wait_for(websocket.recv(),args.inference_timeout))
        validate_metadata(metadata,cfg,args.control_hz)
        if args.action_steps is None:
            args.action_steps = metadata.get('action_steps', 15)
            if not args.rtc:
                args.action_steps = min(args.action_steps, filter_config['sync_action_steps'])
        if args.action_steps > metadata['horizon']:
            raise ValueError('action-steps exceeds model horizon')
        print(json.dumps(dict(event='connected',checkpoint_step=metadata['checkpoint_step'],
            horizon=metadata['horizon'],rtc=args.rtc,execute=args.execute and not args.mock,
            policy=metadata.get('policy'),n_obs_steps=metadata.get('n_obs_steps',1),action_steps=args.action_steps,
            precision=metadata.get('precision'),input_contract=metadata.get('input_contract'),
            normalization_sha256=metadata.get('normalization_sha256'))),flush=True)
        print(json.dumps(dict(event='action_filter_config',blend_steps=args.blend_steps,
            **filter_config)),flush=True)
        policy = RemotePolicy(websocket,metadata,args.inference_timeout,args.image_codec,args.jpeg_quality,
                              args.allow_lossy_images)
        await policy.synchronize_clock()
        try:
            hardware = MockHardware(cfg) if args.mock else PiperHardware(cfg,args.teleop_root,args.execute)
            await wait_ready(hardware,args.startup_timeout)
            if args.execute and not args.mock:
                await hardware.go_home()
                policy.history.clear()
            result = await control_loop(hardware,policy,args,cfg)
            print(json.dumps(dict(event='complete',**result)),flush=True)
            return result
        finally:
            if hardware is not None:
                hardware.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',default='ws://127.0.0.1:8000')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.pc.json'))
    parser.add_argument('--teleop-root',type=Path,help='PC path to PiperX_WS/PiperX_Teleop')
    parser.add_argument('--rtc',dest='rtc',action='store_true',default=True)
    parser.add_argument('--no-rtc',dest='rtc',action='store_false')
    parser.add_argument('--action-steps',type=int,default=10,
                        help='Steps between RTC replans or maximum no-RTC execution prefix (default: 10)')
    parser.add_argument('--rtc-lookahead-steps',type=int,default=3)
    parser.add_argument('--blend-steps',type=int,default=5)
    parser.add_argument('--control-hz',type=int,default=30)
    parser.add_argument('--inference-timeout',type=float,default=5.)
    parser.add_argument('--startup-timeout',type=float,default=15.)
    parser.add_argument('--image-codec',choices=['jpeg','png','raw'],default='png')
    parser.add_argument('--allow-lossy-images',action='store_true',help='Explicitly permit JPEG input drift; PNG/raw are lossless')
    parser.add_argument('--jpeg-quality',type=int,default=90)
    parser.add_argument('--log-every',type=int,default=30)
    parser.add_argument('--max-steps',type=int,default=0,help='0 runs until Ctrl+C')
    parser.add_argument('--token-env',default='ACE_INFERENCE_TOKEN')
    parser.add_argument('--execute',action='store_true',help='Enable arms and dispatch model commands')
    parser.add_argument('--mock',action='store_true',help='No SDK/cameras; use mock hardware')
    args = parser.parse_args()
    if args.image_codec=='jpeg' and not args.allow_lossy_images:
        parser.error('JPEG changes model inputs; use --image-codec png/raw for consistency')
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print('Stopped; PC hardware closed.')


if __name__ == '__main__':
    main()
