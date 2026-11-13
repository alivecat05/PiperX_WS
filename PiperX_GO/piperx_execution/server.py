"""Serve a cleanup ACT or Diffusion Policy checkpoint over binary WebSocket."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import hmac
import http
import json
import os
from pathlib import Path
import time

import websockets

from protocol import MAX_BYTES, VERSION, pack, stamp, unpack


class PolicyServer:
    def __init__(self, backend, token=None):
        self.backend, self.token = backend, token
        self.lock = asyncio.Lock()
        # One inference thread serializes GPU use across sessions.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-inference')

    async def process_request(self, path, headers):
        if self.token and not hmac.compare_digest(headers.get('Authorization',''), 'Bearer '+self.token):
            return http.HTTPStatus.UNAUTHORIZED, [], b'Unauthorized\n'
        if path == '/healthz':
            return http.HTTPStatus.OK, [('Content-Type','application/json')], b'{"ready":true}\n'

    async def handler(self, websocket, path=None):
        await websocket.send(pack(self.backend.metadata))
        try:
            async for data in websocket:
                received = time.perf_counter()
                request = unpack(data)
                request_id = request.get('request_id')
                if request.get('type') == 'clock_sync':
                    body = pack(dict(type='clock_sync',protocol_version=VERSION,
                        request_id=request_id,server_receive_s=received))
                    await websocket.send(stamp(body))
                    continue
                if request.get('type') != 'infer' or request.get('protocol_version') != VERSION:
                    raise ValueError('Unsupported request type or protocol version')
                if type(request_id) is not int or request_id < 1 or type(request.get('observation_tick')) is not int:
                    raise ValueError('Expected integer request_id and observation_tick')
                started = time.perf_counter()
                async with self.lock:
                    queued = time.perf_counter()
                    def infer():
                        if hasattr(self.backend,'infer_timed'):
                            return self.backend.infer_timed(request)
                        start = time.perf_counter()
                        result = self.backend.infer(request)
                        return result,{'model_infer_ms':None,'backend_total_ms':(time.perf_counter()-start)*1000}
                    result,timings = await asyncio.get_running_loop().run_in_executor(self.executor,infer)
                timing = dict(request_decode_ms=(started-received)*1000,queue_ms=(queued-started)*1000,**timings)
                body = pack(dict(type='action', request_id=request_id,
                    observation_tick=request['observation_tick'], actions=result,
                    server_receive_s=received,server_timing=timing))
                await websocket.send(stamp(body))
        except websockets.ConnectionClosed:
            pass
        except Exception as error:
            print(json.dumps({'event':'request_error','error':str(error)}),flush=True)
            try:
                await websocket.send(pack({'type':'error','error':str(error)}))
                await websocket.close(code=1011,reason='Inference request failed')
            except websockets.ConnectionClosed:
                pass


async def serve(args):
    print(json.dumps({'event':'loading','policy':args.policy,'checkpoint':str(args.checkpoint)}),flush=True)
    if args.policy == 'diffusion':
        from dp_policy import DiffusionBackend
        backend = DiffusionBackend(args.checkpoint, args.device, args.precision,
                                  args.inference_steps, args.weights, args.capture_config)
    else:
        from act_policy import ACTBackend
        backend = ACTBackend(args.checkpoint,args.device,args.precision)
    token = os.environ.get(args.token_env)
    server = PolicyServer(backend,token)
    print(json.dumps({'event':'warming_up','policy':args.policy}),flush=True)
    await asyncio.get_running_loop().run_in_executor(server.executor,backend.warmup)
    async with websockets.serve(server.handler,args.host,args.port,compression=None,
        max_size=MAX_BYTES,max_queue=2,process_request=server.process_request,
        ping_interval=20,ping_timeout=20):
        print(json.dumps({'event':'ready','host':args.host,'port':args.port,
                          'metadata':backend.metadata,'authentication':bool(token)}),flush=True)
        try:
            await asyncio.Future()
        finally:
            server.executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True,help='ACT last.pt or Diffusion Policy .ckpt')
    parser.add_argument('--policy',choices=['act','diffusion'],default='act')
    parser.add_argument('--inference-steps',type=int,help='Diffusion denoising steps; default from checkpoint')
    parser.add_argument('--weights',choices=['auto','ema','model'],default='auto',help='Diffusion weights')
    parser.add_argument('--capture-config',type=Path,help='Diffusion capture config if training data is unavailable')
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto')
    parser.add_argument('--token-env',default='ACE_INFERENCE_TOKEN')
    args = parser.parse_args()
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
