"""30 Hz execution scheduling for RTC and synchronous action chunks."""
import asyncio
from collections import deque
import json
import math
import time

from control import (ActionTimeline, CartesianFilter, StaleActionChunk,
                     blend_chunk_start, fresh_actions, smooth_chunk)
from protocol import CAMERAS, actions


async def control_loop(hardware, policy, args, cfg):
    hz = args.control_hz
    timeline = ActionTimeline(args.blend_steps)
    limiter = CartesianFilter(cfg,hz)
    holding = False
    queue = deque()
    pending = None
    next_request_tick = 0
    next_deadline = origin = time.perf_counter()
    last_tick = -1
    last_request_anchor = None
    last_sync = origin
    count = 0
    rtt = args.rtc_lookahead_steps/hz
    expired_streak = 0
    stats = dict(inferences=0,discarded_actions=0,expired_chunks=0,hold_ticks=0,rate_limited_ticks=0)
    try:
        while not args.max_steps or count < args.max_steps:
            now = time.perf_counter()
            tick = math.floor((now-origin)*hz)
            if tick <= last_tick:
                # Windows timers can wake early: never execute the same action
                # tick twice or count duplicate samples as reversal confirmation.
                await asyncio.sleep(max(.001, origin+(last_tick+1)/hz-now))
                continue
            last_tick = tick
            feedback = hardware.read()
            hardware.check_cameras()
            observed = policy.observe(hardware.observation()) if policy.n_obs_steps > 1 else None
            if pending is not None and pending.done():
                result = pending.result()
                pending = None
                rtt = result['roundtrip_s']
                stats['inferences'] += 1
                accepted = True
                try:
                    # Reject expiration before spending time conditioning a plan
                    # which cannot supply even one current/future action.
                    raw = actions(result['actions'])
                    fresh_actions(raw,result['observation_tick'],tick)
                    limiter.validate_chunk(raw)
                    plan = smooth_chunk(raw, limiter.limits['chunk_smoothing_window'])
                    reference = feedback if holding or limiter.last is None else limiter.last
                    if args.rtc:
                        stats['discarded_actions'] += timeline.install(plan,result['observation_tick'],tick,reference)
                        lead = max(args.rtc_lookahead_steps,math.ceil(rtt*hz)+1)
                        next_request_tick = tick+max(1,args.action_steps-lead)
                    else:
                        fresh, skipped = fresh_actions(plan,result['observation_tick'],tick)
                        stats['discarded_actions'] += skipped
                        fresh = blend_chunk_start(fresh, reference, args.blend_steps)
                        queue.extend((tick+index, value) for index,value in enumerate(fresh[:args.action_steps]))
                    expired_streak = 0
                except StaleActionChunk as error:
                    accepted = False
                    stats['expired_chunks'] += 1
                    stats['discarded_actions'] += error.horizon
                    expired_streak += 1
                    # Preserve the session and all hardware watchdogs, but send
                    # no expired targets. Request a newer observation this tick.
                    timeline.plans.clear()
                    queue.clear()
                    next_request_tick = tick
                    print(json.dumps(dict(event='action_chunk_expired',
                        request_id=result['request_id'],age_steps=error.age_steps,
                        observation_age_ms=round(error.age_steps*1000/hz,2),
                        horizon_ms=round(error.horizon*1000/hz,2),
                        roundtrip_ms=round(rtt*1000,2),consecutive_expired_chunks=expired_streak,
                        action='hold_and_request_fresh_observation')),flush=True)
                print(json.dumps(dict(event='inference',mode='rtc' if args.rtc else 'no_rtc',
                    request_id=result['request_id'],roundtrip_ms=round(rtt*1000,2),
                    accepted=accepted,expired_chunks=stats['expired_chunks'],
                    discarded_actions=stats['discarded_actions'],timing=result.get('timing',{}))),flush=True)
                timing = result.get('timing',{})
                def ms(value):
                    return 'N/A' if value is None else f'{value:.2f}'
                sdk = timing.get('camera_sdk_acquire_ms',{})
                print('[ms] model='+ms(timing.get('model_infer_ms'))+
                    ' upload_est='+ms(timing.get('upload_ms_est'))+
                    ' download_est='+ms(timing.get('download_ms_est'))+
                    ' one_way_error=+/-'+ms(timing.get('one_way_uncertainty_ms'))+
                    ' pc_preprocess='+ms(timing.get('client_preprocess_ms'))+
                    ' encode='+ms(timing.get('image_encode_ms'))+
                    ' bytes='+str(timing.get('request_bytes','N/A'))+
                    ' camera_sdk='+','.join(n+':'+ms(sdk.get(n)) for n in CAMERAS),flush=True)
            # RTC overlaps inference with execution. Synchronous mode requests
            # a new chunk only after executing its action_steps prefix.
            while queue and queue[0][0] < tick:
                queue.popleft()
                stats['discarded_actions'] += 1
            request_due = tick >= next_request_tick if args.rtc else not queue
            if pending is None and request_due:
                observation = observed if policy.n_obs_steps > 1 else hardware.observation()
                if observation is not None:
                    anchor_tick = math.floor((observation['observed_at']-origin)*hz)
                    if last_request_anchor is None or anchor_tick > last_request_anchor:
                        pending = asyncio.create_task(policy.infer(observation,anchor_tick))
                        last_request_anchor = anchor_tick
                        last_sync = now
                elif now-last_sync > cfg.get('camera_timeout_s',.25)+.5:
                    raise TimeoutError('Could not assemble a fresh synchronized observation')
            target = timeline.take(tick) if args.rtc else (queue.popleft()[1] if queue and queue[0][0] == tick else None)
            if target is None:
                # One transition into measured-joint hold; do not alternate modes
                # or erase reversal memory on every network-wait tick.
                if not holding:
                    hardware.hold()
                    limiter.reset(feedback, preserve_direction=True)
                    holding = True
                stats['hold_ticks'] += 1
            else:
                if holding:
                    limiter.reset(feedback, preserve_direction=True)
                    holding = False
                sent,limited = limiter.apply(target,feedback)
                hardware.send(sent)
                limiter.commit(sent)
                count += 1
                stats['rate_limited_ticks'] += int(limited)
                if count % args.log_every == 0:
                    print(json.dumps(dict(event='control',control_step=count,tick=tick,
                        action_filter=limiter.diagnostics,**stats)),flush=True)
            next_deadline += 1/hz
            # A delayed PC tick never bursts old commands to catch up.
            if next_deadline < time.perf_counter():
                next_deadline = time.perf_counter()+1/hz
            await asyncio.sleep(max(0,next_deadline-time.perf_counter()))
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending,return_exceptions=True)
    return dict(control_steps=count,mode='rtc' if args.rtc else 'no_rtc',
                action_filter=limiter.diagnostics,**stats)
