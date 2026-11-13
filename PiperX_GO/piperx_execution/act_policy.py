"""GPU ACT inference only. No SDK, camera device or robot imports."""
import json
import hashlib
from pathlib import Path
import sys
import time

import av
import numpy as np
import torch

from image_inputs import ImageInputs
from protocol import CAMERAS, DIM, SEMANTICS, VERSION, vector


class ACTBackend(ImageInputs):
    def __init__(self, checkpoint, device='cuda:0', precision='auto'):
        root = Path(__file__).resolve().parents[2]
        sys.path[:0] = [str(root), str(root/'act')]
        from policy import ACTPolicy
        from ace_training.preprocessing import IMAGE_PREPROCESS, letterbox_rgb, normalize_state, image_to_float
        from ace_training import preprocessing
        if not torch.cuda.is_available() or not str(device).startswith('cuda'):
            raise RuntimeError('ACT inference requires a CUDA GPU')
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(4)
        path = Path(checkpoint).resolve()
        cfgpath = path.parent/'training_config.json'
        training = json.loads(cfgpath.read_text()) if cfgpath.exists() else {}
        if precision == 'auto':
            precision = training.get('precision')
            if precision not in ('fp32','bf16'):
                raise ValueError('Training precision is unavailable; supply training_config.json or explicit --precision')
        if precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('GPU does not support BF16')
        self.amp = precision == 'bf16'
        ckpt = torch.load(path, map_location='cpu')
        config = ckpt['policy_config'].copy()
        if config['state_dim'] != DIM or tuple(config['camera_names']) != CAMERAS:
            raise ValueError('Checkpoint must use the cleanup 16D/three-camera contract')
        config['pretrained_backbone'] = False  # All weights come from this checkpoint.
        argv = ['--ckpt_dir', str(path.parent), '--policy_class', 'ACT',
                '--task_name', 'cleanup_lerobot', '--seed', '42', '--num_epochs', '1']
        self.policy = ACTPolicy(config, argv=argv).to(self.device).eval()
        self.policy.load_state_dict(ckpt['policy'], strict=True)
        self.policy.requires_grad_(False)
        self.stats = ckpt['norm_stats']
        self.image_size = tuple(ckpt['image_size'])
        self.resize = letterbox_rgb
        self.normalize_state, self.image_to_float = normalize_state, image_to_float
        self.horizon = config['num_queries']
        self.metadata = dict(type='hello', protocol_version=VERSION, policy='act',
            checkpoint_step=int(ckpt['step']), checkpoint=str(path), action_dim=DIM,
            horizon=self.horizon, camera_names=list(CAMERAS), image_size=list(self.image_size),
            control_hz=30, action_semantics=SEMANTICS, image_color='RGB',
            precision=precision,
            training_precision=training.get('precision'),
            input_contract={'preprocess':IMAGE_PREPROCESS,'transport':'lossless_png_or_raw',
                'preprocess_sha256':hashlib.sha256(Path(preprocessing.__file__).read_bytes()).hexdigest(),
                'pyav_version':av.__version__,'resize_location':'pc_once',
                'layout':'N,K,C,H,W','pixel_dtype':'uint8','model_input_dtype':'float32',
                'pixel_scale':1/255.,'imagenet_mean':[.485,.456,.406],
                'imagenet_std':[.229,.224,.225],'imagenet_normalization_count':1,
                'state_normalization':'checkpoint_qpos_mean_std','action_denormalization':'checkpoint_action_mean_std'},
            rtc_algorithm='client_async_delay_compensation_overlap_blend',
            prefix_conditioning=False, tcp_offsets={'left':[0.]*6, 'right':[0.]*6})
        self.metadata['camera_shapes'] = {'head':[800,1280,3],
            'left_wrist':[720,1280,3],'right_wrist':[720,1280,3]}
        # Carry the captured TCP convention when the original dataset is present.
        if cfgpath.exists():
            dataset = Path(training.get('dataset_root', ''))
            manifest = dataset/'provenance/episode_001/manifest.json'
            if manifest.exists():
                capture = json.loads(manifest.read_text())['config']
                self.metadata['tcp_offsets'] = {s:capture['arms'][s]['tcp_offset'] for s in ('left','right')}
                self.metadata['camera_shapes'] = {n:[capture['cameras'][n]['height'],capture['cameras'][n]['width'],3]
                                                  for n in CAMERAS}
        digest = hashlib.sha256()
        for key in ('qpos_mean','qpos_std','action_mean','action_std'):
            value = np.asarray(self.stats[key],dtype=np.float32)
            if value.shape != (DIM,) or not np.isfinite(value).all() or (key.endswith('std') and np.any(value<=0)):
                raise ValueError(f'Invalid checkpoint statistic: {key}')
            digest.update(key.encode());digest.update(value.tobytes())
        self.metadata['normalization_sha256'] = digest.hexdigest()
        torch.backends.cudnn.benchmark = True

    def warmup(self):
        # cuDNN handles/algorithm caches can be thread-local. Warm the same
        # inference worker that will handle WebSocket requests, before ready.
        torch.cuda.set_device(self.device)
        observation = {'state':np.array([0,0,0,0,0,0,1,0]*2, np.float32),
                       'images':{name:np.zeros(self.metadata['camera_shapes'][name],np.uint8) for name in CAMERAS}}
        for _ in range(3):
            self.infer(observation)

    def prepare_inputs(self, observation):
        state = vector(observation['state'])
        images = observation['images']
        if set(images) != set(CAMERAS):
            raise ValueError('Expected head, left_wrist and right_wrist images')
        prepared = observation.get('images_preprocessed',False)
        if prepared and observation.get('input_contract') != self.metadata['input_contract']:
            raise ValueError('PC preprocessing contract differs from server/training')
        image = np.stack([self._image(images[name],observation.get('allow_lossy_images',False),
                                    [*self.image_size,3] if prepared else self.metadata['camera_shapes'][name],
                                    prepared=prepared).transpose(2,0,1)
                          for name in CAMERAS])
        qpos = self.normalize_state(state,self.stats)
        return image,qpos

    def infer(self, observation):
        return self.infer_timed(observation)[0]

    @torch.inference_mode()
    def infer_timed(self, observation):
        started = time.perf_counter()
        image,qpos = self.prepare_inputs(observation)
        qpos = torch.as_tensor(qpos, device=self.device).float()[None]
        image = self.image_to_float(torch.as_tensor(image,device=self.device)[None])
        torch.cuda.synchronize(self.device)
        prepared = time.perf_counter()
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            prediction = self.policy(qpos, image)[0]
        end.record();end.synchronize()
        forwarded = time.perf_counter()
        result = prediction.float().cpu().numpy()*self.stats['action_std']+self.stats['action_mean']
        if result.shape != (self.horizon,DIM) or not np.isfinite(result).all():
            raise RuntimeError('Invalid model output')
        for start in (0,8):
            q = result[:,start+3:start+7]
            norms = np.linalg.norm(q,axis=1,keepdims=True)
            if np.any(norms<1e-6):
                raise RuntimeError('Model predicted zero quaternion')
            result[:,start+3:start+7] = q/norms
        finished = time.perf_counter()
        return result.astype(np.float32),dict(preprocess_ms=(prepared-started)*1000,
            model_infer_ms=begin.elapsed_time(end),model_wall_ms=(forwarded-prepared)*1000,
            postprocess_ms=(finished-forwarded)*1000,backend_total_ms=(finished-started)*1000)
