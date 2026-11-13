"""Cleanup Diffusion Policy backend; weights and normalizers come from one checkpoint."""
import hashlib
import json
from pathlib import Path
import sys
import time

import av
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from image_inputs import ImageInputs
from protocol import CAMERAS, DIM, SEMANTICS, VERSION, actions, vector


class DiffusionBackend(ImageInputs):
    def __init__(self, checkpoint, device='cuda:0', precision='auto', inference_steps=None,
                 weights='auto', capture_config=None):
        root = Path(__file__).resolve().parents[2]
        sys.path[:0] = [str(root), str(root / 'diffusion_policy')]
        from ace_training import preprocessing
        self.resize = preprocessing.letterbox_rgb
        self.image_to_float = preprocessing.image_to_float
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        torch.set_num_threads(4)
        path = Path(checkpoint).resolve()
        payload = torch.load(str(path), map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        if cfg.policy._target_ != 'diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy':
            raise ValueError('Expected a cleanup DiffusionUnetImagePolicy checkpoint')
        shape = cfg.policy.shape_meta
        if list(shape.action.shape) != [DIM] or set(shape.obs) != {*CAMERAS, 'qpos'}:
            raise ValueError('Checkpoint must use the cleanup 16D/three-camera contract')
        if list(shape.obs.qpos.shape) != [DIM]:
            raise ValueError('Expected a 16D qpos observation')
        self.image_size = tuple(shape.obs.head.shape[1:])
        if any(list(shape.obs[name].shape) != [3, *self.image_size] for name in CAMERAS):
            raise ValueError('All cameras must have the same model image size')
        config = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))
        config.obs_encoder.rgb_model.weights = None
        self.policy = hydra.utils.instantiate(config)
        key = ('ema_model' if cfg.training.use_ema else 'model') if weights == 'auto' else weights
        if key == 'ema':
            key = 'ema_model'
        if key not in payload['state_dicts']:
            raise ValueError(f'Checkpoint has no {key} weights')
        self.policy.load_state_dict(payload['state_dicts'][key], strict=True)
        self.policy.to(self.device).eval().requires_grad_(False)
        if inference_steps is not None:
            if not 1 <= inference_steps <= self.policy.noise_scheduler.config.num_train_timesteps:
                raise ValueError('inference-steps must be within the scheduler training timestep range')
            self.policy.num_inference_steps = inference_steps
        self.n_obs_steps = self.policy.n_obs_steps
        self.horizon = self.policy.horizon - self.n_obs_steps + 1
        precision = 'fp32' if precision == 'auto' else precision
        if precision == 'bf16' and (self.device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 requires a supported CUDA GPU')
        self.amp = precision == 'bf16'
        if capture_config is not None:
            capture = json.loads(Path(capture_config).read_text(encoding='utf-8-sig'))
        else:
            manifest = Path(cfg.task.dataset.root) / 'provenance/episode_001/manifest.json'
            if not manifest.is_file():
                raise ValueError('Capture metadata is missing; pass --capture-config config.pc.json')
            capture = json.loads(manifest.read_text())['config']
        if capture['record_hz'] != 30:
            raise ValueError('This PC controller requires the trained 30Hz capture frequency')
        digest = hashlib.sha256()
        for name, value in sorted(self.policy.normalizer.state_dict().items()):
            array = value.detach().cpu().float().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f'Invalid normalizer: {name}')
            digest.update(name.encode())
            digest.update(array.tobytes())
        for name in ('qpos', 'action'):
            params = self.policy.normalizer[name].params_dict
            if params['scale'].shape != (DIM,) or torch.any(params['scale'] == 0):
                raise ValueError(f'Invalid {name} normalizer scale')
        step = dill.loads(payload['pickles']['global_step'])
        self.metadata = dict(type='hello', protocol_version=VERSION, policy='diffusion',
            checkpoint=str(path), checkpoint_step=int(step), weights=key,
            checkpoint_epoch=int(dill.loads(payload['pickles']['epoch'])),
            action_dim=DIM, horizon=self.horizon, training_horizon=self.policy.horizon,
            n_obs_steps=self.n_obs_steps, action_steps=self.policy.n_action_steps,
            inference_steps=self.policy.num_inference_steps, precision=precision, training_precision='fp32',
            camera_names=list(CAMERAS), image_size=list(self.image_size), control_hz=30,
            action_semantics=SEMANTICS, image_color='RGB', normalization_sha256=digest.hexdigest(),
            tcp_offsets={side: capture['arms'][side]['tcp_offset'] for side in ('left', 'right')},
            camera_shapes={name: [capture['cameras'][name]['height'], capture['cameras'][name]['width'], 3]
                           for name in CAMERAS},
            input_contract=dict(preprocess=preprocessing.IMAGE_PREPROCESS, transport='lossless_png_or_raw',
                preprocess_sha256=hashlib.sha256(Path(preprocessing.__file__).read_bytes()).hexdigest(),
                pyav_version=av.__version__, resize_location='pc_once', layout='N,T,C,H,W_per_camera',
                pixel_dtype='uint8', model_input_dtype='float32', pixel_scale=1 / 255.,
                imagenet_normalization_count=int(config.obs_encoder.imagenet_norm),
                state_normalization='checkpoint_linear_limits', action_denormalization='checkpoint_linear_limits'),
            rtc_algorithm='client_async_delay_compensation_overlap_blend', prefix_conditioning=False)

    def prepare_inputs(self, observation):
        history = observation.get('observation_history', [observation] if self.n_obs_steps == 1 else None)
        if not isinstance(history, list) or len(history) != self.n_obs_steps:
            raise ValueError(f'Expected {self.n_obs_steps} consecutive observations; update the PC client')
        prepared = observation.get('images_preprocessed', False)
        if prepared and observation.get('input_contract') != self.metadata['input_contract']:
            raise ValueError('PC preprocessing contract differs from server/training')
        result = {'qpos': np.stack([vector(sample['state']) for sample in history])}
        for sample in history:
            if set(sample['images']) != set(CAMERAS):
                raise ValueError('Expected head, left_wrist and right_wrist images in every observation')
        for name in CAMERAS:
            shape = [*self.image_size, 3] if prepared else self.metadata['camera_shapes'][name]
            result[name] = np.stack([
                self._image(sample['images'][name], observation.get('allow_lossy_images', False),
                            shape, prepared=prepared).transpose(2, 0, 1) for sample in history])
        return result

    def warmup(self):
        sample = dict(state=np.array([0, 0, 0, 0, 0, 0, 1, 0] * 2, np.float32),
                      images={name: np.zeros(self.metadata['camera_shapes'][name], np.uint8) for name in CAMERAS})
        for _ in range(2):
            self.infer(dict(observation_history=[sample] * self.n_obs_steps))

    def infer(self, observation):
        return self.infer_timed(observation)[0]

    @torch.inference_mode()
    def infer_timed(self, observation):
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        arrays = self.prepare_inputs(observation)
        inputs = {name: torch.as_tensor(value, device=self.device)[None] for name, value in arrays.items()}
        inputs = {name: value.float() if name == 'qpos' else self.image_to_float(value)
                  for name, value in inputs.items()}
        cuda = self.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize(self.device)
        prepared = time.perf_counter()
        if cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            # predict_action already applies the checkpoint normalizer and its inverse.
            prediction = self.policy.predict_action(inputs)['action_pred'][0, self.n_obs_steps - 1:]
        if cuda:
            end.record()
            end.synchronize()
        forwarded = time.perf_counter()
        result = prediction.float().cpu().numpy().copy()
        actions(result, self.horizon)
        for start in (0, 8):
            quaternion = result[:, start + 3:start + 7]
            result[:, start + 3:start + 7] = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
        finished = time.perf_counter()
        return result, dict(preprocess_ms=(prepared - started) * 1000,
            model_infer_ms=begin.elapsed_time(end) if cuda else (forwarded - prepared) * 1000,
            model_wall_ms=(forwarded - prepared) * 1000, postprocess_ms=(finished - forwarded) * 1000,
            backend_total_ms=(finished - started) * 1000)
