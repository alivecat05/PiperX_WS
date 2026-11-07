"""Adapter for the official ACE-Ego-Hand K-free release.

Full RGB context is retained for occluded frames. Presence outputs are saved
separately; unlike the upstream CLI, the raw 3D predictions are not discarded
just because the in-frame visibility head is low.
"""

import os
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from humanlabel.utils.tracking import contiguous_ranges


def compatibility():
    # The base environment contains unrelated optional quantization plugins.
    # No quantized weights are used by this inference path.
    import diffusers.utils.import_utils as di
    import transformers.utils.import_utils as ti

    di._torchao_available = False
    ti._torchao_available = False
    ti._gptqmodel_available = False
    import peft.import_utils as pi

    pi.is_gptqmodel_available = lambda: False
    import peft.tuners.lora.gptq as pg

    pg.is_gptqmodel_available = lambda: False


def encode_causal(vae, pixels, device):
    """Execute the native causal VAE's 1+4-frame loop without a full GPU clip.

    The encoder caches, pointwise latent projection and scaling are identical
    to AutoencoderKLWan3_8.model.encode; only input/output storage is streamed.
    """
    import torch
    from videox_fun.models.wan_vae3_8 import patchify

    core = vae.model
    core.clear_cache()
    scale = [item.to(device, torch.bfloat16) for item in vae.scale]
    latents = []
    slices = [(0, 1), *[(i, min(i + 4, len(pixels))) for i in range(1, len(pixels), 4)]]
    with torch.inference_mode():
        for a, b in slices:
            chunk = (
                pixels[a:b].permute(1, 0, 2, 3).unsqueeze(0).to(device, torch.bfloat16)
            )
            chunk = patchify(chunk, patch_size=2)
            core._enc_conv_idx = [0]
            encoded = core.encoder(
                chunk, feat_cache=core._enc_feat_map, feat_idx=core._enc_conv_idx
            )
            mu = core.conv1(encoded).chunk(2, dim=1)[0]
            mu = (mu - scale[0].view(1, core.z_dim, 1, 1, 1)) * scale[1].view(
                1, core.z_dim, 1, 1, 1
            )
            latents.append(mu[0].to(torch.float16).cpu())
    core.clear_cache()
    return torch.cat(latents, dim=1)


class ACEBackend:
    def __init__(self, root):
        import smplx
        import torch
        import yaml

        compatibility()
        self.torch = torch
        self.root = Path(root).resolve()
        self.repo = self.root / ".cache/ACE-Ego-Hand"
        sys.path.insert(0, str(self.repo))
        from humanlabel.utils.hand_model import prepare_mano

        prepare_mano(
            self.root / ".cache/mano/MANO_RIGHT.pkl", self.repo / "data/mano/mano"
        )
        os.environ["ACE_EGO_HAND_MANO_DIR"] = str(self.repo / "data/mano")
        torch.set_num_threads(8)
        if not (self.repo / "cache/caption_embed.pt").exists():
            module = runpy.run_path(str(self.repo / "scripts/precompute_caption.py"))
            module["cache_caption"](torch.device("cuda"))
        from ace_ego_hand.models.geodit_model import GeoDitModel

        self.options = yaml.safe_load(
            (self.repo / "options/ace_ego_hand_kfree.yml").read_text()
        )
        for key in ("model_root", "videox_config", "caption_embed"):
            self.options["paths"][key] = str(self.repo / self.options["paths"][key])
        self.model = GeoDitModel(self.options, torch.device("cpu"))
        self.model.load_inference(str(self.repo / "checkpoints/ace_ego_hand_kfree.pt"))
        self.model.net.eval().requires_grad_(False)
        # The published forward stops at tap 15. Drop unused blocks only after
        # loading every checkpoint delta, preserving the executed computation.
        self.model.net.backbone.dit.blocks = torch.nn.ModuleList(
            list(self.model.net.backbone.dit.blocks)[:16]
        )
        self.model.net.backbone._grad_ckpt = False
        self.model.device = torch.device("cuda")
        # Stream transformer blocks to bound VRAM when this GPU is shared.
        # Every block still executes on CUDA with the original weights/dtype.
        self._offload_hooks = []

        def enter_block(module, inputs):
            module.cuda()

        def leave_block(module, inputs, output):
            module.cpu()

        for block in self.model.net.backbone.dit.blocks:
            self._offload_hooks.append(block.register_forward_pre_hook(enter_block))
            self._offload_hooks.append(block.register_forward_hook(leave_block))
        for mano in self.model.mano_models.values():
            mano.cuda()
        torch.cuda.empty_cache()
        self.layer = (
            smplx.MANOLayer(str(self.repo / "data/mano/mano"), is_rhand=True)
            .eval()
            .requires_grad_(False)
            .cuda()
        )
        self.faces = self.layer.faces

    def activate_decoder(self):
        torch = self.torch
        dit = self.model.net.backbone.dit
        blocks = dit.blocks
        dit.blocks = torch.nn.ModuleList()
        try:
            self.model.net.cuda()
        finally:
            dit.blocks = blocks
        for block in blocks:
            block.cpu()

    def geometry(self, rotmat, betas, trans, side, grad=False):
        matrices = rotmat
        if side == 0:
            f = matrices.new_tensor([-1.0, 1.0, 1.0])
            matrices = matrices * f[:, None] * f[None, :]
        out = self.layer(
            global_orient=matrices[:, :1],
            hand_pose=matrices[:, 1:],
            betas=betas,
            pose2rot=False,
        )
        tips = out.vertices[:, [744, 320, 443, 554, 671]]
        order = [
            0,
            13,
            14,
            15,
            16,
            1,
            2,
            3,
            17,
            4,
            5,
            6,
            18,
            10,
            11,
            12,
            19,
            7,
            8,
            9,
            20,
        ]
        joints = self.torch.cat((out.joints[:, :16], tips), 1)[:, order]
        vertices = out.vertices
        if side == 0:
            f = joints.new_tensor([-1.0, 1.0, 1.0])
            joints, vertices = joints * f, vertices * f
        return joints + trans[:, None], vertices + trans[:, None]

    def inference(
        self, episode, paths, boxes, observed, supported, grid, matrix, output
    ):
        from humanlabel.utils.episode import save_npz

        torch = self.torch
        cache, metadata = (
            output / "prediction_raw.npz",
            output / "camera_prediction.json",
        )
        if cache.exists() and metadata.exists():
            import json

            with np.load(cache) as f:
                data = {k: f[k] for k in f.files}
            details = json.loads(metadata.read_text())
            return data, np.asarray(details["camera_matrix"]), details
        import json

        from ace_ego_hand.inference import predict_video
        from ace_ego_hand.video_vae import load_vae
        from infer_video import _encode_size

        n = len(paths)
        image = cv2.imread(paths[0])
        height, width = image.shape[:2]
        ew, eh = _encode_size(width, height, 832)
        focal = 0.5 * ew / np.tan(np.deg2rad(30.0))
        intr = {
            "fx": focal,
            "fy": focal,
            "cx": ew / 2,
            "cy": eh / 2,
            "image_width": ew,
            "image_height": eh,
            "source": "unused nominal placeholder for K-free",
        }
        data = {
            "trans": np.full((2, n, 3), np.nan, np.float32),
            "rotmat": np.full((2, n, 16, 3, 3), np.nan, np.float32),
            "betas": np.full((2, n, 10), np.nan, np.float32),
            "visible": np.zeros((2, n), np.float32),
            "presence": np.zeros((2, n), np.float32),
            "completion": np.zeros((2, n), bool),
            "direct_joints2d": np.full((2, n, 21, 2), np.nan, np.float32),
        }
        cameras, accept = [], []
        self.model.net.cpu()
        torch.cuda.empty_cache()
        latent_path = output / "video_latents.pt"
        cached_latents = (
            torch.load(latent_path, map_location="cpu", weights_only=True)
            if latent_path.exists()
            else None
        )
        vae = load_vae(torch.device("cpu")) if cached_latents is None else None
        if vae is not None:
            # Only encoding is used. The decoder has no executed path here.
            vae.model.decoder = torch.nn.Identity()
            vae.model.conv2 = torch.nn.Identity()
            vae.cuda()
        latent_records = []
        # Splitting at actual dropped grid ticks prevents a 30 Hz video model
        # from treating nonconsecutive source observations as adjacent frames.
        for a, b in contiguous_ranges(grid):
            length = b - a
            if cached_latents is not None:
                continue
            pixels = [
                cv2.cvtColor(
                    cv2.resize(
                        cv2.imread(paths[i]), (ew, eh), interpolation=cv2.INTER_AREA
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                for i in range(a, b)
            ]
            arr = np.stack(pixels).astype(np.float32) / 255 * 2 - 1
            px = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            n_enc = max(5, 4 * ((length + 2) // 4) + 1)
            if n_enc > length:
                px = torch.cat((px, px[-1:].repeat(n_enc - length, 1, 1, 1)))
            with torch.inference_mode():
                latent = encode_causal(vae, px, torch.device("cuda")).float()
            del px, pixels, arr
            latent_records.append((int(a), int(b), latent.half()))
            print(f"  ACE encoded {episode.name}: {a}:{b}/{n}", flush=True)
        del vae
        if cached_latents is not None:
            latent_records = cached_latents
        else:
            torch.save(latent_records, latent_path)
        torch.cuda.empty_cache()
        self.activate_decoder()
        for a, b, latent in latent_records:
            length = b - a
            pred = predict_video(
                self.model, 15, latent, intr, "tiled", torch.device("cuda"), tile_w=22
            )
            # Upstream MANO() adds its mean AFTER converting the decoder's
            # rotations to axis-angle. Standardize to absolute local rotations
            # so later archives reconstruct identically with MANOLayer.
            for side in range(2):
                go = pred["global_orient"][:length, side].reshape(length, 3, 3)
                hp = pred["hand_pose"][:length, side].reshape(length, 15, 3, 3)
                mean = (
                    self.model.mano_models[bool(side)]
                    .hand_mean.detach()
                    .cpu()
                    .numpy()
                    .reshape(15, 3)
                )
                aa = (
                    Rotation.from_matrix(hp.reshape(-1, 3, 3))
                    .as_rotvec()
                    .reshape(length, 15, 3)
                    + mean
                )
                hp_full = (
                    Rotation.from_rotvec(aa.reshape(-1, 3))
                    .as_matrix()
                    .reshape(length, 15, 3, 3)
                )
                data["rotmat"][side, a:b] = np.concatenate((go[:, None], hp_full), 1)
                data["trans"][side, a:b] = pred["cam_trans"][:length, side]
                data["betas"][side, a:b] = pred["betas"][:length, side]
                data["visible"][side, a:b] = pred["exists_2d"][:length, side]
                data["presence"][side, a:b] = pred["exists_3d"][:length, side]
                data["direct_joints2d"][side, a:b] = pred["direct_joints2d"][
                    :length, side
                ]
                data["completion"][side, a:b] = ~observed[side, a:b]
            if pred.get("_pred_K"):
                cameras.append((int(length), pred["_pred_K"]))
            if pred.get("_pnp_accept_frac") is not None:
                accept.append(pred["_pnp_accept_frac"])
            print(f"  ACE {episode.name}: segment {a}:{b}/{n}", flush=True)
        del latent_records
        torch.cuda.empty_cache()
        if not cameras:
            raise ValueError("K-free model did not predict a camera")
        # Fixed camera: robust episode estimate. This estimate is not calibration.
        k = max(cameras, key=lambda x: x[0])[1]
        predicted = np.array(
            [
                [k["fx"] * width / ew, 0, k["cx"] * width / ew],
                [0, k["fy"] * height / eh, k["cy"] * height / eh],
                [0, 0, 1.0],
            ]
        )
        details = {
            "camera_matrix": predicted.tolist(),
            "per_capture_segment_cameras": cameras,
            "pnp_accept_fraction": float(np.mean(accept)) if accept else None,
            "model": "official ACE-Ego-Hand K-free, Wan2.2 5B, tap 15, tiled 22 latents, bf16 backbone + fp32 hand decoder",
            "code_revision": "59da83b58d58c3041b9ee0b35b5054866e33a990",
            "executed_backbone_blocks": 16,
            "encode_size": [ew, eh],
            "gpu_memory_policy": "CUDA inference with one transformer block resident at a time; no quantization or resolution reduction",
            "camera_warning": "predicted intrinsics are estimates, not measured calibration",
        }
        save_npz(cache, data)
        metadata.write_text(json.dumps(details, indent=2))
        return data, predicted, details
