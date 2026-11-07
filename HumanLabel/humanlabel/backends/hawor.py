"""GPU adapter for upstream HaWoR, without its moving-camera SLAM stage."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanlabel.utils.tracking import contiguous_ranges, project_rotations, runs


class HaWoRBackend:
    def __init__(self, root, device="cuda"):
        import torch

        self.root = Path(root).resolve()
        self.repo = self.root / ".cache/HaWoR"
        sys.path.insert(0, str(self.repo))
        from hawor.configs import get_config
        from lib.models.hawor import HAWOR

        cfg = get_config(str(self.root / ".cache/weights/hawor/model_config.yaml"))
        cfg.defrost()
        cfg.MANO.MODEL_PATH = str(self.repo / "_DATA/data/mano")
        cfg.MANO.MEAN_PARAMS = str(self.repo / "_DATA/data/mano_mean_params.npz")
        cfg.freeze()
        torch.set_num_threads(8)
        self.device = torch.device(device)
        self.model = HAWOR(cfg)
        checkpoint = torch.load(
            self.root / ".cache/weights/hawor/checkpoints/hawor.ckpt",
            map_location="cpu",
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        self.model.eval().requires_grad_(False).to(device)
        # The image encoder has no cross-frame operations. Microbatch only
        # that encoder; both temporal attention modules still see all 16 frames.
        backbone_forward = self.model.backbone.forward

        def image_encoder(images):
            return torch.cat(
                [backbone_forward(chunk) for chunk in images.split(4)], dim=0
            )

        self.model.backbone.forward = image_encoder
        self.torch = torch
        self.faces = self.model.mano.faces
        self.infiller = None

    def inference(self, imgfiles, boxes, observed, supported, grid, matrix):
        """Overlapping 16-frame windows, each confined to a capture segment."""
        from lib.datasets.track_dataset import TrackDatasetEval
        from torch.utils.data import default_collate

        torch = self.torch
        n = len(imgfiles)
        trans = np.full((2, n, 3), np.nan, np.float32)
        rot = np.full((2, n, 16, 3, 3), np.nan, np.float32)
        betas = np.full((2, n, 10), np.nan, np.float32)
        focal = float(np.sqrt(matrix[0, 0] * matrix[1, 1]))

        class StableTrackDataset(TrackDatasetEval):
            def __getitem__(self, index):
                # Upstream flips this NumPy view in-place for left hands.
                # Overlapping windows and padding revisit the same index.
                center = self.centers[index].copy()
                try:
                    return super().__getitem__(index)
                finally:
                    self.centers[index] = center

        for a, b in contiguous_ranges(grid):
            for side in range(2):
                for lo, hi in runs(supported[side, a:b]):
                    lo, hi = lo + a, hi + a
                    ids = np.arange(lo, hi)
                    db = StableTrackDataset(
                        np.asarray(imgfiles)[ids],
                        boxes[side, ids].copy(),
                        img_focal=focal,
                        img_center=matrix[:2, 2].tolist(),
                        normalization=True,
                        dilate=1.2,
                        do_flip=side == 0,
                    )
                    length = hi - lo
                    tr = np.zeros((length, 3), np.float64)
                    rr = np.zeros((length, 16, 3, 3), np.float64)
                    bb = np.zeros((length, 10), np.float64)
                    weights = np.zeros(length)
                    starts = sorted(
                        set([*range(0, max(1, length - 15), 8), max(0, length - 16)])
                    )
                    for start in starts:
                        indices = np.minimum(np.arange(start, start + 16), length - 1)
                        batch = default_collate([db[int(i)] for i in indices])
                        batch = {
                            k: v.to(self.device).unsqueeze(0)
                            for k, v in batch.items()
                            if isinstance(v, torch.Tensor)
                        }
                        with (
                            torch.inference_mode(),
                            torch.cuda.amp.autocast(dtype=torch.float16),
                        ):
                            out = self.model(batch)["out"]
                        count = min(16, length - start)
                        sl = slice(start, start + count)
                        weight = np.minimum(np.arange(1, 17), np.arange(16, 0, -1))[
                            :count
                        ]
                        tr[sl] += (
                            out["trans_full"][:count, 0].float().cpu().numpy()
                            * weight[:, None]
                        )
                        rr[sl] += (
                            out["pred_rotmat"][:count].float().cpu().numpy()
                            * weight[:, None, None, None]
                        )
                        bb[sl] += (
                            out["pred_shape"][:count].float().cpu().numpy()
                            * weight[:, None]
                        )
                        weights[sl] += weight
                    trans[side, ids] = tr / weights[:, None]
                    matrices = project_rotations(rr / weights[:, None, None, None])
                    if side == 0:
                        f = np.diag([-1.0, 1.0, 1.0])
                        matrices = f @ matrices @ f
                    rot[side, ids] = matrices
                    betas[side, ids] = bb / weights[:, None]
                print(f"  HaWoR segment {a}:{b} side {side} done", flush=True)
        return {"trans": trans, "rotmat": rot, "betas": betas}

    def geometry(self, rotmat, betas, trans, side, grad=False):
        """Absolute local rotations; left uses the mirrored right MANO template."""
        matrices = rotmat
        if side == 0:
            f = matrices.new_tensor([-1.0, 1.0, 1.0])
            matrices = matrices * f[:, None] * f[None, :]
        out = self.model.mano(
            global_orient=matrices[:, :1],
            hand_pose=matrices[:, 1:],
            betas=betas,
            pose2rot=False,
        )
        joints, vertices = out.joints, out.vertices
        if side == 0:
            f = joints.new_tensor([-1.0, 1.0, 1.0])
            joints, vertices = joints * f, vertices * f
        return joints + trans[:, None], vertices + trans[:, None]

    def complete(self, data, observed, supported, grid):
        """Learned 120-frame infilling; keep original anchor and support masks."""
        if not (~observed & supported).any():
            return {k: v.copy() for k, v in data.items()}, np.zeros_like(observed)
        import lib.eval_utils.filling_utils as fill
        from hawor.utils.rotation import angle_axis_to_rotation_matrix
        from infiller.lib.model.network import TransformerModel

        from humanlabel.utils.tracking import interpolate_parameters

        torch = self.torch
        if self.infiller is None:
            self.infiller = TransformerModel(
                seq_len=120,
                input_dim=218,
                d_model=384,
                nhead=8,
                d_hid=2048,
                nlayers=8,
                dropout=0.05,
                out_dim=218,
                masked_attention_stage=True,
            )
            ckpt = torch.load(
                self.root / ".cache/weights/hawor/checkpoints/infiller.pt",
                map_location="cpu",
            )
            self.infiller.load_state_dict(
                ckpt["transformer_encoder_state_dict"], strict=True
            )
            self.infiller.eval().requires_grad_(False).to(self.device)

        def forward(
            trans, root_orient, hand_pose, is_right=None, betas=None, side=1, **kwargs
        ):
            shape = trans.shape[:2]
            aa = torch.cat(
                (root_orient.reshape(-1, 1, 3), hand_pose.reshape(-1, 15, 3)), 1
            )
            mats = angle_axis_to_rotation_matrix(aa.to(self.device))
            with torch.no_grad():
                j, v = self.geometry(
                    mats,
                    betas.reshape(-1, 10).to(self.device),
                    trans.reshape(-1, 3).to(self.device),
                    side,
                )
            return {
                "joints": j.reshape(*shape, 21, 3).cpu(),
                "vertices": v.reshape(*shape, 778, 3).cpu(),
            }

        fill.run_mano = forward
        fill.run_mano_left = lambda *args, **kwargs: forward(*args, **kwargs, side=0)
        completed = np.zeros_like(observed)
        n = observed.shape[1]
        params = {
            "trans": data["trans"].copy(),
            "betas": data["betas"].copy(),
            "rot": np.zeros((2, n, 3), np.float32),
            "hand_pose": np.zeros((2, n, 45), np.float32),
        }
        for side in range(2):
            good = supported[side]
            if not good.any():
                continue
            aa = (
                Rotation.from_matrix(data["rotmat"][side, good].reshape(-1, 3, 3))
                .as_rotvec()
                .reshape(-1, 16, 3)
            )
            params["rot"][side, good] = aa[:, 0]
            params["hand_pose"][side, good] = aa[:, 1:].reshape(-1, 45)
        for a, b in contiguous_ranges(grid):
            for lo, hi in runs((~observed & supported).any(0)[a:b]):
                lo, hi = a + lo, a + hi
                starts = set(range(max(a, lo - 30), hi, 60))
                for start in sorted(starts):
                    start = max(a, min(start, b - 120))
                    end = min(b, start + 120)
                    valid = observed[:, start:end].copy()
                    if (valid.sum(1) < 2).any() or not valid.all(0).any():
                        continue
                    chunk = {k: v[:, start:end].copy() for k, v in params.items()}
                    chunk = interpolate_parameters(chunk, valid)
                    chunk["valid"] = valid
                    inputs, transforms = fill.filling_preprocess(chunk)
                    length = end - start
                    inputs = np.pad(inputs, ((0, 120 - length), (0, 0)), mode="edge")
                    valid_pad = np.pad(
                        valid, ((0, 0), (0, 120 - length)), constant_values=True
                    )
                    both = torch.from_numpy(valid_pad.all(0)).to(self.device)
                    src = torch.from_numpy(inputs).float().to(self.device).unsqueeze(1)
                    mask = (~both).reshape(1, 1, 1, 120).expand(1, 1, 120, 120)
                    with torch.inference_mode():
                        out = self.infiller(
                            src,
                            torch.zeros(
                                (120, 120), dtype=torch.bool, device=self.device
                            ),
                            both.float().reshape(120, 1, 1),
                            mask,
                        )
                    out = out[:, 0].reshape(120, 2, 109)[:length].cpu()
                    result = fill.filling_postprocess(out, transforms)
                    use = ~valid & supported[:, start:end]
                    for key in params:
                        target = params[key][:, start:end]
                        target[use] = result[key][use]
                    completed[:, start:end] |= use
        aa = np.concatenate(
            (params["rot"][:, :, None], params["hand_pose"].reshape(2, n, 15, 3)), 2
        )
        result = {
            "trans": params["trans"],
            "betas": params["betas"],
            "rotmat": data["rotmat"].copy(),
        }
        if completed.any():
            result["rotmat"][completed] = (
                Rotation.from_rotvec(aa[completed].reshape(-1, 3))
                .as_matrix()
                .reshape(-1, 16, 3, 3)
            )
        return result, completed
