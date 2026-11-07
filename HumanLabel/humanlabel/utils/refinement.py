"""Local CUDA MANO fitting with shared shape and masked temporal objectives.

Projection anchors are model estimates, not ground truth. Acquired frames can
share a GPU batch across capture gaps; temporal losses never span those gaps.
This implementation is not the official Dyn-HaMR pipeline.
"""

import numpy as np

from humanlabel.utils.tracking import project_rotations


def temporal_triples(ids, grid):
    ids = np.asarray(ids)
    ticks = np.asarray(grid)[ids]
    return (
        (np.diff(ids)[:-1] == 1)
        & (np.diff(ids)[1:] == 1)
        & (np.diff(ticks)[:-1] == 1)
        & (np.diff(ticks)[1:] == 1)
    )


def refine_sequence(backend, prediction, observed, supported, grid, matrix, steps=60):
    import torch
    import torch.nn.functional as F

    def to_rotation(x):
        a = F.normalize(x[..., 0], dim=-1)
        b = F.normalize(x[..., 1] - (a * x[..., 1]).sum(-1, keepdim=True) * a, dim=-1)
        return torch.stack((a, b, torch.cross(a, b, dim=-1)), -1)

    result = {k: v.copy() for k, v in prediction.items()}
    n = prediction["trans"].shape[1]
    for side in range(2):
        valid = (
            np.isfinite(prediction["trans"][side]).all(-1)
            & np.isfinite(prediction["rotmat"][side]).all((1, 2, 3))
            & np.isfinite(prediction["betas"][side]).all(-1)
            & supported[side]
        )
        good = valid & observed[side]
        if not good.any():
            continue
        beta = np.median(prediction["betas"][side, good], axis=0).clip(-3, 3)
        all_ids = np.flatnonzero(valid)
        accum_t = np.zeros((n, 3), np.float64)
        accum_r = np.zeros((n, 16, 3, 3), np.float64)
        count = np.zeros(n, np.float64)
        starts = sorted(
            set(
                [*range(0, max(1, len(all_ids) - 255), 224), max(0, len(all_ids) - 256)]
            )
        )
        for first in starts:
            ids = all_ids[first : first + 256]
            target_t = torch.as_tensor(prediction["trans"][side, ids], device="cuda")
            target_r = torch.as_tensor(prediction["rotmat"][side, ids], device="cuda")
            target_b = torch.as_tensor(prediction["betas"][side, ids], device="cuda")
            shared_b = torch.as_tensor(beta, device="cuda").expand(len(ids), -1)
            weight = torch.as_tensor(
                np.where(observed[side, ids], 1.0, 0.2),
                device="cuda",
                dtype=torch.float32,
            )
            triple = torch.as_tensor(temporal_triples(ids, grid), device="cuda")
            focal = (
                prediction["camera_matrix_per_frame"][ids, 0, 0]
                if "camera_matrix_per_frame" in prediction
                else np.full(len(ids), matrix[0, 0])
            )
            focal = torch.as_tensor(focal, device="cuda", dtype=torch.float32)[
                :, None, None
            ]
            with torch.no_grad():
                target_j, _ = backend.geometry(target_r, target_b, target_t, side)
                target_uv = target_j[..., :2] / target_j[..., 2:].clamp_min(0.01)
            t = target_t.clone().requires_grad_(True)
            r = target_r[..., :2].clone().requires_grad_(True)
            optimizer = torch.optim.Adam([t, r], lr=0.002)
            has_triples = bool(triple.any())
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                rotations = to_rotation(r)
                joints, _ = backend.geometry(rotations, shared_b, t, side)
                uv = joints[..., :2] / joints[..., 2:].clamp_min(0.01)
                data = F.smooth_l1_loss(
                    (joints - target_j) / 0.01,
                    torch.zeros_like(joints),
                    reduction="none",
                ).mean((1, 2))
                image_delta = (uv - target_uv) * focal / 5.0
                image = F.smooth_l1_loss(
                    image_delta, torch.zeros_like(image_delta), reduction="none"
                ).mean((1, 2))
                loss = ((data + image) * weight).mean() + 0.1 * (
                    (rotations - target_r) ** 2
                ).mean()
                if has_triples:
                    acceleration = (joints[2:] - 2 * joints[1:-1] + joints[:-2])[
                        triple
                    ] / 0.005
                    loss = loss + 0.15 * F.smooth_l1_loss(
                        acceleration, torch.zeros_like(acceleration)
                    )
                    loss = (
                        loss
                        + 0.02 * ((r[2:] - 2 * r[1:-1] + r[:-2])[triple] ** 2).mean()
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_([t, r], 10.0)
                optimizer.step()
            weights = np.minimum(
                np.minimum(np.arange(1, len(ids) + 1), np.arange(len(ids), 0, -1)), 16
            )
            accum_t[ids] += t.detach().cpu().numpy() * weights[:, None]
            accum_r[ids] += (
                to_rotation(r).detach().cpu().numpy() * weights[:, None, None, None]
            )
            count[ids] += weights
        result["trans"][side, all_ids] = accum_t[all_ids] / count[all_ids, None]
        result["rotmat"][side, all_ids] = project_rotations(
            accum_r[all_ids] / count[all_ids, None, None, None]
        )
        result["betas"][side, all_ids] = beta
    return result
