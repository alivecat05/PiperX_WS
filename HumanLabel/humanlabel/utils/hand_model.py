"""MANO asset preparation; an explicit reflected-right left template.

The left template is generated mathematically from the already available right
asset, rather than pretending that an official MANO_LEFT asset was supplied.
"""

import pickle
from pathlib import Path

import numpy as np

from humanlabel.utils.assets import convert_mano


def mirror_mano(data):
    result = {k: v.copy() if hasattr(v, "copy") else v for k, v in data.items()}
    reflection = np.array([-1.0, 1.0, 1.0])
    result["v_template"] = data["v_template"] * reflection
    result["shapedirs"] = data["shapedirs"] * reflection[None, :, None]
    # vec(F R F - I) = (F tensor F) vec(R - I).
    pose_sign = np.tile((reflection[:, None] * reflection[None, :]).ravel(), 15)
    result["posedirs"] = data["posedirs"] * reflection[None, :, None] * pose_sign
    aa_sign = np.tile([1.0, -1.0, -1.0], 15)
    for key in ("hands_components", "hands_mean"):
        result[key] = data[key] * aa_sign
    result["f"] = np.asarray(data["f"])[:, [0, 2, 1]]
    return result


def prepare_mano(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    right = destination / "MANO_RIGHT.pkl"
    if not right.exists():
        convert_mano(source, right)
    with right.open("rb") as f:
        data = pickle.load(f)
    left = destination / "MANO_LEFT.pkl"
    if not left.exists():
        with left.open("wb") as f:
            pickle.dump(mirror_mano(data), f, protocol=4)
    return right, left


class MANOGeometry:
    """Small CUDA geometry-only backend, independent of image-network weights."""

    def __init__(self, model_dir, device="cuda"):
        import smplx
        import torch

        self.torch = torch
        self.layer = (
            smplx.MANOLayer(str(model_dir), is_rhand=True)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        self.faces = self.layer.faces

    def geometry(self, rotmat, betas, trans, side, grad=False):
        r = rotmat
        if side == 0:
            f = r.new_tensor([-1.0, 1.0, 1.0])
            r = r * f[:, None] * f[None, :]
        out = self.layer(
            global_orient=r[:, :1], hand_pose=r[:, 1:], betas=betas, pose2rot=False
        )
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
        tips = out.vertices[:, [744, 320, 443, 554, 671]]
        joints = self.torch.cat((out.joints[:, :16], tips), 1)[:, order]
        vertices = out.vertices
        if side == 0:
            f = joints.new_tensor([-1.0, 1.0, 1.0])
            joints, vertices = joints * f, vertices * f
        return joints + trans[:, None], vertices + trans[:, None]
