"""Existing WiLoR/MANO estimator as a fair single-frame head-view baseline."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class WiLoRVideoBackend:
    def __init__(self, root):
        import os

        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from humanlabel.backends.wilor_model import WiLoRBackend

        self.root = Path(root)
        self.backend = WiLoRBackend(
            self.root / ".cache/wilor", self.root / ".cache/WiLoR-mini", device="cuda"
        )
        self.model = self.backend.model
        self.faces = self.backend.faces
        self.device = "cuda"

    def inference(
        self, episode, paths, boxes, observed, supported, grid, matrix, output
    ):
        from humanlabel.utils.episode import save_npz

        cache = output / "prediction_raw.npz"
        if cache.exists():
            with np.load(cache) as f:
                return (
                    {k: f[k] for k in f.files},
                    matrix,
                    {"model": self.backend.provenance()},
                )
        n = len(paths)
        result = {
            "trans": np.full((2, n, 3), np.nan, np.float32),
            "rotmat": np.full((2, n, 16, 3, 3), np.nan, np.float32),
            "betas": np.full((2, n, 10), np.nan, np.float32),
            "completion": np.zeros((2, n), bool),
            "visible": observed.astype(np.float32),
            "presence": observed.astype(np.float32),
        }
        args = SimpleNamespace(swap_handedness=False, hand_scale=1.0)
        for a in range(0, n, 2):
            prepared = {}
            for i in range(a, min(n, a + 2)):
                image = cv2.imread(paths[i])
                height, width = image.shape[:2]
                ids = np.flatnonzero(observed[:, i])
                bs = boxes[ids, i]
                centers = (bs[:, :2] + bs[:, 2:]) / 2
                sizes = 2.5 * np.max(bs[:, 2:] - bs[:, :2], axis=-1)
                patches = []
                for side, center, size in zip(ids, centers, sizes):
                    view = image
                    downsampling = size / 256 / 2
                    if downsampling > 1.1:
                        from skimage.filters import gaussian

                        view = gaussian(
                            image,
                            sigma=(downsampling - 1) / 2,
                            channel_axis=2,
                            preserve_range=True,
                        )
                    patches.append(
                        self.backend.crop(
                            view,
                            *center,
                            size,
                            size,
                            256,
                            256,
                            side == 0,
                            1.0,
                            0,
                            border_mode=cv2.BORDER_CONSTANT,
                        )[0]
                    )
                prepared[str(i)] = {
                    "patches": patches,
                    "centers": centers,
                    "sizes": sizes,
                    "classes": ids,
                    "scores": np.ones(len(ids)),
                    "width": width,
                    "height": height,
                    "matrix": matrix,
                    "distortion": np.zeros(5),
                }
            detected = self.backend.infer_many(prepared, args)
            for key, hands in detected.items():
                i = int(key)
                for side, name in enumerate(("left", "right")):
                    if not hands[name]:
                        continue
                    mano = hands[name][0]["mano"]
                    aa = np.r_[
                        mano["global_orient"].reshape(1, 3),
                        mano["hand_pose"].reshape(15, 3),
                    ]
                    result["rotmat"][side, i] = Rotation.from_rotvec(aa).as_matrix()
                    result["trans"][side, i] = mano["camera_translation_m"]
                    result["betas"][side, i] = mano["betas"]
            if a % 100 == 0:
                print(f"  WiLoR {episode.name}: {a}/{n}", flush=True)
        save_npz(cache, result)
        return result, matrix, {"model": self.backend.provenance()}

    def geometry(self, rotmat, betas, trans, side, grad=False):
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
