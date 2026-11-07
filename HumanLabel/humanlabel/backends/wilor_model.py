"""WiLoR CUDA model and perspective geometry used by the single-frame baseline."""

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from humanlabel.utils.assets import CODE_REVISION, WEIGHTS_REVISION

THUMB_TIP_VERTEX = 744
INDEX_TIP_VERTEX = 320


def camera_translation(pred_cam, center, box_size, matrix, is_right):
    """Weak-perspective crop camera -> optical translation, using supplied K."""
    scale, tx, ty = np.asarray(pred_cam, dtype=float)
    denominator = box_size * scale
    if not np.isfinite(denominator) or denominator <= 1e-8:
        raise ValueError("invalid_mano_camera_scale")
    focal = np.sqrt(matrix[0, 0] * matrix[1, 1])
    z = 2 * focal / denominator
    offset = (np.asarray(center) - matrix[:2, 2]) * z / np.diag(matrix)[:2]
    return np.array([offset[0] + tx * (1 if is_right else -1), offset[1] + ty, z])


def fit_perspective_camera(
    pred_cam, center, box_size, matrix, image_size, joints, is_right
):
    """Preserve WiLoR's nominal image projection when changing to physical K.

    Fitting targets are model projections, not independently observed landmarks.
    The returned translation matches MANO's rotation pivot at the wrist root.
    """
    joints = np.asarray(joints, dtype=np.float64).copy()
    if not is_right:
        joints[:, 0] *= -1
    nominal = np.array(
        [
            [5000.0 / 256 * max(image_size), 0, image_size[0] / 2],
            [0, 5000.0 / 256 * max(image_size), image_size[1] / 2],
            [0, 0, 1.0],
        ]
    )
    nominal_t = camera_translation(pred_cam, center, box_size, nominal, is_right)
    targets = cv2.projectPoints(joints, np.zeros(3), nominal_t, nominal, None)[
        0
    ].reshape(-1, 2)
    initial_t = camera_translation(
        pred_cam, center, box_size, matrix, is_right
    ).reshape(3, 1)
    ok, rotation, translation = cv2.solvePnP(
        joints,
        targets,
        matrix,
        None,
        np.zeros((3, 1)),
        initial_t,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("mano_camera_conversion_failed")
    rotation, translation = cv2.solvePnPRefineLM(
        joints, targets, matrix, None, rotation, translation
    )
    projected = cv2.projectPoints(joints, rotation, translation, matrix, None)[
        0
    ].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - targets) ** 2, axis=1))))
    if not np.isfinite(error) or error > 12:
        raise ValueError("mano_camera_conversion_residual")
    rotation = cv2.Rodrigues(rotation)[0]
    # MANO global_orient rotates about J0, whereas solvePnP rotates about (0,0,0).
    translation = translation.reshape(3) + rotation @ joints[0] - joints[0]
    return translation, rotation, error


def mano_geometry(
    joints,
    vertices,
    root_rotvec,
    hand_rotvecs,
    translation,
    is_right,
    hand_scale=1.0,
    camera_rotation=None,
):
    """Left reflection changes geometry and canonical axes; the rotation stays proper."""
    joints, vertices = (
        np.asarray(joints, dtype=float).copy(),
        np.asarray(vertices, dtype=float).copy(),
    )
    root_rotvec, hand_rotvecs = (
        np.asarray(root_rotvec).reshape(3).copy(),
        np.asarray(hand_rotvecs).reshape(15, 3).copy(),
    )
    if not is_right:
        joints[:, 0] *= -1
        vertices[:, 0] *= -1
        root_rotvec[1:] *= -1  # F R F, F=diag(-1,1,1), NOT the improper F R.
        hand_rotvecs[:, 1:] *= -1
    if camera_rotation is not None:
        pivot = joints[0].copy()
        joints = (joints - pivot) @ camera_rotation.T + pivot
        vertices = (vertices - pivot) @ camera_rotation.T + pivot
        root_rotvec = Rotation.from_matrix(
            camera_rotation @ Rotation.from_rotvec(root_rotvec).as_matrix()
        ).as_rotvec()
    camera_joints = (joints + translation) * hand_scale
    camera_vertices = (vertices + translation) * hand_scale
    if (
        camera_joints.shape != (21, 3)
        or camera_vertices.shape != (778, 3)
        or not np.isfinite(camera_joints).all()
        or not np.isfinite(camera_vertices).all()
        or np.any(camera_joints[:, 2] <= 0.01)
        or np.any(camera_joints[:, 2] > 3.0)
    ):
        raise ValueError("invalid_mano_geometry")
    vector = np.r_[camera_joints[0], Rotation.from_rotvec(root_rotvec).as_quat()]
    if not np.allclose(
        camera_joints[[4, 8]],
        camera_vertices[[THUMB_TIP_VERTEX, INDEX_TIP_VERTEX]],
        atol=1e-6,
    ):
        raise ValueError("MANO fingertip indexing mismatch")
    distance = float(
        np.linalg.norm(
            camera_vertices[THUMB_TIP_VERTEX] - camera_vertices[INDEX_TIP_VERTEX]
        )
    )
    return vector, distance, camera_joints, camera_vertices, root_rotvec, hand_rotvecs


class WiLoRBackend:
    def __init__(self, cache, repo, device="cuda", threads=8):
        import torch

        sys.path.insert(0, str(Path(repo).resolve()))
        from wilor_mini.models.wilor import WiLor
        from wilor_mini.utils.utils import generate_image_patch_cv2

        self.torch = torch
        self.execution_device = device
        self.device = torch.device(device)
        if device not in ("cuda", "cpu"):
            raise ValueError("WiLoR comparison supports CUDA or CPU")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        torch.set_num_threads(threads)
        self.cache = Path(cache).resolve()
        folder = self.cache / "pretrained_models"
        self.model = WiLor(
            mano_model_path=str(folder / "MANO_RIGHT_numpy.pkl"),
            mano_mean_path=str(folder / "mano_mean_params.npz"),
            focal_length=5000,
            image_size=256,
        )
        checkpoint = torch.load(
            folder / "wilor_final.ckpt", map_location="cpu", weights_only=True
        )
        mismatch = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        bad = [
            key
            for key in mismatch.missing_keys
            if key.startswith(("backbone.", "refine_net."))
        ]
        unexpected = [
            key
            for key in mismatch.unexpected_keys
            if key != "initialized" and not key.startswith("discriminator.")
        ]
        if bad or unexpected:
            raise ValueError(f"WiLoR checkpoint mismatch: {bad[:5]}, {unexpected[:5]}")
        self.model.eval().to(self.device, dtype=torch.float32)
        self.crop = generate_image_patch_cv2
        import pickle

        with (folder / "MANO_RIGHT_numpy.pkl").open("rb") as stream:
            self.faces = np.asarray(pickle.load(stream)["f"], dtype=np.int32)

    def provenance(self):
        import json
        from importlib.metadata import version

        return {
            "backend": "WiLoR-mini + MANO",
            "device": self.execution_device,
            "dtype": "float32",
            "code_revision": CODE_REVISION,
            "weights_revision": WEIGHTS_REVISION,
            "libraries": {
                name: version(name)
                for name in ("torch", "torchvision", "numpy", "scipy", "smplx", "timm")
            },
            "assets": json.loads(
                (self.cache / "weights.json").read_text(encoding="utf-8")
            ),
        }

    def infer_many(self, prepared, args):
        patches = [
            patch for context in prepared.values() for patch in context["patches"]
        ]
        if not patches:
            return {camera: {"left": [], "right": []} for camera in prepared}
        patches = np.asarray(patches, dtype=np.float32)
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(patches).to(self.device)
            predictions = {
                key: value.cpu().numpy() for key, value in self.model(tensor).items()
            }
        result, start = {}, 0
        for camera, context in prepared.items():
            count = len(context["patches"])
            result[camera] = self.decode_image(
                context,
                {
                    key: value[start : start + count]
                    for key, value in predictions.items()
                },
                args,
            )
            start += count
        return result

    def decode_image(self, context, predictions, args):
        centers, sizes, classes, scores = (
            context[key] for key in ("centers", "sizes", "classes", "scores")
        )
        matrix, distortion, width, height = (
            context[key] for key in ("matrix", "distortion", "width", "height")
        )
        hands = {"left": [], "right": []}
        for i, (center, size, is_right, score) in enumerate(
            zip(centers, sizes, classes, scores)
        ):
            side = "right" if is_right else "left"
            if args.swap_handedness:
                side = "left" if side == "right" else "right"
            try:
                translation, view_rotation, conversion_error = fit_perspective_camera(
                    predictions["pred_cam"][i],
                    center,
                    size,
                    matrix,
                    (width, height),
                    predictions["pred_keypoints_3d"][i],
                    bool(is_right),
                )
                vector, distance, joints, vertices, root, articulation = mano_geometry(
                    predictions["pred_keypoints_3d"][i],
                    predictions["pred_vertices"][i],
                    predictions["global_orient"][i],
                    predictions["hand_pose"][i],
                    translation,
                    bool(is_right),
                    args.hand_scale,
                    view_rotation,
                )
            except ValueError:
                continue
            pixels = cv2.projectPoints(
                joints, np.zeros(3), np.zeros(3), matrix, distortion
            )[0].reshape(-1, 2)
            inside = (
                np.isfinite(pixels).all(axis=1)
                & (pixels >= 0).all(axis=1)
                & (pixels < [width, height]).all(axis=1)
            )
            hands[side].append(
                {
                    "pixels": pixels,
                    "world": joints,
                    "score": float(score),
                    "handedness_score": np.nan,
                    "inside": inside,
                    "image_height": height,
                    "wrist_in_image": bool(inside[0]),
                    "pinch_valid": bool(inside[[4, 8]].all()),
                    "distance": distance if inside[[4, 8]].all() else np.nan,
                    "distance_px": float(np.linalg.norm(pixels[4] - pixels[8]))
                    if inside[[4, 8]].all()
                    else np.nan,
                    "reason": "ok",
                    "pose": vector,
                    "mesh": vertices,
                    "faces": self.faces,
                    "camera_conversion_rmse_px": conversion_error,
                    "mano": {
                        "global_orient": root,
                        "hand_pose": articulation,
                        "betas": predictions["betas"][i],
                        "camera_translation_m": translation * args.hand_scale,
                        "joints_camera_m": joints,
                        "is_right": bool(is_right),
                    },
                }
            )
        return hands
