"""WiLoR -> MANO wrist/root pose, mesh and thumb/index distances."""
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from mano_assets import CODE_REVISION, WEIGHTS_REVISION

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


def fit_perspective_camera(pred_cam, center, box_size, matrix, image_size, joints, is_right):
    """Preserve WiLoR's nominal image projection when changing to physical K.

    Fitting targets are model projections, not independently observed landmarks.
    The returned translation matches MANO's rotation pivot at the wrist root.
    """
    joints = np.asarray(joints, dtype=np.float64).copy()
    if not is_right:
        joints[:, 0] *= -1
    nominal = np.array([[5000. / 256 * max(image_size), 0, image_size[0] / 2],
                        [0, 5000. / 256 * max(image_size), image_size[1] / 2], [0, 0, 1.]])
    nominal_t = camera_translation(pred_cam, center, box_size, nominal, is_right)
    targets = cv2.projectPoints(joints, np.zeros(3), nominal_t, nominal, None)[0].reshape(-1, 2)
    initial_t = camera_translation(pred_cam, center, box_size, matrix, is_right).reshape(3, 1)
    ok, rotation, translation = cv2.solvePnP(joints, targets, matrix, None, np.zeros((3, 1)), initial_t,
                                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("mano_camera_conversion_failed")
    rotation, translation = cv2.solvePnPRefineLM(joints, targets, matrix, None, rotation, translation)
    projected = cv2.projectPoints(joints, rotation, translation, matrix, None)[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - targets) ** 2, axis=1))))
    if not np.isfinite(error) or error > 12:
        raise ValueError("mano_camera_conversion_residual")
    rotation = cv2.Rodrigues(rotation)[0]
    # MANO global_orient rotates about J0, whereas solvePnP rotates about (0,0,0).
    translation = translation.reshape(3) + rotation @ joints[0] - joints[0]
    return translation, rotation, error


def mano_geometry(joints, vertices, root_rotvec, hand_rotvecs, translation, is_right, hand_scale=1., camera_rotation=None):
    """Left reflection changes geometry and canonical axes; the rotation stays proper."""
    joints, vertices = np.asarray(joints, dtype=float).copy(), np.asarray(vertices, dtype=float).copy()
    root_rotvec, hand_rotvecs = np.asarray(root_rotvec).reshape(3).copy(), np.asarray(hand_rotvecs).reshape(15, 3).copy()
    if not is_right:
        joints[:, 0] *= -1
        vertices[:, 0] *= -1
        root_rotvec[1:] *= -1  # F R F, F=diag(-1,1,1), NOT the improper F R.
        hand_rotvecs[:, 1:] *= -1
    if camera_rotation is not None:
        pivot = joints[0].copy()
        joints = (joints - pivot) @ camera_rotation.T + pivot
        vertices = (vertices - pivot) @ camera_rotation.T + pivot
        root_rotvec = Rotation.from_matrix(camera_rotation @ Rotation.from_rotvec(root_rotvec).as_matrix()).as_rotvec()
    camera_joints = (joints + translation) * hand_scale
    camera_vertices = (vertices + translation) * hand_scale
    if (camera_joints.shape != (21, 3) or camera_vertices.shape != (778, 3)
            or not np.isfinite(camera_joints).all() or not np.isfinite(camera_vertices).all()
            or np.any(camera_joints[:, 2] <= .01) or np.any(camera_joints[:, 2] > 3.0)):
        raise ValueError("invalid_mano_geometry")
    vector = np.r_[camera_joints[0], Rotation.from_rotvec(root_rotvec).as_quat()]
    if not np.allclose(camera_joints[[4, 8]], camera_vertices[[THUMB_TIP_VERTEX, INDEX_TIP_VERTEX]], atol=1e-6):
        raise ValueError("MANO fingertip indexing mismatch")
    distance = float(np.linalg.norm(camera_vertices[THUMB_TIP_VERTEX] - camera_vertices[INDEX_TIP_VERTEX]))
    return vector, distance, camera_joints, camera_vertices, root_rotvec, hand_rotvecs


class WiLoRBackend:
    def __init__(self, cache, repo, device="cpu", threads=8, detection_score=.3):
        import torch

        self.torch = torch
        if device == "auto":
            try:
                import openvino as ov
                device = "igpu" if "GPU" in ov.Core().available_devices else "cpu"
            except ImportError:
                device = "cpu"
        self.execution_device = device
        self.device = torch.device("cpu" if device == "igpu" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; use --device cpu")
        torch.set_num_threads(threads)
        self.detection_score = detection_score
        cache, repo = Path(cache).resolve(), Path(repo).resolve()
        self.cache = cache
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "yolo"))
        sys.path.insert(0, str(repo))
        from wilor_mini.utils.utils import generate_image_patch_cv2
        from ultralytics import YOLO

        folder = cache / "pretrained_models"
        gpu_folder = cache / "openvino_v2"
        self.model, self.gpu_model = None, None
        if device != "igpu" or not (gpu_folder / "outputs.json").exists():
            from wilor_mini.models.wilor import WiLor
            self.model = WiLor(mano_model_path=str(folder / "MANO_RIGHT_numpy.pkl"),
                           mano_mean_path=str(folder / "mano_mean_params.npz"), focal_length=5000, image_size=256)
            checkpoint = torch.load(folder / "wilor_final.ckpt", map_location="cpu", weights_only=True)
            mismatch = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            bad = [key for key in mismatch.missing_keys if key.startswith(("backbone.", "refine_net."))]
            unexpected = [key for key in mismatch.unexpected_keys
                      if key != "initialized" and not key.startswith("discriminator.")]
            if bad or unexpected:
                raise ValueError(f"WiLoR checkpoint mismatch: {bad[:5]}, {unexpected[:5]}")
            del checkpoint
            self.model.eval().to(self.device, dtype=torch.float32)
        if device == "igpu":
            from openvino_mano import IntelGPUModel, export_model
            if not (gpu_folder / "outputs.json").exists():
                print("Exporting full FP32 WiLoR/MANO for Intel GPU...", flush=True)
                export_model(self.model, gpu_folder)
            self.model = None
            import gc
            gc.collect()
            self.gpu_model = IntelGPUModel(gpu_folder)
        self.detector = YOLO(str(folder / "detector.pt"))
        self.crop = generate_image_patch_cv2
        import pickle
        with (folder / "MANO_RIGHT_numpy.pkl").open("rb") as stream:
            self.faces = np.asarray(pickle.load(stream)["f"], dtype=np.int32)

    def provenance(self):
        import json
        from importlib.metadata import version
        return {"backend": "WiLoR-mini + MANO", "device": self.execution_device, "dtype": "float32",
                "accelerator": self.gpu_model.provenance() if self.gpu_model else None,
                "code_revision": CODE_REVISION, "weights_revision": WEIGHTS_REVISION,
                "libraries": {name: version(name) for name in ("torch", "torchvision", "numpy", "scipy", "smplx", "timm")},
                "assets": json.loads((self.cache / "weights.json").read_text(encoding="utf-8"))}

    def detect(self, image, matrix, distortion, args):
        return self.detect_many({"image": image}, {"image": (matrix, distortion)}, args)["image"]

    def prepare_image(self, image, matrix, distortion):
        from skimage.filters import gaussian

        torch = self.torch
        height, width = image.shape[:2]
        if abs(matrix[0, 1]) > 1e-8:
            raise ValueError("MANO backend requires zero-skew pinhole intrinsics")
        # WiLoR sees pinhole pixels. Rectification with R=I preserves the optical axes.
        if np.any(distortion):
            image = cv2.undistort(image, matrix, distortion, None, matrix)
        with torch.inference_mode():
            result = self.detector.predict(image, conf=self.detection_score, verbose=False,
                                           device=str(self.device))[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            patches, centers, sizes = [], (boxes[:, :2] + boxes[:, 2:]) / 2, 2.5 * (boxes[:, 2:] - boxes[:, :2]).max(axis=1)
            for center, size, is_right in zip(centers, sizes, classes):
                view = image
                downsampling = size / 256 / 2
                if downsampling > 1.1:
                    view = gaussian(view, sigma=(downsampling - 1) / 2, channel_axis=2, preserve_range=True)
                patch, _ = self.crop(view, *center, size, size, 256, 256, not bool(is_right), 1., 0,
                                     border_mode=cv2.BORDER_CONSTANT)
                patches.append(patch)
        return {"patches": patches, "centers": centers, "sizes": sizes, "classes": classes, "scores": scores,
                "width": width, "height": height, "matrix": matrix, "distortion": distortion}

    def detect_many(self, images, camera_models, args):
        return self.infer_many(self.prepare_many(images, camera_models), args)

    def prepare_many(self, images, camera_models):
        return {camera: self.prepare_image(image, *camera_models[camera][:2]) for camera, image in images.items()}

    def infer_many(self, prepared, args):
        patches = [patch for context in prepared.values() for patch in context["patches"]]
        if not patches:
            return {camera: {"left": [], "right": []} for camera in prepared}
        patches = np.asarray(patches, dtype=np.float32)
        if self.gpu_model is not None:
            predictions = self.gpu_model.infer(patches)
        else:
            with self.torch.inference_mode():
                tensor = self.torch.from_numpy(patches).to(self.device)
                predictions = {key: value.cpu().numpy() for key, value in self.model(tensor).items()}
        result, start = {}, 0
        for camera, context in prepared.items():
            count = len(context["patches"])
            result[camera] = self.decode_image(context, {key: value[start:start + count] for key, value in predictions.items()}, args)
            start += count
        return result

    def iter_frames(self, caps, camera_models, args, count):
        """Overlap CPU preparation, GPU inference and the caller's video rendering."""
        from concurrent.futures import ThreadPoolExecutor

        def prepare_frame(index):
            images = {}
            for camera, video in caps.items():
                ok, image = video.read()
                if not ok:
                    raise ValueError(f"Cannot decode {camera} source frame {index}")
                images[camera] = image
            return images, self.prepare_many(images, camera_models)

        def infer_frame(prepared):
            images, contexts = prepared.result()
            return images, self.infer_many(contexts, args)

        with (ThreadPoolExecutor(max_workers=1, thread_name_prefix="mano-preprocess") as preparer,
              ThreadPoolExecutor(max_workers=1, thread_name_prefix="mano-gpu") as inference):
            def submit(index):
                return inference.submit(infer_frame, preparer.submit(prepare_frame, index))

            pending = submit(0) if count else None
            following = submit(1) if count > 1 else None
            for index in range(count):
                result = pending.result()
                pending = following
                if index + 2 < count:
                    following = submit(index + 2)
                yield result

    def decode_image(self, context, predictions, args):
        centers, sizes, classes, scores = (context[key] for key in ("centers", "sizes", "classes", "scores"))
        matrix, distortion, width, height = (context[key] for key in ("matrix", "distortion", "width", "height"))
        hands = {"left": [], "right": []}
        for i, (center, size, is_right, score) in enumerate(zip(centers, sizes, classes, scores)):
            side = "right" if is_right else "left"
            if args.swap_handedness:
                side = "left" if side == "right" else "right"
            try:
                translation, view_rotation, conversion_error = fit_perspective_camera(
                    predictions["pred_cam"][i], center, size, matrix, (width, height),
                    predictions["pred_keypoints_3d"][i], bool(is_right))
                vector, distance, joints, vertices, root, articulation = mano_geometry(
                    predictions["pred_keypoints_3d"][i], predictions["pred_vertices"][i],
                    predictions["global_orient"][i], predictions["hand_pose"][i], translation, bool(is_right),
                    args.hand_scale, view_rotation)
            except ValueError:
                continue
            pixels = cv2.projectPoints(joints, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
            inside = np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1) & (pixels < [width, height]).all(axis=1)
            hands[side].append({"pixels": pixels, "world": joints, "score": float(score), "handedness_score": np.nan,
                "inside": inside, "image_height": height, "wrist_in_image": bool(inside[0]),
                "pinch_valid": bool(inside[[4, 8]].all()), "distance": distance if inside[[4, 8]].all() else np.nan,
                "distance_px": float(np.linalg.norm(pixels[4] - pixels[8])) if inside[[4, 8]].all() else np.nan,
                "reason": "ok", "pose": vector, "mesh": vertices, "faces": self.faces,
                "camera_conversion_rmse_px": conversion_error,
                "mano": {"global_orient": root, "hand_pose": articulation, "betas": predictions["betas"][i],
                         "camera_translation_m": translation * args.hand_scale, "joints_camera_m": joints,
                         "is_right": bool(is_right)}})
        return hands


def draw_mesh(frame, hand, matrix, distortion, side):
    if "mesh" not in hand:
        return
    pixels = cv2.projectPoints(hand["mesh"], np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    polygons = pixels[hand["faces"]]
    height, width = frame.shape[:2]
    valid = np.isfinite(polygons).all(axis=(1, 2)) & (hand["mesh"][hand["faces"], 2] > .01).all(axis=1)
    valid &= (np.abs(polygons).max(axis=(1, 2)) < 4 * max(width, height))
    overlay = frame.copy()
    color = (70, 220, 90) if side == "left" else (30, 170, 255)
    cv2.polylines(overlay, np.rint(polygons[valid]).astype(np.int32), True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, .65, frame, .35, 0, dst=frame)


class ManoArchive:
    def __init__(self, count, cameras, hand_scale):
        self.cameras, self.hand_scale = cameras, hand_scale
        shapes = {"global_orient": (3,), "hand_pose": (15, 3), "betas": (10,),
                  "camera_translation_m": (3,), "joints_camera_m": (21, 3)}
        self.data = {key: np.full((count, len(cameras), 2, *shape), np.nan, np.float32)
                     for key, shape in shapes.items()}
        self.valid = np.zeros((count, len(cameras), 2), bool)
        self.is_right = np.full((count, len(cameras), 2), -1, np.int8)

    def add(self, frame, camera, side, hand):
        if "mano" not in hand:
            return
        index = (frame, self.cameras.index(camera), ("left", "right").index(side))
        self.valid[index] = True
        self.is_right[index] = int(hand["mano"]["is_right"])
        for key in self.data:
            self.data[key][index] = hand["mano"][key]

    def save(self, path, count):
        np.savez_compressed(path, **{key: value[:count] for key, value in self.data.items()},
                            model_valid=self.valid[:count], mano_is_right=self.is_right[:count], frame_index=np.arange(count),
                            camera_names=np.asarray(self.cameras), hand_sides=np.asarray(["left", "right"]),
                            hand_scale=np.asarray(self.hand_scale))
