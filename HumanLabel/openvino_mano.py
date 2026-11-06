"""Full FP32 WiLoR/MANO inference on Intel GPU; rotations are decoded on CPU."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OUTPUTS = ("global_orient_matrix", "hand_pose_matrix", "betas", "pred_cam", "pred_keypoints_3d", "pred_vertices")
EXPORT_REVISION = "explicit_cross_fp32_v2"


def export_model(torch_model, folder):
    import openvino as ov
    import torch

    def rot6d_to_rotmat(x):
        # Explicit components avoid the Intel GPU Roll lowering of linalg.cross.
        columns = x.reshape(-1, 2, 3).permute(0, 2, 1)
        first = torch.nn.functional.normalize(columns[:, :, 0], dim=1)
        second = columns[:, :, 1]
        second = torch.nn.functional.normalize(second - (first * second).sum(1, keepdim=True) * first, dim=1)
        third = torch.stack((first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1],
                             first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2],
                             first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]), dim=1)
        return torch.stack((first, second, third), dim=-1)

    class Export(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, patches):
            model = self.model
            pixels = patches.flip(dims=[-1]) / 255.
            pixels = (pixels - model.IMAGE_MEAN.to(pixels.dtype)) / model.IMAGE_STD.to(pixels.dtype)
            pixels = pixels.permute(0, 3, 1, 2)
            batch = pixels.shape[0]
            temporary, camera, features, latent = model.backbone(pixels[:, :, :, 32:-32])
            temporary["global_orient"] = temporary["global_orient"].reshape(batch, -1, 3, 3)
            temporary["hand_pose"] = temporary["hand_pose"].reshape(batch, -1, 3, 3)
            temporary["betas"] = temporary["betas"].reshape(batch, -1)
            vertices = model.mano(**temporary, pose2rot=False).vertices
            focal = torch.full((batch, 2), 5000., dtype=pixels.dtype)
            parameters = model.refine_net(latent, vertices, camera, features, focal)
            mesh = model.mano(**parameters, pose2rot=False)
            return (parameters["global_orient"], parameters["hand_pose"], parameters["betas"],
                    parameters["pred_cam"], mesh.joints, mesh.vertices)

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    wrapper = Export(torch_model).eval()
    from importlib import import_module
    modules = [import_module("wilor_mini.models." + name) for name in ("vit", "refinement_net")]
    originals = [module.rot6d_to_rotmat for module in modules]
    try:
        for module in modules:
            module.rot6d_to_rotmat = rot6d_to_rotmat
        with torch.inference_mode():
            model = ov.convert_model(wrapper, example_input=torch.zeros(1, 256, 256, 3),
                                     input=ov.PartialShape([ov.Dimension(1, 8), 256, 256, 3]))
    finally:
        for module, original in zip(modules, originals):
            module.rot6d_to_rotmat = original
    ov.save_model(model, folder / "wilor_fp32.xml", compress_to_fp16=False)
    (folder / "outputs.json").write_text(json.dumps({"outputs": OUTPUTS, "precision": "FP32", "export_revision": EXPORT_REVISION}), encoding="utf-8")


def decode_outputs(values):
    result = dict(zip(OUTPUTS, values))
    for output, shape in (("global_orient", (1, 3)), ("hand_pose", (15, 3))):
        matrices = result.pop(output + "_matrix")
        if (not np.isfinite(matrices).all() or np.max(np.abs(np.linalg.det(matrices) - 1)) > .01
                or np.max(np.abs(matrices @ matrices.swapaxes(-1, -2) - np.eye(3))) > .01):
            raise ValueError("Intel GPU returned invalid MANO rotation matrices")
        result[output] = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(-1, *shape).astype(np.float32)
    return result


class IntelGPUModel:
    def __init__(self, folder):
        import openvino as ov

        folder = Path(folder)
        self.core = ov.Core()
        if "GPU" not in self.core.available_devices:
            raise ValueError("OpenVINO cannot find an Intel GPU; use --device cpu")
        metadata = json.loads((folder / "outputs.json").read_text(encoding="utf-8"))
        if tuple(metadata["outputs"]) != OUTPUTS or metadata.get("export_revision") != EXPORT_REVISION:
            raise ValueError("OpenVINO model output layout mismatch")
        self.compiled = self.core.compile_model(str(folder / "wilor_fp32.xml"), "GPU", {
            "INFERENCE_PRECISION_HINT": ov.Type.f32,
            "EXECUTION_MODE_HINT": "ACCURACY",
            "PERFORMANCE_HINT": "THROUGHPUT",
            "PERFORMANCE_HINT_NUM_REQUESTS": 2,
            "CACHE_DIR": str((folder / "gpu_cache").resolve())})
        self.name = self.core.get_property("GPU", "FULL_DEVICE_NAME")
        self.request = self.compiled.create_infer_request()
        print(f"Intel GPU FP32: {self.name}; batching up to 8 hand crops", flush=True)

    def infer(self, patches):
        collected = [[] for _ in OUTPUTS]
        for start in range(0, len(patches), 8):
            self.request.infer({self.compiled.input(0): np.ascontiguousarray(patches[start:start + 8], dtype=np.float32)})
            for i in range(len(OUTPUTS)):
                collected[i].append(self.request.get_output_tensor(i).data.copy())
        return decode_outputs([np.concatenate(parts, axis=0) for parts in collected])

    def provenance(self):
        import openvino as ov
        return {"runtime": "OpenVINO", "version": ov.__version__, "device": self.name,
                "execution_devices": self.compiled.get_property("EXECUTION_DEVICES"),
                "precision": "FP32", "export_revision": EXPORT_REVISION,
                "streams": self.compiled.get_property("NUM_STREAMS"), "max_hand_batch": 8}
