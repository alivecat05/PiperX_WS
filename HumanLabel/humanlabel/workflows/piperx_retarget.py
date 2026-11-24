"""Convert camera-frame EgoHand wrist eposes and SOMA meshes to Piper X actions."""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import osqp
import yaml
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from humanlabel.export import SIDES, frame_aligned_actions, read_action_rows


WRIST, THUMB_KNUCKLE, THUMB_TIP, INDEX_KNUCKLE, INDEX_TIP = 0, 2, 4, 6, 9
SWAP_JAWS = Rotation.from_rotvec([0, 0, np.pi])


class PiperKinematics:
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for geom in link.findall(tag):
                    link.remove(geom)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        if self.model.nq != 6 or self.model.nv != 6:
            raise ValueError("Expected the six-joint Piper X arm-only URDF")
        self.data = mujoco.MjData(self.model)
        self.tip = self.model.body("link6").id
        self.low, self.high = self.model.jnt_range.T
        self.speed_limits = np.array([
            float(root.find(f"joint[@name='{self.model.joint(i).name}']/limit").get("velocity"))
            for i in range(self.model.njnt)
        ])
        self.position_jacobian = np.zeros((3, 6))
        self.rotation_jacobian = np.zeros((3, 6))

    def evaluate(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacBody(
            self.model, self.data, self.position_jacobian, self.rotation_jacobian, self.tip
        )
        return self.data.xpos[self.tip].copy(), Rotation.from_matrix(
            self.data.xmat[self.tip].reshape(3, 3)
        )


def load_grasp(path, side):
    with np.load(path) as archive:
        hand = list(archive["sides"]).index(side)
        joints = archive["native_joints25_camera_m"][hand].astype(float)
        vertices = archive["vertices_camera_m"][hand]
        valid = archive["model_valid"][hand].copy()
        grid_index = archive["grid_index"].copy()

    thumb, index = joints[:, THUMB_TIP], joints[:, INDEX_TIP]
    center = (thumb + index) / 2
    span = index - thumb
    width = np.linalg.norm(span, axis=1)
    valid &= np.isfinite(joints).all(axis=(1, 2)) & (width > .001)
    if not valid.any():
        raise ValueError(f"SOMA has no valid {side}-hand grasp estimate")

    # Select a fixed patch on the mesh near the first web space in open-hand
    # frames. Use the same vertices after the hand deforms in every frame.
    knuckle_midpoint = (joints[:, THUMB_KNUCKLE] + joints[:, INDEX_KNUCKLE]) / 2
    web_proxy = .88 * knuckle_midpoint + .12 * joints[:, WRIST]
    mesh_valid = valid & np.isfinite(vertices).all(axis=(1, 2))
    if not mesh_valid.any():
        raise ValueError(f"SOMA has no usable {side}-hand mesh")
    open_hand = mesh_valid & (width >= np.percentile(width[mesh_valid], 65))
    patch_distance = np.median(
        np.linalg.norm(vertices[open_hand] - web_proxy[open_hand, None, :], axis=2),
        axis=0,
    )
    patch = np.argsort(patch_distance)[:12]
    web = vertices[:, patch].mean(axis=1).astype(float)

    jaw = np.divide(span, width[:, None], out=np.zeros_like(span), where=width[:, None] > .001)
    approach = center - web
    approach -= np.sum(approach * jaw, axis=1, keepdims=True) * jaw
    approach_norm = np.linalg.norm(approach, axis=1)
    valid &= mesh_valid & (approach_norm > .008)
    if not valid.any():
        raise ValueError(f"No usable SOMA {side}-hand thumb web to fingertip direction")
    known = np.flatnonzero(valid)
    source = known[np.maximum(0, np.searchsorted(known, np.arange(len(joints)), side="right") - 1)]
    jaw = jaw[source]
    approach = approach[source] / approach_norm[source, None]
    axes = np.stack((jaw, np.cross(approach, jaw), approach), axis=-1)
    return center[source], Rotation.from_matrix(axes), width[source], grid_index


def flange_to_jaw_midpoint(path):
    root = ET.parse(path)
    offsets = {}
    rotations = {}
    for name in ("flange_joint", "gripper_base_joint", "gripper_joint1", "gripper_joint2"):
        origin = root.find(f"joint[@name='{name}']/origin")
        offsets[name] = np.fromstring(origin.get("xyz"), sep=" ")
        rotations[name] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" "))
    fixed = offsets["gripper_base_joint"]
    midpoint = (offsets["gripper_joint1"] + offsets["gripper_joint2"]) / 2
    return offsets["flange_joint"] + rotations["flange_joint"].apply(
        fixed + rotations["gripper_base_joint"].apply(midpoint)
    )


def skew(vectors):
    x, y, z = vectors.T
    zeros = np.zeros_like(x)
    return np.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), axis=1).reshape(-1, 3, 3)


def smooth_grasp_rotations(rotations, sigma_frames):
    if sigma_frames <= 0:
        return rotations
    direct = rotations.as_quat()
    swapped = (rotations * SWAP_JAWS).as_quat()
    continuous = direct.copy()
    for i in range(1, len(continuous)):
        previous = continuous[i - 1]
        alternatives = np.stack((direct[i], swapped[i]))
        dots = alternatives @ previous
        best = np.argmax(np.abs(dots))
        continuous[i] = np.copysign(1, dots[best]) * alternatives[best]
    filtered = gaussian_filter1d(continuous, sigma_frames, axis=0, mode="nearest")
    return Rotation.from_quat(filtered)


def rotation_log_jacobian(angle):
    angle = np.atleast_2d(angle)
    theta = np.linalg.norm(angle, axis=1)
    hat = skew(angle)
    coeff = np.full_like(theta, 1 / 12)
    big = theta > 1e-4
    coeff[big] = (1 - theta[big] / (2 * np.tan(theta[big] / 2))) / theta[big]**2
    return np.eye(3) - .5 * hat + coeff[:, None, None] * (hat @ hat)


class GraspPath:
    def __init__(self, arm, offset, positions, rotations, width, fps, rotation_scale,
                 smoothing_seconds=.10, joint_smoothing=.12):
        self.arm = arm
        self.offset = offset
        self.positions = positions
        self.rotations = smooth_grasp_rotations(rotations, smoothing_seconds * fps)
        self.primary = self.rotations.inv()
        self.swapped = (self.rotations * SWAP_JAWS).inv()
        self.count = len(positions)
        # The fingertips constrain rotation most strongly while pinching.
        pinch_weight = np.clip((.065 - width) / .045, 0, 1)
        self.rotation_weight = rotation_scale * (.48 + .52 * pinch_weight)
        self.low = np.tile(arm.low, self.count)
        self.high = np.tile(arm.high, self.count)
        self.difference = sparse.csc_matrix((0, 6 * self.count))
        if self.count > 1:
            ones = np.ones(self.count - 1)
            diff = sparse.diags([-ones, ones], [0, 1], shape=(self.count - 1, self.count))
            self.difference = sparse.kron(diff, sparse.eye(6), format="csc")
        acceleration = self.difference[6:] - self.difference[:-6]
        self.smoothness = (
            .016**2 * (self.difference.T @ self.difference)
            + joint_smoothing**2 * (acceleration.T @ acceleration)
        ).tocsc()
        self.step_limit = np.tile(arm.speed_limits * .999 / fps, self.count - 1)
        self.constraint = sparse.vstack((sparse.eye(6 * self.count), self.difference), format="csc")

    def model(self, q):
        n = self.count
        flange = np.empty((n, 3))
        rotation = np.empty((n, 4))
        position_jac = np.empty((n, 3, 6))
        rotation_jac = np.empty((n, 3, 6))
        for i, frame in enumerate(q):
            flange[i], rot = self.arm.evaluate(frame)
            rotation[i] = rot.as_quat()
            position_jac[i] = self.arm.position_jacobian
            rotation_jac[i] = self.arm.rotation_jacobian
        rotation = Rotation.from_quat(rotation)
        pad = rotation.apply(np.broadcast_to(self.offset, flange.shape))
        jaw_position_jac = position_jac - skew(pad) @ rotation_jac
        return flange, rotation, flange + pad, jaw_position_jac, rotation_jac

    def evaluate(self, q):
        _, actual, jaw, jaw_jac, rot_jac = self.model(q)
        first = (actual * self.primary).as_rotvec()
        second = (actual * self.swapped).as_rotvec()
        use_second = np.sum(second**2, axis=1) < np.sum(first**2, axis=1)
        angle = np.where(use_second[:, None], second, first)
        spatial_log_jac = rotation_log_jacobian(angle)
        weight = self.rotation_weight
        jaw_error = jaw - self.positions
        # Keep acceleration smoothing from pulling the jaw off the grasp point
        # during brief, fast reaches.
        distance = np.linalg.norm(jaw_error, axis=1)
        outside = distance > .004
        unit = jaw_error[outside] / distance[outside, None]
        ratio = .004 / distance[outside]
        gain = 1 + 1.6 * (1 - ratio)
        jaw_error[outside] *= gain[:, None]
        jaw_gradient = gain[:, None, None] * np.eye(3)
        jaw_gradient += (1.6 * ratio)[:, None, None] * (unit[:, :, None] * unit[:, None, :])
        jaw_jac[outside] = jaw_gradient @ jaw_jac[outside]
        residual = np.column_stack((jaw_error, weight[:, None] * angle))
        jac = np.concatenate((jaw_jac, weight[:, None, None] * (spatial_log_jac @ rot_jac)), axis=1)
        flat = q.ravel()
        cost = .5 * (np.sum(residual**2) + flat @ (self.smoothness @ flat))
        return cost, residual, jac

    def feasible_seed(self, q):
        flat = np.clip(q.ravel(), self.low, self.high)
        if np.all(np.abs(self.difference @ flat) <= self.step_limit):
            return flat.reshape(self.count, 6)
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.eye(len(flat), format="csc"), q=-flat, A=self.constraint,
            l=np.r_[self.low, -self.step_limit], u=np.r_[self.high, self.step_limit],
            verbose=False, polishing=True, eps_abs=1e-7, eps_rel=1e-7,
        )
        result = solver.solve(raise_error=True)
        steps = np.abs(self.difference @ result.x)
        if np.any(steps > self.step_limit + 1e-6):
            raise RuntimeError("Could not construct a velocity-feasible IK seed")
        return result.x.reshape(self.count, 6)

    def initial_path(self):
        """Find a branch-continuous seed using both valid jaw assignments."""
        arm = self.arm
        neutral = (arm.low + arm.high) / 2
        rng = np.random.default_rng(2391)
        restarts = [neutral, *rng.uniform(arm.low, arm.high, (5, 6))]
        candidates = np.empty((self.count, 2, 6))
        costs = np.empty((self.count, 2))
        swap = self.rotations * SWAP_JAWS
        step_limit = self.step_limit[:6]
        parent = np.zeros((self.count, 2), dtype=int)
        total = np.zeros(2)
        for frame, position in enumerate(self.positions):
            starts = restarts if frame == 0 else [*candidates[frame - 1], *restarts]
            weight = self.rotation_weight[frame]
            for branch, target in enumerate((self.rotations[frame], swap[frame])):
                target_inv = target.inv()

                def residual(q):
                    flange, actual = arm.evaluate(q)
                    jaw = flange + actual.apply(self.offset)
                    error = (actual * target_inv).as_rotvec()
                    return np.r_[jaw - position, weight * error]

                def jacobian(q):
                    _, actual = arm.evaluate(q)
                    pad = actual.apply(self.offset)
                    error = (actual * target_inv).as_rotvec()
                    jaw_jac = arm.position_jacobian - skew(pad[None])[0] @ arm.rotation_jacobian
                    rotation_jac = rotation_log_jacobian(error)[0] @ arm.rotation_jacobian
                    return np.vstack((jaw_jac, weight * rotation_jac))

                best = None
                for seed in starts:
                    fit = least_squares(residual, np.clip(seed, arm.low, arm.high), jac=jacobian,
                                        bounds=(arm.low, arm.high), max_nfev=40, gtol=1e-7)
                    if best is None or fit.cost < best.cost:
                        best = fit
                    if np.linalg.norm(fit.fun[:3]) < .001 and np.linalg.norm(fit.fun[3:]) < weight * .03:
                        break
                candidates[frame, branch] = best.x
                costs[frame, branch] = best.cost
            if frame:
                change = np.abs(candidates[frame, None] - candidates[frame - 1, :, None])
                excess = np.maximum(change - step_limit, 0)
                transition = np.sum(.016**2 * change**2 + .10**2 * excess**2, axis=2) / 2
                options = total[:, None] + transition
                parent[frame] = np.argmin(options, axis=0)
                total = options.min(axis=0) + costs[frame]
            else:
                total = costs[0].copy()
        branch = np.argmin(total)
        selected = np.empty((self.count, 6))
        for frame in range(self.count - 1, -1, -1):
            selected[frame] = candidates[frame, branch]
            branch = parent[frame, branch]
        return selected

    def optimize(self, initial, iterations):
        q = self.feasible_seed(initial)
        cost, residual, jac = self.evaluate(q)
        solver, pattern = None, None
        damping = 3e-4
        for _ in range(iterations):
            flat = q.ravel()
            delta = self.difference @ flat
            gram = jac.transpose(0, 2, 1) @ jac
            gram += damping * np.eye(6)
            blocks = sparse.bsr_matrix(
                (gram, np.arange(self.count), np.arange(self.count + 1)),
                shape=(len(flat), len(flat)),
            )
            hessian = sparse.triu(blocks.tocsc() + self.smoothness, format="csc")
            hessian.sort_indices()
            gradient = (jac.transpose(0, 2, 1) @ residual[:, :, None]).ravel()
            gradient += self.smoothness @ flat
            low = np.r_[np.maximum(self.low - flat, -.30), -self.step_limit - delta]
            high = np.r_[np.minimum(self.high - flat, .30), self.step_limit - delta]
            current_pattern = (hessian.indices.tobytes(), hessian.indptr.tobytes())
            if pattern != current_pattern:
                solver = osqp.OSQP()
                solver.setup(
                    P=hessian, q=gradient, A=self.constraint, l=low, u=high, verbose=False,
                    polishing=False, eps_abs=5e-7, eps_rel=5e-7, max_iter=3500,
                )
                pattern = current_pattern
            else:
                solver.update(Px=hessian.data, q=gradient, l=low, u=high)
            result = solver.solve(raise_error=True)
            step = result.x.reshape(self.count, 6)
            accepted = False
            for scale in (1., .5, .25, .125):
                candidate = q + scale * step
                next_cost, next_residual, next_jac = self.evaluate(candidate)
                if next_cost < cost:
                    accepted = True
                    improvement = cost - next_cost
                    q, cost, residual, jac = candidate, next_cost, next_residual, next_jac
                    damping = max(1e-6, damping * .6)
                    break
            if not accepted:
                damping *= 4
            elif improvement < 1e-8 or np.max(np.abs(step)) < 1e-4:
                break
        return q, cost



def load_base_transforms(path):
    """Return D555-to-base transforms for the two calibrated arms."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid calibration YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a mapping from calibration names to transforms")
    transforms = {}
    for side in SIDES:
        key = f"T_Head_{side.title()}Base"
        head_from_base = np.asarray(data[key], dtype=float)
        if (
            head_from_base.shape != (4, 4)
            or not np.isfinite(head_from_base).all()
            or not np.allclose(head_from_base[3], [0, 0, 0, 1])
        ):
            raise ValueError(f"{key} must be a finite homogeneous transform")
        rotation = head_from_base[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
        ):
            raise ValueError(f"{key} must contain a proper rotation matrix")
        transforms[side] = np.linalg.inv(head_from_base)
    return transforms


class PiperXRetargeter:
    """Fit both grippers to a calibrated episode, reusing the model per worker."""

    def __init__(self, transforms, urdf, gripper_xacro, gripper_limits):
        self.transforms = transforms
        self.arm = PiperKinematics(urdf)
        self.offset = flange_to_jaw_midpoint(gripper_xacro)
        hardware_limit = float(ET.parse(gripper_xacro).find("joint[@name='gripper']/limit").get("upper"))
        if not np.isfinite(hardware_limit) or hardware_limit <= 0:
            raise ValueError("Invalid Piper X gripper limit")
        self.gripper_limits = {side: min(gripper_limits[side], hardware_limit) for side in SIDES}

    def actions(self, folder, timestamp_file):
        """Return one base-frame flange action per original recorded frame."""
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        fps = float(meta["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source video FPS")
        if not meta.get("camera_calibration"):
            raise ValueError("Piper X retargeting requires calibrated EgoHand results")

        with timestamp_file.open(encoding="utf-8-sig", newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        stamps = [row["timestamp_unix_s"] for row in source_rows]
        grid = np.asarray([int(row["grid_index"]) for row in source_rows], dtype=np.int64)
        camera_rows = read_action_rows(folder / "epose.csv")
        if [row[0] for row in camera_rows] != stamps:
            raise ValueError("Epose and source video timestamps differ")
        camera_poses = np.asarray(
            [row[1:] for row in frame_aligned_actions(camera_rows, stamps)], dtype=float
        ).reshape(len(stamps), 2, 8)
        commands = [
            self.fit_hand(side, folder / "soma.npz", camera_poses[:, index], grid, fps)
            for index, side in enumerate(SIDES)
        ]
        return [[stamp, *left, *right] for stamp, left, right in zip(stamps, *commands)]

    def fit_hand(self, side, soma, wrist_camera, grid_indices, fps):
        center, rotation, width, grid = load_grasp(soma, side)
        if not np.array_equal(grid, grid_indices):
            raise ValueError("SOMA and source video frame indices differ")

        # Convert the wrist epose to this robot's base, then attach the SOMA
        # wrist-to-grasp transform and solve for the flange behind the jaws.
        base_from_head = self.transforms[side]
        head_to_base_rotation = Rotation.from_matrix(base_from_head[:3, :3])
        camera_wrist_rotation = Rotation.from_quat(wrist_camera[:, 3:7])
        base_wrist_rotation = head_to_base_rotation * camera_wrist_rotation
        base_wrist_position = head_to_base_rotation.apply(wrist_camera[:, :3]) + base_from_head[:3, 3]
        wrist_from_camera = camera_wrist_rotation.inv()
        grasp_offset = wrist_from_camera.apply(center - wrist_camera[:, :3])
        grasp_rotation = wrist_from_camera * rotation
        positions = base_wrist_position + base_wrist_rotation.apply(grasp_offset)
        rotations = base_wrist_rotation * grasp_rotation

        path = GraspPath(self.arm, self.offset, positions, rotations, width, fps, .055)
        joints, _ = path.optimize(path.initial_path(), 70)
        flange, achieved, _, _, _ = path.model(joints)
        speed = np.abs(np.diff(joints, axis=0)) * fps
        if (
            not np.isfinite(joints).all()
            or np.any(joints < self.arm.low - 2e-6)
            or np.any(joints > self.arm.high + 2e-6)
            or np.any(speed > self.arm.speed_limits + 2e-6)
        ):
            raise ValueError(f"{side.title()} Piper X angle/velocity validation failed")

        quaternion = achieved.as_quat()
        for i in range(1, len(quaternion)):
            if quaternion[i] @ quaternion[i - 1] < 0:
                quaternion[i] *= -1
        widths = np.clip(width, 0, self.gripper_limits[side])
        commands = np.column_stack((flange, quaternion, widths))
        if not np.isfinite(commands).all():
            raise ValueError(f"{side.title()} Piper X action contains non-finite values")
        return commands
