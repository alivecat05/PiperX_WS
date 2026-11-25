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


THUMB_CHAIN = slice(2, 5)
INDEX_CHAIN = slice(6, 10)
INDEX_ROOT, MIDDLE_ROOT = 6, 11
# Front face and half span of the jaw carrier in the supplied gripper_base STL.
CROSSBAR_FRONT_Z_M = .065
CROSSBAR_HALF_SPAN_M = .082
SWAP_JAWS = Rotation.from_rotvec([0, 0, np.pi])
MAX_JOINT_ACCELERATION_RAD_S2 = 18.
QP_BOUND_TOLERANCE_RAD = 1e-5
IK_CANDIDATES_PER_JAW_ASSIGNMENT = 2


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


def finger_direction(chain):
    """Fit a straight finger axis through its tip using the posed phalanges."""
    lengths = np.linalg.norm(np.diff(chain, axis=1), axis=2)
    remaining = np.cumsum(lengths[:, ::-1], axis=1)[:, ::-1]
    toward_tip = chain[:, -1, None] - chain[:, :-1]
    direction = np.sum(remaining[:, :, None] * toward_tip, axis=1)
    magnitude = np.linalg.norm(direction, axis=1)
    return np.divide(direction, magnitude[:, None], out=np.zeros_like(direction),
                     where=magnitude[:, None] > 1e-8)


def fit_crossbar_line(points):
    """Fit the transverse line through the web, index root and middle root."""
    center = points.mean(axis=1)
    centered = points - center[:, None]
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    axis = directions[:, 0]
    sign = np.sum(axis * (points[:, 2] - points[:, 0]), axis=1)
    axis = np.where(sign[:, None] < 0, -axis, axis)
    return center, axis


def load_grasp(path, side):
    with np.load(path) as archive:
        hand = list(archive["sides"]).index(side)
        joints = archive["native_joints25_camera_m"][hand].astype(float)
        vertices = archive["vertices_camera_m"][hand]
        valid = archive["model_valid"][hand].copy()
        grid_index = archive["grid_index"].copy()

    thumb, index = joints[:, THUMB_CHAIN], joints[:, INDEX_CHAIN]
    index_root, middle_root = joints[:, INDEX_ROOT], joints[:, MIDDLE_ROOT]
    wrist = joints[:, 0]
    proximal_span = np.linalg.norm(index[:, 0] - thumb[:, 0], axis=1)
    web_proxy = .44 * (thumb[:, 0] + index[:, 0]) + .12 * wrist
    valid &= (
        np.isfinite(thumb).all(axis=(1, 2)) & np.isfinite(index).all(axis=(1, 2))
        & np.isfinite(wrist).all(axis=1) & np.isfinite(middle_root).all(axis=1)
        & np.isfinite(vertices).all(axis=(1, 2))
    )
    if not valid.any():
        raise ValueError(f"No usable SOMA {side}-hand mesh for the thumb web")
    open_hand = valid & (proximal_span >= np.percentile(proximal_span[valid], 65))
    distance = np.median(
        np.linalg.norm(vertices[open_hand] - web_proxy[open_hand, None], axis=2), axis=0
    )
    patch = np.argsort(distance)[:12]
    valid_frames = np.flatnonzero(valid)
    thumb, index = thumb[valid], index[valid]
    web = vertices[valid_frames[:, None], patch].mean(axis=1).astype(float)
    points = np.stack((web, index_root[valid], middle_root[valid]), axis=1)
    crossbar_center, crossbar_axis = fit_crossbar_line(points)
    center = (thumb[:, -1] + index[:, -1]) / 2
    span = index[:, -1] - thumb[:, -1]
    approach = center - crossbar_center
    approach -= np.sum(approach * crossbar_axis, axis=1, keepdims=True) * crossbar_axis
    approach_norm = np.linalg.norm(approach, axis=1)
    degenerate = approach_norm < .004
    if degenerate.any():
        fallback = finger_direction(thumb[degenerate]) + finger_direction(index[degenerate])
        axis = crossbar_axis[degenerate]
        fallback -= np.sum(fallback * axis, axis=1, keepdims=True) * axis
        approach[degenerate] = fallback
        approach_norm[degenerate] = np.linalg.norm(fallback, axis=1)
    usable = (np.linalg.norm(points[:, 2] - points[:, 0], axis=1) >= .008) & (approach_norm >= .004)
    if not usable.any():
        raise ValueError(f"No usable SOMA {side}-hand thumb web and finger root line")
    known = valid_frames[usable]
    source = np.flatnonzero(usable)[
        np.maximum(0, np.searchsorted(known, np.arange(len(joints)), side="right") - 1)
    ]
    crossbar_axis = crossbar_axis[source]
    approach = approach[source] / approach_norm[source, None]
    axes = np.stack((crossbar_axis, np.cross(approach, crossbar_axis), approach), axis=-1)
    span = span[source]
    return center[source], Rotation.from_matrix(axes), np.linalg.norm(span, axis=1), points[source], span, grid_index


def gripper_offsets(assembly):
    """Return the jaw-tip midpoint and the crossbar center in flange coordinates."""
    offsets = {}
    rotations = {}
    for name in ("flange_joint", "gripper_base_joint", "gripper_joint1", "gripper_joint2"):
        origin = assembly.find(f"joint[@name='{name}']/origin")
        offsets[name] = np.fromstring(origin.get("xyz"), sep=" ")
        rotations[name] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" "))
    fixed = offsets["gripper_base_joint"]
    midpoint = (offsets["gripper_joint1"] + offsets["gripper_joint2"]) / 2
    crossbar = np.array([0., 0., CROSSBAR_FRONT_Z_M])
    return offsets["flange_joint"] + rotations["flange_joint"].apply(
        fixed + rotations["gripper_base_joint"].apply(np.stack((midpoint, crossbar)))
    )


def flange_to_jaw_midpoint(path):
    return gripper_offsets(ET.parse(path))[0]


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
                 smoothing_seconds=.10, joint_smoothing=.12, *,
                 crossbar_offset=None, crossbar_points=None, tip_span=None, opening=None):
        self.arm = arm
        self.offset = offset
        self.positions = positions
        self.crossbar_points = crossbar_points
        if crossbar_points is not None:
            self.jaw_to_crossbar = crossbar_offset - offset
            self.crossbar_center, self.crossbar_axis = fit_crossbar_line(crossbar_points)
            projections = (crossbar_points - self.crossbar_center[:, None]) @ self.crossbar_axis[:, :, None]
            self.crossbar_min = projections.min(axis=1)
            self.crossbar_max = projections.max(axis=1)
            self.crossbar_mid = (self.crossbar_min + self.crossbar_max) / 2
            self.crossbar_slide = np.maximum(
                0, CROSSBAR_HALF_SPAN_M - (self.crossbar_max - self.crossbar_min) / 2
            )
            self.crossbar_axis_projector = self.crossbar_axis[:, :, None] * self.crossbar_axis[:, None, :]
            self.target_half_span = tip_span / 2
            self.half_opening = opening[:, None] / 2
        self.rotations = smooth_grasp_rotations(rotations, smoothing_seconds * fps)
        self.primary = self.rotations.inv()
        self.swapped = (self.rotations * SWAP_JAWS).inv()
        self.count = len(positions)
        # Regularize orientation most strongly while pinching.
        pinch_weight = np.clip((.065 - width) / .045, 0, 1)
        self.rotation_weight = rotation_scale * (.48 + .52 * pinch_weight)
        self.low = np.tile(arm.low + QP_BOUND_TOLERANCE_RAD, self.count)
        self.high = np.tile(arm.high - QP_BOUND_TOLERANCE_RAD, self.count)
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
        self.step_limit = arm.speed_limits / fps
        self.acceleration_limit = MAX_JOINT_ACCELERATION_RAD_S2 / fps**2
        # Leave room for the solver residual inside the physical bounds.
        self.qp_tolerance = min(QP_BOUND_TOLERANCE_RAD,
                                .01 * min(self.step_limit.min(), self.acceleration_limit))
        step_bound = np.tile(self.step_limit - self.qp_tolerance, self.count - 1)
        acceleration_bound = np.full(acceleration.shape[0], self.acceleration_limit - self.qp_tolerance)
        self.constraint = sparse.vstack(
            (sparse.eye(6 * self.count), self.difference, acceleration), format="csc"
        )
        self.constraint_low = np.r_[self.low, -step_bound, -acceleration_bound]
        self.constraint_high = np.r_[self.high, step_bound, acceleration_bound]

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

    def crossbar_line_error(self, crossbar, frames=slice(None)):
        axis = self.crossbar_axis[frames]
        midpoint, allowance = self.crossbar_mid[frames], self.crossbar_slide[frames]
        displacement = crossbar - self.crossbar_center[frames]
        along = np.sum(displacement * axis, axis=-1, keepdims=True)
        slide = along - midpoint
        allowed = midpoint + np.clip(slide, -allowance, allowance)
        return displacement - allowed * axis, np.abs(slide) <= allowance

    def evaluate(self, q):
        _, actual, jaw, jaw_jac, rot_jac = self.model(q)
        first = (actual * self.primary).as_rotvec()
        second = (actual * self.swapped).as_rotvec()
        use_second = np.sum(second**2, axis=1) < np.sum(first**2, axis=1)
        angle = np.where(use_second[:, None], second, first)
        spatial_log_jac = rotation_log_jacobian(angle)
        weight = self.rotation_weight
        jaw_error = jaw - self.positions
        if self.crossbar_points is not None:
            crossbar_delta = actual.apply(self.jaw_to_crossbar)
            # Align the finite rail with the hand segment. The human landmarks
            # may slide along the longer robot rail while it covers all three.
            bar_error, within_rail = self.crossbar_line_error(jaw + crossbar_delta)
            bar_jac = jaw_jac - skew(crossbar_delta) @ rot_jac
            bar_jac = (np.eye(3) - within_rail[:, :, None] * self.crossbar_axis_projector) @ bar_jac
            jaw_axis = actual.as_matrix()[:, :, 0]
            bar_half_span = CROSSBAR_HALF_SPAN_M * jaw_axis
            same_direction = np.sum(jaw_axis * self.crossbar_axis, axis=1) >= 0
            target_axis = np.where(same_direction[:, None], self.crossbar_axis, -self.crossbar_axis)
            axis_error = bar_half_span - CROSSBAR_HALF_SPAN_M * target_axis
            axis_jac = -skew(bar_half_span) @ rot_jac
            half_span = jaw_axis * self.half_opening
            same_tips = np.sum(jaw_axis * self.target_half_span, axis=1) >= 0
            target_span = np.where(same_tips[:, None], self.target_half_span, -self.target_half_span)
            span_error = half_span - target_span
            span_jac = -skew(half_span) @ rot_jac
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
        if self.crossbar_points is not None:
            # The bar and jaws share an axis. Keep the rail direction primary
            # when the measured span between fingertips is oblique to it.
            residual = np.column_stack((residual, bar_error, axis_error, .25 * span_error))
            jac = np.concatenate((jac, bar_jac, axis_jac, .25 * span_jac), axis=1)
        flat = q.ravel()
        cost = .5 * (np.sum(residual**2) + flat @ (self.smoothness @ flat))
        return cost, residual, jac

    def solve_qp(self, solver, low, high, stage):
        solved = (osqp.SolverStatus.OSQP_SOLVED, osqp.SolverStatus.OSQP_SOLVED_INACCURATE)
        iteration_limit = osqp.SolverStatus.OSQP_MAX_ITER_REACHED
        retryable = (*solved, iteration_limit)
        for attempt in range(2):
            if attempt:
                solver.update_settings(max_iter=14000)
            result = solver.solve(raise_error=False)
            status = result.info.status_val
            # A feasible iterate can seed the fit or go through its cost-decrease check.
            candidate_status = status in solved or (attempt == 1 and status == iteration_limit)
            if candidate_status and np.isfinite(result.x).all():
                values = self.constraint @ result.x
                violation = max(np.max(low - values), np.max(values - high))
                if violation <= self.qp_tolerance:
                    return result.x.reshape(self.count, 6)
            if status not in retryable:
                break
        info = result.info
        raise RuntimeError(
            f"{stage} QP: {info.status} (OSQP {info.status_val}, "
            f"primal residual {info.prim_res:.3g}, dual residual {info.dual_res:.3g})"
        )

    def feasible_seed(self, q):
        flat = np.clip(q.ravel(), self.low, self.high)
        values = self.constraint @ flat
        low, high = self.constraint_low, self.constraint_high
        if np.all((low <= values) & (values <= high)):
            return flat.reshape(self.count, 6)
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.eye(len(flat), format="csc"), q=-flat, A=self.constraint,
            l=low, u=high,
            verbose=False, polishing=True, eps_abs=self.qp_tolerance / 20, eps_rel=1e-7,
        )
        return self.solve_qp(solver, low, high, "initial trajectory")

    def ik_candidates(self, position, target_inv, weight, seeds, restarts, explore, frame=None):
        arm = self.arm
        track_rail = frame is not None and self.crossbar_points is not None
        if track_rail:
            target_axis = self.crossbar_axis[frame].copy()
            if target_axis @ target_inv.inv().as_matrix()[:, 0] < 0:
                target_axis *= -1
        previous_q = None

        def sample(q):
            nonlocal previous_q, errors, derivatives
            if previous_q is not None and np.array_equal(q, previous_q):
                return
            flange, actual = arm.evaluate(q)
            pad = actual.apply(self.offset)
            jaw = flange + pad
            error = (actual * target_inv).as_rotvec()
            rotation_jac = arm.rotation_jacobian
            jaw_jac = arm.position_jacobian - skew(pad[None])[0] @ rotation_jac
            rotation_error_jac = rotation_log_jacobian(error)[0] @ rotation_jac
            errors = np.r_[jaw - position, weight * error]
            derivatives = np.vstack((jaw_jac, weight * rotation_error_jac))
            if track_rail:
                bar_delta = actual.apply(self.jaw_to_crossbar)
                bar_error, inside = self.crossbar_line_error(jaw + bar_delta, frame)
                bar_jac = jaw_jac - skew(bar_delta[None])[0] @ rotation_jac
                bar_jac = (np.eye(3) - inside[0] * self.crossbar_axis_projector[frame]) @ bar_jac
                bar_span = actual.as_matrix()[:, 0] * CROSSBAR_HALF_SPAN_M
                axis_error = bar_span - CROSSBAR_HALF_SPAN_M * target_axis
                axis_jac = -skew(bar_span[None])[0] @ rotation_jac
                errors = np.r_[errors, bar_error, axis_error]
                derivatives = np.vstack((derivatives, bar_jac, axis_jac))
            previous_q = q.copy()

        errors = derivatives = None

        def residual(q):
            sample(q)
            return errors

        def jacobian(q):
            sample(q)
            return derivatives

        def solve(seed):
            return least_squares(
                residual, np.clip(seed, arm.low, arm.high), jac=jacobian,
                bounds=(arm.low, arm.high), max_nfev=40, gtol=1e-7,
            )

        fits = [solve(seed) for seed in seeds]
        near_target_cost = .5 * (.001**2 + (weight * .03)**2)
        if explore or not any(fit.cost <= near_target_cost for fit in fits):
            fits.extend(solve(seed) for seed in restarts)
        distinct = []
        for fit in sorted(fits, key=lambda fit: fit.cost):
            if all(np.linalg.norm(fit.x - other.x) > .025 for other in distinct):
                distinct.append(fit)
        return distinct

    def initial_path(self):
        """Keep distinct IK solutions and score how each extends the joint sequence."""
        arm = self.arm
        neutral = (arm.low + arm.high) / 2
        rng = np.random.default_rng(2391)
        restarts = [neutral, *rng.uniform(arm.low, arm.high, (5, 6))]
        history = []
        previous = ((), ())
        for frame, position in enumerate(self.positions):
            groups = [
                self.ik_candidates(position, target, self.rotation_weight[frame],
                                   seeds, restarts, explore=frame % 30 == 0, frame=frame)
                for seeds, target in zip(previous, (self.primary[frame], self.swapped[frame]))
            ]
            fits = [fit for group in groups for fit in group]
            candidates = np.array([fit.x for fit in fits])
            costs = np.array([fit.cost for fit in fits])
            if frame:
                last = np.concatenate(previous)
                change = candidates[None, :] - last[:, None]
                acceleration = change - velocity[:, None]
                speed_excess = np.maximum(np.abs(change) - self.step_limit, 0)
                acceleration_excess = np.maximum(np.abs(acceleration) - self.acceleration_limit, 0)
                transition = np.sum(
                    (.016 * change)**2 + (.12 * acceleration)**2
                    + (.10 * speed_excess)**2 + (.12 * acceleration_excess)**2, axis=2
                ) / 2
                options = total[:, None] + transition
                parents = np.argmin(options, axis=0)
                costs += options.min(axis=0)
                incoming = change[parents, np.arange(len(fits))]
            else:
                parents = np.zeros(len(fits), dtype=int)
                incoming = np.zeros_like(candidates)
            split = len(groups[0])
            keep_primary = np.argsort(costs[:split])[:IK_CANDIDATES_PER_JAW_ASSIGNMENT]
            keep_swapped = split + np.argsort(costs[split:])[:IK_CANDIDATES_PER_JAW_ASSIGNMENT]
            keep = np.r_[keep_primary, keep_swapped]
            previous = (candidates[keep_primary], candidates[keep_swapped])
            total, velocity = costs[keep], incoming[keep]
            history.append((candidates[keep], parents[keep]))
        branch = np.argmin(total)
        selected = np.empty((self.count, 6))
        for frame in range(self.count - 1, -1, -1):
            candidates, parents = history[frame]
            selected[frame] = candidates[branch]
            branch = parents[branch]
        return selected

    def optimize(self, initial, iterations):
        q = self.feasible_seed(initial)
        cost, residual, jac = self.evaluate(q)
        solver, pattern = None, None
        damping = 3e-4
        for _ in range(iterations):
            flat = q.ravel()
            values = self.constraint @ flat
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
            low = self.constraint_low - values
            high = self.constraint_high - values
            low[:len(flat)] = np.maximum(low[:len(flat)], -.30)
            high[:len(flat)] = np.minimum(high[:len(flat)], .30)
            current_pattern = (hessian.indices.tobytes(), hessian.indptr.tobytes())
            if pattern != current_pattern:
                solver = osqp.OSQP()
                solver.setup(
                    P=hessian, q=gradient, A=self.constraint, l=low, u=high, verbose=False,
                    polishing=False, eps_abs=self.qp_tolerance / 20, eps_rel=5e-7, max_iter=3500,
                )
                pattern = current_pattern
            else:
                solver.update(Px=hessian.data, q=gradient, l=low, u=high)
            step = self.solve_qp(solver, low, high, "trajectory update")
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
        assembly = ET.parse(gripper_xacro)
        self.offset, self.crossbar_offset = gripper_offsets(assembly)
        hardware_limit = float(assembly.find("joint[@name='gripper']/limit").get("upper"))
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
        center, rotation, width, crossbar_points, tip_span, grid = load_grasp(soma, side)
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
        crossbar_points = head_to_base_rotation.apply(crossbar_points.reshape(-1, 3)).reshape(-1, 3, 3)
        crossbar_points += base_from_head[:3, 3]
        widths = np.clip(width, 0, self.gripper_limits[side])

        path = GraspPath(
            self.arm, self.offset, positions, rotations, width, fps, .015,
            crossbar_offset=self.crossbar_offset, crossbar_points=crossbar_points,
            tip_span=head_to_base_rotation.apply(tip_span), opening=widths,
        )
        try:
            joints, _ = path.optimize(path.initial_path(), 70)
        except RuntimeError as exc:
            raise RuntimeError(f"{side.title()} Piper X retargeting: {exc}") from exc
        flange, achieved, _, _, _ = path.model(joints)
        speed = np.abs(np.diff(joints, axis=0)) * fps
        acceleration = np.abs(np.diff(joints, n=2, axis=0)) * fps**2
        if (
            not np.isfinite(joints).all()
            or np.any(joints < self.arm.low - 2e-6)
            or np.any(joints > self.arm.high + 2e-6)
            or np.any(speed > self.arm.speed_limits + 2e-6)
            or np.any(acceleration > MAX_JOINT_ACCELERATION_RAD_S2 + 2e-6)
        ):
            raise ValueError(f"{side.title()} Piper X joint angle, velocity or acceleration limit exceeded")

        quaternion = achieved.as_quat()
        for i in range(1, len(quaternion)):
            if quaternion[i] @ quaternion[i - 1] < 0:
                quaternion[i] *= -1
        commands = np.column_stack((flange, quaternion, widths))
        if not np.isfinite(commands).all():
            raise ValueError(f"{side.title()} Piper X action contains non-finite values")
        return commands
