"""Local, directional joint-limit avoidance; this module never sends commands."""
import numpy as np
from scipy.spatial.transform import Rotation


class MotionGuard:
    def __init__(self, fk_tcp, limits, cfg):
        self.fk_tcp = fk_tcp
        self.bounds = np.asarray([limits[f"joint{i}"] for i in range(1, 7)], dtype=float)
        self.margin = np.deg2rad(cfg.get("joint_limit_slow_margin_deg", 8.))
        self.stop_margin = np.deg2rad(cfg.get("joint_limit_stop_margin_deg", 1.))
        self.speed = cfg.get("joint_max_speed_rad_s", 2.)

    def jacobian(self, joints):
        q = np.asarray(joints, dtype=float)
        reference = np.asarray(self.fk_tcp(q), dtype=float)
        rotation = Rotation.from_quat(reference[3:7])
        jacobian = np.empty((6, 6))
        epsilon = 1e-5
        for i in range(6):
            shifted = q.copy()
            shifted[i] += epsilon
            pose = np.asarray(self.fk_tcp(shifted), dtype=float)
            jacobian[:3, i] = (pose[:3]-reference[:3])/epsilon
            jacobian[3:, i] = (Rotation.from_quat(pose[3:7])*rotation.inv()).as_rotvec()/epsilon
        return jacobian

    def apply(self, target, state, dt):
        q = np.asarray(state["joints"], dtype=float)
        actual, desired = np.asarray(state["vector"], dtype=float), np.asarray(target, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(actual).all():
            raise ValueError("Joint guard requires finite measured joints and TCP feedback")
        delta = np.r_[desired[:3]-actual[:3],
                      (Rotation.from_quat(desired[3:7])*Rotation.from_quat(actual[3:7]).inv()).as_rotvec()]
        weight = np.diag([1., 1., 1., .15, .15, .15])
        jacobian = self.jacobian(q)
        weighted = weight@jacobian
        motion = weight@delta
        # Damping keeps singular poses from producing enormous predicted joint steps.
        dq = weighted.T@np.linalg.solve(weighted@weighted.T+np.eye(6)*.005**2, motion)
        limited = dq.copy()
        affected = []
        maximum = self.speed*max(0., min(dt, .05))
        speed_limited = np.max(np.abs(limited)) > maximum+1e-9
        if speed_limited:
            limited *= maximum/max(np.max(np.abs(limited)), 1e-12)
        for i, step in enumerate(limited):
            if abs(step) <= 1e-8:
                continue
            room = self.bounds[i, 1]-q[i] if step > 0 else q[i]-self.bounds[i, 0]
            factor = np.clip((room-self.stop_margin)/max(self.margin-self.stop_margin, 1e-9), 0., 1.)
            if factor < 1.:
                affected.append(f"J{i+1}")
                limited[i] *= factor
            limited[i] = np.sign(step)*min(abs(limited[i]), max(0., room-self.stop_margin))
        projection_error = np.linalg.norm(weighted@dq-motion)
        unreachable = projection_error > max(2e-5, np.linalg.norm(motion)*.15)
        margins = np.minimum(q-self.bounds[:, 0], self.bounds[:, 1]-q)
        nearest = int(np.argmin(margins))
        info = {"limited": bool(affected or unreachable),
                "joints": affected, "nearest_joint": f"J{nearest+1}",
                "remaining_deg": round(float(np.rad2deg(margins[nearest])), 2),
                "reason": ("joint_boundary" if affected else "singularity" if unreachable else ""),
                "predicted_joint_delta_rad": limited.tolist()}
        if not info["limited"]:
            return list(target), info
        step = jacobian@limited
        rotation = Rotation.from_rotvec(step[3:])*Rotation.from_quat(actual[3:7])
        quaternion = rotation.as_quat()
        if np.dot(quaternion, actual[3:7]) < 0:
            quaternion = -quaternion
        return [*(actual[:3]+step[:3]), *quaternion, float(desired[7])], info
