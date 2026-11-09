"""Validate recorded TCP action CSVs and interpolate them on their original timeline."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .core import ARMS, VECTOR_NAMES


def interpolate_action(first, second, alpha):
    first, second = np.asarray(first), np.asarray(second)
    result = (1-alpha)*first+alpha*second
    for offset in (0, 8):
        rotation = Slerp([0., 1.], Rotation.from_quat(np.stack(
            (first[offset+3:offset+7], second[offset+3:offset+7]))))
        result[offset+3:offset+7] = rotation([alpha]).as_quat()[0]
    return result


@dataclass(frozen=True)
class ReplayTrajectory:
    path: Path
    times: np.ndarray
    actions: np.ndarray

    @property
    def duration(self):
        return float(self.times[-1])

    @classmethod
    def load(cls, filename, cfg):
        path = Path(filename.strip().strip('"')).resolve()
        manifest = path.parent/"manifest.json"
        if path.name == "action.csv" and manifest.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if meta.get("placeholder_frames", 0):
                raise ValueError("此片段含缺帧占位，不能将占位动作作为真实示教轨迹回放")
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != ["timestamp_unix_s", *VECTOR_NAMES]:
                raise ValueError("CSV 列必须为 timestamp_unix_s 和左右各 8 维 action")
            for line, row in enumerate(reader, 2):
                if len(row) != 17:
                    raise ValueError(f"CSV 第 {line} 行必须有 17 列")
                try:
                    rows.append([float(value) for value in row])
                except ValueError:
                    raise ValueError(f"CSV 第 {line} 行包含无效数字") from None
                if len(rows) > 500_000:
                    raise ValueError("单条回放轨迹最多支持 500000 帧")
        if len(rows) < 2:
            raise ValueError("回放轨迹至少需要两帧")
        data = np.asarray(rows, dtype=float)
        if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
            raise ValueError("CSV 必须包含有限数值和严格递增的时间戳")
        times = data[:, 0]-data[0, 0]
        actions = data[:, 1:].copy()
        for side, offset in zip(ARMS, (0, 8)):
            arm = cfg["arms"][side]
            part = actions[:, offset:offset+8]
            if (np.any(part[:, :3] < np.asarray(arm["workspace_min"])-1e-9)
                    or np.any(part[:, :3] > np.asarray(arm["workspace_max"])+1e-9)):
                raise ValueError(f"{side}: CSV 轨迹超出当前配置的工作空间")
            norms = np.linalg.norm(part[:, 3:7], axis=1)
            if np.any(np.abs(norms-1) > .01):
                raise ValueError(f"{side}: CSV 四元数未归一化")
            part[:, 3:7] /= norms[:, None]
            if np.any(part[:, 7] < 0) or np.any(part[:, 7] > arm["gripper_max_m"]+1e-9):
                raise ValueError(f"{side}: CSV 夹爪宽度超限")
        times.setflags(write=False)
        actions.setflags(write=False)
        return cls(path, times, actions)

    def check_speed(self, cfg, speed):
        if not np.isfinite(speed) or not .1 <= speed <= 1.:
            raise ValueError("回放倍率必须在 0.1～1.0 之间")
        dt = np.diff(self.times)
        for side, offset in zip(ARMS, (0, 8)):
            part, arm = self.actions[:, offset:offset+8], cfg["arms"][side]
            if (not np.isfinite(arm["max_speed_m_s"]) or arm["max_speed_m_s"] <= 0
                    or not np.isfinite(arm["max_rotation_rad_s"]) or arm["max_rotation_rad_s"] < 0):
                raise ValueError(f"{side}: 回放速度上限配置无效")
            linear = np.linalg.norm(np.diff(part[:, :3], axis=0), axis=1)/dt*speed
            rotations = Rotation.from_quat(part[:, 3:7])
            angular = (rotations[1:]*rotations[:-1].inv()).magnitude()/dt*speed
            if np.any(linear > arm["max_speed_m_s"]+1e-3):
                raise ValueError(f"{side}: 回放平移速度超过配置上限，请降低回放倍率")
            if np.any(angular > arm["max_rotation_rad_s"]+1e-3):
                raise ValueError(f"{side}: 回放旋转速度超过配置上限，请降低回放倍率")

    def sample(self, elapsed):
        if elapsed <= 0:
            return self.actions[0].copy()
        if elapsed >= self.duration:
            return self.actions[-1].copy()
        hi = int(np.searchsorted(self.times, elapsed, side="right"))
        lo = hi-1
        alpha = (elapsed-self.times[lo])/(self.times[hi]-self.times[lo])
        return interpolate_action(self.actions[lo], self.actions[hi], alpha)
