"""Convert absolute TCP poses from the head optical frame to each arm's base."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Head-frame action.csv (meters, quaternion xyzw)")
    parser.add_argument("--output", type=Path, required=True, help="Base-frame action.csv")
    parser.add_argument("--extrinsics", type=Path, default=Path("calibration/extrinsics.yaml"))
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    matrices = {}
    for line in args.extrinsics.read_text(encoding="utf-8-sig").splitlines():
        if line and not line.startswith(" "):
            key = line.rstrip(":")
            matrices[key] = []
        elif line.strip():
            matrices[key].append(json.loads(line.strip().removeprefix("- ")))
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    fields = ("x", "y", "z", "qx", "qy", "qz", "qw")
    required = {"timestamp_unix_s", *[f"{side}.{field}" for side in ("left", "right") for field in (*fields, "gripper_m")]}
    if not columns or not required.issubset(columns):
        raise ValueError("Expected timestamp_unix_s and left/right x,y,z,qx,qy,qz,qw,gripper_m columns")
    for side, key in (("left", "T_Head_LeftBase"), ("right", "T_Head_RightBase")):
        head_base = np.asarray(matrices[key], float)
        if (head_base.shape != (4, 4) or not np.isfinite(head_base).all()
                or not np.allclose(head_base[3], [0, 0, 0, 1])
                or not np.allclose(head_base[:3, :3].T @ head_base[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(head_base[:3, :3]), 1)):
            raise ValueError(f"{key} is not a valid rigid transform")
        base_head = np.linalg.inv(head_base)
        rotation = Rotation.from_matrix(base_head[:3, :3])
        for index, row in enumerate(rows, start=2):
            pose = np.array([float(row[f"{side}.{field}"]) for field in fields])
            if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-12:
                raise ValueError(f"Invalid {side} pose at CSV line {index}")
            position = base_head[:3, :3] @ pose[:3] + base_head[:3, 3]
            quaternion = (rotation * Rotation.from_quat(pose[3:])).as_quat()
            for field, value in zip(fields, np.r_[position, quaternion]):
                row[f"{side}.{field}"] = format(value, ".16g")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} base-frame actions to {args.output}")


if __name__ == "__main__":
    main()
