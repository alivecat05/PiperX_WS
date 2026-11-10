"""Export current-frame camera wrist eposes using PiperCapture's CSV schema."""

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanlabel.utils.paths import MODEL, ROOT

SIDES = ("left", "right")
HEADER = [
    "timestamp_unix_s",
    *[
        f"{s}.{k}"
        for s in SIDES
        for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper_m")
    ],
]


def capture_contract(config=None):
    capture = ROOT.parent / "PiperCapture"
    path = Path(config) if config else capture / "config.json"
    basis = "provided_config" if config else "PiperCapture/config.json"
    if not config and not path.exists():
        path = capture / "config.example.json"
        basis = "PiperCapture/config.example.json; no local config.json exists"
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    maxima = {s: float(data["arms"][s]["gripper_max_m"]) for s in SIDES}
    if not all(np.isfinite(v) and v > 0 for v in maxima.values()):
        raise ValueError("PiperCapture gripper limits must be positive finite meters")
    return {
        "schema_reference": str(capture / "piper_capture/core.py"),
        "columns": HEADER,
        "config": str(path.resolve()),
        "config_basis": basis,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "gripper_max_m": maxima,
        "gripper_mapping": "clip(raw fingertip distance in meters, 0, side gripper_max_m); no rescaling",
        "reference_frame": "head optical camera: +X right, +Y down, +Z forward",
        "transform": "T_C_W maps native bind-wrist coordinates into the head camera; origin is joints_camera_m[0]",
        "quaternion_order": "xyzw",
        "length_unit": "meter",
        "time_semantics": "current RGB frame wrist pose; original source timestamp retained",
        "pipercapture_time_difference": "PiperCapture action is next-grid measured robot TCP in each arm base. These are current-frame estimated human wrists in the camera frame; only the column schema and width limits match.",
        "raw_distance_file": "pinch.csv",
        "epose_file": "epose.csv",
        "transform_file": "wrist_epose.npz",
    }


@lru_cache(maxsize=1)
def soma_axes():
    data = json.loads((ROOT / "configs/soma_wrist_axes.json").read_text())
    for name, digest in data["asset_sha256"].items():
        asset = MODEL / "egohand/soma" / name
        if asset.exists() and hashlib.sha256(asset.read_bytes()).hexdigest() != digest:
            raise ValueError(f"SOMA reference axes do not match asset: {name}")
    axes = np.asarray([data["A"][s] for s in SIDES])
    if not np.allclose(axes @ axes.swapaxes(-1, -2), np.eye(3), atol=1e-6):
        raise ValueError("Invalid SOMA wrist reference axes")
    return axes


def camera_wrist_rotations(native_roots, representation):
    if representation != "SOMA":
        raise ValueError("Only EgoHand SOMA predictions are supported")
    # SOMA's FK applies the T-pose joint orientation, including at root 0.
    # Root parent is itself: R_camera = A.T @ R_parameter @ A.
    a = soma_axes()[:, None]
    return a.swapaxes(-1, -2) @ native_roots @ a


def rotation_contract(contract, representation):
    if representation != "SOMA":
        raise ValueError("Only EgoHand SOMA predictions are supported")
    return {
        **contract,
        "rotation_conversion": "SOMA FK root: A.T @ native_root_rotation @ A; A from soma_wrist_axes.json",
        "wrist_axes_reference": str(ROOT / "configs/soma_wrist_axes.json"),
    }


def wrist_values(joints, roots, valid, maxima):
    """Return CSV vectors, unbounded fingertip distances and T_camera_wrist."""
    n = joints.shape[1]
    if (
        joints.shape != (2, n, 21, 3)
        or roots.shape != (2, n, 3, 3)
        or valid.shape != (2, n)
    ):
        raise ValueError("Expected side/frame/joint camera geometry")
    values = np.full((n, 16), np.nan)
    pinch = np.linalg.norm(joints[:, :, 4] - joints[:, :, 8], axis=-1).astype(
        np.float64
    )
    transforms = np.full((2, n, 4, 4), np.nan)
    for side, name in enumerate(SIDES):
        ids = np.flatnonzero(valid[side])
        if not len(ids):
            continue
        r = roots[side, ids]
        if (
            not np.isfinite(joints[side, ids]).all()
            or not np.isfinite(r).all()
            or not np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(r), 1.0, atol=1e-4)
        ):
            raise ValueError(
                "Valid wrist geometry requires finite points and proper rotations"
            )
        q = Rotation.from_matrix(r).as_quat()
        for i in range(1, len(q)):
            if q[i - 1] @ q[i] < 0:
                q[i] *= -1
        width = np.clip(pinch[side, ids], 0.0, maxima[name])
        values[ids, side * 8 : side * 8 + 8] = np.c_[joints[side, ids, 0], q, width]
        transforms[side, ids] = np.eye(4)
        transforms[side, ids, :3, :3] = r
        transforms[side, ids, :3, 3] = joints[side, ids, 0]
    return values, pinch, transforms


def write_csv(path, header, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def write_products(
    folder, timestamps, grid, values, pinch, transforms, valid, contract
):
    records = [[stamp, *row] for stamp, row in zip(timestamps, values)]
    write_csv(folder / "epose.csv", HEADER, records)
    write_csv(
        folder / "pinch.csv",
        [
            "timestamp_unix_s",
            "left.pinch_distance_m",
            "right.pinch_distance_m",
            "left.gripper_m",
            "right.gripper_m",
        ],
        (
            [stamp, pinch[0, i], pinch[1, i], values[i, 7], values[i, 15]]
            for i, stamp in enumerate(timestamps)
        ),
    )
    temporary = folder / "wrist_epose.tmp.npz"
    np.savez_compressed(
        temporary,
        T_camera_wrist=transforms,
        pinch_distance_m=pinch,
        gripper_m=values[:, [7, 15]].T,
        model_valid=valid,
        timestamp_unix_s=np.asarray(timestamps, float),
        grid_index=np.asarray(grid, np.int64),
        sides=np.asarray(SIDES),
        axis_order=np.asarray("side,frame,..."),
        reference_frame=np.asarray(contract["reference_frame"]),
        quaternion_order=np.asarray("xyzw"),
        gripper_max_m=np.asarray([contract["gripper_max_m"][s] for s in SIDES]),
    )
    temporary.replace(folder / "wrist_epose.npz")


def update_episode(folder, contract):
    meta_path = folder / "manifest.json"
    meta = json.loads(meta_path.read_text())
    if meta["status"] != "complete":
        raise ValueError(f"Cannot export incomplete predictions: {folder}")
    source = Path(meta["source"])
    with (source / "timestamps.csv").open() as f:
        timestamps = list(csv.DictReader(f))
    n = meta["processed_frames"]
    if len(timestamps) != n:
        raise ValueError("Source/result frame counts differ")
    archive = folder / "soma.npz"
    with np.load(archive) as d:
        joints, native_roots, valid = (
            d["joints_camera_m"],
            d["rotmat"][:, :, 0],
            d["model_valid"],
        )
        roots = camera_wrist_rotations(native_roots, meta["representation"])
        values, pinch, transforms = wrist_values(
            joints, roots, valid, contract["gripper_max_m"]
        )
    with (folder / "action_raw.csv").open() as f:
        old = list(csv.reader(f))
    if old[0] != HEADER or len(old) != n + 1:
        raise ValueError("Unexpected existing output schema/count")
    for i, row in enumerate(old[1:]):
        if row[0] != timestamps[i]["timestamp_unix_s"]:
            raise ValueError(
                "Existing output is not aligned to current-frame timestamps"
            )
        for side in range(2):
            if valid[side, i]:
                offset = 1 + side * 8
                if not np.allclose(
                    np.asarray(row[offset : offset + 3], float),
                    joints[side, i, 0],
                    atol=1e-6,
                ):
                    raise ValueError(
                        "Existing wrist origin disagrees with model joint 0"
                    )
                old_r = Rotation.from_quat(
                    np.asarray(row[offset + 3 : offset + 7], float)
                ).as_matrix()
                if not (
                    np.allclose(old_r, native_roots[side, i], atol=1e-5)
                    or np.allclose(old_r, roots[side, i], atol=1e-5)
                ):
                    raise ValueError(
                        "Existing wrist rotation disagrees with native root"
                    )
    with (folder / "frame_map.csv").open() as f:
        mapping = list(csv.DictReader(f))
    with (folder / "quality.csv").open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        quality = list(reader)
    if len(quality) != n:
        raise ValueError("Quality/frame count disagreement")
    stamps = [r["timestamp_unix_s"] for r in timestamps]
    records = [[stamp, *row] for stamp, row in zip(stamps, values)]
    for i, row in enumerate(quality):
        for side, name in enumerate(SIDES):
            row[name + ".gripper_m"] = values[i, 7 + side * 8]
    write_csv(folder / "action_raw.csv", HEADER, records)
    write_csv(
        folder / "action.csv", HEADER, (records[int(m["frame_index"])] for m in mapping)
    )
    write_csv(
        folder / "quality.csv", fields, ([row[k] for k in fields] for row in quality)
    )
    write_products(
        folder,
        stamps,
        [r["grid_index"] for r in timestamps],
        values,
        pinch,
        transforms,
        valid,
        contract,
    )
    meta.update(
        epose_export=rotation_contract(contract, meta["representation"]),
        pose_convention="T_C_W: bind-wrist axes to head optical camera; origin=joint 0; joint-orient conversion in epose_export; +X right +Y down +Z forward; meters; quaternion xyzw",
        action_semantics="current-frame estimated camera wrist epose; not PiperCapture next-grid measured TCP",
        robot_base_calibrated=False,
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    report = {
        "status": "passed",
        "frames": n,
        "columns_match_pipercapture": True,
        "reference_frame": "head_optical",
        "wrist_origin_joint": 0,
        "native_rotation_parameters_preserved": True,
        "rotation_conversion": meta["epose_export"]["rotation_conversion"],
        "raw_pinch_preserved": True,
        "gripper_max_m": contract["gripper_max_m"],
        "timestamp_semantics": "current_frame",
        "original_timestamps_preserved": True,
        "width_observed_range_m": {
            s: [
                float(np.nanmin(values[:, side * 8 + 7])),
                float(np.nanmax(values[:, side * 8 + 7])),
            ]
            for side, s in enumerate(SIDES)
        },
    }
    (folder / "epose_validation.json").write_text(json.dumps(report, indent=2))
    return meta, report


def validate_products(folder):
    meta = json.loads((folder / "manifest.json").read_text())
    contract = meta["epose_export"]
    with (folder / "epose.csv").open() as f:
        epose = list(csv.reader(f))
    with (folder / "action_raw.csv").open() as f:
        raw = list(csv.reader(f))
    assert epose == raw and epose[0] == HEADER
    values = np.asarray([r[1:] for r in epose[1:]], float)
    archive = folder / "soma.npz"
    with np.load(archive) as d:
        joints, valid = d["joints_camera_m"], d["model_valid"]
        roots = camera_wrist_rotations(d["rotmat"][:, :, 0], meta["representation"])
    with (folder / "pinch.csv").open() as f:
        pinch_rows = list(csv.DictReader(f))
    with (folder / "quality.csv").open() as f:
        quality = list(csv.DictReader(f))
    with np.load(folder / "wrist_epose.npz") as d:
        for side, name in enumerate(SIDES):
            ids = np.flatnonzero(valid[side])
            offset = 8 * side
            if not len(ids):
                continue
            q = values[ids, offset + 3 : offset + 7]
            assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-6)
            assert np.allclose(
                Rotation.from_quat(q).as_matrix(), roots[side, ids], atol=1e-5
            )
            assert np.allclose(
                values[ids, offset : offset + 3], joints[side, ids, 0], atol=1e-6
            )
            distance = np.linalg.norm(joints[side, :, 4] - joints[side, :, 8], axis=-1)
            expected = np.clip(distance[ids], 0.0, contract["gripper_max_m"][name])
            assert np.allclose(values[ids, offset + 7], expected, atol=1e-8)
            assert (values[ids, offset + 7] >= 0).all()
            assert (values[ids, offset + 7] <= contract["gripper_max_m"][name]).all()
            assert np.allclose(
                [float(pinch_rows[i][name + ".pinch_distance_m"]) for i in ids],
                distance[ids],
                atol=1e-8,
            )
            assert np.allclose(
                [float(quality[i][name + ".gripper_m"]) for i in ids],
                expected,
                atol=1e-8,
            )
            assert np.allclose(
                d["T_camera_wrist"][side, ids, :3, :3], roots[side, ids], atol=1e-5
            )
            assert np.allclose(
                d["T_camera_wrist"][side, ids, :3, 3], joints[side, ids, 0], atol=1e-6
            )
            assert np.allclose(d["T_camera_wrist"][side, ids, 3], [0, 0, 0, 1])
    source = list(csv.DictReader((Path(meta["source"]) / "timestamps.csv").open()))
    assert len(source) == len(values) == meta["source_frames"]
    assert [r[0] for r in epose[1:]] == [r["timestamp_unix_s"] for r in source]
    mapping = list(csv.DictReader((folder / "frame_map.csv").open()))
    actions = list(csv.reader((folder / "action.csv").open()))
    assert actions[0] == HEADER and actions[1:] == [
        epose[1 + int(m["frame_index"])] for m in mapping
    ]
    return {
        "status": "passed",
        "frames": len(values),
        "header_columns": len(HEADER),
        "camera_wrist_transform_verified": True,
        "quaternion_xyzw_verified": True,
        "gripper_limits_verified": contract["gripper_max_m"],
        "raw_pinch_preserved": True,
        "current_frame_timestamps_preserved": True,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="*", type=Path, default=[ROOT / "result"])
    p.add_argument("--piper-config", type=Path)
    args = p.parse_args(argv)
    contract = capture_contract(args.piper_config)
    roots = []
    for root in args.roots:
        root = root.resolve()
        if (root / "manifest.json").exists() or any(
            root.glob("episode_*/manifest.json")
        ):
            roots.append(root)
        else:
            groups = (
                sorted(
                    folder
                    for folder in root.iterdir()
                    if folder.is_dir() and any(folder.glob("episode_*/manifest.json"))
                )
                if root.is_dir()
                else []
            )
            if not groups:
                p.error(f"No episodes in {root}")
            roots.extend(groups)
    for root in roots:
        folders = (
            [root]
            if (root / "manifest.json").exists()
            else sorted(root.glob("episode_*"))
        )
        if not folders:
            p.error(f"No episodes in {root}")
        reports, metas = {}, []
        for folder in folders:
            meta, report = update_episode(folder, contract)
            report.update(validate_products(folder))
            (folder / "epose_validation.json").write_text(json.dumps(report, indent=2))
            reports[folder.name] = report
            metas.append(meta)
            print(f"Exported {folder}: {report['frames']} eposes", flush=True)
        if not (root / "manifest.json").exists():
            (root / "summary.json").write_text(
                json.dumps(metas, ensure_ascii=False, indent=2)
            )
            (root / "epose_validation_all.json").write_text(
                json.dumps(reports, indent=2)
            )


if __name__ == "__main__":
    main()
