"""Three RGB videos + measured robot poses -> head-to-base extrinsics (meters)."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import root
from scipy.spatial.transform import Rotation

CAMERAS = ("head", "left_wrist", "right_wrist")


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, np.asarray(translation).reshape(3)
    return result


def project(xy, camera):
    K, d = np.array(camera["camera_matrix"], float), np.array(camera["dist_coeffs"], float)
    if camera["distortion_model"] in ("inverse_brown_conrady", "modified_brown_conrady"):
        x, y = xy.T
        r2 = x*x + y*y
        f = 1 + r2*(d[0] + r2*(d[1] + r2*d[4]))
        x, y = x*f, y*f
        xy = np.column_stack((x + 2*d[2]*x*y + d[3]*(r2+2*x*x), y + 2*d[3]*x*y + d[2]*(r2+2*y*y)))
        return xy * np.diag(K)[:2] + K[:2, 2]
    if camera["distortion_model"] not in ("opencv", "brown_conrady", "none"):
        raise ValueError("Unsupported distortion model")
    return cv2.projectPoints(np.column_stack((xy, np.ones(len(xy)))), np.zeros(3), np.zeros(3), K, d)[0].reshape(-1, 2)


def board_pose(image, board, camera):
    if (image.shape[1], image.shape[0]) != (camera["width"], camera["height"]):
        raise ValueError("RGB image/intrinsics resolution mismatch")
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if ids is None or len(ids) < 8 or board.checkCharucoCornersCollinear(ids):
        return None
    objects, pixels = board.getChessboardCorners()[ids.ravel()].astype(float), corners.reshape(-1, 2)
    K = np.array(camera["camera_matrix"])
    initial = ((pixels-K[:2, 2])/np.diag(K)[:2]).ravel()
    normalized = root(lambda xy: (project(xy.reshape(-1, 2), camera)-pixels).ravel(), initial)
    if np.max(np.abs(normalized.fun)) > .01 or not np.isfinite(normalized.x).all():
        raise ValueError("Distortion inversion failed")
    ok, rvec, tvec = cv2.solvePnP(objects, normalized.x.reshape(-1, 2), np.eye(3), None)
    if not ok:
        return None
    pose = transform(cv2.Rodrigues(rvec)[0], tvec)
    points = objects @ pose[:3, :3].T + pose[:3, 3]
    rms = np.sqrt(np.mean(np.sum((project(points[:, :2]/points[:, 2:], camera)-pixels)**2, axis=1)))
    return pose if np.all(points[:, 2] > 0) and rms < 2 else None


def residual(poses, reference):
    return (np.linalg.norm(poses[:, :3, 3]-reference[:3, 3], axis=1),
            Rotation.from_matrix(reference[:3, :3].T @ poses[:, :3, :3]).magnitude())


def fuse(poses):
    poses = np.asarray(poses)
    reference = min(poses, key=lambda p: np.median(residual(poses, p)[0]/.015 + residual(poses, p)[1]/np.deg2rad(3)))
    distance, angle = residual(poses, reference)
    keep = (distance < .015) & (angle < np.deg2rad(3))
    if keep.sum() < 3:
        raise ValueError("Fewer than three consistent board poses")
    return transform(Rotation.from_matrix(poses[keep, :3, :3]).mean().as_matrix(), poses[keep, :3, 3].mean(axis=0)), keep


def solve_base(robot, wrist):
    robot, wrist = np.asarray(robot), np.asarray(wrist)
    if len(robot) < 20:
        raise ValueError("Need at least 20 matched wrist poses per arm")
    train = np.arange(len(robot)) % 5 != 0
    test = ~train
    motion = Rotation.from_matrix(robot[train][0, :3, :3].T @ robot[train, :3, :3]).as_rotvec()
    if np.linalg.svd(motion, compute_uv=False)[1]/np.sqrt(train.sum()) < .1:
        raise ValueError("Need robot rotations about two nonparallel axes")
    r, t = cv2.calibrateHandEye(list(robot[train, :3, :3]), list(robot[train, :3, 3]),
                              list(wrist[train, :3, :3]), list(wrist[train, :3, 3]), method=cv2.CALIB_HAND_EYE_PARK)
    eye = transform(r, t)
    fixed, keep = fuse(robot[train] @ eye @ wrist[train])
    if keep.mean() < .7:
        raise ValueError("Inconsistent board poses; check input data")
    distance, angle = residual(robot[test] @ eye @ wrist[test], fixed)
    print(f"held-out board consistency: {np.sqrt(np.mean(distance**2))*1000:.2f} mm, "
          f"{np.rad2deg(np.sqrt(np.mean(angle**2))):.2f} degree")
    if not np.isfinite(eye).all() or np.sqrt(np.mean(distance**2)) > .015 or np.sqrt(np.mean(angle**2)) > np.deg2rad(3):
        raise ValueError("Independent validation failed; output not saved")
    return fixed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Episode directory containing videos and samples.jsonl")
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--board", nargs=4, type=float, required=True, metavar=("NX", "NY", "SQUARE_M", "MARKER_M"))
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--output", type=Path, default=Path("calibration/extrinsics.yaml"))
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in (args.data/"samples.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    cameras = json.loads(args.intrinsics.read_text(encoding="utf-8-sig"))
    nx, ny, square, marker = args.board
    board = cv2.aruco.CharucoBoard((int(nx), int(ny)), square, marker,
                                  cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary)))
    samples = {name: [] for name in CAMERAS}
    for name in CAMERAS:
        video = cv2.VideoCapture(str(args.data/f"{name}.mp4"))
        try:
            if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
                raise ValueError(f"{name}: video/JSONL frame count mismatch")
            for index, row in enumerate(rows):
                ok, image = video.read()
                if not ok or row["frame_index"] != index:
                    raise ValueError(f"{name}: frame alignment error")
                if index % 5 or not row.get("sample_valid", True):
                    continue
                pose = np.eye(4)
                if name != "head":
                    side, offset = ("left", 0) if name == "left_wrist" else ("right", 8)
                    stamps = row["source_timestamps_unix_s"]
                    times = [v for k, v in stamps["states"][side].items() if k == "tcp" or k.startswith("end_pose_")]
                    if not times or max(abs(t-stamps["cameras"][name]) for t in times) > .02:
                        continue
                    state = row["observation.state"][offset:offset+7]
                    pose = transform(Rotation.from_quat(state[3:]).as_matrix(), state[:3])
                detected = board_pose(image, board, cameras[name])
                if detected is not None:
                    samples[name].append((pose, detected))
        finally:
            video.release()
        print(f"{name}: {len(samples[name])} valid board poses")
    head_board, _ = fuse([pair[1] for pair in samples["head"]])
    result = {}
    for side, key in (("left_wrist", "T_Head_LeftBase"), ("right_wrist", "T_Head_RightBase")):
        pairs = np.asarray(samples[side])
        result[key] = head_board @ np.linalg.inv(solve_base(pairs[:, 0], pairs[:, 1]))
    text = "".join(key+":\n"+"".join("  - "+json.dumps(row)+"\n" for row in value.tolist()) for key, value in result.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
