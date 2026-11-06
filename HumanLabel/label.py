"""Head wrist poses + synchronized pinch refinement, with MediaPipe or WiLoR/MANO."""
import argparse
from contextlib import ExitStack
import csv
from importlib.metadata import version
import json
from pathlib import Path
import urllib.request
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
SIDES = ("left", "right")
VECTOR_NAMES = [f"{s}.{f}" for s in SIDES
                for f in ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper_m")]
HEADER = ["timestamp_unix_s", *VECTOR_NAMES]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
REVISION = "wrist_origin_with_wrist_camera_pinch"


def pipeline_revision(args):
    return REVISION + ("_mano_root_perspective_v2" if getattr(args, "backend", "mediapipe") == "mano" else "")


def camera_model(width, height, path=None):
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if (data["width"], data["height"]) != (width, height):
            raise ValueError("Intrinsics resolution must match head.mp4; do not rescale implicitly")
        matrix = np.asarray(data["camera_matrix"], dtype=float)
        distortion = np.asarray(data.get("distortion_coeffs", [0.] * 5), dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1])
                or distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()):
            raise ValueError("Invalid OpenCV pinhole intrinsics/distortion")
        basis = "provided_opencv_pinhole"
    else:
        focal = float(max(width, height))
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
        distortion, basis = np.zeros(5), "ASSUMED_fx_fy_max_width_height_zero_distortion"
    return matrix, distortion, basis


def wrist_pose(points):
    """T_C_W: origin at WRIST #0; +z wrist->middle MCP; +x pinky->index MCP."""
    def unit(v):
        length = np.linalg.norm(v)
        if length < 1e-6:
            raise ValueError("degenerate_palm")
        return v / length

    z = unit(points[9] - points[0])
    lateral = points[5] - points[17]
    x = unit(lateral - np.dot(lateral, z) * z)
    y = unit(np.cross(z, x))
    x = np.cross(y, z)
    rotation = np.column_stack((x, y, z))
    return points[0].copy(), Rotation.from_matrix(rotation).as_quat()


def gripper_width(distance, args):
    return float(args.gripper_max_m * np.clip(
        (distance - args.pinch_closed_m) / (args.pinch_open_m - args.pinch_closed_m), 0, 1))


def estimate(world, pixels, matrix, distortion, args):
    """PnP maps hand-centered model coordinates into the optical camera frame."""
    world = np.asarray(world, dtype=np.float64) * args.hand_scale
    pixels = np.asarray(pixels, dtype=np.float64)
    if world.shape != (21, 3) or pixels.shape != (21, 2) or not np.isfinite(world).all():
        raise ValueError("invalid_landmarks")
    if not np.isfinite(pixels).all():
        raise ValueError("invalid_landmarks")
    ok, rvec, tvec = cv2.solvePnP(world, pixels, matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise ValueError("pnp_failed")
    rvec, tvec = cv2.solvePnPRefineLM(world, pixels, matrix, distortion, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    points = world @ rotation.T + tvec.reshape(3)
    projected = cv2.projectPoints(world, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
    if (not np.isfinite(points).all() or not np.isfinite(error)
            or np.any(points[:, 2] < .05) or np.any(points[:, 2] > 3.0)):
        raise ValueError("invalid_depth")
    if error > args.max_reprojection_px:
        raise ValueError("high_reprojection_error")
    position, quaternion = wrist_pose(points)
    distance = float(np.linalg.norm(points[4] - points[8]))
    return np.array([*position, *quaternion, gripper_width(distance, args)]), distance, error


def timestamps(episode, count, manifest, cameras=("head",)):
    with (episode / "timestamps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != count or manifest["frames"] != count or not rows:
        raise ValueError("Video/manifest/timestamps frame counts differ or episode is empty")
    for i, row in enumerate(rows):
        for key in ("timestamp_unix_s", *[f"{c}_timestamp_unix_s" for c in cameras]):
            if not np.isfinite(float(row[key])):
                raise ValueError("Non-finite source timestamp")
            if i and float(row[key]) <= float(rows[i - 1][key]):
                raise ValueError("Source timestamps must increase strictly")
        if int(row["frame_index"]) != i or (i and int(row["grid_index"]) <= int(rows[i - 1]["grid_index"])):
            raise ValueError("Invalid source frame/grid order")
    return rows


def observations(result, width, height, args):
    """Keep anatomical identity and fingertip validity separate from wrist pose."""
    hands = {s: [] for s in SIDES}
    for cats, image_lm, world_lm in zip(result.handedness, result.hand_landmarks, result.hand_world_landmarks):
        side = cats[0].category_name.lower()
        if args.swap_handedness:
            side = "right" if side == "left" else "left"
        pixels = np.array([[p.x * width, p.y * height] for p in image_lm])
        world = np.array([[p.x, p.y, p.z] for p in world_lm])
        inside = np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1) & (pixels < [width, height]).all(axis=1)
        reliable = cats[0].score >= args.min_handedness
        pinch_valid = bool(reliable and inside[[4, 8]].all() and np.isfinite(world[[4, 8]]).all())
        hands[side].append({"pixels": pixels, "world": world, "score": cats[0].score,
                            "image_height": height,
                            "inside": inside, "wrist_in_image": bool(reliable and inside[0]),
                            "pinch_valid": pinch_valid,
                            "distance": float(np.linalg.norm(world[4] - world[8]) * args.hand_scale) if pinch_valid else np.nan,
                            "distance_px": float(np.linalg.norm(pixels[4] - pixels[8])) if pinch_valid else np.nan,
                            "reason": "ok" if reliable else "low_handedness_score"})
    return hands


def unique_hand(hands):
    if len(hands) == 1:
        return hands[0]
    return {"pixels": None, "world": None, "score": np.nan, "inside": np.zeros(21, bool),
            "wrist_in_image": False, "pinch_valid": False, "distance": np.nan, "distance_px": np.nan,
            "reason": "ambiguous_handedness" if hands else "not_detected"}


def fuse_pinch(head, wrist, side, args):
    """Refine a scalar distance only; never mix positions from different cameras."""
    # Own hand lies near the bottom of a camera mounted on that wrist. Require
    # both full finger chains inside the real image, not just two hallucinated tips.
    required = [3, 4, 7, 8] if "mano" in wrist and head["pinch_valid"] else list(range(1, 9))
    supported = bool(wrist["pinch_valid"] and wrist["inside"][required].all()
                     and wrist["pixels"][0, 1] >= wrist["image_height"] * .55)
    if not supported:
        return head["distance"], "head" if head["pinch_valid"] else "none", "wrist_unavailable"
    if not head["pinch_valid"]:
        return wrist["distance"], f"{side}_wrist", "wrist_fallback"
    if abs(head["distance"] - wrist["distance"]) > args.max_wrist_disagreement_m:
        return head["distance"], "head", "wrist_disagreement"
    return (head["distance"] + wrist["distance"]) / 2, f"head+{side}_wrist", "two_view_agreement"


def mark_wrist(frame, hand, side):
    if not hand["wrist_in_image"]:
        return
    point = tuple(np.rint(hand["pixels"][0]).astype(int))
    cv2.circle(frame, point, 9, (255, 255, 0), 2)
    cv2.drawMarker(frame, point, (255, 255, 0), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(frame, f"{side} WRIST #0", (max(5, point[0] - 70), max(45, point[1] - 14)),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 2)


def draw(frame, hand, vector, distance, side, matrix, distortion, source):
    color = (80, 220, 80) if side == "left" else (0, 180, 255)
    mark_wrist(frame, hand, side)
    if hand["pinch_valid"]:
        a, b = np.rint(hand["pixels"][[4, 8]]).astype(int)
        cv2.circle(frame, tuple(a), 5, color, -1)
        cv2.circle(frame, tuple(b), 5, color, -1)
        cv2.line(frame, tuple(a), tuple(b), color, 2)
    if np.isfinite(vector[:7]).all():
        rvec = Rotation.from_quat(vector[3:7]).as_rotvec()
        cv2.drawFrameAxes(frame, matrix, distortion, rvec, vector[:3], .04, 2)
    cv2.putText(frame, f"{side}: grip {vector[7] * 1000:.0f}mm | {source}",
                (20, 65 if side == "left" else 95), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)


def process(episode, output, model, args, predictor=None):
    mano_mode = getattr(args, "backend", "mediapipe") == "mano"
    if not mano_mode:
        import mediapipe as mp

    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "human_capture.rgb.v1":
        raise ValueError(f"Unsupported source schema: {episode}")
    if manifest.get("simulated"):
        print(f"Skip simulated video: {episode}", flush=True)
        return
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)  # Never replace an existing label run.
    meta = {"schema": "human_label.camera_pseudo_action.v1", "pipeline_revision": pipeline_revision(args), "status": "processing",
            "source": str(episode), "source_status": manifest["status"], "task": manifest["task"],
            "robot_base_calibrated": False, "action_semantics": "same-frame estimated human WRIST pose + refined pinch width; NOT robot commands",
            "vector_names": VECTOR_NAMES, "reference_frame": "fixed_head_color_optical", "camera_fixed": True,
            "pose_convention": "T_C_W; origin=WRIST landmark #0; right-handed; head camera +x right +y down +z forward; meters; quaternion xyzw",
            "tool_convention": "origin=P0; +z=WRIST->MIDDLE_MCP; +x=orthogonalized PINKY_MCP->INDEX_MCP; +y=z cross x",
            "pinch_refinement": "own wrist camera; same anatomical side; near-field wrist and full two-finger chains; average with head only within disagreement threshold; otherwise head fallback",
            "backend": "mano" if mano_mode else "mediapipe",
            "parameters": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, bool))},
            "missing_policy": "action_raw.csv keeps NaN; action.csv includes only both-valid frames; no interpolation",
            "mask_semantics": "valid=pose geometry passed; pinch_valid=two tips in image and handedness passed; neither proves unoccluded visibility",
            "frame_mapping": "action.csv row N maps through frame_map.csv row N to source MP4 frame_index",
            "training_contract": "Use frame_map to select RGB and split by segment_id; use quality masks for per-arm losses; calibrate/retarget before mixing robot actions"}
    for key in ("derived_from", "source_frame_indices", "source_ranges"):
        if key in manifest:
            meta[key] = manifest[key]
    if mano_mode:
        if predictor is None:
            raise ValueError("MANO predictor missing; run setup_mano.cmd then label_mano.cmd")
        meta.update(model=predictor.provenance(),
                    tool_convention="origin=MANO joints[0]; rotation=MANO canonical wrist global_orient; left uses F R F with F=diag(-1,1,1); quaternion xyzw",
                    pose_convention="T_C_W; MANO wrist root origin and canonical root axes; fixed head optical +x right +y down +z forward; meters; quaternion xyzw",
                    fingertip_convention="MANO vertex 744=thumb, 320=index; OpenPose21 joints 4/8; distance=Euclidean meters",
                    mano_parameters="mano.npz; camera,side axes explicit; left canonical parameters use the mirrored right template; hand_scale stored",
                    camera_conversion="Fit rigid perspective camera to WiLoR nominal model projections (f=5000/256*max_image_size); update root orientation and wrist-pivot translation; residual is NOT image landmark accuracy",
                    score_semantics="detector_score is YOLO box/class confidence; handedness_score and reprojection_rmse_px unavailable, stored NaN")
    else:
        meta.update(model_url=MODEL_URL, mediapipe_version=version("mediapipe"))
    meta_path = output / "manifest.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    cameras = ["head"]
    if not getattr(args, "head_only", False):
        cameras += [f"{s}_wrist" for s in SIDES if (episode / f"{s}_wrist.mp4").exists()]
    caps = {c: cv2.VideoCapture(str(episode / f"{c}.mp4")) for c in cameras}
    cap, preview, wrist_preview = caps["head"], None, None
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open head.mp4")
        width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if abs(fps - manifest["fps"]) > .01:
            raise ValueError("Video FPS differs from manifest")
        for camera, video in caps.items():
            if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                raise ValueError(f"Missing/truncated synchronized video: {camera}")
            if abs(video.get(cv2.CAP_PROP_FPS) - fps) > .01:
                raise ValueError(f"Synchronized video FPS mismatch: {camera}")
        rows = timestamps(episode, count, manifest, cameras)
        frame_limit = getattr(args, "max_frames", 0)
        limit = min(frame_limit, count) if frame_limit else count
        matrix, distortion, basis = camera_model(width, height, args.intrinsics)
        camera_models = {"head": (matrix, distortion, basis)}
        for camera in cameras[1:]:
            video = caps[camera]
            camera_models[camera] = camera_model(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        meta.update(fps=fps, source_frames=count, cameras_used=cameras, intrinsics_basis=basis,
                    camera_matrix=matrix.tolist(), distortion_coeffs=distortion.tolist(),
                    camera_intrinsics={c: {"matrix": k.tolist(), "distortion_coeffs": d.tolist(), "basis": b}
                                       for c, (k, d, b) in camera_models.items()})
        archive = None
        if mano_mode:
            from mano_backend import ManoArchive, draw_mesh
            archive = ManoArchive(limit, cameras, args.hand_scale)
        if not args.no_preview:
            preview = cv2.VideoWriter(str(output / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not preview.isOpened():
                raise OSError("Cannot create preview.mp4")
            if len(cameras) > 1:
                wrist_preview = cv2.VideoWriter(str(output / "wrist_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
                if not wrist_preview.isOpened():
                    raise OSError("Cannot create wrist_preview.mp4")
        quality_fields = ["frame_index", "grid_index", "timestamp_unix_s", "head_timestamp_unix_s"]
        side_fields = ("valid", "pose_valid", "reason", "handedness_score", "detector_score", "wrist_u_px", "wrist_v_px", "wrist_in_image",
                       "pinch_valid", "pinch_source", "pinch_reason", "pinch_distance_m", "pinch_distance_px", "gripper_m",
                       "head_pinch_distance_m", "wrist_pinch_distance_m", "wrist_pinch_distance_px",
                       "wrist_handedness_score", "wrist_detector_score", "wrist_timestamp_unix_s", "reprojection_rmse_px", "camera_conversion_rmse_px")
        quality_fields += [f"{s}.{f}" for s in SIDES for f in
                           side_fields]
        previous_q, valid_counts = {}, dict.fromkeys(SIDES, 0)
        pinch_counts = dict.fromkeys(SIDES, 0)
        source_counts = {s: {} for s in SIDES}
        saved, segment, last_grid, last_frame = 0, -1, None, None
        with (ExitStack() as stack,
              (output / "action.csv").open("w", newline="", encoding="utf-8") as actions,
              (output / "action_raw.csv").open("w", newline="", encoding="utf-8") as raw,
              (output / "quality.csv").open("w", newline="", encoding="utf-8") as quality,
              (output / "frame_map.csv").open("w", newline="", encoding="utf-8") as mapping):
            prefetched = None
            if mano_mode and getattr(predictor, "gpu_model", None) is not None:
                prefetched = predictor.iter_frames(caps, camera_models, args, limit)
                stack.callback(prefetched.close)
            detectors = {}
            for camera in ([] if mano_mode else cameras):
                threshold = .5 if camera == "head" else .3
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
                    min_hand_detection_confidence=threshold, min_hand_presence_confidence=threshold,
                    min_tracking_confidence=.5)
                detectors[camera] = stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(options))
            aw, rw = csv.writer(actions), csv.writer(raw)
            qw = csv.DictWriter(quality, fieldnames=quality_fields)
            mw = csv.writer(mapping)
            aw.writerow(HEADER)
            rw.writerow(HEADER)
            qw.writeheader()
            mw.writerow(["action_row_index", "frame_index", "grid_index", "segment_id", "timestamp_unix_s", "head_timestamp_unix_s"])
            for i, row in enumerate(rows[:limit]):
                images, detected = {}, {}
                if prefetched is not None:
                    images, detected = next(prefetched)
                for camera, video in ([] if prefetched is not None else caps.items()):
                    ok, image = video.read()
                    if not ok:
                        raise ValueError(f"Cannot decode {camera} source frame {i}")
                    images[camera] = image
                    key = f"{camera}_timestamp_unix_s"
                    ms = round((float(row[key]) - float(rows[0][key])) * 1000)
                    if mano_mode and not hasattr(predictor, "detect_many"):
                        detected[camera] = predictor.detect(image, *camera_models[camera][:2], args)
                    elif not mano_mode:
                        result = detectors[camera].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                                   data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), ms)
                        detected[camera] = observations(result, image.shape[1], image.shape[0], args)
                if mano_mode:
                    if prefetched is None and hasattr(predictor, "detect_many"):
                        detected = predictor.detect_many(images, camera_models, args)
                    for camera, image in images.items():
                        for side in SIDES:
                            hand = unique_hand(detected[camera][side])
                            archive.add(i, camera, side, hand)
                            if preview:
                                draw_mesh(image, hand, *camera_models[camera][:2], side)
                frame = images["head"]
                qrow = {k: row[k] for k in quality_fields[:4]}
                vectors = []
                for side in SIDES:
                    head = unique_hand(detected["head"][side])
                    camera = f"{side}_wrist"
                    wrist = unique_hand(detected.get(camera, {}).get(side, []))
                    value, reason, error = np.full(8, np.nan), head["reason"], np.nan
                    if reason == "ok":
                        reason = "landmarks_outside_image"
                        # Wrist pose no longer depends on fingertips being in the head image.
                        if head["inside"][[0, 5, 9, 17]].all():
                            try:
                                if mano_mode:
                                    value[:7] = head["pose"]
                                else:
                                    value, _, error = estimate(head["world"], head["pixels"], matrix, distortion, args)
                                reason = "ok"
                            except (ValueError, cv2.error) as exc:
                                reason = str(exc) if isinstance(exc, ValueError) else "pnp_failed"
                    pose_valid = bool(np.isfinite(value[:7]).all())
                    distance, source, pinch_reason = fuse_pinch(head, wrist, side, args)
                    pinch_valid = bool(np.isfinite(distance))
                    value[7] = gripper_width(distance, args) if pinch_valid else np.nan
                    if pinch_valid:
                        pinch_counts[side] += 1
                    source_counts[side][source] = source_counts[side].get(source, 0) + 1
                    valid = bool(np.isfinite(value).all())
                    if pose_valid:
                        if side in previous_q and np.dot(value[3:7], previous_q[side]) < 0:
                            value[3:7] *= -1
                        previous_q[side] = value[3:7].copy()
                    if valid:
                        valid_counts[side] += 1
                    if preview:
                        draw(frame, head, value, distance, side, matrix, distortion, source)
                    if wrist_preview and camera in images:
                        mark_wrist(images[camera], wrist, side)
                        if wrist["pinch_valid"]:
                            a, b = np.rint(wrist["pixels"][[4, 8]]).astype(int)
                            cv2.circle(images[camera], tuple(a), 6, (0, 220, 255), -1)
                            cv2.circle(images[camera], tuple(b), 6, (0, 220, 255), -1)
                            cv2.line(images[camera], tuple(a), tuple(b), (0, 220, 255), 2)
                        cv2.putText(images[camera], f"{camera}: {wrist['distance'] * 1000:.0f}mm | {pinch_reason}",
                                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
                    vectors.extend(value.tolist())
                    wrist_pixel = head["pixels"][0] if head["pixels"] is not None else [np.nan, np.nan]
                    for field, field_value in zip(side_fields,
                        (int(valid), int(pose_valid), reason, head.get("handedness_score", head["score"]),
                         head["score"] if mano_mode else np.nan, *wrist_pixel, int(head["wrist_in_image"]),
                         int(pinch_valid), source, pinch_reason, distance, head["distance_px"], value[7], head["distance"],
                         wrist["distance"], wrist["distance_px"], wrist.get("handedness_score", wrist["score"]),
                         wrist["score"] if mano_mode else np.nan, row.get(f"{camera}_timestamp_unix_s", ""), error,
                         head.get("camera_conversion_rmse_px", np.nan))):
                        qrow[f"{side}.{field}"] = field_value
                record = [row["timestamp_unix_s"], *vectors]
                rw.writerow(record)
                qw.writerow(qrow)
                if np.isfinite(vectors).all():
                    grid = int(row["grid_index"])
                    if last_grid is None or grid != last_grid + 1 or i != last_frame + 1:
                        segment += 1
                    aw.writerow(record)
                    mw.writerow([saved, i, grid, segment, row["timestamp_unix_s"], row["head_timestamp_unix_s"]])
                    saved, last_grid, last_frame = saved + 1, grid, i
                if preview:
                    cv2.putText(frame, f"WRIST POSE #0 | head optical | frame {i} | L {qrow['left.reason']} R {qrow['right.reason']}",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2)
                    preview.write(frame)
                if wrist_preview:
                    panels = [cv2.resize(images[f"{s}_wrist"], (640, 360)) if f"{s}_wrist" in images
                              else np.zeros((360, 640, 3), np.uint8) for s in SIDES]
                    wrist_preview.write(cv2.hconcat(panels))
                if (i + 1) % (10 if mano_mode else 300) == 0:
                    print(f"  {episode.name}: {i + 1}/{limit} frames", flush=True)
            for camera, video in ([] if limit < count else caps.items()):
                if video.read()[0]:
                    raise ValueError(f"{camera} contains more frames than declared")
        if archive is not None:
            archive.save(output / "mano.npz", limit)
        meta.update(status="complete" if limit == count else "partial", processed_frames=limit,
                    labeling_wall_s=time.perf_counter() - started,
                    labeling_frames_per_s=limit / (time.perf_counter() - started),
                    stop_reason="end_of_video" if limit == count else "frame_limit", valid_frames_per_hand=valid_counts,
                    pinch_valid_frames_per_hand=pinch_counts, pinch_source_counts=source_counts,
                    both_valid_frames=saved, contiguous_segments=segment + 1)
        print(f"Saved {output}: {limit}/{count} frames, valid L/R={valid_counts}, both={saved}", flush=True)
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        raise
    finally:
        for video in caps.values():
            video.release()
        if preview:
            preview.release()
        if wrist_preview:
            wrist_preview.release()
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT.parent / "HumanCapture/outputs/episodes")
    parser.add_argument("--output", type=Path, help="Defaults to result for MediaPipe, result/mano for MANO")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/hand_landmarker.task")
    parser.add_argument("--backend", choices=("mediapipe", "mano"), default="mediapipe")
    parser.add_argument("--mano-cache", type=Path, default=ROOT / ".cache/wilor")
    parser.add_argument("--mano-repo", type=Path, default=ROOT / ".cache/WiLoR-mini")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "igpu"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-detection-score", type=float, default=.3, help="MANO backend: YOLO detector confidence")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit a validation run; 0 processes the whole episode")
    parser.add_argument("--intrinsics", type=Path, help="OpenCV pinhole JSON for the head COLOR stream")
    parser.add_argument("--hand-scale", type=float, default=1.)
    parser.add_argument("--gripper-max-m", type=float, default=.1, help="Matches current PiperX_Teleop config")
    parser.add_argument("--pinch-closed-m", type=float, default=0., help="Human fingertip distance mapped to closed gripper")
    parser.add_argument("--pinch-open-m", type=float, default=.1, help="Human fingertip distance mapped to max gripper width")
    parser.add_argument("--min-handedness", type=float, default=.8)
    parser.add_argument("--max-reprojection-px", type=float, default=12.)
    parser.add_argument("--swap-handedness", action="store_true", help="Only for mirrored input after visual verification")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--head-only", action="store_true", help="Disable wrist-camera pinch refinement")
    parser.add_argument("--max-wrist-disagreement-m", type=float, default=.025,
                        help="Keep head distance if the matching wrist estimate differs by more than this")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("result/mano" if args.backend == "mano" else "result")
    if (not all(np.isfinite(x) for x in (args.hand_scale, args.gripper_max_m, args.pinch_closed_m,
                                         args.pinch_open_m, args.min_handedness, args.max_reprojection_px,
                                         args.max_wrist_disagreement_m))
            or args.hand_scale <= 0 or args.gripper_max_m <= 0 or args.pinch_closed_m < 0
            or args.pinch_open_m <= args.pinch_closed_m or not 0 <= args.min_handedness <= 1
            or args.max_reprojection_px <= 0 or args.max_wrist_disagreement_m <= 0
            or args.threads < 1 or args.max_frames < 0 or not 0 < args.min_detection_score <= 1):
        parser.error("Invalid scale, distance, confidence or reprojection threshold")
    source = args.source.resolve()
    episodes = [source] if (source / "head.mp4").exists() else sorted(p.parent for p in source.rglob("head.mp4"))
    if not episodes:
        parser.error(f"No head.mp4 found: {source}")
    if args.backend == "mediapipe" and not args.model.exists():
        args.model.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.model.with_suffix(".download")
        print("Downloading official MediaPipe model...", flush=True)
        urllib.request.urlretrieve(MODEL_URL, temporary)
        temporary.replace(args.model)
    predictor = None
    for episode in episodes:
        # Preserve task/episode layout. Single-episode calls still keep task and episode names.
        relative = Path(episode.parent.name) / episode.name if episode == source else episode.relative_to(source)
        target = args.output.resolve() / relative
        if target.exists():
            previous = target / "manifest.json"
            if not previous.exists() or json.loads(previous.read_text(encoding="utf-8")).get("status") != "complete":
                raise ValueError(f"Unfinished result exists; choose a new --output directory: {target}")
            if json.loads(previous.read_text(encoding="utf-8")).get("pipeline_revision") != pipeline_revision(args):
                raise ValueError(f"Result uses an older pose definition; archive it or use --output: {target}")
            print(f"Skip existing result (use --output for a new run): {target}", flush=True)
            continue
        if args.backend == "mano" and predictor is None:
            source_meta = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
            if not source_meta.get("simulated"):
                from mano_backend import WiLoRBackend
                print("Loading WiLoR/MANO...", flush=True)
                predictor = WiLoRBackend(args.mano_cache, args.mano_repo, args.device, args.threads, args.min_detection_score)
        process(episode, target, args.model, args, predictor)


if __name__ == "__main__":
    main()
