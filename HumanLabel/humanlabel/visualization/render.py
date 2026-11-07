"""Render ACE outputs with the camera estimate used for each capture segment.

The physical head camera is fixed. These are varying model estimates of that
camera, not a claim of changing measured intrinsics. Raw 3D predictions stay in
the optical frame. A single segment's K must not be used to score the overlays
of every other independently inferred segment.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from humanlabel.utils.episode import ROOT, export, read_episode
from humanlabel.utils.tracking import contiguous_ranges


def reexport(folder):
    meta = json.loads((folder / "manifest.json").read_text())
    if meta.get("status") != "complete":
        raise ValueError(f"Incomplete output {folder}")
    episode = Path(meta["source"])
    _, rows, size, fps, n = read_episode(episode)
    representation = meta.get("representation", "MANO")
    with np.load(
        folder / ("soma.npz" if representation == "SOMA" else "mano.npz")
    ) as d:
        keys = (
            "trans",
            "rotmat",
            "betas",
            "visible",
            "presence",
            "completion",
            "identity_coeffs",
            "scale_params",
            "global_scale",
            "native_joints25_camera_m",
            "direct_joints2d",
            "camera_matrix_per_frame",
        )
        prediction = {k: d[k] for k in keys if k in d}
        joints, vertices = d["joints_camera_m"], d["vertices_camera_m"]
        faces = {0: d["faces_left"], 1: d["faces_right"]} if "faces_left" in d else None
    with np.load(folder / "tracking.npz") as t:
        observed, supported, scores, gap_length, grid = (
            t[k] for k in ("observed", "supported", "scores", "gap_length", "grid")
        )
    if meta["backend"] == "ace":
        prediction["completion"] = ~observed
        meta["validity_support"] = (
            "independent RGB geometry and predicted 3D presence; detector support is only a review flag"
        )
    ranges = contiguous_ranges(grid)
    details_path = folder / "camera_prediction.json"
    details = json.loads(details_path.read_text()) if details_path.exists() else None
    cameras = details["per_capture_segment_cameras"] if details else []
    if not cameras:
        cameras = [(b - a, None) for a, b in ranges]
    if len(cameras) != len(ranges):
        raise ValueError("Some capture segments lack a camera prediction")
    matrices = np.empty((n, 3, 3), np.float64)
    for (a, b), (length, k) in zip(ranges, cameras):
        if b - a != length:
            raise ValueError("Camera/source segment mismatch")
        matrix = (
            np.array(
                [
                    [
                        k["fx"] * size[0] / k["image_width"],
                        0,
                        k["cx"] * size[0] / k["image_width"],
                    ],
                    [
                        0,
                        k["fy"] * size[1] / k["image_height"],
                        k["cy"] * size[1] / k["image_height"],
                    ],
                    [0, 0, 1.0],
                ]
            )
            if k is not None
            else np.asarray(meta["camera_matrix"])
        )
        if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
            raise ValueError("Invalid predicted pinhole")
        matrices[a:b] = matrix
    if details or "camera_matrix_per_frame" not in prediction:
        prediction["camera_matrix_per_frame"] = matrices
    # Retain the independent direct 2D readout in the re-exported archive.
    if faces is None:
        import pickle

        with (ROOT / ".cache/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl").open("rb") as f:
            faces = np.asarray(pickle.load(f)["f"])
    if details:
        meta["overlay_camera"] = (
            "per-capture-segment model-predicted K; physical camera fixed; estimates are not calibration"
        )
        meta["intrinsics_basis"] = "model_predicted_per_capture_segment"
    source_meta = json.loads((episode / "manifest.json").read_text())
    meta.update(
        source_status=source_meta["status"],
        source_dropped_grid_ticks=source_meta.get("dropped_grid_ticks", 0),
    )
    export(
        episode,
        folder,
        rows,
        size,
        fps,
        prediction,
        joints,
        vertices,
        faces,
        observed,
        supported,
        scores,
        gap_length,
        np.asarray(meta["camera_matrix"]),
        meta,
    )
    (folder / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(f"Native-camera overlay exported: {folder.name}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, nargs="?", default=ROOT / "result/ace")
    args = p.parse_args(argv)
    for folder in sorted(args.root.glob("episode_*")):
        reexport(folder)


if __name__ == "__main__":
    main()
