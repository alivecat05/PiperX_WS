"""Refine a finished MANO run on CUDA without reloading an image network."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from humanlabel.utils.episode import (
    ROOT,
    export,
    geometry_arrays,
    read_episode,
    save_npz,
)
from humanlabel.utils.hand_model import MANOGeometry
from humanlabel.utils.refinement import refine_sequence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args(argv)
    import torch

    torch.set_num_threads(8)
    backend = MANOGeometry(ROOT / ".cache/HaWoR/_DATA/data/mano")
    summaries = []
    for folder in sorted(args.source.glob("episode_*")):
        meta = json.loads((folder / "manifest.json").read_text())
        if meta["status"] != "complete" or meta.get("representation", "MANO") != "MANO":
            raise ValueError("Refinement requires a complete MANO run")
        output = args.output / folder.name
        if (output / "manifest.json").exists() and json.loads(
            (output / "manifest.json").read_text()
        ).get("status") == "complete":
            summaries.append(json.loads((output / "manifest.json").read_text()))
            continue
        output.mkdir(parents=True, exist_ok=True)
        meta.update(
            status="processing",
            parent_result=str(folder.resolve()),
            refinement="CUDA shared-shape MANO fit with robust model-projection anchors and joint acceleration; local implementation inspired by temporal optimization, not official Dyn-HaMR",
        )
        started = time.monotonic()
        with np.load(folder / "mano.npz") as d:
            prediction = {
                k: d[k]
                for k in (
                    "trans",
                    "rotmat",
                    "betas",
                    "visible",
                    "presence",
                    "completion",
                    "camera_matrix_per_frame",
                    "direct_joints2d",
                )
                if k in d
            }
            original_joints = d["joints_camera_m"]
        with np.load(folder / "tracking.npz") as d:
            observed, supported, scores, gap_length, grid = (
                d[k] for k in ("observed", "supported", "scores", "gap_length", "grid")
            )
        matrix = np.asarray(meta["camera_matrix"])
        try:
            # Check parameter-to-geometry equivalence before any optimization.
            reconstructed, _ = geometry_arrays(backend, prediction)
            good = np.isfinite(original_joints).all(-1)
            if not np.allclose(reconstructed[good], original_joints[good], atol=2e-5):
                raise ValueError(
                    "Canonical MANO parameters do not reproduce the source geometry"
                )
            prediction = refine_sequence(
                backend, prediction, observed, supported, grid, matrix, args.steps
            )
            joints, vertices = geometry_arrays(backend, prediction)
            delta = np.linalg.norm(joints - original_joints, axis=-1)
            meta["refinement_displacement_mm"] = {
                "median": float(np.nanmedian(delta) * 1000),
                "p95": float(np.nanpercentile(delta, 95) * 1000),
            }
            _, rows, size, fps, _ = read_episode(Path(meta["source"]))
            export(
                Path(meta["source"]),
                output,
                rows,
                size,
                fps,
                prediction,
                joints,
                vertices,
                backend.faces,
                observed,
                supported,
                scores,
                gap_length,
                matrix,
                meta,
            )
            shutil.copy2(folder / "tracking.npz", output / "tracking.npz")
            if (folder / "camera_prediction.json").exists():
                shutil.copy2(
                    folder / "camera_prediction.json", output / "camera_prediction.json"
                )
            save_npz(output / "prediction_raw.npz", prediction)
            meta.update(status="complete", refinement_wall_s=time.monotonic() - started)
            summaries.append(meta)
            print(
                f"Refinement saved: {folder.name}, displacement={meta['refinement_displacement_mm']}",
                flush=True,
            )
        except BaseException as exc:
            meta.update(status="failed", error=str(exc))
            raise
        finally:
            (output / "manifest.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )
    (args.output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
