"""Optional HaWoR/ACE/WiLoR comparison inference, separate from the default EgoHand workflow."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from humanlabel.utils.camera import camera_model
from humanlabel.utils.episode import (
    detections,
    export,
    geometry_arrays,
    infer_hawor,
    read_episode,
    save_npz,
)
from humanlabel.utils.paths import ROOT
from humanlabel.utils.tracking import track_boxes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path, default=ROOT / "human demonstration"
    )
    parser.add_argument("--backend", choices=["hawor", "ace", "wilor"], default="hawor")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-gap", type=int, default=90)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Fit shared hand shape and smooth MANO joint motion on CUDA",
    )
    args = parser.parse_args(argv)
    if args.max_frames < 0 or args.max_gap < 0:
        parser.error("Frame counts must be nonnegative")
    episodes = (
        [args.source]
        if (args.source / "head.mp4").exists()
        else sorted(args.source.glob("episode_*/"))
    )
    if not episodes:
        parser.error("No episodes found")
    output_root = args.output or ROOT / "result" / (
        args.backend + ("_refined" if args.refine else "")
    )
    backend = None
    summary = []
    for episode in episodes:
        source_meta, rows, size, fps, source_count = read_episode(
            episode, args.max_frames
        )
        output = output_root / episode.name
        if (output / "manifest.json").exists():
            previous = json.loads((output / "manifest.json").read_text())
            if previous.get("status") == "complete":
                summary.append(previous)
                print(f"Skip complete {output}", flush=True)
                continue
        output.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        meta = {
            "schema": "human_label.camera_pseudo_action.v2",
            "backend": args.backend,
            "status": "processing",
            "source": str(episode.resolve()),
            "source_frames": source_count,
            "source_status": source_meta["status"],
            "source_dropped_grid_ticks": source_meta.get("dropped_grid_ticks", 0),
            "fps": fps,
            "size": size,
            "camera_fixed": True,
            "robot_base_calibrated": False,
            "time_semantics": "unaltered source timestamps and grid indexes; processing split at every grid discontinuity",
            "mask_semantics": "detected is detector support, not ground-truth visibility; completion is an estimate requiring review",
            "pose_convention": "MANO wrist joints[0] in head optical frame, meters, proper rotation quaternion xyzw",
            "left_template": "mathematically reflected MANO_RIGHT; not an independently supplied MANO_LEFT",
            "parameters": {
                "max_gap": args.max_gap,
                "max_frames": args.max_frames,
                "refine": args.refine,
            },
        }
        mp = output / "manifest.json"
        mp.write_text(json.dumps(meta, indent=2))
        try:
            ds, paths = detections(
                episode, rows, size, ROOT / ".cache/weights/external/detector.pt"
            )
            grid = np.array([int(r["grid_index"]) for r in rows])
            boxes, scores, observed, supported, gap_length = track_boxes(
                ds, grid, args.max_gap
            )
            save_npz(
                output / "tracking.npz",
                {
                    "boxes": boxes,
                    "scores": scores,
                    "observed": observed,
                    "supported": supported,
                    "gap_length": gap_length,
                    "grid": grid,
                },
            )
            if args.detect_only:
                print(
                    f"Detection cached: {episode.name} observed={observed.sum(1)} / {len(rows)}",
                    flush=True,
                )
                continue
            matrix, _, basis = camera_model(*size)
            if backend is None:
                if args.backend == "hawor":
                    from humanlabel.backends.hawor import HaWoRBackend

                    backend = HaWoRBackend(ROOT)
                elif args.backend == "ace":
                    from humanlabel.backends.ace import ACEBackend

                    backend = ACEBackend(ROOT)
                else:
                    from humanlabel.backends.wilor import WiLoRVideoBackend

                    backend = WiLoRVideoBackend(ROOT)
            if args.backend == "hawor":
                prediction, raw = infer_hawor(
                    backend, paths, boxes, observed, supported, grid, matrix, output
                )
            else:
                prediction, matrix, details = backend.inference(
                    episode, paths, boxes, observed, supported, grid, matrix, output
                )
                meta.update(details)
            import torch

            meta.update(
                device="cuda",
                gpu_name=torch.cuda.get_device_name(0),
                torch_version=torch.__version__,
                adapter_revision="temporal_head_v3",
                ground_truth_available=False,
            )
            if args.refine:
                from humanlabel.utils.refinement import refine_sequence

                prediction = refine_sequence(
                    backend, prediction, observed, supported, grid, matrix
                )
                meta["refinement"] = (
                    "local GPU MANO fitting: shared shape, robust model-projection anchors, joint acceleration; inspired by temporal optimization, not official Dyn-HaMR"
                )
            joints, vertices = geometry_arrays(backend, prediction)
            meta.update(
                camera_matrix=matrix.tolist(),
                intrinsics_basis="model_predicted" if args.backend == "ace" else basis,
            )
            export(
                episode,
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
                args.no_preview,
            )
            meta.update(
                status="complete" if len(rows) == source_count else "partial",
                wall_s=time.monotonic() - started,
            )
            summary.append(meta)
            print(
                f"Saved {output}: {meta['processed_frames']} frames, recovered {meta['recovered_frames']}",
                flush=True,
            )
        except BaseException as exc:
            meta.update(status="failed", error=str(exc))
            raise
        finally:
            mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    if not args.detect_only:
        (output_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )


if __name__ == "__main__":
    main()
