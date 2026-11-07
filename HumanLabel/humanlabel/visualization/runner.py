"""Reproduce all four full-length GPU runs and synchronized comparison videos."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from humanlabel.utils.paths import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, nargs="?", default=ROOT / "human demonstration"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "result")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["wilor", "hawor", "ace", "egohand"],
        default=["wilor", "hawor", "ace", "egohand"],
    )
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Include local ACE refinement and context-crop EgoHand",
    )
    for method in ("wilor", "hawor", "ace", "egohand"):
        default = (
            ROOT / ("." + method + "-venv/bin/python")
            if method in ("wilor", "hawor")
            else Path(sys.executable)
        )
        parser.add_argument(
            "--" + method + "-python",
            type=Path,
            default=Path(os.environ.get(method.upper() + "_PYTHON", str(default))),
        )
    args = parser.parse_args(argv)
    if args.enhanced and not {"ace", "egohand"}.issubset(args.methods):
        parser.error("--enhanced requires ace and egohand")
    environments = {
        m: getattr(args, m + "_python") for m in ("wilor", "hawor", "ace", "egohand")
    }
    names = {
        "wilor": "WiLoR-MANO",
        "hawor": "HaWoR + infiller",
        "ace": "ACE-Ego-Hand K-free",
        "egohand": "NVIDIA EgoHand + temporal smoothing",
    }
    log_root = ROOT / ".cache/comparison_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    source, output = args.source.resolve(), args.output.resolve()
    env = {**os.environ, "OMP_NUM_THREADS": "8", "YOLO_AUTOINSTALL": "false"}
    for method in args.methods:
        python = environments[method]
        if not args.render_only and not python.exists():
            raise FileNotFoundError(
                f"Missing prepared runtime: {python}; see README.md and configs/model_sources.json"
            )
        if not args.render_only:
            command = [
                str(python),
                "-u",
                "-m",
                "humanlabel.workflows.egohand"
                if method == "egohand"
                else "humanlabel.visualization.temporal",
                str(source),
                "--output",
                str(output / method),
            ]
            if method != "egohand":
                command += ["--backend", method]
            print(
                f"Running {names[method]}; completed episodes are reused. Log: {log_root / (method + '.log')}",
                flush=True,
            )
            with (log_root / f"{method}.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        subprocess.run(
            [
                str(environments["ace"]),
                "-m",
                "humanlabel.visualization.render",
                str(output / method),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    if args.enhanced:
        if not args.render_only:
            subprocess.run(
                [
                    str(environments["ace"]),
                    "-m",
                    "humanlabel.visualization.refine",
                    str(output / "ace"),
                    "--output",
                    str(output / "ace_refined"),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            subprocess.run(
                [
                    str(environments["ace"]),
                    "-m",
                    "humanlabel.workflows.recover",
                    "--source",
                    str(output / "egohand"),
                    "--output",
                    str(output / "egohand_context"),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
        args.methods += ["ace_refined", "egohand_context"]
        names.update(
            ace_refined="ACE + local temporal fit",
            egohand_context="EgoHand + context crops",
        )
    subprocess.run(
        [
            str(environments["ace"]),
            "-m",
            "humanlabel.export",
            *[str(output / m) for m in args.methods],
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            str(environments["ace"]),
            "-m",
            "humanlabel.visualization.compare",
            "--models",
            *[str(output / m) for m in args.methods],
            "--names",
            *[names[m] for m in args.methods],
            "--output",
            str(output / ("comparison_all" if args.enhanced else "comparison")),
            "--tile-width",
            "960" if args.enhanced else "1280",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print(
        f"Full comparison: {output / ('comparison_all' if args.enhanced else 'comparison') / 'all_episodes_comparison.mp4'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
