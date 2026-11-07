"""Default end-to-end EgoHand camera-wrist annotation workflow."""

import argparse
import json
from pathlib import Path

from humanlabel.utils.paths import ROOT


def complete_for(folder, source):
    path = folder / "manifest.json"
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    return (
        meta.get("status") == "complete"
        and Path(meta.get("source", "")).resolve() == source.resolve()
        and meta.get("processed_frames")
        == json.loads((source / "manifest.json").read_text())["frames"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path, default=ROOT / "human demonstration"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "result")
    parser.add_argument("--piper-config", type=Path)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    episodes = (
        [source] if (source / "head.mp4").exists() else sorted(source.glob("episode_*"))
    )
    if not episodes:
        parser.error(f"No head-view episodes in {source}")
    native, recovered = output / "egohand", output / "egohand_context"
    if not all(complete_for(native / p.name, p) for p in episodes):
        from humanlabel.workflows.egohand import main as infer

        infer([str(source), "--output", str(native)])
    if not all(complete_for(recovered / p.name, p) for p in episodes):
        from humanlabel.workflows.recover import main as recover

        recover(["--source", str(native), "--output", str(recovered)])
    from humanlabel.export import main as export

    arguments = [str(recovered / p.name) for p in episodes]
    if args.piper_config:
        arguments += ["--piper-config", str(args.piper_config)]
    export(arguments)
    from humanlabel.visualization.compare import validate

    reports = {p.name: validate(recovered / p.name) for p in episodes}
    (recovered / "validation_all.json").write_text(json.dumps(reports, indent=2))
    best = output / "best"
    if best.is_symlink():
        if best.resolve() != recovered.resolve():
            raise ValueError(f"Existing best link points elsewhere: {best}")
    elif best.exists():
        raise ValueError(f"Preserve existing non-symlink result directory: {best}")
    else:
        best.symlink_to("egohand_context", target_is_directory=True)
    print(
        f"Camera wrist eposes ready: {best}; {sum(r['processed_frames'] for r in reports.values())} frames",
        flush=True,
    )
