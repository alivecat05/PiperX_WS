"""Default end-to-end EgoHand camera-wrist annotation workflow."""

import argparse
import json
from pathlib import Path

from humanlabel.utils.paths import ROOT, temporary_workspace


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
        "source", nargs="?", type=Path, default=ROOT / "human_demonstration"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "result")
    parser.add_argument("--piper-config", type=Path)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    single_episode = (source / "head.mp4").exists()
    dataset = source.parent if single_episode else source
    episodes = (
        [source]
        if single_episode
        else sorted(p for p in source.glob("episode_*") if p.is_dir())
    )
    if not episodes:
        parser.error(f"No head-view episodes in {source}")
    final = output / dataset.name
    for episode in episodes:
        manifest = final / episode.name / "manifest.json"
        if manifest.exists():
            recorded_source = Path(json.loads(manifest.read_text()).get("source", ""))
            if recorded_source.resolve() != episode.resolve():
                parser.error(f"Output belongs to another source: {manifest}")
    pending = [p for p in episodes if not complete_for(final / p.name, p)]
    if pending:
        from humanlabel.workflows.egohand import main as infer
        from humanlabel.workflows.recover import main as recover

        with temporary_workspace() as work:
            native = work / "egohand"
            infer(
                [
                    str(source),
                    "--output",
                    str(native),
                    "--episodes",
                    *[p.name for p in pending],
                ]
            )
            recover(
                [
                    "--source",
                    str(native),
                    "--output",
                    str(final),
                    "--episodes",
                    *[p.name for p in pending],
                ]
            )
    from humanlabel.export import main as export

    arguments = [str(final / p.name) for p in episodes]
    if args.piper_config:
        arguments += ["--piper-config", str(args.piper_config)]
    export(arguments)
    from humanlabel.validation import validate

    reports_path = final / "validation_all.json"
    reports = json.loads(reports_path.read_text()) if reports_path.exists() else {}
    current_reports = {p.name: validate(final / p.name) for p in episodes}
    reports.update(current_reports)
    reports_path.write_text(json.dumps(reports, indent=2))
    summaries = [
        json.loads(path.read_text())
        for path in sorted(final.glob("episode_*/manifest.json"))
    ]
    (final / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2)
    )
    print(
        f"Camera wrist eposes ready: {final}; {sum(r['processed_frames'] for r in current_reports.values())} frames",
        flush=True,
    )
