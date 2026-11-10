"""Prepare pinned upstream source trees without redistributing model assets."""

import argparse
import json
import subprocess

from humanlabel.utils.paths import ROOT

DEPENDENCIES = {
    "egohand": ["NVIDIA-IsaacCapture", "DINOv3"],
}
PATCHES = {
    "NVIDIA-IsaacCapture": "IsaacCapture.patch",
}


def run(*args, **kwargs):
    return subprocess.run(list(args), check=True, **kwargs)


def prepare(name, record):
    folder = ROOT / record["path"]
    cloned = not (folder / ".git").exists()
    if cloned:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Preserve existing non-repository directory: {folder}")
        folder.parent.mkdir(parents=True, exist_ok=True)
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            record["url"],
            str(folder),
        )
    actual = subprocess.check_output(
        ["git", "-C", str(folder), "rev-parse", "HEAD"], text=True
    ).strip()
    if cloned or actual != record["revision"]:
        dirty = subprocess.check_output(
            ["git", "-C", str(folder), "status", "--porcelain"], text=True
        )
        if not cloned and dirty.strip():
            raise ValueError(
                f"Upstream checkout contains changes; preserve them before changing revision: {folder}"
            )
        run("git", "-C", str(folder), "fetch", "origin", record["revision"])
        run("git", "-C", str(folder), "checkout", "--detach", record["revision"])
    if name in PATCHES:
        patch = ROOT / "patches" / PATCHES[name]
        applied = subprocess.run(
            ["git", "-C", str(folder), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
        )
        if applied.returncode != 0:
            run("git", "-C", str(folder), "apply", "--check", str(patch))
            run("git", "-C", str(folder), "apply", str(patch))
    print(f"{name}: {record['revision']} ready at {folder}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", choices=DEPENDENCIES, nargs="+", default=["egohand"]
    )
    args = parser.parse_args(argv)
    sources = json.loads((ROOT / "configs/model_sources.json").read_text())[
        "git_revisions"
    ]
    for name in dict.fromkeys(n for model in args.models for n in DEPENDENCIES[model]):
        prepare(name, sources[name])


if __name__ == "__main__":
    main()
