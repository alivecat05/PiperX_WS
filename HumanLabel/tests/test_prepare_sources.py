import subprocess

import pytest

from humanlabel.utils import prepare_sources


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_fresh_clone_checks_out_pinned_files_and_preserves_later_local_changes(
    tmp_path, monkeypatch
):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "main")
    (upstream / "module.py").write_text("VERSION = 1\n")
    git(upstream, "add", "module.py")
    git(
        upstream,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "first",
    )
    first = git(upstream, "rev-parse", "HEAD")
    target = tmp_path / "workspace"
    target.mkdir()
    monkeypatch.setattr(prepare_sources, "ROOT", target)
    record = {"path": ".cache/source", "url": str(upstream), "revision": first}
    prepare_sources.prepare("fixture", record)
    checkout = target / ".cache/source"
    assert (checkout / "module.py").read_text() == "VERSION = 1\n"
    prepare_sources.prepare("fixture", record)
    (upstream / "module.py").write_text("VERSION = 2\n")
    git(upstream, "add", "module.py")
    git(
        upstream,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "second",
    )
    (checkout / "module.py").write_text("LOCAL_CHANGE = True\n")
    with pytest.raises(ValueError, match="contains changes"):
        prepare_sources.prepare(
            "fixture", {**record, "revision": git(upstream, "rev-parse", "HEAD")}
        )
    assert (checkout / "module.py").read_text() == "LOCAL_CHANGE = True\n"
