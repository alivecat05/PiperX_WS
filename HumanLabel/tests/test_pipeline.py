import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from humanlabel.utils import paths
from humanlabel.workflows import pipeline


@pytest.mark.parametrize("batch", [False, True])
def test_label_groups_final_results_and_cleans_intermediate_data(
    tmp_path, monkeypatch, batch
):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    source = tmp_path / "recordings"
    episodes = [source / f"episode_{i:03d}" for i in range(2 if batch else 1)]
    for episode in episodes:
        episode.mkdir(parents=True)
        (episode / "head.mp4").touch()
        (episode / "manifest.json").write_text(json.dumps({"frames": 3}))
    source_arg = source if batch else episodes[0]
    output = tmp_path / "result"
    final = output / source.name
    calls = []
    selected = [p.name for p in episodes]

    def complete_stage(stage_output, names):
        for episode in episodes:
            if episode.name not in names:
                continue
            folder = stage_output / episode.name
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "source": str(episode),
                        "processed_frames": 3,
                        "representation": "SOMA",
                    }
                )
            )

    def infer(argv):
        calls.append("egohand")
        assert Path(argv[0]) == source_arg
        assert argv[argv.index("--episodes") + 1 :] == selected
        native = Path(argv[argv.index("--output") + 1])
        assert native.is_relative_to(tmp_path / ".cache")
        complete_stage(native, selected)
        frames = native.parent / "frames"
        frames.mkdir()
        (frames / "temporary.jpg").touch()

    def recover(argv):
        calls.append("recover")
        native = Path(argv[argv.index("--source") + 1])
        assert (native.parent / "frames/temporary.jpg").is_file()
        assert argv[argv.index("--episodes") + 1 :] == selected
        assert Path(argv[argv.index("--output") + 1]) == final
        complete_stage(final, selected)

    def export(argv):
        calls.append("export")
        assert argv == [str(final / p.name) for p in episodes]

    def validate(folder):
        calls.append("validate")
        assert folder.parent == final
        return {"status": "passed", "processed_frames": 3}

    for name, function in (
        ("humanlabel.workflows.egohand", infer),
        ("humanlabel.workflows.recover", recover),
        ("humanlabel.export", export),
        ("humanlabel.validation", validate),
    ):
        module = ModuleType(name)
        setattr(module, "validate" if name.endswith("validation") else "main", function)
        monkeypatch.setitem(sys.modules, name, module)

    arguments = [str(source_arg), "--output", str(output)]
    pipeline.main(arguments)
    assert calls == ["egohand", "recover", "export", *["validate"] * len(episodes)]
    assert {path.name for path in output.iterdir()} == {source.name}
    assert {path.name for path in final.iterdir()} == {
        *[p.name for p in episodes],
        "summary.json",
        "validation_all.json",
    }
    assert not (tmp_path / ".cache").exists()
    reports = json.loads((final / "validation_all.json").read_text())
    assert set(reports) == {p.name for p in episodes}

    calls.clear()
    pipeline.main(arguments)
    assert calls == ["export", *["validate"] * len(episodes)]
    assert not (tmp_path / ".cache").exists()

    if batch:
        calls.clear()
        selected[:] = [episodes[1].name]
        shutil.rmtree(final / episodes[1].name)
        marker = final / episodes[0].name / "keep.txt"
        marker.write_text("already completed")
        pipeline.main(arguments)
        assert calls == ["egohand", "recover", "export", "validate", "validate"]
        assert marker.read_text() == "already completed"
        assert not (tmp_path / ".cache").exists()


def test_failed_inference_also_removes_its_temporary_data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    source = tmp_path / "recordings/episode_001"
    source.mkdir(parents=True)
    (source / "head.mp4").touch()
    (source / "manifest.json").write_text(json.dumps({"frames": 3}))

    def infer(argv):
        native = Path(argv[argv.index("--output") + 1])
        native.mkdir()
        (native / "partial.npz").touch()
        raise RuntimeError("inference failed")

    for name in ("egohand", "recover"):
        module = ModuleType(name)
        module.main = infer
        monkeypatch.setitem(sys.modules, f"humanlabel.workflows.{name}", module)
    with pytest.raises(RuntimeError, match="inference failed"):
        pipeline.main([str(source), "--output", str(tmp_path / "result")])
    assert not (tmp_path / ".cache").exists()
