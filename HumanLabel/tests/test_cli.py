import subprocess
import sys

import pytest

from humanlabel.utils.paths import ROOT


@pytest.mark.parametrize("command", ["label", "export", "prepare-sources"])
def test_public_command_help_without_model_weights(command):
    result = subprocess.run(
        [sys.executable, "-m", "humanlabel", command, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "command",
    ["compare", "compare-videos", "temporal", "refine", "render", "egohand", "recover"],
)
def test_removed_workflows_are_not_public_commands(command):
    result = subprocess.run(
        [sys.executable, "-m", "humanlabel", command, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
