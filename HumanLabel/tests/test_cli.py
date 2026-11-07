import subprocess
import sys

import pytest

from humanlabel.utils.paths import ROOT


@pytest.mark.parametrize("command", ["label", "export", "compare", "prepare-sources"])
def test_public_command_help_without_model_weights(command):
    result = subprocess.run(
        [sys.executable, "-m", "humanlabel", command, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
