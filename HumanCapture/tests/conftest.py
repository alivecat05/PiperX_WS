import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def cfg(tmp_path):
    cfg = json.loads((ROOT/"config.example.json").read_text(encoding="utf-8"))
    cfg["output"] = str(tmp_path/"episodes")
    for camera in cfg["cameras"].values():
        camera.update(width=64, height=48)
    return cfg
