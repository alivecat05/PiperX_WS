"""Workspace-relative locations shared by command-line workflows."""

import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
MODEL = ROOT / "model"


@contextmanager
def temporary_workspace():
    """Remove this run's intermediate data on success or failure."""
    cache = ROOT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="label-", dir=cache) as folder:
            yield Path(folder)
    finally:
        try:
            cache.rmdir()
        except OSError:
            pass  # Another active run may still own a temporary directory.
