"""Workspace-relative locations shared by command-line workflows."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
