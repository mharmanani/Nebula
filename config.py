"""Reads configs/nebula.yaml and your configs/paths.yaml (or the files named by NEBULA_CONFIG / NEBULA_PATHS)."""

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / os.environ.get("NEBULA_CONFIG", "configs/nebula.yaml")).read_text())
    paths = Path(os.environ.get("NEBULA_PATHS", ROOT / "configs" / "paths.yaml"))
    if not paths.exists():
        raise FileNotFoundError(
            f"{paths} not found: copy configs/paths.example.yaml to configs/paths.yaml and fill in your paths"
        )
    return {**cfg, **yaml.safe_load(paths.read_text())}
