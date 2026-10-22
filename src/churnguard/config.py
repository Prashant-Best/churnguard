"""Configuration loading. Every path in the config is resolved relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config file into a plain dictionary."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Turn a config path such as 'models/x.joblib' into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else root / p
