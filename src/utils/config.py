from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config


def save_yaml(data: Mapping[str, Any], output_path: str | Path) -> None:
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=False)


def ensure_parent(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
