from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_log(message: str, log_path: str | Path) -> None:
    path = resolve_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
