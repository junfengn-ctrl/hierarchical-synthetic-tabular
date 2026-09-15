from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path_fields(config: dict[str, Any], fields: list[str]) \
        -> dict[str, Any]:
    resolved = dict(config)
    for field in fields:
        if field in resolved and resolved[field] is not None:
            resolved[field] = resolve_project_path(str(resolved[field]))
    return resolved
