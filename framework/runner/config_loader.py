from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _as_config_mapping(payload: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Config at {path} must be a mapping")
    return dict(payload)


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _extends_paths(value: Any, *, path: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    raise TypeError(f"Config extends must be a string or list of strings at {path}")


def _load_config(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if resolved_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"Detected cyclic config extends: {cycle}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config fragment not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = _as_config_mapping(yaml.safe_load(handle), path=resolved_path)

    merged: dict[str, Any] = {}
    next_stack = (*stack, resolved_path)
    for raw_parent in _extends_paths(payload.pop("extends", None), path=resolved_path):
        parent_path = Path(raw_parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = resolved_path.parent / parent_path
        merged = _merge_config(merged, _load_config(parent_path, stack=next_stack))
    return _merge_config(merged, payload)


def load_composed_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and recursively resolve its optional ``extends`` list."""
    return _load_config(Path(path), stack=())


__all__ = ["load_composed_yaml"]