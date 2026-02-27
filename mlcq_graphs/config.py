from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts. Values in `override` take precedence."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_include_path(current_file: Path, include_value: str) -> Path:
    include_path = Path(include_value)
    if not include_path.is_absolute():
        include_path = (current_file.parent / include_path).resolve()
    if include_path.exists():
        return include_path
    if include_path.suffix == "":
        yml_path = include_path.with_suffix(".yml")
        if yml_path.exists():
            return yml_path
        yaml_path = include_path.with_suffix(".yaml")
        if yaml_path.exists():
            return yaml_path
    raise ConfigError(f"Included config file not found: {include_value}")


def load_yaml_recursive(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if seen is None:
        seen = set()
    if resolved in seen:
        raise ConfigError(f"Recursive include detected at: {resolved}")
    seen.add(resolved)

    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    raw_obj = yaml.safe_load(resolved.read_text())
    data = raw_obj if isinstance(raw_obj, dict) else {}

    include_obj = data.pop("_include_yml", None)
    if include_obj is None:
        return data

    include_values: list[str]
    if isinstance(include_obj, list):
        include_values = [str(value) for value in include_obj]
    else:
        include_values = [str(include_obj)]

    merged: dict[str, Any] = {}
    for include_value in include_values:
        include_path = _resolve_include_path(resolved, include_value)
        included_cfg = load_yaml_recursive(include_path, seen=seen)
        merged = deep_merge_dicts(merged, included_cfg)

    return deep_merge_dicts(merged, data)


def parse_override_value(raw: str) -> Any:
    parsed = yaml.safe_load(raw)
    return parsed


def parse_cli_overrides(tokens: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ConfigError(f"Unexpected argument: {token}")

        stripped = token[2:]
        if "=" in stripped:
            key, raw_value = stripped.split("=", 1)
            index += 1
        else:
            key = stripped
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                raw_value = tokens[index + 1]
                index += 2
            else:
                raw_value = "true"
                index += 1

        if not key:
            raise ConfigError(f"Invalid override token: {token}")
        if "." not in key:
            raise ConfigError(
                f"Override '{key}' must use dotted notation, e.g. --training.lr=5e-4"
            )

        overrides[key] = parse_override_value(raw_value)

    return overrides


def set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor:
            cursor[part] = {}
        node = cursor[part]
        if not isinstance(node, dict):
            raise ConfigError(
                f"Cannot assign '{dotted_key}': '{part}' is not a mapping in config"
            )
        cursor = node
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    for dotted_key, value in overrides.items():
        set_nested_value(out, dotted_key, value)
    return out


def load_config(config_path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    config = load_yaml_recursive(config_path)
    return apply_overrides(config, overrides)


def dump_resolved_config(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))
