from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import copy

import yaml  # akan dipasang nanti di requirements


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dict override ke base secara rekursif."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path, base_path: str | Path = "configs/base.yaml") -> Dict[str, Any]:
    base = load_yaml(base_path)
    override = load_yaml(config_path)
    cfg = deep_merge(base, override)
    cfg["_config_path"] = str(config_path)
    cfg["_base_path"] = str(base_path)
    return cfg


@dataclass(frozen=True)
class ConfigPaths:
    data_raw_dir: Path
    runs_dir: Path
    persist_dir: Path

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "ConfigPaths":
        return ConfigPaths(
            data_raw_dir=Path(cfg["paths"]["data_raw_dir"]),
            runs_dir=Path(cfg["paths"]["runs_dir"]),
            persist_dir=Path(cfg["index"]["persist_dir"]),
        )