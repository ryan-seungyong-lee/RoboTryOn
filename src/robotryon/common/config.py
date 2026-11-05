"""YAML configs with dotted command-line overrides.

Every script takes ``--config path.yaml`` followed by any number of ``key.sub=value``
overrides, e.g.::

    python scripts/train_image.py --config configs/image/klein9b_lora.yaml \
        train.max_steps=20 data.root=/path/to/data

A config may name a parent with ``_base_: other.yaml`` (relative to itself); the child is merged
on top of it. Values are parsed with YAML rules, so ``1e-5``, ``true``, ``[1, 2]`` and ``null``
work as expected on the command line.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Sequence

import yaml


class Config(dict):
    """A dict whose keys are also attributes (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def to_dict(self) -> dict:
        return _plain(self)


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    v = yaml.safe_load(raw)
    if isinstance(v, str):          # YAML 1.1 reads "1e-5" as a string; accept it as a float
        try:
            return float(v)
        except ValueError:
            pass
    return v


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = merge(load_config(path.parent / base).to_dict(), cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key, _parse_value(raw))
    return _wrap(cfg)


def parse_args(description: str = "", argv: Sequence[str] | None = None) -> Config:
    """``--config file.yaml [key=value ...]`` -> config."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-5")
    a = p.parse_args(argv)
    return load_config(a.config, a.overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(_plain(cfg), sort_keys=False))
