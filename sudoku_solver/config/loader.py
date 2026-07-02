"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import AppConfig, RuntimeConfig, TrainingConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at config root, got {type(raw).__name__}")
    return raw


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_training_config(raw: dict[str, Any]) -> TrainingConfig:
    datasets = raw.get("datasets", [])
    if not isinstance(datasets, list) or not all(
        isinstance(name, str) for name in datasets
    ):
        raise ValueError("training.datasets must be a list of strings")

    batch_size = raw.get("batch_size", 64)
    num_workers = raw.get("num_workers", 0)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("training.batch_size must be a positive integer")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("training.num_workers must be a non-negative integer")

    return TrainingConfig(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def _build_runtime_config(raw: dict[str, Any]) -> RuntimeConfig:
    input_size = raw.get("input_size", 64)
    device = raw.get("device", "cpu")
    debug = raw.get("debug", False)

    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError("runtime.input_size must be a positive integer")
    if not isinstance(device, str) or not device:
        raise ValueError("runtime.device must be a non-empty string")
    if not isinstance(debug, bool):
        raise ValueError("runtime.debug must be a boolean")

    return RuntimeConfig(
        input_size=input_size,
        device=device,
        debug=debug,
    )


def load_config(path: str | None) -> AppConfig:
    base = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        override_path = Path(path)
        override = _read_yaml(override_path)
        base = _merge_dicts(base, override)

    training_raw = base.get("training", {})
    runtime_raw = base.get("runtime", {})
    if not isinstance(training_raw, dict):
        raise ValueError("training section must be a mapping")
    if not isinstance(runtime_raw, dict):
        raise ValueError("runtime section must be a mapping")

    return AppConfig(
        training=_build_training_config(training_raw),
        runtime=_build_runtime_config(runtime_raw),
    )
