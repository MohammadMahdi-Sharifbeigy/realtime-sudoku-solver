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
    epochs = raw.get("epochs", 1)
    learning_rate = raw.get("learning_rate", 1e-3)
    weight_decay = raw.get("weight_decay", 0.0)
    val_split = raw.get("val_split", 0.15)
    random_seed = raw.get("random_seed", 2023)
    data_root = Path(raw.get("data_root", "data"))
    checkpoint_dir = Path(raw.get("checkpoint_dir", "models/checkpoints"))
    save_best_only = raw.get("save_best_only", False)
    raw_paths = raw.get("dataset_paths", {})
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("training.batch_size must be a positive integer")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("training.num_workers must be a non-negative integer")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("training.epochs must be a positive integer")
    if not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0:
        raise ValueError("training.learning_rate must be a positive float")
    if not isinstance(weight_decay, (int, float)) or float(weight_decay) < 0:
        raise ValueError("training.weight_decay must be a non-negative float")
    if not isinstance(val_split, (int, float)) or not 0 < float(val_split) < 1:
        raise ValueError("training.val_split must be a float between 0 and 1")
    if not isinstance(random_seed, int):
        raise ValueError("training.random_seed must be an integer")
    if not isinstance(save_best_only, bool):
        raise ValueError("training.save_best_only must be a boolean")
    if not isinstance(raw_paths, dict):
        raise ValueError("training.dataset_paths must be a mapping")

    dataset_paths: dict[str, Path] = {}
    for key, value in raw_paths.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("training.dataset_paths keys and values must be strings")
        dataset_paths[key.strip().lower()] = Path(value)

    return TrainingConfig(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        val_split=float(val_split),
        random_seed=random_seed,
        data_root=data_root,
        dataset_paths=dataset_paths,
        checkpoint_dir=checkpoint_dir,
        save_best_only=save_best_only,
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
