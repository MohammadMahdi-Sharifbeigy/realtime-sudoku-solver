"""Typed configuration models for the Sudoku solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    datasets: list[str] = field(default_factory=list)
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    val_split: float = 0.15
    random_seed: int = 2023
    data_root: Path = Path("data")
    dataset_paths: dict[str, Path] = field(default_factory=dict)
    checkpoint_dir: Path = Path("models/checkpoints")
    save_best_only: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    input_size: int = 64
    device: str = "cpu"
    debug: bool = False


@dataclass(slots=True)
class AppConfig:
    training: TrainingConfig
    runtime: RuntimeConfig
