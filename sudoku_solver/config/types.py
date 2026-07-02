"""Typed configuration models for the Sudoku solver."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingConfig:
    datasets: list[str] = field(default_factory=list)
    batch_size: int = 64
    num_workers: int = 0


@dataclass(slots=True)
class RuntimeConfig:
    input_size: int = 64
    device: str = "cpu"
    debug: bool = False


@dataclass(slots=True)
class AppConfig:
    training: TrainingConfig
    runtime: RuntimeConfig
