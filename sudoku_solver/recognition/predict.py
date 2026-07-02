"""Prediction helpers for converting cell crops into Sudoku boards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
import torch

from sudoku_solver.config.types import AppConfig
from sudoku_solver.recognition.model import build_model
from sudoku_solver.recognition.transforms import build_eval_transforms


Board = list[list[int]]


class CellPredictor(Protocol):
    """Callable protocol for batched cell digit prediction."""

    def __call__(self, batch: Sequence[np.ndarray]) -> Sequence[int]:
        """Return one canonical class id per cell crop."""


@dataclass(frozen=True)
class EmptyCellHeuristic:
    """Thresholds for routing obviously blank cells to class 0."""

    min_foreground_ratio: float = 0.02
    min_foreground_pixels: int = 24
    threshold_value: int = 200


@dataclass(slots=True)
class TorchCheckpointPredictor:
    """PyTorch-backed batched digit predictor loaded from a checkpoint."""

    model: torch.nn.Module
    device: torch.device
    input_size: int

    def __call__(self, batch: Sequence[np.ndarray]) -> Sequence[int]:
        if not batch:
            return []

        tensors = [torch.as_tensor(item, dtype=torch.float32) for item in batch]
        stacked = torch.stack(tensors, dim=0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(stacked)
            predictions = torch.argmax(logits, dim=1)
        return predictions.detach().cpu().tolist()


def predict_board(
    cells: Sequence[np.ndarray],
    predictor: CellPredictor,
    input_size: int = 64,
    heuristic: EmptyCellHeuristic | None = None,
) -> Board:
    """Convert 81 cell crops into a canonical 9x9 Sudoku board."""
    if len(cells) != 81:
        raise ValueError("predict_board expects exactly 81 cell crops.")

    empty_heuristic = heuristic or EmptyCellHeuristic()
    prepared_cells = [_prepare_cell(cell) for cell in cells]
    board_values = [0] * 81

    pending_indices: list[int] = []
    pending_batch: list[np.ndarray] = []
    eval_transform = build_eval_transforms(input_size)

    for index, cell in enumerate(prepared_cells):
        if _is_obviously_empty(cell, empty_heuristic):
            board_values[index] = 0
            continue

        transformed = eval_transform(cell)
        pending_indices.append(index)
        pending_batch.append(transformed.numpy())

    if pending_batch:
        predictions = predictor(pending_batch)
        predicted_values = [int(value) for value in predictions]
        if len(predicted_values) != len(pending_indices):
            raise ValueError("Predictor returned a mismatched number of outputs.")
        for index, value in zip(pending_indices, predicted_values, strict=True):
            if value < 0 or value > 9:
                raise ValueError(f"Predictor returned invalid class id {value}.")
            board_values[index] = value

    return [board_values[row_index * 9 : (row_index + 1) * 9] for row_index in range(9)]


def load_checkpoint_predictor(
    config: AppConfig,
    checkpoint_path: str | None = None,
) -> CellPredictor:
    """Load a checkpoint-backed predictor for runtime board recognition."""
    resolved_path = resolve_checkpoint_path(config, checkpoint_path)
    if resolved_path is None:
        raise FileNotFoundError(
            "No recognition checkpoint was found. Train the model first or pass --checkpoint."
        )

    device = _resolve_device(config.runtime.device)
    checkpoint = torch.load(resolved_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {resolved_path}")

    input_size = (
        int(checkpoint.get("input_size", config.runtime.input_size))
        if isinstance(checkpoint, dict)
        else config.runtime.input_size
    )
    model = build_model(num_classes=10).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return TorchCheckpointPredictor(model=model, device=device, input_size=input_size)


def resolve_checkpoint_path(
    config: AppConfig,
    checkpoint_path: str | None = None,
) -> Path | None:
    """Resolve an explicit or default checkpoint path for runtime inference."""
    if checkpoint_path:
        path = Path(checkpoint_path)
        return path if path.exists() else None

    checkpoint_dir = getattr(config.training, "checkpoint_dir", Path("models/checkpoints"))
    for filename in ("best.pt", "last.pt"):
        candidate = checkpoint_dir / filename
        if candidate.exists():
            return candidate
    return None


def _prepare_cell(cell: np.ndarray) -> np.ndarray:
    array = np.asarray(cell)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    raise ValueError("Each cell crop must be a 2D grayscale or 3D color image.")


def _is_obviously_empty(cell: np.ndarray, heuristic: EmptyCellHeuristic) -> bool:
    if cell.size == 0:
        return True

    trimmed = _trim_border(cell)
    if trimmed.size == 0:
        return True

    foreground_mask = trimmed < heuristic.threshold_value
    foreground_pixels = int(np.count_nonzero(foreground_mask))
    foreground_ratio = foreground_pixels / float(trimmed.size)
    return (
        foreground_pixels < heuristic.min_foreground_pixels
        or foreground_ratio < heuristic.min_foreground_ratio
    )


def _trim_border(cell: np.ndarray) -> np.ndarray:
    height, width = cell.shape[:2]
    border_y = max(1, height // 10)
    border_x = max(1, width // 10)
    top = min(border_y, height)
    bottom = max(top, height - border_y)
    left = min(border_x, width)
    right = max(left, width - border_x)
    return cell[top:bottom, left:right]


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
