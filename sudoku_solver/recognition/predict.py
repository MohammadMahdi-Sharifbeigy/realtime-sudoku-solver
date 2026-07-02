"""Prediction helpers for converting cell crops into Sudoku boards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np

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
