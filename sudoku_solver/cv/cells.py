"""Cell extraction helpers for warped Sudoku board images."""

from __future__ import annotations

import numpy as np


def extract_cells(board_image: np.ndarray) -> list[np.ndarray]:
    """Split a square board image into 81 row-major cell crops."""
    if board_image.ndim < 2:
        raise ValueError("Board image must have at least two dimensions.")

    height, width = board_image.shape[:2]
    if height != width:
        raise ValueError("Board image must be square before cell extraction.")
    if height < 9:
        raise ValueError("Board image is too small to split into 81 cells.")

    y_edges = np.linspace(0, height, 10, dtype=int)
    x_edges = np.linspace(0, width, 10, dtype=int)

    cells: list[np.ndarray] = []
    for row_index in range(9):
        for column_index in range(9):
            top = y_edges[row_index]
            bottom = y_edges[row_index + 1]
            left = x_edges[column_index]
            right = x_edges[column_index + 1]
            cells.append(board_image[top:bottom, left:right].copy())
    return cells
