"""Overlay helpers for projecting solved digits onto source images."""

from __future__ import annotations

import cv2
import numpy as np

from sudoku_solver.cv.warp import order_corners


Board = list[list[int]]


def should_render_digit(original_value: int, solved_value: int) -> bool:
    """Return True only for digits filled by the solver."""
    return original_value == 0 and solved_value != 0


def overlay_solution(
    image: np.ndarray,
    corners: np.ndarray,
    original_board: Board,
    solved_board: Board,
    *,
    color: tuple[int, int, int] = (0, 180, 0),
    alpha: float = 0.85,
) -> np.ndarray:
    """Project solved digits for empty cells back onto the original image."""
    _validate_board(original_board, "original_board")
    _validate_board(solved_board, "solved_board")

    if image.ndim == 2:
        base_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        base_image = image.copy()
    else:
        raise ValueError("image must be a 2D grayscale or 3D color image.")

    ordered_corners = order_corners(np.asarray(corners, dtype=np.float32))
    side_length = _compute_overlay_side_length(ordered_corners)
    if side_length <= 0:
        raise ValueError("Overlay side length must be positive.")

    overlay_board = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    cell_size = side_length / 9.0

    for row_index in range(9):
        for column_index in range(9):
            original_value = original_board[row_index][column_index]
            solved_value = solved_board[row_index][column_index]
            if not should_render_digit(original_value, solved_value):
                continue
            _draw_digit(
                overlay_board,
                digit=solved_value,
                row_index=row_index,
                column_index=column_index,
                cell_size=cell_size,
                color=color,
            )

    destination = np.array(
        [
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1],
        ],
        dtype=np.float32,
    )
    inverse_transform = cv2.getPerspectiveTransform(destination, ordered_corners)
    warped_overlay = cv2.warpPerspective(
        overlay_board,
        inverse_transform,
        (base_image.shape[1], base_image.shape[0]),
    )

    overlay_mask = np.any(warped_overlay > 0, axis=2)
    blended = base_image.copy()
    if np.any(overlay_mask):
        mask_3d = overlay_mask[:, :, None]
        blended_region = cv2.addWeighted(base_image, 1.0 - alpha, warped_overlay, alpha, 0.0)
        blended = np.where(mask_3d, blended_region, base_image)
    return blended


def _validate_board(board: Board, name: str) -> None:
    if len(board) != 9 or any(len(row) != 9 for row in board):
        raise ValueError(f"{name} must be a 9x9 board.")


def _compute_overlay_side_length(corners: np.ndarray) -> int:
    top_left, top_right, bottom_right, bottom_left = corners
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    return int(round(max(width_top, width_bottom, height_left, height_right)))


def _draw_digit(
    image: np.ndarray,
    *,
    digit: int,
    row_index: int,
    column_index: int,
    cell_size: float,
    color: tuple[int, int, int],
) -> None:
    text = str(digit)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(cell_size / 42.0, 0.45)
    thickness = max(1, int(round(cell_size / 18.0)))
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)

    center_x = int(round((column_index + 0.5) * cell_size))
    center_y = int(round((row_index + 0.5) * cell_size))
    origin_x = center_x - text_size[0] // 2
    origin_y = center_y + (text_size[1] - baseline) // 2

    cv2.putText(
        image,
        text,
        (origin_x, origin_y),
        font,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )
