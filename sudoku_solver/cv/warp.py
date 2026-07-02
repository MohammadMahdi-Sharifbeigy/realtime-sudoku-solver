"""Geometry helpers for ordering and warping Sudoku boards."""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners ordered as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Expected corners with shape (4, 2).")

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def warp_board(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the detected board into a square top-down image."""
    ordered = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = ordered

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    side_length = int(round(max(width_top, width_bottom, height_left, height_right)))
    if side_length <= 0:
        raise ValueError("Board side length must be positive.")

    destination = np.array(
        [
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (side_length, side_length))
