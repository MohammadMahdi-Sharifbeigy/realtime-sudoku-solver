"""Contour-based Sudoku board detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sudoku_solver.cv.preprocess import build_board_mask
from sudoku_solver.cv.warp import order_corners


@dataclass(slots=True)
class BoardDetection:
    corners: np.ndarray
    contour: np.ndarray
    area: float
    bounding_rect: tuple[int, int, int, int]


def detect_board(image: np.ndarray) -> BoardDetection | None:
    """Detect the largest quadrilateral contour that resembles a Sudoku board."""
    if image.size == 0:
        return None

    mask = build_board_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(image.shape[0] * image.shape[1])

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.1:
            continue

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4:
            continue

        corners = approximation.reshape(4, 2).astype(np.float32)
        ordered = order_corners(corners)
        x, y, width, height = cv2.boundingRect(approximation)
        return BoardDetection(
            corners=ordered,
            contour=contour,
            area=area,
            bounding_rect=(x, y, width, height),
        )

    return None
