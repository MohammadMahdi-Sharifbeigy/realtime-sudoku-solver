"""OpenCV preprocessing helpers for Sudoku board detection."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a grayscale view of the input image."""
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a 2D grayscale or 3D BGR image.")


def build_board_mask(image: np.ndarray) -> np.ndarray:
    """Create a high-contrast binary mask suited for contour detection."""
    grayscale = to_grayscale(image)
    blurred = cv2.GaussianBlur(grayscale, (7, 7), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return cv2.bitwise_not(thresholded)
