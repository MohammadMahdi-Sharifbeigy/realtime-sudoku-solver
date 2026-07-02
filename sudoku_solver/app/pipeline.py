"""End-to-end runtime pipeline for Sudoku board inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from sudoku_solver.config.types import AppConfig
from sudoku_solver.cv.board_detect import detect_board
from sudoku_solver.cv.cells import extract_cells
from sudoku_solver.cv.overlay import overlay_solution
from sudoku_solver.cv.warp import warp_board
from sudoku_solver.recognition.predict import (
    CellPredictor,
    load_checkpoint_predictor,
    predict_board,
)
from sudoku_solver.solver.backtracking import solve_board
from sudoku_solver.solver.board import Board


class FramePredictorFactory(Protocol):
    """Build a predictor instance from runtime configuration."""

    def __call__(
        self,
        config: AppConfig,
        checkpoint_path: str | None = None,
    ) -> CellPredictor:
        """Return a cell predictor compatible with `predict_board`."""


@dataclass(slots=True)
class PipelineResult:
    """Serializable image-pipeline result."""

    success: bool
    message: str
    original_image: np.ndarray
    warped_board: np.ndarray | None = None
    overlay_image: np.ndarray | None = None
    board: Board | None = None
    solved_board: Board | None = None


@dataclass(slots=True)
class DebugPaths:
    """Structured debug artifact paths rooted in a caller-supplied directory."""

    root: Path
    warped_boards: Path
    cells: Path
    overlays: Path
    failures: Path
    reports: Path

    def image_path(self, category: str, stem: str, suffix: str = ".png") -> Path:
        """Return a path within a known debug image category."""
        categories = {
            "warped_boards": self.warped_boards,
            "cells": self.cells,
            "overlays": self.overlays,
            "failures": self.failures,
        }
        if category not in categories:
            raise ValueError(f"Unsupported debug artifact category: {category}")
        return categories[category] / f"{stem}{suffix}"

    def report_path(self, stem: str, suffix: str = ".json") -> Path:
        """Return a path for non-image debug reports."""
        return self.reports / f"{stem}{suffix}"


def build_runtime_mode(image_path: str | None, source: int | None) -> str:
    """Return the requested runtime mode."""
    if image_path:
        return "image"
    if source is not None:
        return "webcam"
    return "webcam"


def build_debug_paths(root: str | Path) -> DebugPaths:
    """Create and return concrete debug artifact directories under `root/debug`."""
    base_root = Path(root)
    debug_root = base_root / "debug"
    paths = DebugPaths(
        root=debug_root,
        warped_boards=debug_root / "warped_boards",
        cells=debug_root / "cells",
        overlays=debug_root / "overlays",
        failures=debug_root / "failures",
        reports=debug_root / "reports",
    )
    for path in (
        paths.root,
        paths.warped_boards,
        paths.cells,
        paths.overlays,
        paths.failures,
        paths.reports,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_image_pipeline(
    config: AppConfig,
    image_path: str,
    *,
    checkpoint_path: str | None = None,
    predictor: CellPredictor | None = None,
    predictor_factory: FramePredictorFactory | None = None,
    output_path: str | None = None,
) -> PipelineResult:
    """Run one end-to-end inference pass on an image path."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = process_frame(
        image=image,
        config=config,
        checkpoint_path=checkpoint_path,
        predictor=predictor,
        predictor_factory=predictor_factory,
    )
    if output_path and result.overlay_image is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result.overlay_image)
    return result


def process_frame(
    image: np.ndarray,
    config: AppConfig,
    *,
    checkpoint_path: str | None = None,
    predictor: CellPredictor | None = None,
    predictor_factory: FramePredictorFactory | None = None,
) -> PipelineResult:
    """Run the current CV, prediction, solver, and overlay path on one frame."""
    if image.size == 0:
        return PipelineResult(success=False, message="Empty image provided.", original_image=image)

    detection = detect_board(image)
    if detection is None:
        return PipelineResult(
            success=False,
            message="No Sudoku board detected.",
            original_image=image,
        )

    warped_board = warp_board(image, detection.corners)
    cells = extract_cells(warped_board)

    resolved_predictor = predictor or _resolve_predictor(
        config=config,
        checkpoint_path=checkpoint_path,
        predictor_factory=predictor_factory,
    )
    board = predict_board(
        cells,
        resolved_predictor,
        input_size=config.runtime.input_size,
    )
    solved_board = solve_board(board)
    if solved_board is None:
        return PipelineResult(
            success=False,
            message="Recognized board is invalid or unsolvable.",
            original_image=image,
            warped_board=warped_board,
            board=board,
            solved_board=None,
        )

    overlay_image = overlay_solution(image, detection.corners, board, solved_board)
    return PipelineResult(
        success=True,
        message="Sudoku solved successfully.",
        original_image=image,
        warped_board=warped_board,
        overlay_image=overlay_image,
        board=board,
        solved_board=solved_board,
    )


def _resolve_predictor(
    *,
    config: AppConfig,
    checkpoint_path: str | None,
    predictor_factory: FramePredictorFactory | None,
) -> CellPredictor:
    if predictor_factory is not None:
        return predictor_factory(config, checkpoint_path)
    return load_checkpoint_predictor(config, checkpoint_path)


def build_placeholder_predictor() -> CellPredictor:
    """Return a deterministic predictor stub until model loading is wired in."""

    def _predict(batch: list[np.ndarray]) -> list[int]:
        return [0 for _ in batch]

    return _predict
