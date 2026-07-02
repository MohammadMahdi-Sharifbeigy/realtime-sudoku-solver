"""Tests for runtime mode selection."""

from __future__ import annotations

from sudoku_solver.app.pipeline import build_runtime_mode


def test_build_runtime_mode_returns_image_for_image_argument() -> None:
    mode = build_runtime_mode(image_path="sample.jpg", source=None)
    assert mode == "image"


def test_build_runtime_mode_returns_webcam_for_explicit_source() -> None:
    mode = build_runtime_mode(image_path=None, source=1)
    assert mode == "webcam"


def test_build_runtime_mode_defaults_to_webcam_when_no_args_are_set() -> None:
    mode = build_runtime_mode(image_path=None, source=None)
    assert mode == "webcam"
