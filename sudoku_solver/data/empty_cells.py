"""Helpers for explicitly modeling empty Sudoku cell samples."""

from __future__ import annotations

from pathlib import Path

from .base import SampleRecord


def build_empty_cell_records(paths: list[str | Path]) -> list[SampleRecord]:
    """Create canonical empty-cell records mapped to class 0."""
    return [
        SampleRecord(
            source="digit_images",
            raw_label=0,
            path=Path(path),
        )
        for path in paths
    ]
