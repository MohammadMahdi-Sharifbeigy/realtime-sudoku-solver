"""Tests for canonical dataset label mapping."""

from __future__ import annotations

import pytest

from sudoku_solver.data.labels import map_label


@pytest.mark.parametrize(
    ("source", "raw_label", "expected"),
    [
        ("digit_images", "7", 7),
        ("digit_images", 0, 0),
        ("mnist", 9, 9),
        ("mnist", "3", 3),
        ("hoda", 4, 4),
        ("hoda", "۸", 8),
    ],
)
def test_map_label_accepts_supported_labels(
    source: str, raw_label: str | int, expected: int
) -> None:
    assert map_label(source, raw_label) == expected


@pytest.mark.parametrize(
    ("source", "raw_label"),
    [
        ("digit_images", "eleven"),
        ("digit_images", 11),
        ("digit_images", -1),
        ("mnist", "ten"),
        ("hoda", "A"),
    ],
)
def test_map_label_rejects_invalid_labels(source: str, raw_label: str | int) -> None:
    with pytest.raises(ValueError):
        map_label(source, raw_label)

