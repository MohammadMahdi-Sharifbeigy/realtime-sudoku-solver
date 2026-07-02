"""Tests for dataset label validation reports."""

from __future__ import annotations

import pytest

from sudoku_solver.data.base import SampleRecord
from sudoku_solver.data.validate import validate_records


def test_validate_records_counts_canonical_labels() -> None:
    records = [
        SampleRecord(source="digit_images", raw_label="0", path="0/a.png"),
        SampleRecord(source="digit_images", raw_label="7", path="7/b.png"),
        SampleRecord(source="digit_images", raw_label=7, path="7/c.png"),
    ]

    report = validate_records("digit_images", records)

    assert report.source == "digit_images"
    assert report.total_records == 3
    assert report.discovered_labels == ["0", "7"]
    assert report.canonical_counts[0] == 1
    assert report.canonical_counts[7] == 2
    assert report.invalid_labels == []
    assert 1 in report.missing_labels
    assert 7 not in report.missing_labels


def test_validate_records_accepts_hoda_persian_digits() -> None:
    records = [
        SampleRecord(source="hoda", raw_label="۱", path="one.png"),
        SampleRecord(source="hoda", raw_label="۹", path="nine.png"),
    ]

    report = validate_records("hoda", records)

    assert report.canonical_counts[1] == 1
    assert report.canonical_counts[9] == 1


def test_validate_records_rejects_invalid_digit_images_label() -> None:
    records = [SampleRecord(source="digit_images", raw_label="eleven", path="bad.png")]

    with pytest.raises(ValueError, match="Label validation failed"):
        validate_records("digit_images", records)


def test_validate_records_rejects_source_mismatch() -> None:
    records = [SampleRecord(source="mnist", raw_label=3, path="sample.png")]

    with pytest.raises(ValueError, match="Record source mismatch"):
        validate_records("digit_images", records)
