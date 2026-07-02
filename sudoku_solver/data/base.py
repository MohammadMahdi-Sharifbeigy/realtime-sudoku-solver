"""Shared dataset record and validation report types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True, frozen=True)
class SampleRecord:
    """Canonical sample description shared by dataset adapters."""

    source: str
    raw_label: str | int
    path: Path | str | None = None
    image: object | None = None


@dataclass(slots=True)
class ValidationIssue:
    """Captured invalid label information for reporting and exceptions."""

    raw_label: str | int
    reason: str
    path: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """Summary of label validation for a single dataset source."""

    source: str
    total_records: int
    discovered_labels: list[str] = field(default_factory=list)
    canonical_counts: dict[int, int] = field(default_factory=dict)
    invalid_labels: list[ValidationIssue] = field(default_factory=list)
    missing_labels: list[int] = field(default_factory=list)

