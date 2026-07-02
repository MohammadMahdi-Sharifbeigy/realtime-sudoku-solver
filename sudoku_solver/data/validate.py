"""Dataset label validation helpers."""

from __future__ import annotations

from .base import SampleRecord, ValidationIssue, ValidationReport
from .labels import CANONICAL_LABELS, map_label


def validate_records(source: str, records: list[SampleRecord]) -> ValidationReport:
    """Validate source labels and return a summary report.

    Raises:
        ValueError: If any record label cannot be mapped into the canonical 0-9
            label space or if a record source does not match the validation source.
    """
    canonical_counts = {label: 0 for label in sorted(CANONICAL_LABELS)}
    discovered_labels: set[str] = set()
    invalid_labels: list[ValidationIssue] = []

    for record in records:
        if record.source != source:
            invalid_labels.append(
                ValidationIssue(
                    raw_label=record.raw_label,
                    reason=f"Record source mismatch: expected {source}, got {record.source}",
                    path=_stringify_path(record.path),
                )
            )
            continue

        discovered_labels.add(str(record.raw_label))
        try:
            canonical_label = map_label(source, record.raw_label)
        except ValueError as exc:
            invalid_labels.append(
                ValidationIssue(
                    raw_label=record.raw_label,
                    reason=str(exc),
                    path=_stringify_path(record.path),
                )
            )
            continue

        canonical_counts[canonical_label] += 1

    report = ValidationReport(
        source=source,
        total_records=len(records),
        discovered_labels=sorted(discovered_labels),
        canonical_counts=canonical_counts,
        invalid_labels=invalid_labels,
        missing_labels=[
            label for label, count in canonical_counts.items() if count == 0
        ],
    )
    if invalid_labels:
        raise ValueError(_format_validation_error(report))
    return report


def _stringify_path(path: str | object | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _format_validation_error(report: ValidationReport) -> str:
    invalid_summary = ", ".join(
        _format_issue(issue) for issue in report.invalid_labels
    )
    return f"Label validation failed for source '{report.source}': {invalid_summary}"


def _format_issue(issue: ValidationIssue) -> str:
    location = f" at {issue.path}" if issue.path else ""
    return f"{issue.reason}{location}"

