"""Canonical label mapping for supported digit datasets."""

from __future__ import annotations

import re


SUPPORTED_SOURCES = {"digit_images", "mnist", "hoda"}
CANONICAL_LABELS = set(range(10))
HODA_PERSIAN_DIGITS = {
    "۰": 0,
    "۱": 1,
    "۲": 2,
    "۳": 3,
    "۴": 4,
    "۵": 5,
    "۶": 6,
    "۷": 7,
    "۸": 8,
    "۹": 9,
}


def map_label(source: str, raw_label: str | int) -> int:
    """Map a source-specific label to the canonical 0-9 label space."""
    normalized_source = source.strip().lower()
    if normalized_source not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported dataset source: {source}")

    if normalized_source == "hoda":
        return _map_hoda_label(raw_label)
    return _map_ascii_digit_label(raw_label, normalized_source)


def _map_ascii_digit_label(raw_label: str | int, source: str) -> int:
    if isinstance(raw_label, bool):
        raise ValueError(f"Unsupported label for {source}: {raw_label!r}")

    if source == "digit_images" and isinstance(raw_label, str):
        stripped = raw_label.strip()
        match = re.fullmatch(r"sample0*([1-9])", stripped, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    try:
        canonical = int(raw_label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported label for {source}: {raw_label!r}") from exc

    if canonical not in CANONICAL_LABELS:
        raise ValueError(f"Unsupported label for {source}: {raw_label!r}")
    return canonical


def _map_hoda_label(raw_label: str | int) -> int:
    if isinstance(raw_label, str):
        stripped = raw_label.strip()
        if stripped in HODA_PERSIAN_DIGITS:
            return HODA_PERSIAN_DIGITS[stripped]
        return _map_ascii_digit_label(stripped, "hoda")
    return _map_ascii_digit_label(raw_label, "hoda")
