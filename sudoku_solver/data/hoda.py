"""Lightweight Hoda/DigitDB adapter entrypoint."""

from __future__ import annotations

from pathlib import Path

from .base import SampleRecord
from .mnist import DatasetAdapter


def load_hoda_dataset(
    split: str,
    *,
    root: str | Path | None = None,
    records: list[SampleRecord] | None = None,
) -> DatasetAdapter:
    """Return a concrete Hoda adapter.

    This is intentionally lightweight for the rewrite stage. Real binary file
    parsing can replace the folder scan later without changing the interface.
    """
    resolved_root = Path(root) if root is not None else None
    dataset_records = list(records) if records is not None else _scan_hoda_records(
        split,
        resolved_root,
    )
    return DatasetAdapter(name="hoda", split=split, records=dataset_records)


def _scan_hoda_records(split: str, root: Path | None) -> list[SampleRecord]:
    if root is None:
        return []

    split_root = root / split
    if not split_root.exists():
        return []

    records: list[SampleRecord] = []
    for label_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in label_dir.iterdir() if path.is_file()):
            records.append(
                SampleRecord(
                    source="hoda",
                    raw_label=label_dir.name,
                    path=image_path,
                )
            )
    return records
