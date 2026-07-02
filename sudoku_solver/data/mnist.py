"""Lightweight MNIST adapter entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import SampleRecord


@dataclass(slots=True)
class DatasetAdapter:
    """Concrete dataset container shared by composition helpers."""

    name: str
    split: str
    records: list[SampleRecord] = field(default_factory=list)

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


def load_mnist_dataset(
    split: str,
    *,
    root: str | Path | None = None,
    records: list[SampleRecord] | None = None,
) -> DatasetAdapter:
    """Return a concrete MNIST adapter.

    The initial rewrite keeps this loader filesystem-light and network-free.
    Callers may inject prepared records for tests, or point the loader at a
    local directory containing class-named subfolders.
    """
    resolved_root = Path(root) if root is not None else None
    dataset_records = list(records) if records is not None else _scan_digit_folders(
        "mnist", split, resolved_root
    )
    return DatasetAdapter(name="mnist", split=split, records=dataset_records)


def _scan_digit_folders(
    source: str,
    split: str,
    root: Path | None,
) -> list[SampleRecord]:
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
                    source=source,
                    raw_label=label_dir.name,
                    path=image_path,
                )
            )
    return records
