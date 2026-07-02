"""Filesystem-backed adapter for local digit image folders."""

from __future__ import annotations

from pathlib import Path

from .base import SampleRecord
from .data_utils import scan_labeled_image_tree, split_records
from .mnist import DatasetAdapter


def load_digit_images_dataset(
    split: str,
    *,
    root: str | Path | None = None,
    records: list[SampleRecord] | None = None,
    val_split: float = 0.15,
    random_seed: int = 2023,
) -> DatasetAdapter:
    """Return a concrete digit_images adapter."""
    resolved_root = Path(root) if root is not None else None
    dataset_records = list(records) if records is not None else _scan_digit_images(
        split,
        resolved_root,
        val_split=val_split,
        random_seed=random_seed,
    )
    return DatasetAdapter(name="digit_images", split=split, records=dataset_records)


def _scan_digit_images(
    split: str,
    root: Path | None,
    *,
    val_split: float,
    random_seed: int,
) -> list[SampleRecord]:
    if root is None:
        return []

    split_root = root / split
    if split_root.exists():
        return scan_labeled_image_tree(split_root, "digit_images")

    all_records = scan_labeled_image_tree(root, "digit_images")
    return split_records(
        all_records,
        split,
        val_ratio=val_split,
        random_seed=random_seed,
    )
