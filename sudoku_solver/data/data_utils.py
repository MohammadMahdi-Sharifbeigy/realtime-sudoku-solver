"""Shared dataset loading utilities for local files and tensor-backed datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .base import SampleRecord


def resolve_dataset_root(
    dataset_name: str,
    *,
    configured_paths: dict[str, Path],
    default_root: Path,
) -> Path:
    """Resolve the on-disk root for one dataset."""
    normalized = dataset_name.strip().lower()
    if normalized in configured_paths:
        return configured_paths[normalized]
    return default_root / normalized


def list_class_dirs(root: Path) -> list[Path]:
    """Return class directories only, sorted by name."""
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def scan_labeled_image_tree(root: Path, source: str) -> list[SampleRecord]:
    """Load images from a `label_name/image.*` tree."""
    records: list[SampleRecord] = []
    for label_dir in list_class_dirs(root):
        for image_path in sorted(path for path in label_dir.iterdir() if path.is_file()):
            records.append(
                SampleRecord(
                    source=source,
                    raw_label=label_dir.name,
                    path=image_path,
                )
            )
    return records


def split_records(
    records: Sequence[SampleRecord],
    split: str,
    *,
    val_ratio: float,
    test_ratio: float = 0.15,
    random_seed: int = 2023,
) -> list[SampleRecord]:
    """Derive train/val/test slices from a flat record list."""
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if not records:
        return []

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    test_count = int(round(len(records) * test_ratio))
    val_count = int(round(len(records) * val_ratio))
    test_count = min(test_count, len(records))
    val_count = min(val_count, max(0, len(records) - test_count))

    test_indices = indices[:test_count]
    val_indices = indices[test_count : test_count + val_count]
    train_indices = indices[test_count + val_count :]

    selected = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }[normalized_split]
    return [records[int(index)] for index in selected]


def build_records_from_arrays(
    *,
    source: str,
    images: Iterable[np.ndarray],
    labels: Iterable[int],
) -> list[SampleRecord]:
    """Convert in-memory arrays into canonical sample records."""
    records: list[SampleRecord] = []
    for image, label in zip(images, labels, strict=True):
        records.append(
            SampleRecord(
                source=source,
                raw_label=int(label),
                image=np.asarray(image),
            )
        )
    return records
