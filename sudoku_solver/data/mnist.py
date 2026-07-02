"""MNIST adapter entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

from .base import SampleRecord
from .data_utils import build_records_from_arrays, scan_labeled_image_tree


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
    val_split: float = 0.15,
    random_seed: int = 2023,
) -> DatasetAdapter:
    """Return a concrete MNIST adapter."""
    resolved_root = Path(root) if root is not None else None
    if records is not None:
        dataset_records = list(records)
    else:
        dataset_records = _load_mnist_records(
            split,
            resolved_root,
            val_split=val_split,
            random_seed=random_seed,
        )
    return DatasetAdapter(name="mnist", split=split, records=dataset_records)


def _load_mnist_records(
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
        return scan_labeled_image_tree(split_root, "mnist")

    train_dataset = MNIST(root=str(root), train=True, download=True)
    test_dataset = MNIST(root=str(root), train=False, download=True)

    train_images = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    test_images = test_dataset.data.numpy()
    test_labels = test_dataset.targets.numpy()

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(train_images))
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_split))
    val_indices = indices[:val_count]
    fit_indices = indices[val_count:]

    normalized_split = split.strip().lower()
    if normalized_split == "train":
        return build_records_from_arrays(
            source="mnist",
            images=train_images[fit_indices],
            labels=train_labels[fit_indices],
        )
    if normalized_split == "val":
        return build_records_from_arrays(
            source="mnist",
            images=train_images[val_indices],
            labels=train_labels[val_indices],
        )
    if normalized_split == "test":
        return build_records_from_arrays(
            source="mnist",
            images=test_images,
            labels=test_labels,
        )
    raise ValueError(f"Unsupported split: {split}")
