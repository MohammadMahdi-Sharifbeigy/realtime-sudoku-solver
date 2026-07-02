"""Dataset composition helpers that preserve requested source order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import SampleRecord
from .digit_images import load_digit_images_dataset
from .hoda import load_hoda_dataset
from .mnist import DatasetAdapter, load_mnist_dataset


DatasetRoots = dict[str, str | Path]
DatasetRecordMap = dict[str, list[SampleRecord]]


@dataclass(slots=True)
class CombinedDataset:
    """Ordered composition of multiple dataset adapters."""

    datasets: list[DatasetAdapter]

    @property
    def source_names(self) -> list[str]:
        return [dataset.name for dataset in self.datasets]

    @property
    def records(self) -> list[SampleRecord]:
        merged: list[SampleRecord] = []
        for dataset in self.datasets:
            merged.extend(dataset.records)
        return merged

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)


def load_dataset(
    name: str,
    split: str,
    *,
    root: str | Path | None = None,
    records: list[SampleRecord] | None = None,
    val_split: float = 0.15,
    random_seed: int = 2023,
) -> DatasetAdapter:
    """Load a single supported dataset adapter."""
    normalized_name = name.strip().lower()
    if normalized_name == "mnist":
        return load_mnist_dataset(
            split,
            root=root,
            records=records,
            val_split=val_split,
            random_seed=random_seed,
        )
    if normalized_name == "hoda":
        return load_hoda_dataset(
            split,
            root=root,
            records=records,
            val_split=val_split,
            random_seed=random_seed,
        )
    if normalized_name == "digit_images":
        return load_digit_images_dataset(
            split,
            root=root,
            records=records,
            val_split=val_split,
            random_seed=random_seed,
        )
    raise ValueError(f"Unsupported dataset name: {name}")


def compose_datasets(
    names: list[str],
    split: str,
    *,
    roots: DatasetRoots | None = None,
    record_overrides: DatasetRecordMap | None = None,
    val_split: float = 0.15,
    random_seed: int = 2023,
) -> CombinedDataset:
    """Compose datasets in the exact order requested by the caller."""
    datasets: list[DatasetAdapter] = []
    for name in names:
        normalized_name = name.strip().lower()
        dataset_root = None if roots is None else roots.get(normalized_name)
        dataset_records = None
        if record_overrides is not None and normalized_name in record_overrides:
            dataset_records = record_overrides[normalized_name]
        datasets.append(
            load_dataset(
                normalized_name,
                split,
                root=dataset_root,
                records=dataset_records,
                val_split=val_split,
                random_seed=random_seed,
            )
        )
    return CombinedDataset(datasets=datasets)
