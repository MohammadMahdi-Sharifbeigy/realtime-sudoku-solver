"""Minimal training entrypoint for Sudoku digit recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sudoku_solver.config.types import AppConfig
from sudoku_solver.data.base import SampleRecord, ValidationReport
from sudoku_solver.data.compose import CombinedDataset, compose_datasets
from sudoku_solver.data.labels import map_label
from sudoku_solver.data.validate import validate_records
from sudoku_solver.recognition.model import build_model
from sudoku_solver.recognition.transforms import (
    build_eval_transforms,
    build_train_transforms,
)


@dataclass(slots=True)
class TrainingArtifacts:
    """Summary of a training setup or run."""

    model: nn.Module
    device: str
    dataset_names: list[str]
    split_sizes: dict[str, int]
    validation_reports: list[ValidationReport]
    dry_run: bool
    train_steps: int = 0


class RecognitionDataset(Dataset[tuple[torch.Tensor, int]]):
    """Torch dataset wrapper over canonical sample records."""

    def __init__(self, records: list[SampleRecord], input_size: int, split: str) -> None:
        self._records = list(records)
        self._transform = (
            build_train_transforms(input_size)
            if split == "train"
            else build_eval_transforms(input_size)
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._records[index]
        image = _load_record_image(record)
        label = map_label(record.source, record.raw_label)
        return self._transform(image), label


def train_model(config: AppConfig, *, dry_run: bool = False) -> TrainingArtifacts:
    """Build datasets, model, and optionally run a minimal training epoch."""
    train_dataset = compose_datasets(config.training.datasets, "train")
    validation_reports = _validate_combined_dataset(train_dataset)
    split_sizes = {"train": len(train_dataset)}

    eval_dataset = compose_datasets(config.training.datasets, "val")
    if len(eval_dataset) == 0:
        eval_dataset = compose_datasets(config.training.datasets, "test")
    split_sizes["eval"] = len(eval_dataset)

    device = _resolve_device(config.runtime.device)
    model = build_model(num_classes=10).to(device)

    if dry_run:
        return TrainingArtifacts(
            model=model,
            device=device.type,
            dataset_names=train_dataset.source_names,
            split_sizes=split_sizes,
            validation_reports=validation_reports,
            dry_run=True,
            train_steps=0,
        )

    if len(train_dataset) == 0:
        raise ValueError(
            "No training records were found. Configure local dataset roots or pass --dry-run."
        )

    torch_dataset = RecognitionDataset(
        train_dataset.records,
        input_size=config.runtime.input_size,
        split="train",
    )
    dataloader = DataLoader(
        torch_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    train_steps = 0
    for batch_inputs, batch_labels in dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        train_steps += 1
        break

    return TrainingArtifacts(
        model=model,
        device=device.type,
        dataset_names=train_dataset.source_names,
        split_sizes=split_sizes,
        validation_reports=validation_reports,
        dry_run=False,
        train_steps=train_steps,
    )


def _validate_combined_dataset(dataset: CombinedDataset) -> list[ValidationReport]:
    reports: list[ValidationReport] = []
    for source_name, records in _group_records_by_source(dataset).items():
        reports.append(validate_records(source_name, records))
    return reports


def _group_records_by_source(dataset: CombinedDataset) -> dict[str, list[SampleRecord]]:
    grouped: dict[str, list[SampleRecord]] = {}
    for record in dataset.records:
        grouped.setdefault(record.source, []).append(record)
    return grouped


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_record_image(record: SampleRecord) -> np.ndarray:
    if record.image is not None:
        return _coerce_image(record.image)
    if record.path is None:
        raise ValueError(
            f"Record from source '{record.source}' does not contain image data or a path"
        )

    image = cv2.imread(str(record.path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image from path: {record.path}")
    return image


def _coerce_image(image: object) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported image array shape: {image.shape}")

    if torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 2:
            return tensor.numpy().astype(np.uint8)
        if tensor.ndim == 3:
            array = tensor.permute(1, 2, 0).numpy()
            return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")

    raise ValueError(f"Unsupported image payload type: {type(image).__name__}")
