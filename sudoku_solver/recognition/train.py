"""Minimal training entrypoint for Sudoku digit recognition."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sudoku_solver.config.types import AppConfig
from sudoku_solver.data.base import SampleRecord, ValidationReport
from sudoku_solver.data.compose import CombinedDataset, compose_datasets
from sudoku_solver.data.data_utils import resolve_dataset_root
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
    epochs_completed: int = 0
    train_loss: float | None = None
    val_accuracy: float | None = None
    checkpoint_path: Path | None = None
    best_checkpoint_path: Path | None = None
    epoch_metrics: list[dict[str, float | int]] | None = None


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
    configured_paths = getattr(config.training, "dataset_paths", {})
    data_root = getattr(config.training, "data_root", Path("data"))
    val_split = getattr(config.training, "val_split", 0.15)
    random_seed = getattr(config.training, "random_seed", 2023)
    epochs = getattr(config.training, "epochs", 1)
    learning_rate = getattr(config.training, "learning_rate", 1e-3)
    weight_decay = getattr(config.training, "weight_decay", 0.0)
    checkpoint_dir = getattr(config.training, "checkpoint_dir", Path("models/checkpoints"))
    save_best_only = getattr(config.training, "save_best_only", False)

    roots = {
        dataset_name: resolve_dataset_root(
            dataset_name,
            configured_paths=configured_paths,
            default_root=data_root,
        )
        for dataset_name in config.training.datasets
    }
    train_dataset = compose_datasets(
        config.training.datasets,
        "train",
        roots=roots,
        val_split=val_split,
        random_seed=random_seed,
    )
    validation_reports = _validate_combined_dataset(train_dataset)
    split_sizes = {"train": len(train_dataset)}

    eval_dataset = compose_datasets(
        config.training.datasets,
        "val",
        roots=roots,
        val_split=val_split,
        random_seed=random_seed,
    )
    if len(eval_dataset) == 0:
        eval_dataset = compose_datasets(
            config.training.datasets,
            "test",
            roots=roots,
            val_split=val_split,
            random_seed=random_seed,
        )
    split_sizes["eval"] = len(eval_dataset)

    device = _resolve_device(config.runtime.device)
    model = build_model(num_classes=10).to(device)

    print(
        "training_setup "
        f"device={device.type} "
        f"datasets={','.join(train_dataset.source_names)} "
        f"train_records={split_sizes['train']} "
        f"eval_records={split_sizes['eval']} "
        f"epochs={epochs} "
        f"batch_size={config.training.batch_size} "
        f"num_workers={config.training.num_workers} "
        f"learning_rate={learning_rate} "
        f"weight_decay={weight_decay} "
        f"checkpoint_dir={checkpoint_dir}"
    )
    for report in validation_reports:
        print(
            "dataset_validation "
            f"source={report.source} "
            f"total={report.total_records} "
            f"missing={report.missing_labels}"
        )

    if dry_run:
        return TrainingArtifacts(
            model=model,
            device=device.type,
            dataset_names=train_dataset.source_names,
            split_sizes=split_sizes,
            validation_reports=validation_reports,
            dry_run=True,
            train_steps=0,
            epochs_completed=0,
            train_loss=None,
            val_accuracy=None,
            checkpoint_path=None,
            best_checkpoint_path=None,
            epoch_metrics=[],
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

    eval_dataloader = _build_eval_dataloader(
        eval_dataset.records,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        input_size=config.runtime.input_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_batches = len(dataloader)
    eval_batches = len(eval_dataloader) if eval_dataloader is not None else 0
    progress_interval = _resolve_progress_interval(total_batches)

    print(
        "training_start "
        f"total_batches={total_batches} "
        f"eval_batches={eval_batches} "
        f"progress_interval={progress_interval}"
    )

    train_steps = 0
    best_val_accuracy = float("-inf")
    latest_train_loss = 0.0
    latest_val_accuracy = 0.0
    epoch_metrics: list[dict[str, float | int]] = []
    training_started_at = perf_counter()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "last.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        epoch_started_at = perf_counter()
        print(
            "epoch_start "
            f"epoch={epoch + 1}/{epochs} "
            f"batches={total_batches}"
        )
        for batch_index, (batch_inputs, batch_labels) in enumerate(dataloader, start=1):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_steps += 1
            batch_size = int(batch_labels.size(0))
            epoch_loss_sum += float(loss.item()) * batch_size
            epoch_sample_count += batch_size

            if (
                batch_index == 1
                or batch_index % progress_interval == 0
                or batch_index == total_batches
            ):
                running_loss = epoch_loss_sum / max(epoch_sample_count, 1)
                elapsed_epoch_seconds = perf_counter() - epoch_started_at
                average_batch_seconds = elapsed_epoch_seconds / max(batch_index, 1)
                epoch_eta_seconds = average_batch_seconds * max(total_batches - batch_index, 0)
                total_elapsed_seconds = perf_counter() - training_started_at
                completed_epoch_fraction = epoch + (batch_index / max(total_batches, 1))
                average_epoch_seconds = total_elapsed_seconds / max(completed_epoch_fraction, 1e-9)
                total_eta_seconds = average_epoch_seconds * max(epochs - completed_epoch_fraction, 0.0)
                print(
                    "batch_progress "
                    f"epoch={epoch + 1}/{epochs} "
                    f"batch={batch_index}/{total_batches} "
                    f"running_loss={running_loss:.4f} "
                    f"elapsed={_format_duration(elapsed_epoch_seconds)} "
                    f"epoch_eta={_format_duration(epoch_eta_seconds)} "
                    f"total_eta={_format_duration(total_eta_seconds)}"
                )

        latest_train_loss = epoch_loss_sum / max(epoch_sample_count, 1)
        latest_val_accuracy = (
            _evaluate_accuracy(model, eval_dataloader, device) if eval_dataloader is not None else 0.0
        )
        epoch_elapsed_seconds = perf_counter() - epoch_started_at
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": latest_train_loss,
                "val_accuracy": latest_val_accuracy,
            }
        )
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={latest_train_loss:.4f} "
            f"val_accuracy={latest_val_accuracy:.4f} "
            f"elapsed={_format_duration(epoch_elapsed_seconds)}"
        )

        payload = _build_checkpoint_payload(
            model=model,
            dataset_names=train_dataset.source_names,
            epochs_completed=epoch + 1,
            train_steps=train_steps,
            input_size=config.runtime.input_size,
            train_loss=latest_train_loss,
            val_accuracy=latest_val_accuracy,
        )
        torch.save(payload, checkpoint_path)

        if latest_val_accuracy >= best_val_accuracy:
            best_val_accuracy = latest_val_accuracy
            torch.save(payload, best_checkpoint_path)
        elif save_best_only and checkpoint_path.exists():
            checkpoint_path.unlink(missing_ok=True)

    return TrainingArtifacts(
        model=model,
        device=device.type,
        dataset_names=train_dataset.source_names,
        split_sizes=split_sizes,
        validation_reports=validation_reports,
        dry_run=False,
        train_steps=train_steps,
        epochs_completed=epochs,
        train_loss=latest_train_loss,
        val_accuracy=latest_val_accuracy,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else None,
        epoch_metrics=epoch_metrics,
    )


def _build_eval_dataloader(
    records: list[SampleRecord],
    *,
    batch_size: int,
    num_workers: int,
    input_size: int,
) -> DataLoader[tuple[torch.Tensor, int]] | None:
    if not records:
        return None
    dataset = RecognitionDataset(records, input_size=input_size, split="eval")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def _evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            predictions = torch.argmax(logits, dim=1)
            total += int(batch_labels.numel())
            correct += int((predictions == batch_labels).sum().item())
    return correct / total if total > 0 else 0.0


def _build_checkpoint_payload(
    *,
    model: nn.Module,
    dataset_names: list[str],
    epochs_completed: int,
    train_steps: int,
    input_size: int,
    train_loss: float,
    val_accuracy: float,
) -> dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "datasets": dataset_names,
        "epochs": epochs_completed,
        "train_steps": train_steps,
        "input_size": input_size,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
    }


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


def _resolve_progress_interval(total_batches: int) -> int:
    if total_batches <= 0:
        return 1
    return max(1, ceil(total_batches / 10))


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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
