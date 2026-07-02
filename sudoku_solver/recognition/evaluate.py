"""Evaluation helpers for Sudoku digit recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(slots=True)
class EvaluationResult:
    """Structured evaluation outputs for serialization and reporting."""

    accuracy: float
    confusion_matrix: list[list[int]]
    class_names: list[str]
    total_samples: int
    correct_predictions: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_model(
    model: nn.Module | None,
    dataloader: DataLoader[tuple[torch.Tensor, int]] | None,
    class_names: list[str],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Evaluate a classifier and return serializable metrics.

    The current project scaffolding may provide no checkpoint and no records yet,
    so this function returns a zeroed result instead of failing on empty inputs.
    """
    num_classes = len(class_names)
    confusion_matrix = _build_empty_confusion_matrix(num_classes)

    if dataloader is None or _dataloader_is_empty(dataloader):
        return EvaluationResult(
            accuracy=0.0,
            confusion_matrix=confusion_matrix,
            class_names=list(class_names),
            total_samples=0,
            correct_predictions=0,
        ).to_dict()

    if model is None:
        raise ValueError("model must be provided when evaluating a non-empty dataloader")

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(resolved_device)
            batch_labels = batch_labels.to(resolved_device)

            logits = model(batch_inputs)
            predictions = torch.argmax(logits, dim=1)

            total_samples += int(batch_labels.numel())
            correct_predictions += int((predictions == batch_labels).sum().item())

            for actual, predicted in zip(
                batch_labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
            ):
                if 0 <= actual < num_classes and 0 <= predicted < num_classes:
                    confusion_matrix[actual][predicted] += 1

    accuracy = (
        correct_predictions / total_samples if total_samples > 0 else 0.0
    )
    return EvaluationResult(
        accuracy=accuracy,
        confusion_matrix=confusion_matrix,
        class_names=list(class_names),
        total_samples=total_samples,
        correct_predictions=correct_predictions,
    ).to_dict()


def _build_empty_confusion_matrix(num_classes: int) -> list[list[int]]:
    return [[0 for _ in range(num_classes)] for _ in range(num_classes)]


def _dataloader_is_empty(dataloader: DataLoader[tuple[torch.Tensor, int]]) -> bool:
    try:
        return len(dataloader.dataset) == 0
    except TypeError:
        return False


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
