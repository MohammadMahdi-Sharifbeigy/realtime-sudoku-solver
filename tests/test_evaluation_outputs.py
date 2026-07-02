from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from sudoku_solver.recognition.evaluate import evaluate_model


def test_evaluate_model_returns_accuracy_and_confusion_matrix():
    results = evaluate_model(
        model=None,
        dataloader=None,
        class_names=[str(index) for index in range(10)],
    )

    assert "accuracy" in results
    assert "confusion_matrix" in results
    assert results["accuracy"] == 0.0
    assert len(results["confusion_matrix"]) == 10
    assert all(len(row) == 10 for row in results["confusion_matrix"])
    assert results["total_samples"] == 0


def test_evaluate_model_counts_predictions_for_non_empty_dataloader():
    features = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)

    class StaticModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            del inputs
            return torch.tensor(
                [
                    [5.0, 0.1, 0.0],
                    [0.2, 3.0, 0.0],
                ],
                dtype=torch.float32,
            )

    results = evaluate_model(
        model=StaticModel(),
        dataloader=dataloader,
        class_names=["0", "1", "2"],
    )

    assert results["accuracy"] == 1.0
    assert results["total_samples"] == 2
    assert results["correct_predictions"] == 2
    assert results["confusion_matrix"][0][0] == 1
    assert results["confusion_matrix"][1][1] == 1
