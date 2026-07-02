"""CLI entrypoint for recognition evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sudoku_solver.config.loader import load_config
from sudoku_solver.data.base import SampleRecord, ValidationReport
from sudoku_solver.data.compose import CombinedDataset, compose_datasets
from sudoku_solver.data.validate import validate_records
from sudoku_solver.recognition.evaluate import evaluate_model
from sudoku_solver.recognition.model import build_model
from sudoku_solver.recognition.train import RecognitionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Sudoku digit recognizer.")
    parser.add_argument(
        "--config",
        default="sudoku_solver/config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. If omitted, evaluates the randomly initialized model.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/evaluation",
        help="Directory for evaluation metrics JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dataset = compose_datasets(config.training.datasets, "val")
    if len(eval_dataset) == 0:
        eval_dataset = compose_datasets(config.training.datasets, "test")

    validation_reports = _validate_combined_dataset(eval_dataset)
    dataloader = _build_eval_dataloader(eval_dataset, config.runtime.input_size)

    model = build_model(num_classes=10)
    checkpoint_loaded = False
    if args.checkpoint:
        checkpoint_loaded = _try_load_checkpoint(model, Path(args.checkpoint), config.runtime.device)

    class_names = [str(index) for index in range(10)]
    results = evaluate_model(
        model=model if checkpoint_loaded or dataloader is not None else None,
        dataloader=dataloader,
        class_names=class_names,
        device=config.runtime.device,
    )

    payload = {
        "checkpoint_path": args.checkpoint,
        "checkpoint_loaded": checkpoint_loaded,
        "datasets": eval_dataset.source_names,
        "validation_reports": [_report_to_dict(report) for report in validation_reports],
        "results": results,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"checkpoint_loaded={checkpoint_loaded}")
    print(f"eval_records={len(eval_dataset)}")
    print(f"accuracy={results['accuracy']}")
    print(f"metrics_path={metrics_path}")
    return 0


def _build_eval_dataloader(
    dataset: CombinedDataset,
    input_size: int,
) -> DataLoader[tuple[torch.Tensor, int]] | None:
    if len(dataset) == 0:
        return None
    torch_dataset = RecognitionDataset(
        dataset.records,
        input_size=input_size,
        split="eval",
    )
    return DataLoader(torch_dataset, batch_size=32, num_workers=0, shuffle=False)


def _validate_combined_dataset(dataset: CombinedDataset) -> list[ValidationReport]:
    reports: list[ValidationReport] = []
    grouped: dict[str, list[SampleRecord]] = {}
    for record in dataset.records:
        grouped.setdefault(record.source, []).append(record)
    for source_name, records in grouped.items():
        reports.append(validate_records(source_name, records))
    return reports


def _try_load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, requested_device: str) -> bool:
    if not checkpoint_path.exists():
        return False

    map_location = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    model.load_state_dict(state_dict)
    return True


def _report_to_dict(report: ValidationReport) -> dict[str, object]:
    return {
        "source": report.source,
        "total_records": report.total_records,
        "discovered_labels": report.discovered_labels,
        "canonical_counts": report.canonical_counts,
        "invalid_labels": [
            {
                "raw_label": issue.raw_label,
                "reason": issue.reason,
                "path": issue.path,
            }
            for issue in report.invalid_labels
        ],
        "missing_labels": report.missing_labels,
    }


if __name__ == "__main__":
    raise SystemExit(main())
