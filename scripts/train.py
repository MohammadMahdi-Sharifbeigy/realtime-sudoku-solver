"""CLI entrypoint for recognition training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sudoku_solver.config.loader import load_config
from sudoku_solver.recognition.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sudoku digit recognizer.")
    parser.add_argument(
        "--config",
        default="sudoku_solver/config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build config, datasets, and model without running optimization.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    artifacts = train_model(config, dry_run=args.dry_run)

    print(f"device={artifacts.device}")
    print(f"datasets={','.join(artifacts.dataset_names)}")
    print(f"train_records={artifacts.split_sizes['train']}")
    print(f"eval_records={artifacts.split_sizes['eval']}")
    print(f"dry_run={artifacts.dry_run}")
    print(f"train_steps={artifacts.train_steps}")

    for report in artifacts.validation_reports:
        print(
            f"validated_source={report.source} total={report.total_records} "
            f"missing={report.missing_labels}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
