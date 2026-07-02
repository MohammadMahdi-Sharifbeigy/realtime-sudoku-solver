"""Webcam inference entrypoint for the Sudoku solver."""

from __future__ import annotations

import argparse

from sudoku_solver.app.runtime import run_webcam_mode
from sudoku_solver.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live Sudoku inference on a webcam.")
    parser.add_argument("--source", type=int, default=0, help="Webcam source index.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional recognizer checkpoint path placeholder for future use.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the webcam window during inference.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    return run_webcam_mode(
        config,
        source=args.source,
        checkpoint_path=args.checkpoint,
        display=args.display,
    )


if __name__ == "__main__":
    raise SystemExit(main())
