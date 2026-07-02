"""Image inference entrypoint for the Sudoku solver."""

from __future__ import annotations

import argparse

from sudoku_solver.app.runtime import run_image_mode
from sudoku_solver.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sudoku inference on one image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--output", default=None, help="Optional output image path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional recognizer checkpoint path placeholder for future use.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the output image.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    return run_image_mode(
        config,
        image_path=args.image,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        display=args.display,
    )


if __name__ == "__main__":
    raise SystemExit(main())
