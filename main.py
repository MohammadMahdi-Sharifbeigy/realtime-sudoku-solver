"""CLI entrypoint for the Python Sudoku solver runtime."""

from __future__ import annotations

import argparse

from sudoku_solver.app.runtime import (
    build_runtime_mode,
    run_image_mode,
    run_webcam_mode,
)
from sudoku_solver.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Sudoku solver on an image or webcam stream.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Defaults to the packaged config.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image for single-image inference.",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Webcam source index for live inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path for image mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional recognizer checkpoint path placeholder for future model loading.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display image or webcam windows during inference.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    mode = build_runtime_mode(image_path=args.image, source=args.source)

    if mode == "image":
        return run_image_mode(
            config,
            image_path=args.image,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            display=args.display,
        )

    return run_webcam_mode(
        config,
        source=0 if args.source is None else args.source,
        checkpoint_path=args.checkpoint,
        display=args.display,
    )


if __name__ == "__main__":
    raise SystemExit(main())
