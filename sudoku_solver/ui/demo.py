"""Optional demo UI entrypoint, isolated from the core runtime."""

from __future__ import annotations

import argparse
from typing import Any

from sudoku_solver.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the optional Sudoku solver demo UI.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Defaults to the packaged config.",
    )
    parser.add_argument(
        "--backend",
        choices=("streamlit", "gradio"),
        default="streamlit",
        help="Optional UI backend to launch.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path to prefill the demo workflow later.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional recognizer checkpoint path placeholder for future UI wiring.",
    )
    return parser


def launch_demo(
    *,
    backend: str,
    config_path: str | None = None,
    image_path: str | None = None,
    checkpoint_path: str | None = None,
) -> int:
    """Lazy-load an optional UI backend without affecting core runtime imports."""
    _ = load_config(config_path)
    _ = image_path
    _ = checkpoint_path

    try:
        module = _import_backend(backend)
    except ImportError:
        print(
            f"Optional demo backend '{backend}' is not installed. "
            f"Install it separately to launch the UI."
        )
        return 1

    print(
        f"Optional demo backend '{backend}' is available as '{module.__name__}', "
        "but the interactive UI scaffold is not implemented yet."
    )
    return 0


def _import_backend(backend: str) -> Any:
    if backend == "streamlit":
        import streamlit

        return streamlit
    if backend == "gradio":
        import gradio

        return gradio
    raise ValueError(f"Unsupported demo backend: {backend}")


def main() -> int:
    args = build_parser().parse_args()
    return launch_demo(
        backend=args.backend,
        config_path=args.config,
        image_path=args.image,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
