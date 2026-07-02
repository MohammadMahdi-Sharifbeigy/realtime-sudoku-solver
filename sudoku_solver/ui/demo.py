"""Optional demo UI entrypoint, isolated from the core runtime."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import cv2

from sudoku_solver.app.pipeline import run_image_pipeline
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
        help="Optional recognizer checkpoint path.",
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
    config = load_config(config_path)

    try:
        module = _import_backend(backend)
    except ImportError:
        print(
            f"Optional demo backend '{backend}' is not installed. "
            f"Install it separately to launch the UI."
        )
        return 1

    if backend == "streamlit":
        return _launch_streamlit(module, config, image_path=image_path, checkpoint_path=checkpoint_path)
    if backend == "gradio":
        return _launch_gradio(module, config, image_path=image_path, checkpoint_path=checkpoint_path)
    raise ValueError(f"Unsupported demo backend: {backend}")


def _import_backend(backend: str) -> Any:
    if backend == "streamlit":
        import streamlit

        return streamlit
    if backend == "gradio":
        import gradio

        return gradio
    raise ValueError(f"Unsupported demo backend: {backend}")


def _launch_streamlit(
    streamlit: Any,
    config: Any,
    *,
    image_path: str | None,
    checkpoint_path: str | None,
) -> int:
    streamlit.set_page_config(page_title="Sudoku Solver", layout="wide")
    streamlit.title("Sudoku Solver UI")
    streamlit.write("Run image-based Sudoku solving with the trained MobileNet checkpoint.")

    checkpoint_value = streamlit.text_input(
        "Checkpoint path",
        value=checkpoint_path or "",
        help="Leave empty to auto-use models/checkpoints/best.pt or last.pt.",
    )
    source_image_path = streamlit.text_input(
        "Image path",
        value=image_path or "",
        help="Use an existing image path or upload a file below.",
    )
    uploaded_file = streamlit.file_uploader("Upload Sudoku image", type=["png", "jpg", "jpeg", "bmp"])

    if streamlit.button("Solve Sudoku"):
        effective_image_path = source_image_path.strip()
        temp_path: Path | None = None
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(uploaded_file.getbuffer())
                temp_path = Path(handle.name)
            effective_image_path = str(temp_path)

        if not effective_image_path:
            streamlit.error("Provide an image path or upload a file.")
            return 0

        try:
            result = run_image_pipeline(
                config,
                effective_image_path,
                checkpoint_path=checkpoint_value or None,
            )
        except Exception as exc:
            streamlit.error(str(exc))
            return 0

        streamlit.write(result.message)
        streamlit.image(cv2.cvtColor(result.original_image, cv2.COLOR_BGR2RGB), caption="Original image")
        if result.warped_board is not None:
            streamlit.image(result.warped_board, caption="Warped board", clamp=True)
        if result.overlay_image is not None:
            streamlit.image(cv2.cvtColor(result.overlay_image, cv2.COLOR_BGR2RGB), caption="Solved overlay")
        if result.board is not None:
            streamlit.text("\n".join(" ".join(str(value) for value in row) for row in result.board))
        if result.solved_board is not None:
            streamlit.text("\n".join(" ".join(str(value) for value in row) for row in result.solved_board))
    return 0


def _launch_gradio(
    gradio: Any,
    config: Any,
    *,
    image_path: str | None,
    checkpoint_path: str | None,
) -> int:
    def _solve(image: Any, checkpoint: str) -> tuple[Any, str]:
        if image is None:
            return None, "Provide an image."
        result = run_image_pipeline(
            config,
            str(image),
            checkpoint_path=checkpoint or checkpoint_path,
        )
        rendered = None
        if result.overlay_image is not None:
            rendered = cv2.cvtColor(result.overlay_image, cv2.COLOR_BGR2RGB)
        lines = [result.message]
        if result.board is not None:
            lines.append("Recognized board:")
            lines.extend(" ".join(str(value) for value in row) for row in result.board)
        if result.solved_board is not None:
            lines.append("Solved board:")
            lines.extend(" ".join(str(value) for value in row) for row in result.solved_board)
        return rendered, "\n".join(lines)

    demo = gradio.Interface(
        fn=_solve,
        inputs=[
            gradio.Image(type="filepath", value=image_path, label="Sudoku image"),
            gradio.Textbox(
                value=checkpoint_path or "",
                label="Checkpoint path",
                placeholder="Leave empty to auto-use models/checkpoints/best.pt or last.pt",
            ),
        ],
        outputs=[
            gradio.Image(label="Solved overlay"),
            gradio.Textbox(label="Result"),
        ],
        title="Sudoku Solver UI",
        description="Upload a Sudoku image and run the OpenCV + PyTorch pipeline.",
    )
    demo.launch()
    return 0


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
