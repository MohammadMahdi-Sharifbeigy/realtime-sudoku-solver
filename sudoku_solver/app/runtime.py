"""Runtime dispatch for image and webcam inference."""

from __future__ import annotations

from typing import Final

import cv2

from sudoku_solver.app.pipeline import (
    PipelineResult,
    build_runtime_mode,
    process_frame,
    run_image_pipeline,
)
from sudoku_solver.config.types import AppConfig


WINDOW_NAME: Final[str] = "Sudoku Solver"


def run_image_mode(
    config: AppConfig,
    *,
    image_path: str | None,
    output_path: str | None = None,
    checkpoint_path: str | None = None,
    display: bool = True,
) -> int:
    """Run image-mode inference and optionally display the result."""
    if not image_path:
        raise ValueError("image_path is required for image mode.")

    result = run_image_pipeline(
        config,
        image_path,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )
    _print_result(result)

    if display and result.overlay_image is not None:
        cv2.imshow(WINDOW_NAME, result.overlay_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if result.success else 1


def run_webcam_mode(
    config: AppConfig,
    *,
    source: int = 0,
    checkpoint_path: str | None = None,
    display: bool = True,
) -> int:
    """Run live webcam inference until the stream ends or the user quits."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Could not open webcam source {source}.")
        return 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Webcam stream ended or frame capture failed.")
                return 1

            result = process_frame(
                frame,
                config,
                checkpoint_path=checkpoint_path,
            )

            if not display:
                continue

            frame_to_show = result.overlay_image if result.overlay_image is not None else frame
            if not result.success:
                cv2.putText(
                    frame_to_show,
                    result.message,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
            cv2.imshow(WINDOW_NAME, frame_to_show)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def _print_result(result: PipelineResult) -> None:
    print(result.message)
    if result.board is not None:
        print("Recognized board:")
        for row in result.board:
            print(" ".join(str(value) for value in row))
    if result.solved_board is not None:
        print("Solved board:")
        for row in result.solved_board:
            print(" ".join(str(value) for value in row))
