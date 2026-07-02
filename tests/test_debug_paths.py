from pathlib import Path

import pytest

from sudoku_solver.app.pipeline import build_debug_paths


def test_build_debug_paths_creates_reports_debug_directory(tmp_path: Path):
    paths = build_debug_paths(tmp_path)

    assert paths.root.name == "debug"
    assert paths.reports.is_dir()


def test_build_debug_paths_creates_all_expected_subdirectories(tmp_path: Path):
    paths = build_debug_paths(tmp_path)

    assert paths.warped_boards.is_dir()
    assert paths.cells.is_dir()
    assert paths.overlays.is_dir()
    assert paths.failures.is_dir()
    assert paths.reports.is_dir()


def test_build_debug_paths_builds_artifact_paths_under_debug_root(tmp_path: Path):
    paths = build_debug_paths(tmp_path)

    overlay_path = paths.image_path("overlays", "frame_001")
    report_path = paths.report_path("metrics")

    assert overlay_path == paths.overlays / "frame_001.png"
    assert report_path == paths.reports / "metrics.json"
    assert overlay_path.parent == paths.overlays
    assert report_path.parent == paths.reports


def test_build_debug_paths_rejects_unknown_category(tmp_path: Path):
    paths = build_debug_paths(tmp_path)

    with pytest.raises(ValueError):
        paths.image_path("unknown", "artifact")
