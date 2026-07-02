# Python OpenCV PyTorch Sudoku Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Java/OpenCV/DL4J Sudoku solver with a reproducible Python/OpenCV/PyTorch project that supports mixed dataset training, MobileNet digit recognition, image and webcam inference, and solution overlay on the original board.

**Architecture:** Build a modular Python package named `sudoku_solver` with isolated modules for CV, datasets, recognition, solver logic, runtime orchestration, and optional UI. Deliver the rewrite in stages so each stage has passing tests and a usable artifact before the next integration point.

**Tech Stack:** Python 3.11+, OpenCV, PyTorch, torchvision, NumPy, PyYAML, matplotlib, scikit-learn, pytest, optional Streamlit or Gradio

## Global Constraints

- All required project functionality must be implemented in `.py` files.
- The first release must support both image mode and webcam mode.
- The classifier must use a MobileNet-based PyTorch architecture with a 10-class output.
- Canonical labels must be exactly `0-9`, where `0` means empty cell and `1-9` mean Sudoku digits.
- The dataset layer must support any combination of `mnist`, `hoda`, and `digit_images`.
- Training must hard-fail if dataset labels do not map cleanly into the canonical 10-class schema.
- The first end-to-end version targets English digits only.
- Persian digit support must remain possible through isolated dataset adapters and remapping logic.
- The core deliverable is CLI-first and reproducible with `requirements.txt` or `environment.yml`.
- The full application UI is out of scope for the first rewrite; only an optional lightweight demo UI may be added near the end.

---

## File Structure

- Create: `requirements.txt`
- Create: `main.py`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `scripts/infer_image.py`
- Create: `scripts/infer_webcam.py`
- Create: `scripts/export_model.py`
- Create: `sudoku_solver/__init__.py`
- Create: `sudoku_solver/config/default.yaml`
- Create: `sudoku_solver/config/types.py`
- Create: `sudoku_solver/config/loader.py`
- Create: `sudoku_solver/solver/board.py`
- Create: `sudoku_solver/solver/backtracking.py`
- Create: `sudoku_solver/data/base.py`
- Create: `sudoku_solver/data/labels.py`
- Create: `sudoku_solver/data/mnist.py`
- Create: `sudoku_solver/data/hoda.py`
- Create: `sudoku_solver/data/digit_images.py`
- Create: `sudoku_solver/data/compose.py`
- Create: `sudoku_solver/data/validate.py`
- Create: `sudoku_solver/data/empty_cells.py`
- Create: `sudoku_solver/recognition/model.py`
- Create: `sudoku_solver/recognition/transforms.py`
- Create: `sudoku_solver/recognition/train.py`
- Create: `sudoku_solver/recognition/evaluate.py`
- Create: `sudoku_solver/recognition/predict.py`
- Create: `sudoku_solver/cv/preprocess.py`
- Create: `sudoku_solver/cv/board_detect.py`
- Create: `sudoku_solver/cv/warp.py`
- Create: `sudoku_solver/cv/cells.py`
- Create: `sudoku_solver/cv/overlay.py`
- Create: `sudoku_solver/app/pipeline.py`
- Create: `sudoku_solver/app/runtime.py`
- Create: `sudoku_solver/ui/demo.py`
- Create: `tests/test_board_solver.py`
- Create: `tests/test_board_validation.py`
- Create: `tests/test_label_mapping.py`
- Create: `tests/test_dataset_validation.py`
- Create: `tests/test_dataset_composition.py`
- Create: `tests/test_config_loader.py`
- Create: `tests/test_geometry_helpers.py`

## Task 1: Bootstrap Python Package and Configuration

**Files:**
- Create: `requirements.txt`
- Create: `main.py`
- Create: `sudoku_solver/__init__.py`
- Create: `sudoku_solver/config/default.yaml`
- Create: `sudoku_solver/config/types.py`
- Create: `sudoku_solver/config/loader.py`
- Test: `tests/test_config_loader.py`

**Interfaces:**
- Consumes: none
- Produces: `load_config(path: str | None) -> AppConfig`, `AppConfig`, `TrainingConfig`, `RuntimeConfig`

- [ ] **Step 1: Write the failing config test**

```python
from sudoku_solver.config.loader import load_config


def test_load_config_uses_defaults_when_path_is_none():
    config = load_config(None)
    assert config.training.datasets == ["mnist", "digit_images"]
    assert config.runtime.input_size == 64
    assert config.runtime.device == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_loader.py::test_load_config_uses_defaults_when_path_is_none -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol error

- [ ] **Step 3: Write minimal configuration implementation**

```python
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    datasets: list[str]


@dataclass
class RuntimeConfig:
    input_size: int
    device: str


@dataclass
class AppConfig:
    training: TrainingConfig
    runtime: RuntimeConfig
```

```python
from pathlib import Path
import yaml

from .types import AppConfig, RuntimeConfig, TrainingConfig


def load_config(path: str | None) -> AppConfig:
    raw = yaml.safe_load(Path("sudoku_solver/config/default.yaml").read_text())
    return AppConfig(
        training=TrainingConfig(datasets=raw["training"]["datasets"]),
        runtime=RuntimeConfig(
            input_size=raw["runtime"]["input_size"],
            device=raw["runtime"]["device"],
        ),
    )
```

- [ ] **Step 4: Add minimal defaults file**

```yaml
training:
  datasets: ["mnist", "digit_images"]

runtime:
  input_size: 64
  device: "cpu"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config_loader.py::test_load_config_uses_defaults_when_path_is_none -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt main.py sudoku_solver tests
git commit -m "feat: bootstrap python package and config loader"
```

## Task 2: Port Sudoku Board Validation and Backtracking Solver

**Files:**
- Create: `sudoku_solver/solver/board.py`
- Create: `sudoku_solver/solver/backtracking.py`
- Test: `tests/test_board_solver.py`
- Test: `tests/test_board_validation.py`

**Interfaces:**
- Consumes: `list[list[int]]` boards with canonical values `0-9`
- Produces: `is_valid_board(board: list[list[int]]) -> bool`, `solve_board(board: list[list[int]]) -> list[list[int]] | None`

- [ ] **Step 1: Write failing solver tests**

```python
from sudoku_solver.solver.backtracking import solve_board


def test_solve_board_solves_known_puzzle():
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    solved = solve_board(puzzle)
    assert solved is not None
    assert solved[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_board_solver.py tests/test_board_validation.py -v`
Expected: FAIL with missing solver module

- [ ] **Step 3: Implement validation helpers and solver**

```python
def is_valid_board(board: list[list[int]]) -> bool:
    # Validate shape, range, rows, columns, and 3x3 subgrids.
    ...


def solve_board(board: list[list[int]]) -> list[list[int]] | None:
    # Return a solved copy or None if unsolvable.
    ...
```

- [ ] **Step 4: Add unsolvable-board test**

```python
from sudoku_solver.solver.backtracking import solve_board


def test_solve_board_returns_none_for_invalid_puzzle():
    puzzle = [[1] * 9 for _ in range(9)]
    assert solve_board(puzzle) is None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_board_solver.py tests/test_board_validation.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sudoku_solver/solver tests/test_board_solver.py tests/test_board_validation.py
git commit -m "feat: port sudoku validation and solver"
```

## Task 3: Implement Canonical Label Mapping and Dataset Validation

**Files:**
- Create: `sudoku_solver/data/base.py`
- Create: `sudoku_solver/data/labels.py`
- Create: `sudoku_solver/data/validate.py`
- Test: `tests/test_label_mapping.py`
- Test: `tests/test_dataset_validation.py`

**Interfaces:**
- Consumes: dataset records with source labels and source names
- Produces: `map_label(source: str, raw_label: str | int) -> int`, `validate_records(source: str, records: list[SampleRecord]) -> ValidationReport`

- [ ] **Step 1: Write failing label mapping test**

```python
from sudoku_solver.data.labels import map_label


def test_map_label_accepts_digit_images_numeric_folder_names():
    assert map_label("digit_images", "7") == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_label_mapping.py tests/test_dataset_validation.py -v`
Expected: FAIL with missing data label module

- [ ] **Step 3: Implement canonical label mapping**

```python
def map_label(source: str, raw_label: str | int) -> int:
    normalized = int(raw_label)
    if normalized < 0 or normalized > 9:
        raise ValueError(f"Unsupported label: {raw_label}")
    return normalized
```

- [ ] **Step 4: Implement validation report shape**

```python
from dataclasses import dataclass


@dataclass
class ValidationReport:
    source: str
    discovered_labels: list[str]
    canonical_counts: dict[int, int]
    invalid_labels: list[str]
    ignored_files: list[str]
```

- [ ] **Step 5: Add hard-fail validation test**

```python
import pytest

from sudoku_solver.data.validate import validate_records


def test_validate_records_rejects_invalid_digit_images_label():
    records = [{"label": "eleven", "path": "bad.png"}]
    with pytest.raises(ValueError):
        validate_records("digit_images", records)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_label_mapping.py tests/test_dataset_validation.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add sudoku_solver/data tests/test_label_mapping.py tests/test_dataset_validation.py
git commit -m "feat: add canonical label mapping and dataset validation"
```

## Task 4: Add Dataset Adapters and Mixed Dataset Composition

**Files:**
- Create: `sudoku_solver/data/mnist.py`
- Create: `sudoku_solver/data/hoda.py`
- Create: `sudoku_solver/data/digit_images.py`
- Create: `sudoku_solver/data/compose.py`
- Create: `sudoku_solver/data/empty_cells.py`
- Test: `tests/test_dataset_composition.py`

**Interfaces:**
- Consumes: config-selected dataset names and paths
- Produces: `load_dataset(name: str, split: str) -> Dataset`, `compose_datasets(names: list[str], split: str) -> Dataset`

- [ ] **Step 1: Write failing composition test**

```python
from sudoku_solver.data.compose import compose_datasets


def test_compose_datasets_preserves_requested_order():
    dataset = compose_datasets(["mnist", "digit_images"], "train")
    assert dataset.source_names == ["mnist", "digit_images"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_composition.py::test_compose_datasets_preserves_requested_order -v`
Expected: FAIL with missing compose module

- [ ] **Step 3: Implement dataset adapters with shared sample contract**

```python
class SampleRecord(TypedDict):
    image: object
    label: int
    source: str
    raw_label: str | int
```

```python
def compose_datasets(names: list[str], split: str):
    datasets = [load_dataset(name, split) for name in names]
    return CombinedDataset(datasets)
```

- [ ] **Step 4: Add explicit empty-cell dataset helper**

```python
def build_empty_cell_records(paths: list[str]) -> list[SampleRecord]:
    return [{"image": path, "label": 0, "source": "empty_cells", "raw_label": 0} for path in paths]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_dataset_composition.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sudoku_solver/data tests/test_dataset_composition.py
git commit -m "feat: add dataset adapters and composition"
```

## Task 5: Build MobileNet Model, Transforms, and Training Loop

**Files:**
- Create: `sudoku_solver/recognition/model.py`
- Create: `sudoku_solver/recognition/transforms.py`
- Create: `sudoku_solver/recognition/train.py`
- Create: `scripts/train.py`

**Interfaces:**
- Consumes: composed training dataset and `AppConfig`
- Produces: `build_model(num_classes: int = 10) -> torch.nn.Module`, `train_model(config: AppConfig) -> TrainingArtifacts`

- [ ] **Step 1: Write failing model-head test**

```python
from sudoku_solver.recognition.model import build_model


def test_build_model_uses_ten_output_classes():
    model = build_model(10)
    assert model.classifier[-1].out_features == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_head.py::test_build_model_uses_ten_output_classes -v`
Expected: FAIL with missing module

- [ ] **Step 3: Implement MobileNet builder**

```python
from torchvision.models import mobilenet_v3_small
import torch.nn as nn


def build_model(num_classes: int = 10):
    model = mobilenet_v3_small(weights=None)
    model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
```

- [ ] **Step 4: Implement minimal training entrypoint**

```python
def train_model(config):
    # Load datasets, build loaders, train model, save checkpoints and metrics.
    ...
```

- [ ] **Step 5: Run targeted test and a dry training smoke check**

Run: `pytest tests/test_model_head.py::test_build_model_uses_ten_output_classes -v`
Expected: PASS

Run: `python scripts/train.py --config sudoku_solver/config/default.yaml --dry-run`
Expected: exits successfully after dataset/config/model setup

- [ ] **Step 6: Commit**

```bash
git add sudoku_solver/recognition scripts/train.py tests/test_model_head.py
git commit -m "feat: add mobilenet recognizer and training entrypoint"
```

## Task 6: Add Evaluation Pipeline and Dataset Validation Reports

**Files:**
- Create: `sudoku_solver/recognition/evaluate.py`
- Create: `scripts/evaluate.py`
- Modify: `sudoku_solver/data/validate.py`

**Interfaces:**
- Consumes: trained checkpoint, validation dataset, validation reports
- Produces: `evaluate_model(...) -> dict[str, object]`, metrics JSON, confusion matrix PNG, validation report JSON

- [ ] **Step 1: Write failing evaluation artifact test**

```python
from sudoku_solver.recognition.evaluate import evaluate_model


def test_evaluate_model_returns_accuracy_and_confusion_matrix():
    results = evaluate_model(model=None, dataloader=None, class_names=[str(i) for i in range(10)])
    assert "accuracy" in results
    assert "confusion_matrix" in results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation_outputs.py::test_evaluate_model_returns_accuracy_and_confusion_matrix -v`
Expected: FAIL with missing module

- [ ] **Step 3: Implement minimal evaluation result structure**

```python
def evaluate_model(model, dataloader, class_names):
    return {"accuracy": 0.0, "confusion_matrix": [[0] * len(class_names) for _ in class_names]}
```

- [ ] **Step 4: Add artifact serialization in script**

```python
def main():
    # Load checkpoint, run evaluation, write metrics.json and confusion_matrix.png.
    ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_evaluation_outputs.py::test_evaluate_model_returns_accuracy_and_confusion_matrix -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sudoku_solver/recognition/evaluate.py scripts/evaluate.py tests/test_evaluation_outputs.py
git commit -m "feat: add evaluation and reporting pipeline"
```

## Task 7: Implement OpenCV Board Detection, Warping, and Cell Extraction

**Files:**
- Create: `sudoku_solver/cv/preprocess.py`
- Create: `sudoku_solver/cv/board_detect.py`
- Create: `sudoku_solver/cv/warp.py`
- Create: `sudoku_solver/cv/cells.py`
- Test: `tests/test_geometry_helpers.py`

**Interfaces:**
- Consumes: raw `numpy.ndarray` image frames
- Produces: `detect_board(image) -> BoardDetection | None`, `warp_board(image, corners) -> np.ndarray`, `extract_cells(board_image) -> list[np.ndarray]`

- [ ] **Step 1: Write failing corner-order test**

```python
import numpy as np

from sudoku_solver.cv.warp import order_corners


def test_order_corners_returns_top_left_first():
    corners = np.array([[200, 50], [50, 50], [50, 200], [200, 200]], dtype=np.float32)
    ordered = order_corners(corners)
    assert ordered[0].tolist() == [50, 50]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_geometry_helpers.py::test_order_corners_returns_top_left_first -v`
Expected: FAIL with missing geometry helper

- [ ] **Step 3: Implement preprocessing and geometry helpers**

```python
def order_corners(corners):
    # Sort points into top-left, top-right, bottom-right, bottom-left order.
    ...
```

```python
def detect_board(image):
    # Use grayscale, blur, threshold/edges, contours, polygon approximation, and geometry filters.
    ...
```

- [ ] **Step 4: Implement cell extraction contract**

```python
def extract_cells(board_image):
    # Return 81 normalized cell crops in row-major order.
    ...
```

- [ ] **Step 5: Run geometry tests**

Run: `pytest tests/test_geometry_helpers.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sudoku_solver/cv tests/test_geometry_helpers.py
git commit -m "feat: add board detection and cell extraction"
```

## Task 8: Add Digit Isolation, Prediction Wrapper, and Overlay Projection

**Files:**
- Create: `sudoku_solver/recognition/predict.py`
- Create: `sudoku_solver/cv/overlay.py`
- Modify: `sudoku_solver/cv/cells.py`
- Modify: `sudoku_solver/solver/board.py`

**Interfaces:**
- Consumes: extracted cells, trained model checkpoint, detected board geometry
- Produces: `predict_board(cells, predictor) -> list[list[int]]`, `overlay_solution(image, corners, original_board, solved_board) -> np.ndarray`

- [ ] **Step 1: Write failing overlay test**

```python
import numpy as np

from sudoku_solver.cv.overlay import should_render_digit


def test_should_render_digit_only_for_originally_empty_cells():
    assert should_render_digit(original_value=0, solved_value=9) is True
    assert should_render_digit(original_value=5, solved_value=5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_overlay_rules.py::test_should_render_digit_only_for_originally_empty_cells -v`
Expected: FAIL with missing overlay module

- [ ] **Step 3: Implement predictor wrapper**

```python
def predict_board(cells, predictor):
    # Apply empty-cell heuristic, run model on remaining cells, return 9x9 board.
    ...
```

- [ ] **Step 4: Implement overlay rule and projection**

```python
def should_render_digit(original_value: int, solved_value: int) -> bool:
    return original_value == 0 and solved_value != 0
```

```python
def overlay_solution(image, corners, original_board, solved_board):
    # Project solved digits onto original image perspective.
    ...
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_overlay_rules.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sudoku_solver/cv/overlay.py sudoku_solver/recognition/predict.py tests/test_overlay_rules.py
git commit -m "feat: add prediction wrapper and overlay projection"
```

## Task 9: Integrate End-to-End Image and Webcam Runtime

**Files:**
- Create: `sudoku_solver/app/pipeline.py`
- Create: `sudoku_solver/app/runtime.py`
- Modify: `main.py`
- Create: `scripts/infer_image.py`
- Create: `scripts/infer_webcam.py`

**Interfaces:**
- Consumes: config, checkpoint path, input image or webcam source
- Produces: `run_image_pipeline(...) -> PipelineResult`, `run_webcam_pipeline(...) -> int`

- [ ] **Step 1: Write failing pipeline smoke test**

```python
from sudoku_solver.app.pipeline import build_runtime_mode


def test_build_runtime_mode_returns_image_for_image_argument():
    mode = build_runtime_mode(image_path="sample.jpg", source=None)
    assert mode == "image"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_runtime_mode.py::test_build_runtime_mode_returns_image_for_image_argument -v`
Expected: FAIL with missing runtime pipeline

- [ ] **Step 3: Implement runtime mode selection and CLI parsing**

```python
def build_runtime_mode(image_path: str | None, source: int | None) -> str:
    if image_path:
        return "image"
    return "webcam"
```

```python
def main():
    # Parse args, load config, dispatch to image or webcam runner.
    ...
```

- [ ] **Step 4: Implement end-to-end pipeline orchestration**

```python
def run_image_pipeline(...):
    # Detect board, classify digits, solve, overlay result, and save/display outputs.
    ...
```

```python
def run_webcam_pipeline(...):
    # Repeatedly process frames until quit.
    ...
```

- [ ] **Step 5: Run targeted tests and a local smoke command**

Run: `pytest tests/test_runtime_mode.py -v`
Expected: PASS

Run: `python main.py --help`
Expected: shows image and webcam options

- [ ] **Step 6: Commit**

```bash
git add main.py sudoku_solver/app scripts/infer_image.py scripts/infer_webcam.py tests/test_runtime_mode.py
git commit -m "feat: integrate image and webcam runtime"
```

## Task 10: Add Debug Artifacts, Failure Reporting, and Optional Demo UI

**Files:**
- Create: `sudoku_solver/ui/demo.py`
- Modify: `sudoku_solver/app/pipeline.py`
- Modify: `scripts/evaluate.py`

**Interfaces:**
- Consumes: pipeline results and evaluation outputs
- Produces: debug images, metrics files, optional demo UI entrypoint

- [ ] **Step 1: Write failing debug-path test**

```python
from pathlib import Path

from sudoku_solver.app.pipeline import build_debug_paths


def test_build_debug_paths_creates_reports_debug_directory(tmp_path: Path):
    paths = build_debug_paths(tmp_path)
    assert paths.root.name == "debug"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_debug_paths.py::test_build_debug_paths_creates_reports_debug_directory -v`
Expected: FAIL with missing helper

- [ ] **Step 3: Implement debug artifact writing**

```python
def build_debug_paths(root):
    # Create debug output directories for warped boards, cells, overlays, and failures.
    ...
```

- [ ] **Step 4: Add optional demo UI wrapper**

```python
def launch_demo():
    # Thin wrapper over image pipeline, intentionally separate from core runtime.
    ...
```

- [ ] **Step 5: Run tests and demo help command**

Run: `pytest tests/test_debug_paths.py -v`
Expected: PASS

Run: `python -m sudoku_solver.ui.demo --help`
Expected: exits successfully with demo options

- [ ] **Step 6: Commit**

```bash
git add sudoku_solver/ui/demo.py sudoku_solver/app/pipeline.py tests/test_debug_paths.py
git commit -m "feat: add debug artifacts and optional demo ui"
```

## Task 11: Final Documentation and Reproducibility Pass

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `sudoku_solver/config/default.yaml`

**Interfaces:**
- Consumes: finished scripts and package entrypoints
- Produces: final run instructions for training, evaluation, image inference, webcam inference, and optional demo UI

- [ ] **Step 1: Write README acceptance checklist**

```markdown
- install dependencies
- validate datasets
- train model
- evaluate model
- run image inference
- run webcam inference
- launch optional demo ui
```

- [ ] **Step 2: Update README with exact commands**

```bash
python scripts/train.py --config sudoku_solver/config/default.yaml
python scripts/evaluate.py --checkpoint models/checkpoints/best.pt
python main.py --image assets/example.jpg --checkpoint models/checkpoints/best.pt
python main.py --source 0 --checkpoint models/checkpoints/best.pt
```

- [ ] **Step 3: Run final automated test suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 4: Run final type and import smoke checks**

Run: `python main.py --help`
Expected: PASS

Run: `python scripts/train.py --config sudoku_solver/config/default.yaml --dry-run`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt sudoku_solver/config/default.yaml
git commit -m "docs: finalize reproducible python sudoku workflow"
```

## Self-Review

- Spec coverage:
  - Python rewrite: covered by Tasks 1-11
  - MobileNet classifier: covered by Task 5
  - mixed datasets: covered by Tasks 3-4
  - strict label validation: covered by Tasks 3 and 6
  - OpenCV board extraction: covered by Tasks 7-8
  - solver and overlay: covered by Tasks 2 and 8
  - image and webcam support: covered by Task 9
  - optional UI now, fuller app later: covered by Task 10
  - reproducibility and report artifacts: covered by Tasks 6, 10, and 11

- Placeholder scan:
  - Remaining `...` blocks in code snippets are intentional markers for implementation bodies inside the plan and should be replaced with concrete code during execution. If stricter execution is desired, expand these per task before implementation starts.

- Type consistency:
  - Canonical board type remains `list[list[int]]`
  - Canonical class count remains `10`
  - Runtime always dispatches through `run_image_pipeline` and `run_webcam_pipeline`
  - Dataset validation always returns `ValidationReport`
