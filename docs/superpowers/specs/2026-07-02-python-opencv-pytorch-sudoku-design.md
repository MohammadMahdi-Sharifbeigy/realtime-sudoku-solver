# Python OpenCV PyTorch Sudoku Rewrite Design

## Goal

Rewrite the existing Java/OpenCV/DL4J realtime Sudoku solver into a fully Python-based project that satisfies the course brief described in `CV_Project_English.md` by using OpenCV for computer vision, PyTorch with a MobileNet-based classifier for digit recognition, and a reproducible CLI-first pipeline for training, evaluation, image inference, and webcam inference.

## Project Context

The current repository is a Java Gradle project that:

- captures webcam frames in realtime,
- detects a Sudoku grid with OpenCV,
- classifies digits using a small DL4J dense model trained on a local resource dataset,
- solves the board with backtracking,
- overlays the solution on the original frame.

The course brief requires:

- all deliverables in `.py` files,
- reproducible execution with `requirements.txt` or `environment.yml`,
- digit recognition with a CNN or lightweight architecture such as MobileNet,
- OpenCV-based grid extraction,
- use of real Sudoku images for final tests,
- evaluation artifacts such as charts, confusion matrix, and error analysis,
- robust project structure suitable for submission and presentation.

## Scope

The rewrite covers:

- Python-only implementation for all core functionality,
- OpenCV-based Sudoku detection and warping,
- PyTorch MobileNet digit recognizer,
- training pipeline supporting selectable and mixed datasets,
- board solving and solution projection onto the original image,
- CLI entrypoints for both image and webcam usage,
- optional small demo UI after the core pipeline is stable,
- a later extension path toward a full application UI,
- English digits in the first version,
- Persian digit support designed in as a later extension.

The rewrite does not make a full GUI application part of the first delivery. That work is deferred until the core CV and recognition pipeline is validated.

## Recommended Architecture

Use a modular Python package rather than a notebook-first or monolithic script approach.

### Proposed Structure

```text
realtime-sudoku-solver/
  data/
    raw/
    external/
    processed/
  models/
    checkpoints/
    exports/
  reports/
    figures/
    metrics/
    debug/
  scripts/
    train.py
    evaluate.py
    infer_image.py
    infer_webcam.py
    export_model.py
  sudoku_solver/
    __init__.py
    app/
    config/
    cv/
    data/
    recognition/
    solver/
    ui/
    utils/
  tests/
  requirements.txt
  main.py
```

### Module Responsibilities

- `sudoku_solver.cv`
  - frame preprocessing,
  - contour detection,
  - corner ordering,
  - perspective transform,
  - 81-cell extraction,
  - line removal and foreground isolation,
  - solution projection back to source perspective.

- `sudoku_solver.data`
  - dataset adapters for `mnist`, `hoda`/`DigitDB`, and `digit_images`,
  - merged dataset composition,
  - augmentation,
  - label normalization,
  - dataset validation and reporting.

- `sudoku_solver.recognition`
  - MobileNet model construction,
  - training loop,
  - inference wrapper,
  - checkpoint loading,
  - metrics generation,
  - confusion matrix support.

- `sudoku_solver.solver`
  - Sudoku board validation,
  - backtracking solver,
  - handling inconsistent and unsolvable boards.

- `sudoku_solver.app`
  - runtime orchestration for image and webcam sources,
  - CLI argument handling,
  - debug and save-output options.

- `sudoku_solver.ui`
  - optional lightweight demo interface,
  - isolated so core functionality never depends on the UI.

- `scripts`
  - thin wrappers for training, evaluation, inference, and model export workflows.

## Runtime Flow

The core runtime pipeline is:

1. Acquire an image or webcam frame.
2. Preprocess the frame with grayscale conversion and noise reduction.
3. Detect the Sudoku board using a contour-first OpenCV pipeline.
4. Use fallback heuristics only when contour detection is weak.
5. Order the board corners consistently.
6. Warp the board into a top-down square view.
7. Split the warped board into 81 cells.
8. Remove grid lines and isolate the digit foreground.
9. Detect empty cells and prepare non-empty candidates for classification.
10. Classify each candidate with the trained MobileNet recognizer.
11. Build a `9x9` Sudoku board with `0` representing empty cells.
12. Validate the recognized board before solving.
13. Solve the board with backtracking.
14. Project only newly solved digits back onto the original image perspective.
15. Display or save the final result and optional debug artifacts.

## CLI and Runtime Modes

The first release is CLI-first and must support both image and webcam execution.

### Required Modes

- image mode
  - example: `python main.py --image assets/example.jpg`
- webcam mode
  - example: `python main.py --source 0`

### Expected Runtime Options

- input source selection,
- model checkpoint path,
- debug mode,
- save-output path,
- save-intermediate-images toggle,
- confidence thresholds,
- preprocessing profile selection.

The CLI must remain the primary supported interface for reproducibility and grading.

## Model Specification

The digit recognizer will be implemented in PyTorch using a MobileNet-based architecture.

### Model Requirements

- Use `MobileNetV3-Small` or `MobileNetV2` as the base architecture.
- Adapt the network for grayscale cell inputs.
- Use a final 10-class classification head.
- Use `CrossEntropyLoss`.
- Support CPU inference by default.
- Keep export compatibility in mind for later TorchScript or ONNX conversion.

### Canonical Label Space

The unified class schema is:

- `0`: empty cell,
- `1-9`: Sudoku digits.

Board logic must treat `0` as empty and never as a filled Sudoku value.

### Language Scope

- First version: English digits only for the end-to-end runtime.
- Future extension: Persian digit support via a separate dataset adapter and label remapping layer, without requiring architectural changes to the rest of the system.

## Dataset System

The training pipeline must support arbitrary combinations of these sources:

- `mnist`,
- `hoda` / `DigitDB`,
- `digit_images`.

Users must be able to train on:

- one dataset,
- any pair of datasets,
- all three datasets together.

### Dataset Combination Behavior

The configuration layer must support declarations such as:

```yaml
datasets:
  - mnist
```

```yaml
datasets:
  - mnist
  - digit_images
```

```yaml
datasets:
  - mnist
  - hoda
  - digit_images
```

The system should also support optional weighting or sampling controls so small local datasets are not drowned out by large standard datasets.

## Unified Sample Contract

Every dataset adapter must emit samples in the same format:

- grayscale image,
- fixed size, recommended `64x64`,
- tensor-ready numeric label in the canonical class space,
- metadata identifying source dataset and original label.

All transforms used for training and evaluation must be explicit and reproducible.

## Dataset Validation Requirements

Dataset validation is mandatory and must happen before training begins.

### Validation Goals

- Ensure every source maps cleanly into the unified 10-class schema.
- Detect missing classes, invalid labels, ambiguous labels, and inconsistent folder structures.
- Provide auditable reports showing how source labels become canonical labels.
- Prevent silent training on mislabeled or partially broken data.

### Required Validation Stages

1. Validate each dataset source independently.
2. Validate the merged dataset after label remapping and composition.
3. Emit a dataset validation report for each run.

### Required Validation Report Contents

For each selected source, report:

- source name,
- original labels discovered,
- canonical labels after remapping,
- sample count per canonical class,
- missing canonical classes,
- invalid or unmapped labels,
- ignored files and the reason they were ignored.

### Failure Behavior

Training must hard-fail when:

- labels do not map cleanly,
- folders or annotations are ambiguous,
- required classes are missing beyond accepted configuration,
- a selected dataset contains invalid structure that cannot be resolved deterministically.

### Source-Specific Validation Notes

- `mnist`
  - verify labels are numeric,
  - verify the loader documents how source `0` interacts with the canonical `0 = empty` class.

- `hoda` / `DigitDB`
  - validate label IDs and remap them into the canonical class schema,
  - keep the adapter isolated so Persian support can mature later without ripple effects.

- `digit_images`
  - validate folder names or annotation files rigorously,
  - reject non-digit folder names unless explicitly ignored by configuration,
  - detect duplicate aliases and class gaps,
  - treat this as the highest-risk dataset for manual labeling errors.

## Empty Cell Strategy

The old repository inferred emptiness mostly from foreground content. The new design must formalize the empty class.

### Requirements

- train the classifier against explicit empty-cell examples,
- generate or collect empty-cell samples from real Sudoku boards,
- optionally synthesize empty examples using line residue, illumination changes, blur, and slight perspective noise,
- keep empty-cell detection as both:
  - a pre-classification heuristic,
  - and a model-supported class.

This dual approach improves robustness when line removal is imperfect.

## Training and Evaluation Workflow

The training pipeline must be reproducible and configuration-driven.

### Training Outputs

- best checkpoint,
- last checkpoint,
- run configuration snapshot,
- metrics JSON,
- training curves,
- confusion matrix image,
- dataset validation report.

### Evaluation Requirements

- accuracy,
- per-class metrics,
- confusion matrix,
- error analysis on difficult samples,
- validation on selected source datasets,
- explicit validation behavior for `mnist`, `hoda`, and especially `digit_images`.

The project should make it easy to compare:

- single-dataset performance,
- mixed-dataset performance,
- robustness of local digit data against standard datasets.

### Recommended First Baseline

- train on `mnist + digit_images`,
- implement and validate the `hoda` adapter even if it is not used in the first best run,
- establish a clean baseline before attempting wider augmentation or Persian support.

## Computer Vision Pipeline

The OpenCV pipeline should favor reliability and debuggability.

### Board Extraction

Required operations:

- grayscale conversion,
- blur or denoising,
- thresholding and/or edge extraction,
- contour detection,
- largest valid Sudoku-like quadrilateral selection,
- ordered corner extraction,
- perspective transform.

Contour-first detection is preferred because it maps closely to the course goals and the existing repository. Hough-based assistance may be used as a fallback, not as the primary pipeline.

### Cell Processing

For each of the 81 cells:

- crop with stable margins,
- reduce line artifacts,
- isolate the digit foreground,
- estimate whether the cell is empty,
- normalize the cell image for the classifier.

### Error Handling

The runtime must explicitly handle:

- no grid detected,
- invalid corner geometry,
- low-quality warp,
- inconsistent recognized board,
- unsolvable puzzle,
- low-confidence predictions.

These states should produce clear CLI output and optional debug images rather than silent failure.

## Solution Projection

The system must overlay only the missing digits back onto the original image while preserving the original board perspective.

### Requirements

- keep track of originally empty cells,
- solve only after board validation,
- render only inferred solution digits,
- align overlay text to the original quadrilateral geometry,
- preserve perspective so the output satisfies the course bonus requirement for displaying the answer on the original image.

## UI Plan

The UI is phased.

### Phase 1

- CLI-first deliverable,
- optional lightweight demo UI,
- UI must remain optional and isolated from the core pipeline.

Recommended lightweight UI candidates:

- Streamlit,
- Gradio.

### Phase 2

After the core CV and recognition system is validated:

- improve UX,
- add richer debugging views,
- support image upload, webcam controls, result saving, and model selection,
- evolve toward a full application.

The first rewrite must not depend on Phase 2 UI work.

## Testing Strategy

The rewrite should include automated tests where deterministic testing is realistic.

### Required Test Areas

- Sudoku solver correctness,
- board validity checks,
- dataset label remapping,
- dataset validation failure cases,
- merged dataset composition,
- image preprocessing helpers,
- corner ordering and geometry helpers,
- configuration parsing.

### Practical Testing Scope

End-to-end image and webcam quality will still require manual validation with real Sudoku examples, but the reusable logic underneath should be covered by automated tests.

## Reproducibility and Delivery

The repository must include:

- `requirements.txt` or `environment.yml`,
- clear commands for training, evaluation, image inference, and webcam inference,
- stored model weights or instructions to reproduce them,
- `.py` implementation files for every project component,
- saved evaluation artifacts and report-ready figures.

## Non-Goals for the First Rewrite

- full production-grade GUI,
- Persian digit support as a required runtime feature,
- deployment optimization as a mandatory first milestone,
- replacing backtracking with a more advanced solver.

These remain valid future enhancements but should not delay the core rewrite.

## Recommended Implementation Order

1. Establish Python package structure and configuration.
2. Port and improve the Sudoku solver logic.
3. Build dataset adapters and validation reporting.
4. Implement MobileNet training and evaluation pipeline.
5. Implement OpenCV board extraction and cell normalization.
6. Integrate recognition with the solver.
7. Add image and webcam CLI entrypoints.
8. Add debug artifact saving and evaluation utilities.
9. Add optional lightweight UI.
10. Plan the later full-application UI phase.

## Success Criteria

The rewrite is successful when the repository provides:

- an all-Python reproducible project,
- working image and webcam Sudoku solving,
- MobileNet-based digit recognition in PyTorch,
- configurable mixed-dataset training,
- strict dataset validation against the canonical 10-class schema,
- final solution projection back onto the original image,
- course-ready outputs for evaluation and reporting,
- a clear extension path for Persian digits and a fuller application UI.
