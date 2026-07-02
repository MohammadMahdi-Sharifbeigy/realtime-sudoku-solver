# Realtime Sudoku Solver

Python rewrite of the original project for a computer vision course project.

This repository now targets:

- OpenCV for Sudoku board detection and perspective correction
- PyTorch with a MobileNet-based classifier for digit recognition
- mixed dataset support for `MNIST`, `Hoda`/`DigitDB`, and local `digit_images`
- image and webcam runtime modes
- CLI-first execution with an optional lightweight demo UI scaffold

## Project Status

The repository has been migrated away from the old Java/DL4J implementation.

Current state:

- Python package structure is in place
- Sudoku solver and board validation are implemented
- dataset mapping, validation, and composition are implemented
- MobileNet model scaffold and training dry-run are implemented
- evaluation scaffold and metrics export are implemented
- OpenCV board detection, warping, cell extraction, and overlay scaffolds are implemented
- image and webcam CLI runtime scaffolds are implemented

Current limitation:

- the runtime still uses a placeholder predictor by default
- real checkpoint-backed digit recognition and full dataset-root wiring still need to be completed for end-to-end recognition quality

## Repository Structure

```text
.
├── docs/
├── scripts/
├── sudoku_solver/
│   ├── app/
│   ├── config/
│   ├── cv/
│   ├── data/
│   ├── recognition/
│   ├── solver/
│   └── ui/
├── tests/
├── main.py
└── requirements.txt
```

## Installation

Use Python 3.11+.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Default configuration lives at:

```text
sudoku_solver/config/default.yaml
```

This config currently defines:

- selected datasets
- batch size and worker count
- input image size
- runtime device
- debug flag

## Supported Dataset Modes

The data layer is designed for any combination of:

- `mnist`
- `hoda`
- `digit_images`

Examples:

- `["mnist"]`
- `["digit_images"]`
- `["mnist", "digit_images"]`
- `["mnist", "hoda", "digit_images"]`

The canonical class mapping is:

- `0` = empty cell
- `1-9` = Sudoku digits

## Training

Dry-run training setup:

```bash
python scripts/train.py --config sudoku_solver/config/default.yaml --dry-run
```

This verifies:

- config loading
- dataset composition
- label validation
- MobileNet model creation

Current behavior in this repo:

- dry-run works
- full real training depends on wiring real dataset roots and image records into the adapters

## Evaluation

Run evaluation scaffold:

```bash
python scripts/evaluate.py --config sudoku_solver/config/default.yaml --output-dir reports/evaluation
```

This currently writes:

- `metrics.json`

If a checkpoint path is later wired in:

```bash
python scripts/evaluate.py --config sudoku_solver/config/default.yaml --checkpoint path/to/model.pt --output-dir reports/evaluation
```

## Runtime Usage

Main runtime entrypoint:

```bash
python main.py --help
```

### Image Mode

```bash
python main.py --image path/to/image.jpg --output outputs/result.jpg
```

### Webcam Mode

```bash
python main.py --source 0
```

### Runtime Notes

- `--display` is enabled by default
- use `--no-display` for non-interactive runs
- `--checkpoint` is already exposed in the CLI, but the default runtime predictor is still a placeholder

## Optional Demo UI

A minimal optional demo CLI scaffold exists:

```bash
python -m sudoku_solver.ui.demo --help
```

This is intentionally isolated from the core pipeline.

## Testing

Run the full current test suite:

```bash
python -m pytest tests -v --basetemp .pytest_tmp
```

Current verified status:

- all scaffold tests pass

## Course Alignment

This repo is structured to support the course deliverables in `CV_Project_English.md`, including:

- Python-only implementation
- OpenCV-based grid extraction
- MobileNet/CNN-based recognition path
- dataset validation
- Sudoku solving
- image/webcam execution
- final answer overlay on the source image

## Next Work

Main remaining implementation steps:

1. wire real dataset roots and image loading for `mnist`, `hoda`, and local `digit_images`
2. train a real MobileNet checkpoint
3. replace the placeholder runtime predictor with checkpoint-backed inference
4. improve debug artifact export and reporting outputs
5. optionally expand the demo UI into a fuller application
