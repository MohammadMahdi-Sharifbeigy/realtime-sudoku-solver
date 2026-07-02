from pathlib import Path

import numpy as np
import pytest

from sudoku_solver.data.base import SampleRecord
from sudoku_solver.data.compose import compose_datasets, load_dataset
from sudoku_solver.data.empty_cells import build_empty_cell_records
from sudoku_solver.data.hoda import load_hoda_dataset


def test_compose_datasets_preserves_requested_order():
    dataset = compose_datasets(
        ["mnist", "digit_images"],
        "train",
        record_overrides={
            "mnist": [SampleRecord(source="mnist", raw_label=1, path="mnist-1.png")],
            "digit_images": [
                SampleRecord(
                    source="digit_images",
                    raw_label=7,
                    path="digit-images-7.png",
                )
            ],
        },
    )

    assert dataset.source_names == ["mnist", "digit_images"]


def test_compose_datasets_flattens_records_in_source_order():
    dataset = compose_datasets(
        ["hoda", "mnist"],
        "train",
        record_overrides={
            "hoda": [
                SampleRecord(source="hoda", raw_label=2, path="hoda-2-a.png"),
                SampleRecord(source="hoda", raw_label=3, path="hoda-3-b.png"),
            ],
            "mnist": [SampleRecord(source="mnist", raw_label=4, path="mnist-4.png")],
        },
    )

    assert [record.source for record in dataset.records] == ["hoda", "hoda", "mnist"]
    assert len(dataset) == 3


def test_load_dataset_digit_images_scans_split_folder(tmp_path: Path):
    train_root = tmp_path / "train" / "8"
    train_root.mkdir(parents=True)
    image_path = train_root / "cell.png"
    image_path.write_bytes(b"fake")

    dataset = load_dataset("digit_images", "train", root=tmp_path)

    assert len(dataset.records) == 1
    assert dataset.records[0].source == "digit_images"
    assert dataset.records[0].raw_label == "8"
    assert dataset.records[0].path == image_path


def test_build_empty_cell_records_assigns_canonical_zero_label():
    records = build_empty_cell_records(["empty/a.png", "empty/b.png"])

    assert [record.raw_label for record in records] == [0, 0]
    assert all(record.source == "digit_images" for record in records)
    assert records[0].path == Path("empty/a.png")


def test_load_dataset_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unsupported dataset name"):
        load_dataset("unknown", "train")


def test_load_dataset_digit_images_auto_splits_flat_tree(tmp_path: Path):
    for label in ("1", "2"):
        label_dir = tmp_path / label
        label_dir.mkdir(parents=True)
        for index in range(4):
            (label_dir / f"{index}.png").write_bytes(b"fake")

    dataset = load_dataset("digit_images", "train", root=tmp_path, val_split=0.25)

    assert len(dataset.records) > 0
    assert all(record.source == "digit_images" for record in dataset.records)


def test_load_hoda_dataset_reads_cdb_from_digitdb_folder(monkeypatch, tmp_path: Path):
    digitdb_root = tmp_path / "DigitDB"
    digitdb_root.mkdir(parents=True)
    (digitdb_root / "Train 60000.cdb").write_bytes(b"train")
    (digitdb_root / "Test 20000.cdb").write_bytes(b"test")

    def fake_read_hoda_dataset(
        path: str,
        images_height: int = 28,
        images_width: int = 28,
        one_hot: bool = False,
        reshape: bool = False,
    ):
        del images_height, images_width, one_hot, reshape
        if "Train" in path:
            x_values = np.zeros((4, 1, 28, 28), dtype=np.float32)
            y_values = np.array([1, 2, 3, 4], dtype=np.int64)
        else:
            x_values = np.zeros((2, 1, 28, 28), dtype=np.float32)
            y_values = np.array([5, 6], dtype=np.int64)
        return x_values, y_values

    monkeypatch.setattr("sudoku_solver.data.hoda.read_hoda_dataset", fake_read_hoda_dataset)

    dataset = load_hoda_dataset("train", root=digitdb_root, val_split=0.25, random_seed=7)

    assert len(dataset.records) == 3
    assert all(record.source == "hoda" for record in dataset.records)
