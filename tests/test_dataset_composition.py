from pathlib import Path

import pytest

from sudoku_solver.data.base import SampleRecord
from sudoku_solver.data.compose import compose_datasets, load_dataset
from sudoku_solver.data.empty_cells import build_empty_cell_records


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
