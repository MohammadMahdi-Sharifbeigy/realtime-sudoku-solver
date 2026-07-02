"""Hoda/DigitDB adapter with native CDB parsing support."""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

from .base import SampleRecord
from .data_utils import build_records_from_arrays, scan_labeled_image_tree
from .mnist import DatasetAdapter


def load_hoda_dataset(
    split: str,
    *,
    root: str | Path | None = None,
    records: list[SampleRecord] | None = None,
    val_split: float = 0.15,
    random_seed: int = 2023,
) -> DatasetAdapter:
    """Return a concrete Hoda adapter."""
    resolved_root = Path(root) if root is not None else None
    if records is not None:
        dataset_records = list(records)
    else:
        dataset_records = _load_hoda_records(
            split,
            resolved_root,
            val_split=val_split,
            random_seed=random_seed,
        )
    return DatasetAdapter(name="hoda", split=split, records=dataset_records)


def _convert_to_one_hot(vector: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros(shape=[len(vector), num_classes])
    result[np.arange(len(vector)), vector] = 1
    return result


def _resize_image(
    src_image: np.ndarray,
    dst_image_height: int,
    dst_image_width: int,
) -> np.ndarray:
    src_image_height = src_image.shape[0]
    src_image_width = src_image.shape[1]

    if src_image_height > dst_image_height or src_image_width > dst_image_width:
        height_scale = dst_image_height / src_image_height
        width_scale = dst_image_width / src_image_width
        scale = min(height_scale, width_scale)
        img = cv2.resize(
            src=src_image,
            dsize=(0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = src_image

    img_height = img.shape[0]
    img_width = img.shape[1]
    dst_image = np.zeros(shape=[dst_image_height, dst_image_width], dtype=np.uint8)

    y_offset = (dst_image_height - img_height) // 2
    x_offset = (dst_image_width - img_width) // 2
    dst_image[y_offset : y_offset + img_height, x_offset : x_offset + img_width] = img
    return dst_image


def read_hoda_cdb(file_name: str) -> tuple[list[np.ndarray], list[int]]:
    with open(file_name, "rb") as binary_file:
        data = binary_file.read()
        offset = 0

        struct.unpack_from("H", data, offset)[0]
        offset += 2
        struct.unpack_from("B", data, offset)[0]
        offset += 1
        struct.unpack_from("B", data, offset)[0]
        offset += 1
        height = struct.unpack_from("B", data, offset)[0]
        offset += 1
        width = struct.unpack_from("B", data, offset)[0]
        offset += 1
        total_records = struct.unpack_from("I", data, offset)[0]
        offset += 4
        struct.unpack_from("128I", data, offset)
        offset += 128 * 4
        img_type = struct.unpack_from("B", data, offset)[0]
        offset += 1
        struct.unpack_from("256c", data, offset)
        offset += 256
        struct.unpack_from("245c", data, offset)
        offset += 245

        normal = bool((width > 0) and (height > 0))
        images: list[np.ndarray] = []
        labels: list[int] = []

        for _ in range(total_records):
            struct.unpack_from("B", data, offset)[0]
            offset += 1
            label = struct.unpack_from("B", data, offset)[0]
            offset += 1

            if not normal:
                width = struct.unpack_from("B", data, offset)[0]
                offset += 1
                height = struct.unpack_from("B", data, offset)[0]
                offset += 1

            struct.unpack_from("H", data, offset)[0]
            offset += 2
            image = np.zeros(shape=[height, width], dtype=np.uint8)

            if img_type == 0:
                for y_index in range(height):
                    white = True
                    counter = 0
                    while counter < width:
                        wb_count = struct.unpack_from("B", data, offset)[0]
                        offset += 1
                        image[y_index, counter : counter + wb_count] = 0 if white else 255
                        white = not white
                        counter += wb_count
            else:
                data_chunk = struct.unpack_from(f"{width * height}B", data, offset)
                offset += width * height
                image = np.asarray(data_chunk, dtype=np.uint8).reshape([width, height]).T

            images.append(image)
            labels.append(label)

        return images, labels


def read_hoda_dataset(
    dataset_path: str,
    images_height: int = 28,
    images_width: int = 28,
    one_hot: bool = False,
    reshape: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    images, labels = read_hoda_cdb(dataset_path)
    if len(images) != len(labels):
        raise ValueError("Hoda images and labels length mismatch")

    x_values = np.zeros(shape=[len(images), images_height, images_width], dtype=np.float32)
    y_values = np.zeros(shape=[len(labels)], dtype=int)

    for index, image in enumerate(images):
        resized = _resize_image(
            src_image=image,
            dst_image_height=images_height,
            dst_image_width=images_width,
        )
        x_values[index] = resized / 255.0
        y_values[index] = labels[index]

    if one_hot:
        y_result = _convert_to_one_hot(y_values, 10).astype(np.float32)
    else:
        y_result = y_values.astype(np.int64)

    if reshape:
        x_result = x_values.reshape(-1, images_height * images_width)
    else:
        x_result = x_values.reshape(-1, 1, images_height, images_width)

    return x_result, y_result


def _load_hoda_records(
    split: str,
    root: Path | None,
    *,
    val_split: float,
    random_seed: int,
) -> list[SampleRecord]:
    if root is None:
        return []

    split_root = root / split
    if split_root.exists():
        return scan_labeled_image_tree(split_root, "hoda")

    train_path = root / "Train 60000.cdb"
    test_path = root / "Test 20000.cdb"
    if not train_path.exists() or not test_path.exists():
        nested_root = root / "DigitDB"
        train_path = nested_root / "Train 60000.cdb"
        test_path = nested_root / "Test 20000.cdb"
        if not train_path.exists() or not test_path.exists():
            return []

    x_train, y_train = read_hoda_dataset(str(train_path), reshape=False)
    x_test, y_test = read_hoda_dataset(str(test_path), reshape=False)

    train_mask = y_train != 0
    test_mask = y_test != 0
    x_train = x_train[train_mask]
    y_train = y_train[train_mask]
    x_test = x_test[test_mask]
    y_test = y_test[test_mask]

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_split))
    val_indices = indices[:val_count]
    fit_indices = indices[val_count:]

    normalized_split = split.strip().lower()
    if normalized_split == "train":
        return build_records_from_arrays(
            source="hoda",
            images=x_train[fit_indices],
            labels=y_train[fit_indices],
        )
    if normalized_split == "val":
        return build_records_from_arrays(
            source="hoda",
            images=x_train[val_indices],
            labels=y_train[val_indices],
        )
    if normalized_split == "test":
        return build_records_from_arrays(
            source="hoda",
            images=x_test,
            labels=y_test,
        )
    raise ValueError(f"Unsupported split: {split}")
