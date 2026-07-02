from __future__ import annotations

from pathlib import Path

from sudoku_solver.config.loader import load_config


def test_load_config_uses_defaults_when_path_is_none() -> None:
    config = load_config(None)

    assert config.training.datasets == ["mnist", "digit_images"]
    assert config.training.data_root.name == "data"
    assert config.training.dataset_paths["hoda"].name == "DigitDB"
    assert config.runtime.input_size == 64
    assert config.runtime.device == "cpu"


def test_load_config_merges_override_values(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                "  datasets:",
                "    - hoda",
                "  dataset_paths:",
                "    hoda: custom_hoda",
                "runtime:",
                "  device: cuda",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.training.datasets == ["hoda"]
    assert config.training.dataset_paths["hoda"] == Path("custom_hoda")
    assert config.training.batch_size == 64
    assert config.runtime.device == "cuda"
    assert config.runtime.input_size == 64
