from pathlib import Path

from sudoku_solver.recognition.model import build_model
from sudoku_solver.recognition.train import train_model


def test_build_model_uses_ten_output_classes():
    model = build_model(10)
    assert model.classifier[-1].out_features == 10


def test_train_model_dry_run_returns_without_training():
    class TrainingConfig:
        datasets = ["mnist"]
        batch_size = 4
        num_workers = 0
        data_root = Path("data")
        dataset_paths = {"mnist": Path("data/mnist")}
        val_split = 0.15
        random_seed = 2023
        epochs = 1
        checkpoint_dir = Path("models/checkpoints")

    class RuntimeConfig:
        input_size = 64
        device = "cpu"
        debug = False

    class AppConfig:
        training = TrainingConfig()
        runtime = RuntimeConfig()

    artifacts = train_model(AppConfig(), dry_run=True)

    assert artifacts.dry_run is True
    assert artifacts.train_steps == 0
    assert artifacts.split_sizes["train"] > 0
