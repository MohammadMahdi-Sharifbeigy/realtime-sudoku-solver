"""Reusable transforms for recognition training and evaluation."""

from __future__ import annotations

from torchvision import transforms


def build_train_transforms(input_size: int) -> transforms.Compose:
    """Build the default training transform pipeline."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def build_eval_transforms(input_size: int) -> transforms.Compose:
    """Build the default evaluation transform pipeline."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
