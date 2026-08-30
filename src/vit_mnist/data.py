"""MNIST datasets and data loaders for the educational Vision Transformer."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def create_mnist_transform() -> transforms.Compose:
    """Create the standard tensor conversion and normalization for MNIST.

    ``ToTensor`` converts a grayscale PIL image with shape ``[H, W]`` and
    integer pixel values in ``[0, 255]`` into a floating-point tensor with
    shape ``[C, H, W]`` and values in ``[0.0, 1.0]``. For MNIST, the channel
    dimension is one, so one image has shape ``[1, 28, 28]``.

    ``Normalize`` then transforms every pixel value ``x`` using
    ``(x - 0.1307) / 0.3081``. This centers and scales MNIST using its standard
    training-set mean and standard deviation.
    """

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def create_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    data_root: str | Path = Path("data"),
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create separate training and test loaders that return CPU tensors.

    A single dataset image has shape ``[1, 28, 28]``. Each loader returns
    image batches shaped ``[B, 1, 28, 28]`` and labels shaped ``[B]``.
    No device transfer happens here.

    Dataset construction and any requested download happen only when this
    function is called, never when this module is imported.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    transform = create_mnist_transform()

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
