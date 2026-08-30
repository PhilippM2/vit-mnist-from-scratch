"""Train and evaluate the educational Vision Transformer on MNIST."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from vit_mnist.data import create_mnist_loaders  # noqa: E402
from vit_mnist.model import VisionTransformer  # noqa: E402


BATCH_SIZE = 128
NORMAL_EPOCHS = 5
TINY_SUBSET_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 0
SEED = 0
TINY_SUBSET_SIZE = 128


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch so repeated CPU runs are reproducible."""

    random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one pass over the loader and return loss and accuracy."""

    model.train()

    summed_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.shape[0]
        summed_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += batch_size

    if total_predictions == 0:
        raise ValueError("train_loader must contain at least one example")

    average_loss = summed_loss / total_predictions
    training_accuracy = correct_predictions / total_predictions
    return average_loss, training_accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return classification accuracy without computing gradients or updates."""

    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.shape[0]

    if total_predictions == 0:
        raise ValueError("test_loader must contain at least one example")

    return correct_predictions / total_predictions


def create_tiny_subset_loader(
    train_loader: DataLoader,
    subset_size: int = TINY_SUBSET_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> DataLoader:
    """Create a reproducibly shuffled loader over fixed leading examples."""

    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if subset_size > len(train_loader.dataset):
        raise ValueError(
            "subset_size cannot exceed the number of training examples"
        )

    fixed_indices = list(range(subset_size))
    tiny_dataset = Subset(train_loader.dataset, fixed_indices)
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        tiny_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )


def save_model_checkpoint(model: nn.Module, path: str | Path) -> None:
    """Save the model parameter and buffer tensors in its ``state_dict``."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_model_checkpoint(
    path: str | Path,
    device: torch.device,
) -> VisionTransformer:
    """Reconstruct the baseline ViT and load a saved ``state_dict``."""

    model = VisionTransformer().to(device)
    state_dict = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model


def parse_args() -> argparse.Namespace:
    """Parse the small set of command-line choices used by this baseline."""

    parser = argparse.ArgumentParser(
        description="Train the explicit Vision Transformer on MNIST.",
    )
    parser.add_argument(
        "--tiny-subset",
        action="store_true",
        help="overfit the first 128 training examples instead of a normal run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override 5 normal epochs or 30 tiny-subset epochs",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="directory containing or receiving the MNIST files",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "vit_mnist.pt",
        help="where to save the final model state_dict",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="optional state_dict checkpoint used to initialize the model",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="require MNIST to exist under --data-root already",
    )
    return parser.parse_args()


def main() -> None:
    """Run either the fixed tiny-subset check or normal MNIST training."""

    args = parse_args()
    epochs = args.epochs
    if epochs is None:
        epochs = TINY_SUBSET_EPOCHS if args.tiny_subset else NORMAL_EPOCHS
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    set_seed(SEED)
    device = torch.device("cpu")
    print(f"Device: {device}")

    train_loader, test_loader = create_mnist_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_root=args.data_root,
        download=not args.no_download,
    )

    if args.tiny_subset:
        train_loader = create_tiny_subset_loader(train_loader)
        print(
            f"Mode: overfit fixed training subset "
            f"({TINY_SUBSET_SIZE} examples)"
        )
    else:
        print("Mode: normal MNIST training")

    if args.load_checkpoint is None:
        model = VisionTransformer().to(device)
    else:
        model = load_model_checkpoint(args.load_checkpoint, device)
        print(f"Loaded checkpoint: {args.load_checkpoint}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch_index in range(epochs):
        average_loss, training_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        progress = (
            f"Epoch {epoch_index + 1:02d}/{epochs:02d} | "
            f"train loss {average_loss:.4f} | "
            f"train accuracy {training_accuracy:.2%}"
        )
        if args.tiny_subset:
            print(progress)
        else:
            test_accuracy = evaluate(model, test_loader, device)
            print(f"{progress} | test accuracy {test_accuracy:.2%}")

    save_model_checkpoint(model, args.checkpoint_path)
    print(f"Saved checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
