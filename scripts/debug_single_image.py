"""Step one real MNIST test image through the trained ViT on CPU."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch


# Make ``src/vit_mnist`` importable when this file is launched directly from
# the repository root or with the VS Code Python debugger.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from vit_mnist.data import create_mnist_loaders  # noqa: E402
from vit_mnist.model import VisionTransformer  # noqa: E402


def main() -> None:
    """Run one deterministic, inference-only forward pass for debugging."""

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "vit_mnist.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Trained model checkpoint not found at "
            f"{checkpoint_path}. Run `python .\\train.py` first or place the "
            "trained state_dict at that path."
        )

    model = VisionTransformer().to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Checkpoint path: {checkpoint_path}")

    _, test_loader = create_mnist_loaders(
        batch_size=1,
        num_workers=0,
        data_root=PROJECT_ROOT / "data",
    )

    image, true_label = test_loader.dataset[0]
    assert image.shape == (1, 28, 28)
    print(f"Image shape before batching: {list(image.shape)}")

    batched_image = image.unsqueeze(0)
    # [C, H, W] -> [B, C, H, W] = [1, 28, 28] -> [1, 1, 28, 28]
    assert batched_image.shape == (1, 1, 28, 28)
    print(f"Image shape after batching:  {list(batched_image.shape)}")
    print(f"True label: {true_label}")

    batched_image = batched_image.to(device)
    with torch.no_grad():
        logits = model(batched_image)
    assert logits.shape == (1, 10)
    assert not logits.requires_grad

    predicted_label = logits.argmax(dim=1).item()
    print(f"Logits: {logits}")
    print(f"Logits shape: {list(logits.shape)}")
    print(f"Predicted label: {predicted_label}")
    print(f"Prediction correct: {predicted_label == true_label}")


if __name__ == "__main__":
    main()
