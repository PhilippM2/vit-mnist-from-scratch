"""Small, explicit building blocks for the educational MNIST ViT."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Convert square images into a sequence of learned patch embeddings.

    Expected input shape: ``[B, C, H, W]``.
    Output shape: ``[B, N, D]``, where ``N`` is the number of patches and
    ``D`` is the embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side * self.patches_per_side
        self.patch_dim = in_channels * patch_size * patch_size

        self.projection = nn.Linear(self.patch_dim, embedding_dim)

    def forward(self, images: Tensor) -> Tensor:
        """Extract, flatten, and linearly project non-overlapping patches."""

        assert images.ndim == 4, (
            "images must have shape [B, C, H, W], "
            f"but received {tuple(images.shape)}"
        )

        batch_size, channels, height, width = images.shape
        assert channels == self.in_channels, (
            f"expected {self.in_channels} channel(s), but received {channels}"
        )
        assert height == self.image_size and width == self.image_size, (
            f"expected square {self.image_size}x{self.image_size} images, "
            f"but received {height}x{width}"
        )

        patches_by_row = images.unfold(
            dimension=2,
            size=self.patch_size,
            step=self.patch_size,
        )
        # [B, C, H/P, W, P] = [B, 1, 4, 28, 7]

        patch_grid = patches_by_row.unfold(
            dimension=3,
            size=self.patch_size,
            step=self.patch_size,
        )
        # [B, C, H/P, W/P, P, P] = [B, 1, 4, 4, 7, 7]
        assert patch_grid.shape == (
            batch_size,
            self.in_channels,
            self.patches_per_side,
            self.patches_per_side,
            self.patch_size,
            self.patch_size,
        )

        patches_in_row_major_order = patch_grid.permute(0, 2, 3, 1, 4, 5)
        # [B, H/P, W/P, C, P, P] = [B, 4, 4, 1, 7, 7]

        flattened_patches = patches_in_row_major_order.reshape(
            batch_size,
            self.num_patches,
            self.patch_dim,
        )
        # [B, N, C*P*P] = [B, 16, 49]
        assert flattened_patches.shape == (
            batch_size,
            self.num_patches,
            self.patch_dim,
        )

        patch_embeddings = self.projection(flattened_patches)
        # [B, N, D] = [B, 16, 64]
        assert patch_embeddings.shape == (
            batch_size,
            self.num_patches,
            self.embedding_dim,
        )

        return patch_embeddings


def _run_smoke_test() -> None:
    """Run a direct shape check without requiring MNIST data."""

    torch.manual_seed(0)

    patch_embedding = PatchEmbedding()
    images = torch.randn(2, 1, 28, 28)
    output = patch_embedding(images)

    expected_shape = (2, 16, 64)
    assert output.shape == expected_shape, (
        f"expected output shape {expected_shape}, but received {tuple(output.shape)}"
    )

    print(f"PatchEmbedding smoke test passed: {tuple(images.shape)} -> {tuple(output.shape)}")


if __name__ == "__main__":
    _run_smoke_test()
