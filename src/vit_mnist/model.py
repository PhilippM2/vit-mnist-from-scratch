"""Small, explicit building blocks for the educational MNIST ViT."""

from __future__ import annotations

import math

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


class MultiHeadSelfAttention(nn.Module):
    """Apply explicit multi-head scaled dot-product self-attention.

    Expected input shape: ``[B, T, D]``.
    Output shape: ``[B, T, D]``.

    The implemented operation is:
    ``softmax(Q @ K^T / sqrt(Dh)) @ V``.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_projection = nn.Linear(embedding_dim, embedding_dim)
        self.k_projection = nn.Linear(embedding_dim, embedding_dim)
        self.v_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Mix information among all tokens through explicit self-attention."""

        assert tokens.ndim == 3, (
            "tokens must have shape [B, T, D], "
            f"but received {tuple(tokens.shape)}"
        )

        batch_size, num_tokens, embedding_dim = tokens.shape
        assert embedding_dim == self.embedding_dim, (
            f"expected embedding dimension {self.embedding_dim}, "
            f"but received {embedding_dim}"
        )

        queries = self.q_projection(tokens)
        # [B, T, D] = [B, 17, 64]
        keys = self.k_projection(tokens)
        # [B, T, D] = [B, 17, 64]
        values = self.v_projection(tokens)
        # [B, T, D] = [B, 17, 64]

        queries = queries.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )
        # [B, T, A, Dh] = [B, 17, 4, 16]
        keys = keys.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )
        # [B, T, A, Dh] = [B, 17, 4, 16]
        values = values.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )
        # [B, T, A, Dh] = [B, 17, 4, 16]

        queries = queries.transpose(1, 2)
        # [B, A, T, Dh] = [B, 4, 17, 16]
        keys = keys.transpose(1, 2)
        # [B, A, T, Dh] = [B, 4, 17, 16]
        values = values.transpose(1, 2)
        # [B, A, T, Dh] = [B, 4, 17, 16]
        expected_head_shape = (
            batch_size,
            self.num_heads,
            num_tokens,
            self.head_dim,
        )
        assert queries.shape == expected_head_shape
        assert keys.shape == expected_head_shape
        assert values.shape == expected_head_shape

        transposed_keys = keys.transpose(-2, -1)
        # [B, A, Dh, T] = [B, 4, 16, 17]

        attention_scores = torch.matmul(queries, transposed_keys)
        # [B, A, T, T] = [B, 4, 17, 17]
        expected_attention_shape = (
            batch_size,
            self.num_heads,
            num_tokens,
            num_tokens,
        )
        assert attention_scores.shape == expected_attention_shape

        scale = math.sqrt(self.head_dim)
        scaled_attention_scores = attention_scores / scale
        # [B, A, T, T] = [B, 4, 17, 17]

        attention_probabilities = torch.softmax(
            scaled_attention_scores,
            dim=-1,
        )
        # [B, A, T, T] = [B, 4, 17, 17]
        assert attention_probabilities.shape == expected_attention_shape
        probability_sums = attention_probabilities.sum(dim=-1)
        # [B, A, T] = [B, 4, 17]
        assert torch.allclose(
            probability_sums,
            torch.ones_like(probability_sums),
            atol=1e-6,
        )

        context = torch.matmul(attention_probabilities, values)
        # [B, A, T, Dh] = [B, 4, 17, 16]
        assert context.shape == expected_head_shape

        context_with_tokens_first = context.transpose(1, 2)
        # [B, T, A, Dh] = [B, 17, 4, 16]

        merged_heads = context_with_tokens_first.reshape(
            batch_size,
            num_tokens,
            self.embedding_dim,
        )
        # [B, T, D] = [B, 17, 64]

        output = self.output_projection(merged_heads)
        # [B, T, D] = [B, 17, 64]
        assert output.shape == (
            batch_size,
            num_tokens,
            self.embedding_dim,
        )

        return output


def _run_smoke_test() -> None:
    """Run direct component checks without requiring MNIST data."""

    torch.manual_seed(0)

    patch_embedding = PatchEmbedding()
    images = torch.randn(2, 1, 28, 28)
    output = patch_embedding(images)

    expected_shape = (2, 16, 64)
    assert output.shape == expected_shape, (
        f"expected output shape {expected_shape}, but received {tuple(output.shape)}"
    )

    print(f"PatchEmbedding smoke test passed: {tuple(images.shape)} -> {tuple(output.shape)}")

    attention = MultiHeadSelfAttention()
    tokens = torch.randn(2, 17, 64, requires_grad=True)
    attention_output = attention(tokens)

    expected_output_shape = (2, 17, 64)
    assert attention_output.shape == expected_output_shape, (
        f"expected output shape {expected_output_shape}, "
        f"but received {tuple(attention_output.shape)}"
    )

    expected_attention_shape = (2, 4, 17, 17)
    conceptual_attention_shape = (
        tokens.shape[0],
        attention.num_heads,
        tokens.shape[1],
        tokens.shape[1],
    )
    assert conceptual_attention_shape == expected_attention_shape

    attention_output.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()

    print(
        "MultiHeadSelfAttention smoke test passed: "
        f"{tuple(tokens.shape)} -> {tuple(attention_output.shape)}, "
        f"attention matrix {expected_attention_shape}"
    )


if __name__ == "__main__":
    _run_smoke_test()
