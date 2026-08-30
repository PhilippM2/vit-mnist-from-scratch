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


class FeedForwardMLP(nn.Module):
    """Transform each token's features independently with a small MLP.

    Expected input shape: ``[B, T, D]``.
    Output shape: ``[B, T, D]``.

    The same learned ``D -> M -> D`` transformation is applied to every token;
    unlike self-attention, this component does not mix information between
    token positions.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout_after_activation = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self.dropout_after_output = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply the same feature transformation to each token separately."""

        assert tokens.ndim == 3, (
            "tokens must have shape [B, T, D], "
            f"but received {tuple(tokens.shape)}"
        )

        batch_size, num_tokens, embedding_dim = tokens.shape
        assert embedding_dim == self.embedding_dim, (
            f"expected embedding dimension {self.embedding_dim}, "
            f"but received {embedding_dim}"
        )

        hidden_features = self.input_projection(tokens)
        # [B, T, D] -> [B, T, M] = [B, 17, 64] -> [B, 17, 128]
        assert hidden_features.shape == (
            batch_size,
            num_tokens,
            self.hidden_dim,
        )

        activated_features = self.activation(hidden_features)
        # GELU preserves [B, T, M] = [B, 17, 128]

        activated_features = self.dropout_after_activation(activated_features)
        # Dropout preserves [B, T, M]; it is inactive when dropout=0.0.

        projected_features = self.output_projection(activated_features)
        # [B, T, M] -> [B, T, D] = [B, 17, 128] -> [B, 17, 64]

        output = self.dropout_after_output(projected_features)
        # Dropout preserves [B, T, D]; it is inactive when dropout=0.0.
        assert output.shape == (
            batch_size,
            num_tokens,
            self.embedding_dim,
        )

        return output


class TransformerBlock(nn.Module):
    """Apply one pre-normalized transformer encoder block.

    Expected input shape: ``[B, T, D]``.
    Output shape: ``[B, T, D]``.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.embedding_dim = embedding_dim

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = FeedForwardMLP(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Run pre-norm attention and MLP sublayers with residual additions."""

        assert tokens.ndim == 3, (
            "tokens must have shape [B, T, D], "
            f"but received {tuple(tokens.shape)}"
        )

        batch_size, num_tokens, embedding_dim = tokens.shape
        assert embedding_dim == self.embedding_dim, (
            f"expected embedding dimension {self.embedding_dim}, "
            f"but received {embedding_dim}"
        )
        expected_shape = (batch_size, num_tokens, self.embedding_dim)

        normalized_for_attention = self.norm1(tokens)
        # [B, T, D] = [B, 17, 64]
        attention_output = self.attention(normalized_for_attention)
        # [B, T, D] = [B, 17, 64]
        assert tokens.shape == attention_output.shape == expected_shape

        # Both operands are [B, T, D], so their residual addition is valid.
        tokens_after_attention = tokens + attention_output
        # [B, T, D] = [B, 17, 64]

        normalized_for_mlp = self.norm2(tokens_after_attention)
        # [B, T, D] = [B, 17, 64]
        mlp_output = self.mlp(normalized_for_mlp)
        # [B, T, D] = [B, 17, 64]
        assert tokens_after_attention.shape == mlp_output.shape == expected_shape

        # Both operands are [B, T, D], so their residual addition is valid.
        output = tokens_after_attention + mlp_output
        # [B, T, D] = [B, 17, 64]
        assert output.shape == expected_shape

        return output


class VisionTransformer(nn.Module):
    """Assemble the complete Vision Transformer for image classification.

    Expected input shape: ``[B, C, H, W]``.
    Output shape: ``[B, K]``, where ``K`` is the number of classes.
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        embedding_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.embedding_dim = embedding_dim
        self.depth = depth
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )
        self.num_tokens = self.patch_embedding.num_patches + 1

        # A single learned class-token parameter is shared by every image.
        self.class_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim) * 0.02
        )

        # Self-attention alone does not know where image patches came from, so
        # each token position receives its own learned positional information.
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.num_tokens, embedding_dim) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Convert a batch of images into raw class logits."""

        patch_tokens = self.patch_embedding(images)
        # [B, C, H, W] -> [B, N, D] = [B, 1, 28, 28] -> [B, 16, 64]
        batch_size = patch_tokens.shape[0]

        class_tokens = self.class_token.expand(
            batch_size,
            1,
            self.embedding_dim,
        )
        # [1, 1, D] -> [B, 1, D] = [1, 1, 64] -> [B, 1, 64]
        # expand creates one batch view of the same learned parameter per image.

        tokens = torch.cat((class_tokens, patch_tokens), dim=1)
        # [B, 1, D] + [B, N, D] -> [B, T, D] = [B, 17, 64]
        expected_token_shape = (
            batch_size,
            self.num_tokens,
            self.embedding_dim,
        )
        assert tokens.shape == expected_token_shape

        tokens = tokens + self.positional_embedding
        # [B, T, D] + [1, T, D] -> [B, T, D] = [B, 17, 64].
        # Broadcasting shares the positional table across the batch dimension.
        assert tokens.shape == expected_token_shape

        for block in self.blocks:
            tokens = block(tokens)
            # Each block preserves [B, T, D] = [B, 17, 64].
            assert tokens.shape == expected_token_shape

        normalized_tokens = self.final_norm(tokens)
        # LayerNorm preserves [B, T, D] = [B, 17, 64].

        class_representation = normalized_tokens[:, 0]
        # Select token position zero: [B, T, D] -> [B, D] = [B, 64].
        assert class_representation.shape == (
            batch_size,
            self.embedding_dim,
        )

        logits = self.classifier(class_representation)
        # [B, D] -> [B, K] = [B, 64] -> [B, 10].
        # No softmax: CrossEntropyLoss later expects these raw logits.
        assert logits.shape == (batch_size, self.num_classes)

        return logits


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

    mlp = FeedForwardMLP()
    mlp_output = mlp(tokens)

    assert mlp_output.shape == expected_output_shape, (
        f"expected output shape {expected_output_shape}, "
        f"but received {tuple(mlp_output.shape)}"
    )
    assert torch.isfinite(mlp_output).all()

    print(
        "FeedForwardMLP smoke test passed: "
        f"{tuple(tokens.shape)} -> {tuple(mlp_output.shape)}"
    )

    transformer_block = TransformerBlock()
    transformer_output = transformer_block(tokens)

    assert transformer_output.shape == expected_output_shape, (
        f"expected output shape {expected_output_shape}, "
        f"but received {tuple(transformer_output.shape)}"
    )
    assert torch.isfinite(transformer_output).all()

    tokens.grad = None
    transformer_output.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()

    print(
        "TransformerBlock smoke test passed: "
        f"{tuple(tokens.shape)} -> {tuple(transformer_output.shape)}, "
        "finite input gradients"
    )

    vision_transformer = VisionTransformer()
    model_images = torch.randn(2, 1, 28, 28, requires_grad=True)
    logits = vision_transformer(model_images)

    expected_logits_shape = (2, 10)
    assert logits.shape == expected_logits_shape, (
        f"expected output shape {expected_logits_shape}, "
        f"but received {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all()
    assert vision_transformer.class_token.shape == (1, 1, 64)
    assert vision_transformer.positional_embedding.shape == (1, 17, 64)
    assert len(vision_transformer.blocks) == 2
    assert all(
        isinstance(block, TransformerBlock)
        for block in vision_transformer.blocks
    )

    logits.sum().backward()
    assert model_images.grad is not None
    assert torch.isfinite(model_images.grad).all()
    for parameter in vision_transformer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    print(
        "VisionTransformer smoke test passed: "
        f"{tuple(model_images.shape)} -> {tuple(logits.shape)}, "
        "two blocks and finite end-to-end gradients"
    )


if __name__ == "__main__":
    _run_smoke_test()
