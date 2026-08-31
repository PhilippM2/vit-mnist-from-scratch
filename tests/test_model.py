"""Focused tests for the explicit educational Vision Transformer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from vit_mnist.model import (  # noqa: E402
    FeedForwardMLP,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerBlock,
    VisionTransformer,
)


class PatchEmbeddingTests(unittest.TestCase):
    """Check the visible patch layout before learned projection."""

    def test_patch_extraction_uses_row_major_order(self) -> None:
        patch_embedding = PatchEmbedding(
            image_size=4,
            patch_size=2,
            in_channels=1,
            embedding_dim=4,
        )
        patch_embedding.projection = nn.Identity()
        image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

        patches = patch_embedding(image)

        expected_patches = torch.tensor(
            [
                [
                    [0.0, 1.0, 4.0, 5.0],
                    [2.0, 3.0, 6.0, 7.0],
                    [8.0, 9.0, 12.0, 13.0],
                    [10.0, 11.0, 14.0, 15.0],
                ]
            ]
        )
        self.assertEqual(patches.shape, (1, 4, 4))
        self.assertTrue(torch.equal(patches, expected_patches))

    def test_baseline_patch_embedding_shape(self) -> None:
        patch_embedding = PatchEmbedding()
        images = torch.randn(2, 1, 28, 28)

        patch_tokens = patch_embedding(images)

        self.assertEqual(patch_tokens.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(patch_tokens).all())


class ComponentShapeTests(unittest.TestCase):
    """Exercise the baseline tensor contracts and finite values."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.tokens = torch.randn(2, 17, 64)

    def test_attention_preserves_shape_and_has_finite_values(self) -> None:
        attention = MultiHeadSelfAttention()

        output = attention(self.tokens)

        self.assertEqual(output.shape, (2, 17, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_mlp_preserves_shape_and_has_finite_values(self) -> None:
        mlp = FeedForwardMLP()

        output = mlp(self.tokens)

        self.assertEqual(output.shape, (2, 17, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_transformer_block_preserves_shape(self) -> None:
        block = TransformerBlock()

        output = block(self.tokens)

        self.assertEqual(output.shape, (2, 17, 64))
        self.assertTrue(torch.isfinite(output).all())


class VisionTransformerTests(unittest.TestCase):
    """Check complete-model output and gradient contracts."""

    def test_end_to_end_forward_and_backward_are_finite(self) -> None:
        torch.manual_seed(0)
        model = VisionTransformer()
        images = torch.randn(2, 1, 28, 28, requires_grad=True)

        logits = model(images)
        loss = logits.square().mean()
        loss.backward()

        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class InvalidConfigurationTests(unittest.TestCase):
    """Verify useful constructor and input failures already supported."""

    def test_image_size_must_be_divisible_by_patch_size(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "image_size must be divisible by patch_size",
        ):
            PatchEmbedding(image_size=28, patch_size=6)

    def test_embedding_dimension_must_be_divisible_by_heads(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "embedding_dim must be divisible by num_heads",
        ):
            MultiHeadSelfAttention(embedding_dim=10, num_heads=3)

    def test_depth_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "depth must be positive"):
            VisionTransformer(depth=0)

    def test_dropout_must_be_in_supported_range(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"dropout must be in the range \[0.0, 1.0\)",
        ):
            FeedForwardMLP(dropout=1.0)

    def test_forward_rejects_wrong_channel_count(self) -> None:
        patch_embedding = PatchEmbedding()
        images = torch.randn(1, 3, 28, 28)

        with self.assertRaisesRegex(AssertionError, "expected 1 channel"):
            patch_embedding(images)

    def test_forward_rejects_wrong_image_size(self) -> None:
        patch_embedding = PatchEmbedding()
        images = torch.randn(1, 1, 32, 32)

        with self.assertRaisesRegex(AssertionError, "expected square 28x28"):
            patch_embedding(images)


if __name__ == "__main__":
    unittest.main()
