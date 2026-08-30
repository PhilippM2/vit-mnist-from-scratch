"""Tests for the explicit training loop and checkpoint helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from train import (
    evaluate,
    load_model_checkpoint,
    save_model_checkpoint,
    train_one_epoch,
)
from vit_mnist.model import VisionTransformer


class TrainingTests(unittest.TestCase):
    """Exercise one update, evaluation, and state_dict round trip on CPU."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")

    def test_one_training_batch_updates_parameters(self) -> None:
        model = VisionTransformer().to(self.device)
        images = torch.randn(2, 1, 28, 28)
        labels = torch.tensor([2, 7], dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        average_loss, training_accuracy = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            self.device,
        )

        self.assertTrue(torch.isfinite(torch.tensor(average_loss)))
        self.assertGreaterEqual(training_accuracy, 0.0)
        self.assertLessEqual(training_accuracy, 1.0)
        self.assertTrue(
            any(
                not torch.equal(parameters_before[name], parameter)
                for name, parameter in model.named_parameters()
            )
        )
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_evaluate_returns_accuracy_without_gradients(self) -> None:
        model = VisionTransformer().to(self.device)
        images = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            labels = model(images).argmax(dim=1)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        accuracy = evaluate(model, loader, self.device)

        self.assertEqual(accuracy, 1.0)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_checkpoint_round_trip_preserves_logits(self) -> None:
        model = VisionTransformer().to(self.device)
        model.eval()
        images = torch.randn(2, 1, 28, 28)

        with torch.no_grad():
            expected_logits = model(images)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            save_model_checkpoint(model, checkpoint_path)
            loaded_model = load_model_checkpoint(checkpoint_path, self.device)
            loaded_model.eval()

            with torch.no_grad():
                loaded_logits = loaded_model(images)

        self.assertTrue(torch.equal(expected_logits, loaded_logits))


if __name__ == "__main__":
    unittest.main()
