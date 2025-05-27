import sys
import os
import unittest
import torch
import torch.nn as nn
import random

from train import prepare_batch, get_loss, loss_err
from model import TFModel
from generator import ScenarioGenerator
from entities import AGENT_POOL, OBJECT_POOL, LOCATION_POOL
from utils import build_vocab


class SimpleConfig:
    """Simple configuration class for testing."""

    def __init__(self):
        self.d_model = 128
        self.num_heads = 4
        self.num_layers = 2
        self.ff_dim = 256
        self.dropout = 0.1
        self.max_seq_len = 100
        self.linear_attn = False
        self.mlp = True
        self.residual = True
        self.norm = True
        self.output_norm = True
        self.pos = "rotary"
        self.rotary_theta = 10000
        self.device = "cuda"
        self.vocab_size = build_vocab(AGENT_POOL, OBJECT_POOL, LOCATION_POOL).__len__()


class TestTraining(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)

        # Create a small test dataset
        self.generator = ScenarioGenerator(
            num_scenarios=5,
            min_agents=1,
            max_agents=3,
            num_objects=1,
            min_chain_length=1,
            max_chain_length=2,
            move_probability=0.8,
            seed=42,
        )
        self.dataset = self.generator.generate_dataset()

        # Device
        self.device = "cuda"

        # Initialize model
        self.config = SimpleConfig()
        self.model = TFModel(self.config).to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def test_prepare_batch(self):
        """Test batch preparation function."""
        batch_data = self.dataset["data"][:3]  # Take first 3 items
        src, tgt, src_mask = prepare_batch(batch_data, self.device)

        # Check shapes and types
        self.assertIsInstance(src, torch.Tensor)
        self.assertIsInstance(tgt, torch.Tensor)
        self.assertIsInstance(src_mask, torch.Tensor)

        # Check dimensions
        self.assertEqual(src.dim(), 2)  # [batch_size, seq_len]
        self.assertEqual(tgt.dim(), 2)  # [batch_size, 1]
        self.assertEqual(src_mask.dim(), 2)  # [batch_size, seq_len]

        # Check batch size
        self.assertEqual(src.size(0), 3)
        self.assertEqual(tgt.size(0), 3)
        self.assertEqual(src_mask.size(0), 3)

        # Check mask values
        self.assertTrue(torch.all((src_mask == 0) | (src_mask == 1)))

        # Check that target is a single token per example
        self.assertEqual(tgt.size(1), 1)

        print("Prepare batch test passed!")

    def test_get_loss(self):
        """Test loss calculation."""
        # Get loss
        loss = get_loss(self.model, self.criterion, self.dataset, self.device)

        # Check that loss is a tensor and has a valid value
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)  # Loss should be positive
        self.assertFalse(torch.isnan(loss).item())  # Loss should not be NaN

        print(f"Get loss test passed! Loss value: {loss.item():.4f}")

    def test_loss_err(self):
        """Test loss and error rate calculation."""
        # Get loss and error rate
        loss, err = loss_err(
            self.model, self.criterion, self.dataset, self.device, batch_size=2
        )

        # Check that loss and error are tensors and have valid values
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(err, torch.Tensor)

        self.assertTrue(loss.item() > 0)  # Loss should be positive
        self.assertTrue(0 <= err.item() <= 1)  # Error rate should be between 0 and 1

        self.assertFalse(torch.isnan(loss).item())  # Loss should not be NaN
        self.assertFalse(torch.isnan(err).item())  # Error rate should not be NaN

        print(
            f"Loss-error test passed! Loss: {loss.item():.4f}, Error rate: {err.item():.4f}"
        )

    def test_full_training_loop(self):
        """Test a mini training loop."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Run a few training steps
        for _ in range(2):
            self.model.train()
            loss = get_loss(self.model, self.criterion, self.dataset, self.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check evaluation after training
        self.model.eval()
        loss, err = loss_err(self.model, self.criterion, self.dataset, self.device)

        print(
            f"Training loop test passed! Final loss: {loss.item():.4f}, Error rate: {err.item():.4f}"
        )


if __name__ == "__main__":
    unittest.main()
