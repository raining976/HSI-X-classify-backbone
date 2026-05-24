import sys
from pathlib import Path
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import create_experiment_config_from_cli
from src.models.balanced_gmamba_hx.model import (
    BalancedGMambaBlock,
    BalancedGMambaHX,
)


class TrainConfigCliTests(unittest.TestCase):
    def test_cli_fallback_when_config_missing(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "train.py",
                "--config",
                "does-not-exist.json",
                "--model",
                "RSCNet",
                "--dataset",
                "4",
                "--epochs",
                "7",
            ]

            config = create_experiment_config_from_cli()
        finally:
            sys.argv = argv

        self.assertEqual(config.model_name, "RSCNet")
        self.assertEqual(config.dataset_type, 4)
        self.assertEqual(config.epochs, 7)
        self.assertFalse(config.enable_email_notification)
        self.assertEqual(config.email_recipient_override, "")

    def test_json_config_takes_precedence(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "train.py",
                "--config",
                str(PROJECT_ROOT / "configs" / "train.json"),
                "--model",
                "RSCNet",
                "--dataset",
                "4",
            ]

            config = create_experiment_config_from_cli()
        finally:
            sys.argv = argv

        self.assertEqual(config.model_name, "BalancedGMambaHX")
        self.assertEqual(config.dataset_type, 0)
        self.assertTrue(config.enable_email_notification)
        self.assertEqual(config.email_recipient_override, "")

    def test_balanced_gmamba_block_preserves_tensor_contract(self):
        block = BalancedGMambaBlock(embed_dim=16)
        hsi_spatial = torch.randn(2, 16, 7, 7)
        x_spatial = torch.randn(2, 16, 7, 7)
        hsi_tokens = torch.randn(2, 8, 16)
        x_tokens = torch.randn(2, 8, 16)

        outputs = block(hsi_spatial, x_spatial, hsi_tokens, x_tokens)

        self.assertEqual(len(outputs), 6)
        self.assertEqual(tuple(outputs[0].shape), (2, 16, 7, 7))
        self.assertEqual(tuple(outputs[1].shape), (2, 16, 7, 7))
        self.assertEqual(tuple(outputs[2].shape), (2, 16, 7, 7))
        self.assertEqual(tuple(outputs[3].shape), (2, 8, 16))
        self.assertEqual(tuple(outputs[4].shape), (2, 8, 16))
        self.assertEqual(tuple(outputs[5].shape), (2, 8, 16))

    def test_balanced_gmamba_block_uses_frequency_guided_interaction_modules(self):
        block = BalancedGMambaBlock(embed_dim=16)

        self.assertTrue(hasattr(block, "hsi_spectral_guide"))
        self.assertTrue(hasattr(block, "spatial_decomposer"))
        self.assertTrue(hasattr(block, "frequency_mixer"))
        self.assertTrue(hasattr(block, "modulator"))

    def test_balanced_gmamba_uses_simple_aux_encoder(self):
        model = BalancedGMambaHX(
            hsi_channels=144,
            aux_channels=3,
            num_classes=15,
            embed_dim=32,
            stem_dim=8,
            stage_depths=(1, 1, 1),
        )

        aux_encoder_layers = list(model.x_stem.encoder.net)

        self.assertEqual(aux_encoder_layers[0].in_channels, 3)
        self.assertEqual(aux_encoder_layers[0].out_channels, 32)
        self.assertIsInstance(aux_encoder_layers[2], torch.nn.ReLU)
        self.assertIsInstance(aux_encoder_layers[3], torch.nn.Conv2d)
        self.assertEqual(aux_encoder_layers[3].in_channels, 32)
        self.assertEqual(aux_encoder_layers[3].out_channels, 32)
        self.assertIsInstance(aux_encoder_layers[5], torch.nn.ReLU)

    def test_balanced_gmamba_forward_supports_third_stage_default(self):
        model = BalancedGMambaHX(
            hsi_channels=144,
            aux_channels=1,
            num_classes=15,
            embed_dim=32,
            stem_dim=8,
            stage_depths=(1, 1, 1),
        )
        hsi = torch.randn(2, 1, 30, 11, 11)
        aux = torch.randn(2, 1, 11, 11)

        out = model(hsi, aux)

        self.assertEqual(tuple(out.shape), (2, 15))


if __name__ == "__main__":
    unittest.main()
