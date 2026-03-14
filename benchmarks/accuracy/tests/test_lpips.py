"""Unit tests for LPIPS metric.

Tests the LPIPSMetric class to ensure correct functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

# Skip tests if lpips is not installed
try:
    import lpips  # noqa: F401
    import torch

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


@unittest.skipUnless(HAS_LPIPS, "lpips not installed")
class TestLPIPSMetric(unittest.TestCase):
    """Test cases for LPIPSMetric."""

    def setUp(self):
        """Set up test fixtures."""
        from benchmarks.accuracy.i2i import LPIPSMetric

        self.metric = LPIPSMetric(net="alex")

    def test_initialization(self):
        """Test LPIPSMetric initialization."""
        self.assertEqual(self.metric.net, "alex")
        self.assertIsNone(self.metric._lpips_model)

    @patch("lpips.LPIPS")
    def test_load_model(self, mock_lpips):
        """Test lazy loading of LPIPS model."""
        mock_model = MagicMock()
        mock_lpips.return_value = mock_model

        # First call should load the model
        self.metric._load_model()
        mock_lpips.assert_called_once_with(net="alex")
        self.assertIsNotNone(self.metric._lpips_model)

        # Second call should not reload
        mock_lpips.reset_mock()
        self.metric._load_model()
        mock_lpips.assert_not_called()

    def test_pil_to_tensor(self):
        """Test PIL image to tensor conversion."""
        # Create a simple test image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))

        tensor = self.metric._pil_to_tensor(img)

        # Check shape: [1, 3, H, W]
        self.assertEqual(tensor.shape, (1, 3, 64, 64))

        # Check range: should be in [-1, 1]
        self.assertTrue(torch.all(tensor >= -1))
        self.assertTrue(torch.all(tensor <= 1))

    def test_pil_to_tensor_grayscale(self):
        """Test grayscale conversion."""
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))

        tensor = self.metric._pil_to_tensor(img, grayscale=True)

        # Check shape: [1, 1, H, W]
        self.assertEqual(tensor.shape, (1, 1, 64, 64))

    @patch("lpips.LPIPS")
    def test_compute_batch(self, mock_lpips):
        """Test batch LPIPS computation."""
        # Mock the LPIPS model to return batch of distances
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.5], [0.6], [0.7]])
        mock_lpips.return_value = mock_model

        # Create test images
        images1 = [Image.new("RGB", (64, 64), color=(100, 100, 100)) for _ in range(3)]
        images2 = [Image.new("RGB", (64, 64), color=(150, 150, 150)) for _ in range(3)]

        distances = self.metric._compute_batch(images1, images2)

        # Verify batch processing
        self.assertEqual(len(distances), 3)
        self.assertEqual(distances, [0.5, 0.6, 0.7])
        mock_model.assert_called_once()

    @patch("lpips.LPIPS")
    def test_compute_batch_with_mask(self, mock_lpips):
        """Test batch LPIPS computation with masks."""
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.5], [0.6]])
        mock_lpips.return_value = mock_model

        images1 = [Image.new("RGB", (64, 64), color=(100, 100, 100)) for _ in range(2)]
        images2 = [Image.new("RGB", (64, 64), color=(150, 150, 150)) for _ in range(2)]
        masks = [Image.new("L", (64, 64), color=255) for _ in range(2)]

        distances = self.metric._compute_batch(images1, images2, masks)

        self.assertEqual(len(distances), 2)

    @patch("lpips.LPIPS")
    def test_compute_full(self, mock_lpips):
        """Test full compute pipeline with batch processing."""
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.4], [0.5], [0.6]])
        mock_lpips.return_value = mock_model

        # Create test images
        images1 = [Image.new("RGB", (64, 64), color=(100, 100, 100)) for _ in range(3)]
        images2 = [Image.new("RGB", (64, 64), color=(150, 150, 150)) for _ in range(3)]

        result = self.metric.compute(images1, images2)

        # Check result structure
        self.assertIn("lpips_mean", result)
        self.assertIn("lpips_std", result)
        self.assertIn("lpips_per_sample", result)

        # Check values
        self.assertEqual(len(result["lpips_per_sample"]), 3)
        self.assertAlmostEqual(result["lpips_mean"], 0.5, places=5)

    def test_compute_empty_list(self):
        """Test computing with empty image list."""
        result = self.metric.compute([], [])

        self.assertEqual(result["lpips_mean"], 0.0)
        self.assertEqual(result["lpips_std"], 0.0)
        self.assertEqual(result["lpips_per_sample"], [])

    def test_compute_mismatched_lengths(self):
        """Test that mismatched image counts raise ValueError."""
        img = Image.new("RGB", (64, 64))

        with self.assertRaises(ValueError):
            self.metric.compute([img], [img, img])

    def test_different_networks(self):
        """Test initialization with different network backends."""
        from benchmarks.accuracy.i2i import LPIPSMetric

        # Test alex
        metric_alex = LPIPSMetric(net="alex")
        self.assertEqual(metric_alex.net, "alex")

        # Test vgg
        metric_vgg = LPIPSMetric(net="vgg")
        self.assertEqual(metric_vgg.net, "vgg")


@unittest.skipUnless(HAS_LPIPS, "lpips not installed")
class TestI2IEvaluator(unittest.TestCase):
    """Test cases for I2IEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        from benchmarks.accuracy.i2i import I2IEvaluator

        self.evaluator = I2IEvaluator(
            use_lpips=True,
            use_vlm_judge=False,  # Skip VLM for unit tests
        )

    @patch("lpips.LPIPS")
    def test_evaluate_lpips_only(self, mock_lpips):
        """Test evaluation with LPIPS only."""
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.3], [0.4]])
        mock_lpips.return_value = mock_model

        original = [Image.new("RGB", (64, 64)) for _ in range(2)]
        edited = [Image.new("RGB", (64, 64)) for _ in range(2)]
        instructions = ["make it brighter", "add a red circle"]

        result = self.evaluator.evaluate(original, edited, instructions)

        self.assertIn("lpips", result)
        self.assertNotIn("vlm_judge", result)  # VLM judge disabled


class TestLPIPSWithoutOptionalDeps(unittest.TestCase):
    """Test behavior when optional dependencies are missing."""

    def test_import_error_handling(self):
        """Test that ImportError is raised when lpips is not available."""
        # This test runs even without lpips
        try:
            import lpips  # noqa: F401

            self.skipTest("lpips is installed, skipping")
        except ImportError:
            pass

        # The module should still import, but LPIPSMetric will fail at runtime
        from benchmarks.accuracy.i2i import LPIPSMetric

        metric = LPIPSMetric()
        # _load_model should raise ImportError
        with self.assertRaises(ImportError):
            metric._load_model()


if __name__ == "__main__":
    unittest.main()
