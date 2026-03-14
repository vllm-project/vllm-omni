"""Image-to-Image (I2I) Accuracy Evaluation Metrics.

This module implements evaluation metrics for I2I generation:
    - VLM-Judge: Edit success and instruction following
    - LPIPS: Background preservation
"""

import logging
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # noqa: F811

logger = logging.getLogger(__name__)


class LPIPSMetric:
    """LPIPS (Learned Perceptual Image Patch Similarity) for background preservation.

    Measures perceptual similarity between original and edited images,
    focusing on regions that should remain unchanged.

    Reference:
        Zhang, R., et al. (2018). The Unreasonable Effectiveness of Deep Features
        as a Perceptual Metric. CVPR.
    """

    def __init__(self, net: str = "alex", device: str | None = None):
        """Initialize LPIPS metric.

        Args:
            net: Network to use ('alex', 'vgg', or 'squeeze')
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.net = net
        self.device = device
        self._lpips_model = None

    def _load_model(self):
        """Lazy load LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips

                logger.info(f"Loading LPIPS with {self.net} backbone")
                self._lpips_model = lpips.LPIPS(net=self.net)
                if self.device:
                    self._lpips_model = self._lpips_model.to(self.device)
            except ImportError:
                logger.error("lpips not installed. Install with: pip install lpips")
                raise
        return self._lpips_model

    def compute(
        self,
        original_images: list[Any],
        edited_images: list[Any],
        masks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compute LPIPS distance between original and edited images.

        Uses batch processing for efficiency.

        Args:
            original_images: List of original images (before editing)
            edited_images: List of edited images (after editing)
            masks: Optional masks indicating regions that should be preserved.
                   If None, computes LPIPS on the full image.

        Returns:
            Dictionary with LPIPS scores
        """
        if np is None:
            raise ImportError("numpy is required for LPIPS computation")

        if len(original_images) != len(edited_images):
            raise ValueError("Number of original and edited images must match")

        if not original_images:
            return {
                "lpips_mean": 0.0,
                "lpips_std": 0.0,
                "lpips_per_sample": [],
            }

        self._load_model()

        # Batch process all images for efficiency
        distances = self._compute_batch(original_images, edited_images, masks)

        return {
            "lpips_mean": np.mean(distances) if distances else 0.0,
            "lpips_std": np.std(distances) if distances else 0.0,
            "lpips_per_sample": distances,
        }

    def _compute_batch(
        self,
        original_images: list[Any],
        edited_images: list[Any],
        masks: list[Any] | None = None,
    ) -> list[float]:
        """Compute LPIPS for multiple image pairs using batch processing.

        Args:
            original_images: List of original PIL Images
            edited_images: List of edited PIL Images
            masks: Optional list of mask PIL Images

        Returns:
            List of LPIPS distances
        """
        import torch

        # Convert all images to tensors and stack into batches
        orig_tensors = [self._pil_to_tensor(img) for img in original_images]
        edited_tensors = [self._pil_to_tensor(img) for img in edited_images]

        # Stack into batches [N, 3, H, W]
        orig_batch = torch.cat(orig_tensors, dim=0)
        edited_batch = torch.cat(edited_tensors, dim=0)

        # Move to model device
        device = next(self._lpips_model.parameters()).device
        orig_batch = orig_batch.to(device)
        edited_batch = edited_batch.to(device)

        # Compute LPIPS in batch
        with torch.no_grad():
            distances = self._lpips_model(orig_batch, edited_batch)

        # Apply masks if provided
        if masks is not None:
            for i, mask in enumerate(masks):
                mask_tensor = self._pil_to_tensor(mask, grayscale=True)
                mask_tensor = mask_tensor.to(device)
                # Normalize mask to [0, 1]
                mask_tensor = (mask_tensor + 1) / 2
                # Weighted distance
                distances[i] = distances[i] * mask_tensor.mean()

        return distances.cpu().numpy().flatten().tolist()

    def _pil_to_tensor(self, image: Any, grayscale: bool = False) -> Any:
        """Convert PIL Image to torch tensor in [-1, 1] range.

        Args:
            image: PIL Image
            grayscale: Whether to convert to grayscale

        Returns:
            Tensor of shape [1, C, H, W] in range [-1, 1]
        """
        import torch

        if grayscale:
            image = image.convert("L")
            img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        else:
            image = image.convert("RGB")
            img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
            # HWC to CHW
            img_array = np.transpose(img_array, (2, 0, 1))
            tensor = torch.from_numpy(img_array).unsqueeze(0)

        return tensor


class VLMJudge:
    """VLM-as-a-Judge for evaluating edit success.

    Uses a Vision-Language Model to evaluate:
        1. Edit success: Did the edit achieve the instruction?
        2. Instruction following: Does the result match the edit prompt?
    """

    def __init__(self, vlm_model: str | None = None):
        """Initialize VLM judge.

        Args:
            vlm_model: VLM model to use as judge.
                      If None, uses Qwen2.5-VL-7B.
        """
        self.vlm_model = vlm_model or "Qwen2.5-VL-7B"
        self._vlm = None

    def _load_vlm(self):
        """Lazy load VLM."""
        if self._vlm is None:
            try:
                logger.info(f"Loading VLM judge: {self.vlm_model}")
                # TODO: Implement VLM loading
                self._vlm = None
            except ImportError:
                logger.error("transformers not installed")
                raise
        return self._vlm

    def compute(
        self,
        original_images: list[Any],
        edited_images: list[Any],
        edit_instructions: list[str],
    ) -> dict[str, Any]:
        """Evaluate edit success using VLM judge.

        Args:
            original_images: Original images before editing
            edited_images: Edited images after editing
            edit_instructions: Edit instructions (e.g., "make the sky blue")

        Returns:
            Dictionary with evaluation scores
        """
        if not (len(original_images) == len(edited_images) == len(edit_instructions)):
            raise ValueError("Number of images and instructions must match")

        edit_success_scores = []
        instruction_following_scores = []

        for orig, edited, instruction in zip(original_images, edited_images, edit_instructions):
            # Evaluate edit success
            success_score = self._evaluate_edit_success(edited, instruction)
            edit_success_scores.append(success_score)

            # Evaluate instruction following
            following_score = self._evaluate_instruction_following(orig, edited, instruction)
            instruction_following_scores.append(following_score)

        return {
            "edit_success": {
                "mean": np.mean(edit_success_scores) if edit_success_scores else 0.0,
                "per_sample": edit_success_scores,
            },
            "instruction_following": {
                "mean": (np.mean(instruction_following_scores) if instruction_following_scores else 0.0),
                "per_sample": instruction_following_scores,
            },
        }

    def _evaluate_edit_success(self, edited_image: Any, instruction: str) -> float:
        """Evaluate if the edit was successful.

        Returns:
            Score between 0 and 1
        """
        # TODO: Implement VLM-based evaluation
        # Ask VLM: "Did the edit '{instruction}' succeed? Rate 0-10"
        return 0.0

    def _evaluate_instruction_following(self, original: Any, edited: Any, instruction: str) -> float:
        """Evaluate if the edit follows the instruction.

        Returns:
            Score between 0 and 1
        """
        # TODO: Implement VLM-based evaluation
        # Ask VLM: "Does the edited image correctly follow the instruction
        #           '{instruction}'? Rate 0-10"
        return 0.0


class I2IEvaluator:
    """Combined I2I evaluator with multiple metrics."""

    def __init__(
        self,
        use_lpips: bool = True,
        use_vlm_judge: bool = True,
        lpips_net: str = "alex",
        vlm_model: str | None = None,
        device: str | None = None,
    ):
        """Initialize I2I evaluator.

        Args:
            use_lpips: Whether to use LPIPS metric
            use_vlm_judge: Whether to use VLM judge
            lpips_net: LPIPS network backbone
            vlm_model: VLM model for judge
            device: Device to run on
        """
        self.lpips = LPIPSMetric(lpips_net, device) if use_lpips else None
        self.vlm_judge = VLMJudge(vlm_model) if use_vlm_judge else None

    def evaluate(
        self,
        original_images: list[Any],
        edited_images: list[Any],
        edit_instructions: list[str],
        preservation_masks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate I2I generation with all enabled metrics.

        Args:
            original_images: Original images before editing
            edited_images: Edited images after editing
            edit_instructions: Edit instructions
            preservation_masks: Optional masks for regions to preserve

        Returns:
            Dictionary with all metric results
        """
        results = {}

        if self.lpips:
            logger.info("Computing LPIPS...")
            results["lpips"] = self.lpips.compute(original_images, edited_images, preservation_masks)

        if self.vlm_judge:
            logger.info("Computing VLM-Judge scores...")
            results["vlm_judge"] = self.vlm_judge.compute(original_images, edited_images, edit_instructions)

        return results
