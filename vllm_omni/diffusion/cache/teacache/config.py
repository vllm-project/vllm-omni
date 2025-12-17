# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# Model-specific polynomial coefficients for rescaling L1 distances
# These coefficients account for model-specific characteristics in how embeddings change
# Source: TeaCache paper and ComfyUI-TeaCache empirical tuning
_MODEL_COEFFICIENTS = {
    # FLUX model coefficients from TeaCache paper
    "FluxPipeline": [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
    # Qwen-Image model coefficients from ComfyUI-TeaCache
    # Tuned specifically for Qwen's dual-stream transformer architecture
    "QwenImagePipeline": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
}


@dataclass
class TeaCacheConfig:
    """
    Configuration for TeaCache applied to transformer models.

    TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique that speeds up
    diffusion model inference by reusing transformer block computations when consecutive
    timestep embeddings are similar.

    Args:
        rel_l1_thresh: Threshold for accumulated relative L1 distance. When below threshold,
            cached residual is reused. Values in [0.1, 0.3] work best:
            - 0.2: ~1.5x speedup with minimal quality loss
            - 0.4: ~1.8x speedup with slight quality loss
            - 0.6: ~2.0x speedup with noticeable quality loss
        coefficients: Polynomial coefficients for rescaling L1 distance. If None, uses
            model-specific defaults based on model_type.
        model_type: Pipeline class name (e.g., "QwenImagePipeline", "FluxPipeline").
            Must match OmniDiffusionConfig.model_class_name for proper extractor lookup.
            Defaults to "QwenImagePipeline".
    """

    rel_l1_thresh: float = 0.2
    coefficients: list[float] | None = None
    model_type: str = "QwenImagePipeline"

    def __post_init__(self) -> None:
        """Validate and set default coefficients."""
        if self.rel_l1_thresh <= 0:
            raise ValueError(f"rel_l1_thresh must be positive, got {self.rel_l1_thresh}")

        if self.coefficients is None:
            # Use model-specific coefficients, fallback to FluxPipeline if not found
            self.coefficients = _MODEL_COEFFICIENTS.get(self.model_type, _MODEL_COEFFICIENTS["FluxPipeline"])

        if len(self.coefficients) != 5:
            raise ValueError(f"coefficients must contain exactly 5 elements, got {len(self.coefficients)}")
