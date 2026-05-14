# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

# Model-specific polynomial coefficients for rescaling L1 distances
# These coefficients account for model-specific characteristics in how embeddings change
# Source: TeaCache paper and ComfyUI-TeaCache empirical tuning
_MODEL_COEFFICIENTS = {
    # FLUX transformer coefficients from TeaCache paper
    "FluxTransformer2DModel": [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
    # Flux2 Klein transformer coefficients
    # Same as FLUX.1 (similar dual-stream architecture)
    "Flux2Klein": [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
    # Qwen-Image transformer coefficients from ComfyUI-TeaCache
    # Tuned specifically for Qwen's dual-stream transformer architecture
    # Used for all Qwen-Image Family pipelines, in general
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    # Bagel transformer coefficients
    # Using Qwen's coefficients as reasonable default given shared architecture
    "Bagel": [1.33313129e06, -1.68644226e05, 7.95050740e03, -1.63747873e02, 1.26352397e00],
    # Z-Image transformer coefficients
    # Copied from Qwen-Image, need to be tuned specifically for Z-Image in future
    "ZImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    # Estimated TeaCache polynomial coefficients for StableAudioDiTModel.
    "StableAudioDiTModel": [
        121.77490545701518,
        -153.7449426160371,
        68.05368574596551,
        -12.281286412689623,
        1.0733905006198015,
    ],
    # HunyuanImage3 pipeline coefficients
    # Calibrated via polyfit on 3920 data points (80 prompts × 49 steps).
    # Maps time_embed rel_l1 (range ~0.12-0.54) to output rel_l1 (range ~0.01-0.53).
    "HunyuanImage3Pipeline": [
        1.04117826e02,
        -1.26848482e02,
        5.68168652e01,
        -1.04182570e01,
        6.78098549e-01,
    ],
    # Flux2 transformer coefficients
    # Copied from Qwen-Image, need to be tuned specifically for Flux2 in future
    "Flux2Transformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    # LongCat Image transformer coefficients
    "LongCatImageTransformer2DModel": [652.5980, -424.1615, 84.5526, -4.5923, 0.1694],
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
            model-specific defaults based on transformer_type.
        transformer_type: Transformer class name (e.g., "QwenImageTransformer2DModel").
            Auto-detected from pipeline.transformer.__class__.__name__ in backend.
            Defaults to "QwenImageTransformer2DModel".
    """

    rel_l1_thresh: float = 0.2
    coefficients: list[float] | None = None
    transformer_type: str = "QwenImageTransformer2DModel"

    def __post_init__(self) -> None:
        """Validate and set default coefficients."""
        if self.rel_l1_thresh <= 0:
            raise ValueError(f"rel_l1_thresh must be positive, got {self.rel_l1_thresh}")

        if self.coefficients is None:
            # Use model-specific coefficients, explicitly check if the type exists or not
            if self.transformer_type not in _MODEL_COEFFICIENTS:
                raise KeyError(
                    f"Cannot find coefficients for {self.transformer_type}. "
                    f"Supported: {list(_MODEL_COEFFICIENTS.keys())}"
                )
            self.coefficients = _MODEL_COEFFICIENTS[self.transformer_type]

        if len(self.coefficients) != 5:
            raise ValueError(f"coefficients must contain exactly 5 elements, got {len(self.coefficients)}")
