# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization config for diffusion transformers."""

from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from .base import DiffusionQuantizationConfig


class DiffusionFp8Config(DiffusionQuantizationConfig):
    """FP8 quantization config optimized for diffusion transformers.

    Uses dynamic activation scaling (no calibration dataset needed) and
    online weight quantization from BF16/FP16 checkpoints.

    Device Compatibility:
        - Turing (SM 75+): Weight-only FP8 via Marlin kernel
        - Ada/Hopper (SM 89+): Full W8A8 FP8 with native hardware support

    The kernel selection is automatic based on GPU capability.

    Args:
        activation_scheme: Activation quantization scheme.
            - "dynamic": Per-token dynamic scaling (default, no calibration)
            - "static": Single per-tensor scale (requires calibration)
        weight_block_size: Block size for block-wise weight quantization.
            Format: [block_n, block_k]. If None, uses per-tensor scaling.
        ignored_layers: List of layer name patterns to skip quantization.
    """

    # Tight coupling with vLLM's Fp8Config - delegates get_name() and get_min_capability()
    quant_config_cls = Fp8Config

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_block_size: list[int] | None = None,
        ignored_layers: list[str] | None = None,
    ):
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        self.ignored_layers = ignored_layers or []

        # Create underlying vLLM FP8 config
        self._vllm_config = Fp8Config(
            is_checkpoint_fp8_serialized=False,  # Online quantization from BF16
            activation_scheme=activation_scheme,
            weight_block_size=weight_block_size,
            ignored_layers=ignored_layers,
        )
