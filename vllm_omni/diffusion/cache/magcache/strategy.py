# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MagCache strategy definitions for different model architectures.

This module provides model-specific strategies for MagCache, allowing easy
extension to new models by implementing the MagCacheStrategy interface.

Architecture:
- MagCacheStrategy: Abstract base class defining the strategy interface
- FluxMagCacheStrategy: Strategy for Flux (dual-stream) models
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from diffusers.hooks._helpers import TransformerBlockMetadata


class MagCacheStrategy(ABC):
    """
    Abstract base class for MagCache strategies.

    Each model architecture requires a specific strategy to handle:
    - Residual computation (how to calculate the residual for caching)
    - Residual application (how to apply cached residual)
    - Model-specific magnitude ratios

    Implement this class to add support for new model architectures.
    """

    @property
    @abstractmethod
    def mag_ratios(self) -> torch.Tensor:
        """Return the default mag_ratios tensor for this model.

        This tensor defines caching ratios for each transformer block.
        Values should be calibrated for the specific model architecture.

        Returns:
            1D tensor of mag_ratios (one per transformer block).
        """
        pass

    def register_block_metadata(self, block_class: type) -> TransformerBlockMetadata | None:
        """Register model-specific transformer block metadata.

        Override this method to provide custom metadata for transformer blocks
        that have non-standard output formats (e.g., tuple returns).

        Args:
            block_class: The transformer block class to register.

        Returns:
            TransformerBlockMetadata if custom registration is needed, None otherwise.
        """
        return None

    def compute_residual(
        self,
        output: torch.Tensor,
        head_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residual between block output and input.

        Default implementation: output - head_input.
        Override this method for models with non-standard output formats.

        Args:
            output: Output from transformer blocks.
            head_input: Input to the first block.

        Returns:
            Residual tensor for caching.
        """
        return output - head_input

    def apply_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cached residual to hidden states.

        Default implementation: add residual to hidden_states.
        This works for most model architectures.

        Args:
            hidden_states: Current hidden states.
            residual: Cached residual to apply.

        Returns:
            Hidden states with residual added.
        """
        return hidden_states + residual

    def apply_residual_tuple(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        residual: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cached residual tuple to both hidden_states and encoder_hidden_states.

        Default implementation: add residuals separately.
        Override this method for models with specific residual application logic.

        Args:
            hidden_states: Current hidden states.
            encoder_hidden_states: Current encoder hidden states.
            residual: Tuple of (hidden_states_residual, encoder_hidden_states_residual).

        Returns:
            Tuple of (hidden_states, encoder_hidden_states) with residuals applied.
        """
        h_res, e_res = residual
        return hidden_states + h_res, encoder_hidden_states + e_res

    def compute_calibration_metrics(
        self,
        current_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        previous_residual: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[float, float, float]:
        """Compute calibration metrics for mag_ratios generation.

        Default implementation computes norm ratios and cosine dissimilarity.
        Override this method for models with custom metric computation.

        Args:
            current_residual: Residual from the current step.
            previous_residual: Residual from the previous step (None for first step).

        Returns:
            Tuple of (norm_ratio, norm_std, cos_dis):
            - norm_ratio: Mean ratio of current to previous residual norms
            - norm_std: Standard deviation of the norm ratios
            - cos_dis: Mean cosine dissimilarity (1 - cosine_similarity)
        """
        import torch.nn.functional as F

        if previous_residual is None:
            return 1.0, 0.0, 0.0

        curr_norm = torch.linalg.norm(current_residual.float(), dim=-1)
        prev_norm = torch.linalg.norm(previous_residual.float(), dim=-1)

        ratio = (curr_norm / (prev_norm + 1e-8)).mean().item()
        std = (curr_norm / (prev_norm + 1e-8)).std().item()
        cos_dis = (1 - F.cosine_similarity(current_residual, previous_residual, dim=-1, eps=1e-8)).mean().item()

        return ratio, std, cos_dis

    def get_calibration_metrics_names(self) -> tuple[str, str, str]:
        """Return the names of calibration metrics for logging.

        Returns:
            Tuple of metric names in order: (norm_ratio_name, norm_std_name, cos_dis_name)
        """
        return ("norm_ratio", "norm_std", "cos_dis")


class FluxMagCacheStrategy(MagCacheStrategy):
    """
    MagCache strategy for Flux (dual-stream) models.

    Flux architecture:
    - transformer blocks (dual-stream): image tokens and text tokens
      processed independently with separate weights
    - single transformer blocks (single-stream): concatenated sequence
      (image + text tokens) shares the same group of weights
    - Final norm_out and proj_out layers

    This strategy provides:
    - mag_ratios: Pre-computed magnitude ratios for Flux (28 steps)
    - compute_residual: Handles tuple output format
    - apply_residual_tuple: Handles decoder residual only
    """

    FLUX_MAG_RATIOS = torch.tensor(
        [
            1.0,
            1.07313,
            1.21035,
            1.04432,
            1.06818,
            1.05547,
            1.0183,
            1.03405,
            1.02574,
            1.03042,
            1.02739,
            1.01955,
            1.01585,
            1.02439,
            1.01154,
            1.01377,
            1.00994,
            1.01444,
            1.00839,
            1.02269,
            1.0007,
            1.00714,
            1.00484,
            1.01381,
            1.00426,
            0.99764,
            1.00778,
            1.00233,
        ]
    )

    @property
    def mag_ratios(self) -> torch.Tensor:
        """Return default mag_ratios for Flux model."""
        return self.FLUX_MAG_RATIOS

    def compute_residual(
        self,
        output: torch.Tensor,
        head_input: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residual for Flux output format (tuple or single tensor).

        Flux single transformer blocks return a tuple, so we extract
        the decoder output (index 1) before computing residual.
        """
        if isinstance(output, tuple):
            decoder_output = output[1] if len(output) > 1 else output[0]
            return decoder_output - head_input
        else:
            return output - head_input

    def apply_residual_tuple(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        residual: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply residual tuple for Flux - only add decoder residual.

        Flux has separate image and text processing, so the residual
        is only applied to the decoder (image) branch.
        """
        if isinstance(residual, tuple):
            decoder_residual = residual[1]
        else:
            decoder_residual = residual

        output = hidden_states + decoder_residual
        enc_output = encoder_hidden_states

        return output, enc_output

    @staticmethod
    def nearest_interp(src_array: torch.Tensor, target_length: int) -> torch.Tensor:
        """Interpolate mag_ratios to target length using nearest neighbor."""
        src_length = len(src_array)
        if target_length == 1:
            return src_array[-1:]

        scale = (src_length - 1) / (target_length - 1)
        grid = torch.arange(target_length, device=src_array.device, dtype=torch.float32)
        mapped_indices = torch.round(grid * scale).long()

        return src_array[mapped_indices]


class Flux2MagCacheStrategy(FluxMagCacheStrategy):
    """MagCache strategy for Flux2 model.

    Flux2 shares the same dual-stream architecture as Flux, but may have
    different tensor shapes in some transformer blocks, requiring special
    handling in residual computation.
    """

    FLUX2_MAG_RATIOS = torch.tensor(
        [
            1.0,
            0.96528,
            1.11559,
            1.0565,
            1.00425,
            1.0805,
            0.98616,
            1.09289,
            1.03196,
            1.06679,
            1.03941,
            1.05375,
            1.03128,
            1.05349,
            1.01983,
            1.05535,
            1.0662,
            1.05748,
            1.00318,
            1.05222,
            1.04556,
            1.0506,
            1.05058,
            1.05219,
            1.02025,
            1.05052,
            1.04143,
            1.0498,
        ]
    )

    @property
    def mag_ratios(self) -> torch.Tensor:
        """Return default mag_ratios for Flux2 model."""
        return self.FLUX2_MAG_RATIOS

    def register_block_metadata(self, block_class: type) -> TransformerBlockMetadata | None:
        """Register Flux2-specific block metadata based on block type.

        Flux2 has two block types with different output formats:
        - Dual-stream (Flux2TransformerBlock): returns (encoder_hidden_states, hidden_states)
        - Single-stream (Flux2SingleTransformerBlock): returns single tensor
        """
        class_name = block_class.__name__
        is_single_stream = "Single" in class_name

        if is_single_stream:
            return TransformerBlockMetadata(
                return_hidden_states_index=0,
                return_encoder_hidden_states_index=None,
            )
        else:
            return TransformerBlockMetadata(
                return_hidden_states_index=1,
                return_encoder_hidden_states_index=0,
            )

    def compute_residual(
        self,
        output: torch.Tensor,
        head_input: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute residual for Flux2 with dual-stream support.

        For dual-stream blocks, computes residual for both encoder and decoder branches.
        For single-stream blocks: if shapes match, computes output - input; otherwise returns output.
        """
        if isinstance(output, tuple):
            enc_output, dec_output = output[0], output[1]

            if isinstance(head_input, tuple):
                enc_head, dec_head = head_input[0], head_input[1]
            else:
                enc_head = head_input[:, : enc_output.shape[1], ...]
                dec_head = head_input[:, enc_output.shape[1] :, ...]

            enc_residual = enc_output - enc_head
            dec_residual = dec_output - dec_head

            return (enc_residual, dec_residual)
        else:
            if output.shape == head_input.shape:
                return output - head_input
            else:
                return output

    def apply_residual_tuple(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        residual: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply residual tuple for Flux2 with shape mismatch handling.

        For Flux2 dual-stream blocks, handles the case where hidden_states
        and decoder_residual may have different shapes.
        """
        if isinstance(residual, tuple):
            decoder_residual = residual[1]
        else:
            decoder_residual = residual

        if hidden_states.shape == decoder_residual.shape:
            output = hidden_states + decoder_residual
        else:
            output = hidden_states
        enc_output = encoder_hidden_states

        return output, enc_output

    def apply_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply residual for Flux2.

        For single-stream blocks: if shapes match, adds residual; otherwise returns input.
        For dual-stream blocks: applies decoder residual only.
        """
        if isinstance(residual, tuple):
            dec_residual = residual[1]
            if hidden_states.shape == dec_residual.shape:
                return hidden_states + dec_residual
            else:
                return hidden_states
        else:
            if residual.shape == hidden_states.shape:
                return hidden_states + residual
            else:
                return hidden_states


class MagCacheStrategyRegistry:
    """Registry for MagCache strategies by transformer type."""

    _registry: dict[str, MagCacheStrategy] = {}

    @classmethod
    def register(cls, name: str, strategy: MagCacheStrategy) -> None:
        """Register a strategy with explicit name.

        Args:
            name: Transformer model type identifier (e.g., "FluxTransformer2DModel")
            strategy: MagCacheStrategy instance
        """
        cls._registry[name] = strategy

    @classmethod
    def get(cls, transformer_type: str) -> MagCacheStrategy:
        """Get strategy for given transformer type."""
        if transformer_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: '{transformer_type}'. Available types: {available}")
        return cls._registry[transformer_type]

    @classmethod
    def get_if_exists(cls, transformer_type: str) -> MagCacheStrategy | None:
        """Get strategy if exists, None otherwise."""
        return cls._registry.get(transformer_type)


MagCacheStrategyRegistry.register("FluxTransformer2DModel", FluxMagCacheStrategy())

MagCacheStrategyRegistry.register("Flux2Transformer2DModel", Flux2MagCacheStrategy())


def register_strategy(
    transformer_cls_name: str,
    strategy: MagCacheStrategy,
) -> None:
    """Register a MagCache strategy for a model type.

    This allows extending MagCache support to new models without modifying
    the core MagCache code.

    Args:
        transformer_cls_name: Transformer model type identifier (class name or type string)
                               Must match pipeline.transformer.__class__.__name__
        strategy: MagCacheStrategy instance for this model type

    Example:
        >>> class MyModelMagCacheStrategy(MagCacheStrategy):
        ...     @property
        ...     def mag_ratios(self):
        ...         return torch.tensor([...])
        >>> register_strategy("MyModelTransformer", MyModelMagCacheStrategy())
    """
    MagCacheStrategyRegistry.register(transformer_cls_name, strategy)


def get_strategy(transformer_cls_name: str) -> MagCacheStrategy:
    """Get strategy function for given transformer class.

    This function looks up the strategy based on the exact transformer_cls_name string,
    which should match the transformer type in the pipeline (i.e., pipeline.transformer.__class__.__name__).

    Args:
        transformer_cls_name: Transformer class name (e.g., "FluxTransformer2DModel")
                              Must exactly match a registered strategy.

    Returns:
        MagCacheStrategy instance for the model

    Raises:
        ValueError: If model type not found in registry
    """
    return MagCacheStrategyRegistry.get(transformer_cls_name)
