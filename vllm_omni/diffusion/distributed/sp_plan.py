# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM and The HuggingFace Team
# Type definitions in this module are adapted from HuggingFace diffusers library:
#   diffusers/src/diffusers/models/_modeling_parallel.py
"""Sequence Parallelism configuration and plan type definitions.

This module provides:
1. SequenceParallelConfig: Configuration for SP (ulysses_degree, ring_degree)
2. SequenceParallelInput/Output: Type definitions for _sp_plan declarations
3. Validation utilities for _sp_plan

A _sp_plan is a dictionary that specifies how to shard/gather tensors at
different points in a model's forward pass. This allows automatic handling
of sequence parallelism without modifying the model's forward() method.

NOTE: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP)
in diffusers. We use "Sequence Parallelism" to align with vLLM-Omni terminology.

Example:
    class MyTransformer(nn.Module):
        _sp_plan = {
            # Split inputs before model forward
            "": {
                "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
                "encoder_hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
            },
            # Split RoPE embeddings after pos_embed layer
            "pos_embed": {
                0: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True),
            },
            # Gather output after proj_out layer
            "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


# =============================================================================
# Sequence Parallel Configuration
# =============================================================================


@dataclass
class SequenceParallelConfig:
    """Configuration for Sequence Parallelism using vLLM-Omni's parallel state.

    This class provides a unified interface for SP configuration that integrates
    with vLLM-Omni's existing SequenceParallelGroupCoordinator. Unlike diffusers'
    DeviceMesh-based approach (ContextParallelConfig), this uses the existing
    parallel state management.

    Note: This corresponds to `ContextParallelConfig` in diffusers library.

    Args:
        ulysses_degree: Number of devices for Ulysses (All-to-All) attention.
            Sequence is split across devices, with Q/K/V redistributed via
            All-to-All communication. Best for moderate sequences with good
            interconnect bandwidth.
        ring_degree: Number of devices for Ring attention. Sequence is split
            across devices, with K/V passed in a ring topology. Best for long
            sequences with limited memory/bandwidth.
        convert_to_fp32: Whether to convert output and LSE to float32 for
            numerical stability in ring attention.

    Note:
        ulysses_degree * ring_degree = sequence_parallel_size
        vLLM-Omni supports hybrid Ulysses-Ring attention (both > 1).
    """

    ulysses_degree: int = 1
    ring_degree: int = 1
    convert_to_fp32: bool = True

    # Internal state - populated by setup()
    _rank: int | None = None
    _world_size: int | None = None
    _device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.ulysses_degree < 1 or self.ring_degree < 1:
            raise ValueError("`ulysses_degree` and `ring_degree` must be >= 1.")

        if self.ulysses_degree == 1 and self.ring_degree == 1:
            raise ValueError(
                "At least one of `ulysses_degree` or `ring_degree` must be > 1 to use sequence parallelism."
            )

    @property
    def sequence_parallel_size(self) -> int:
        """Total sequence parallel world size."""
        return self.ulysses_degree * self.ring_degree

    def get_world_size(self) -> int:
        """Get the sequence parallel world size from parallel state.

        Returns:
            The world size for sequence parallelism.

        Raises:
            RuntimeError: If parallel state is not initialized.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size

        return get_sequence_parallel_world_size()

    def get_rank(self) -> int:
        """Get the current rank in the sequence parallel group.

        Returns:
            The rank within the sequence parallel group.

        Raises:
            RuntimeError: If parallel state is not initialized.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_rank

        return get_sequence_parallel_rank()

    def get_ulysses_world_size(self) -> int:
        """Get the Ulysses parallel world size.

        Returns:
            The world size for Ulysses (All-to-All) parallelism.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ulysses_parallel_world_size

        return get_ulysses_parallel_world_size()

    def get_ulysses_rank(self) -> int:
        """Get the current rank in the Ulysses parallel group.

        Returns:
            The rank within the Ulysses parallel group.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ulysses_parallel_rank

        return get_ulysses_parallel_rank()

    def get_ring_world_size(self) -> int:
        """Get the Ring parallel world size.

        Returns:
            The world size for Ring attention parallelism.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ring_parallel_world_size

        return get_ring_parallel_world_size()

    def get_ring_rank(self) -> int:
        """Get the current rank in the Ring parallel group.

        Returns:
            The rank within the Ring parallel group.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ring_parallel_rank

        return get_ring_parallel_rank()

    def setup(self, rank: int, world_size: int, device: torch.device) -> None:
        """Initialize the config with runtime parallel state.

        This is called automatically when sequence parallelism is enabled.

        Args:
            rank: The global rank of this process.
            world_size: Total world size.
            device: The device for this rank.
        """
        self._rank = rank
        self._world_size = world_size
        self._device = device

        expected_sp_size = self.ulysses_degree * self.ring_degree
        actual_sp_size = self.get_world_size()

        if expected_sp_size != actual_sp_size:
            raise ValueError(
                f"Configuration mismatch: ulysses_degree ({self.ulysses_degree}) * "
                f"ring_degree ({self.ring_degree}) = {expected_sp_size}, but "
                f"actual sequence parallel world size is {actual_sp_size}."
            )

    def is_initialized(self) -> bool:
        """Check if the config has been initialized with runtime state.

        Returns:
            True if setup() has been called, False otherwise.
        """
        return self._rank is not None


# =============================================================================
# Sequence Parallel Plan Type Definitions
# =============================================================================


@dataclass(frozen=True)
class SequenceParallelInput:
    """Configuration for splitting an input tensor across sequence parallel ranks.

    This specifies how to shard a tensor in the pre-forward or post-forward hook
    of a layer. The tensor will be split along the specified dimension.

    Note: This corresponds to `ContextParallelInput` in diffusers library.

    Args:
        split_dim: The dimension along which to split the tensor.
        expected_dims: Expected number of dimensions. If provided, validates that
            the tensor has this many dimensions before splitting. If the tensor
            has a different number of dimensions, splitting is skipped with a warning.
        split_output: If True, split the output of the layer instead of the input.
            This is useful for layers whose outputs should be split after preprocessing
            (e.g., RoPE embeddings).
        auto_pad: If True, automatically pad the tensor if its size along split_dim
            is not divisible by world_size. Creates an attention mask to indicate
            valid vs padding positions. The mask is stored in ForwardContext.
            Note: Ring attention does not support attention mask, so auto_pad
            should only be used with Ulysses SP.

    Example:
        # Split hidden_states along sequence dimension (dim 1)
        SequenceParallelInput(split_dim=1, expected_dims=3)

        # Split RoPE output along sequence dimension (dim 0)
        SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True)

        # Split with auto-padding for variable-length sequences
        SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True)
    """

    split_dim: int
    expected_dims: int | None = None
    split_output: bool = False
    auto_pad: bool = False

    def __repr__(self) -> str:
        return (
            f"SequenceParallelInput(split_dim={self.split_dim}, "
            f"expected_dims={self.expected_dims}, split_output={self.split_output}, "
            f"auto_pad={self.auto_pad})"
        )


@dataclass(frozen=True)
class SequenceParallelOutput:
    """Configuration for gathering an output tensor across sequence parallel ranks.

    This specifies how to gather a tensor in the post-forward hook of a layer.
    The tensor will be gathered along the specified dimension from all ranks.

    Note: This corresponds to `ContextParallelOutput` in diffusers library.

    Args:
        gather_dim: The dimension along which to gather the tensor.
        expected_dims: Expected number of dimensions. If provided, validates that
            the tensor has this many dimensions before gathering.

    Example:
        # Gather output along sequence dimension (dim 1)
        SequenceParallelOutput(gather_dim=1, expected_dims=3)
    """

    gather_dim: int
    expected_dims: int | None = None

    def __repr__(self) -> str:
        return f"SequenceParallelOutput(gather_dim={self.gather_dim}, expected_dims={self.expected_dims})"


@dataclass(frozen=True)
class SequenceParallelPartialInput:
    """Configuration for partially splitting a tensor (e.g., split image part, keep text part).

    This is designed for models like LongCat/Qwen where RoPE embeddings need special handling:
    - Text portion: kept full across all ranks (for joint attention)
    - Image portion: split across ranks

    The tensor is assumed to be concatenated as [text_part, image_part] along split_dim.

    Note: This is an extension beyond diffusers' standard ContextParallelInput,
    designed for vLLM-Omni's dual-stream attention models.

    Args:
        split_dim: The dimension along which to split the image portion.
        text_len_source: How to determine text length:
            - str: Name of a forward parameter that contains text length
            - int: Fixed text length value
        expected_dims: Expected number of dimensions for validation.
        split_output: If True, split the output instead of input.

    Example:
        # Split RoPE: text portion (from txt_ids.shape[0]) kept full, image portion split
        SequenceParallelPartialInput(
            split_dim=0,
            text_len_source="txt_ids",  # Get text length from txt_ids.shape[0]
            expected_dims=2,
            split_output=True,
        )

        # Or with fixed text length
        SequenceParallelPartialInput(
            split_dim=0,
            text_len_source=512,  # Fixed text length
            expected_dims=2,
            split_output=True,
        )
    """

    split_dim: int
    text_len_source: str | int
    expected_dims: int | None = None
    split_output: bool = False

    def __repr__(self) -> str:
        return (
            f"SequenceParallelPartialInput(split_dim={self.split_dim}, "
            f"text_len_source={self.text_len_source!r}, expected_dims={self.expected_dims}, "
            f"split_output={self.split_output})"
        )


# =============================================================================
# Type Aliases for _sp_plan Structure
# =============================================================================

# Any input config type
AnySequenceParallelInput = SequenceParallelInput | SequenceParallelPartialInput

# Input specification: maps parameter names (str) or output indices (int) to split config
SequenceParallelInputType = dict[
    str | int,
    AnySequenceParallelInput | list[AnySequenceParallelInput] | tuple[AnySequenceParallelInput, ...],
]

# Output specification: single or multiple gather configs
SequenceParallelOutputType = SequenceParallelOutput | list[SequenceParallelOutput] | tuple[SequenceParallelOutput, ...]

# Full model plan: maps module names to input/output specifications
# - Key "" refers to the model itself (root level)
# - Key "module_name" refers to a submodule
# - Key "module_name.*" refers to all children of a ModuleList
#
# Example of a complete _sp_plan:
#
#     _sp_plan = {
#         # Root level: split model inputs before any submodule
#         "": {
#             "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
#         },
#         # Submodule: split outputs of pos_embed (RoPE) layer
#         "pos_embed": {
#             0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
#             1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
#         },
#         # Submodule: gather outputs of proj_out layer
#         "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
#     }
#
SequenceParallelModelPlan = dict[str, SequenceParallelInputType | SequenceParallelOutputType]


# =============================================================================
# Validation Utilities
# =============================================================================


def _is_valid_input_config(value: object) -> bool:
    """Check if a value is a valid input configuration type."""
    return isinstance(value, (SequenceParallelInput, SequenceParallelPartialInput))


def _is_valid_input_config_list(value: object) -> bool:
    """Check if a value is a list/tuple of valid input configurations."""
    if not isinstance(value, (list, tuple)):
        return False
    return all(_is_valid_input_config(x) for x in value)


def validate_sp_plan(plan: SequenceParallelModelPlan) -> None:
    """Validate a _sp_plan dictionary for correctness.

    Args:
        plan: The _sp_plan dictionary to validate.

    Raises:
        ValueError: If the plan is invalid.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"_sp_plan must be a dict, got {type(plan).__name__}")

    for module_id, module_plan in plan.items():
        if not isinstance(module_id, str):
            raise ValueError(f"_sp_plan keys must be strings, got {type(module_id).__name__}")

        # Check if it's an output specification (SequenceParallelOutput or list/tuple thereof)
        if isinstance(module_plan, SequenceParallelOutput):
            continue
        if isinstance(module_plan, (list, tuple)):
            if all(isinstance(x, SequenceParallelOutput) for x in module_plan):
                continue
            if _is_valid_input_config_list(module_plan):
                # List of inputs for a specific parameter (when output is tuple)
                continue

        # Otherwise, should be an input specification dict
        if isinstance(module_plan, dict):
            for key, value in module_plan.items():
                if not isinstance(key, (str, int)):
                    raise ValueError(
                        f"Input spec keys must be str or int, got {type(key).__name__} for module '{module_id}'"
                    )
                if isinstance(key, int) and not _is_valid_input_config(value):
                    raise ValueError(
                        f"Integer keys (output indices) must map to SequenceParallelInput/PartialInput, "
                        f"got {type(value).__name__} for module '{module_id}'[{key}]"
                    )
                if _is_valid_input_config(value):
                    if isinstance(key, int) and not value.split_output:
                        raise ValueError(
                            f"Integer keys (output indices) require split_output=True, "
                            f"got split_output=False for module '{module_id}'[{key}]"
                        )
                elif _is_valid_input_config_list(value):
                    pass  # Valid list of input configs
                else:
                    raise ValueError(
                        f"Input spec values must be SequenceParallelInput/PartialInput or list thereof, "
                        f"got {type(value).__name__} for module '{module_id}'['{key}']"
                    )
        else:
            raise ValueError(
                f"_sp_plan values must be dict (input spec) or SequenceParallelOutput, "
                f"got {type(module_plan).__name__} for module '{module_id}'"
            )


def get_sp_plan_from_model(model: nn.Module) -> SequenceParallelModelPlan | None:
    """Get the _sp_plan from a model if it exists.

    Args:
        model: The model to get the plan from.

    Returns:
        The _sp_plan dictionary, or None if not defined.
    """
    plan = getattr(model, "_sp_plan", None)
    if plan is not None:
        validate_sp_plan(plan)
    return plan
