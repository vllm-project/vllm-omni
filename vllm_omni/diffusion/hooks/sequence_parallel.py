# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and the HuggingFace Team.
# All rights reserved.
#
# This module is adapted from HuggingFace diffusers library:
#   diffusers/src/diffusers/hooks/context_parallel.py
#
# NOTE: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in diffusers.
# We use the term "Sequence Parallelism" to align with vLLM-Omni's existing terminology.
#
# Key adaptations from diffusers:
#   - ModuleForwardMetadata: parameter lookup logic (adapted)
#   - SequenceParallelSplitHook/GatherHook: hook structure (adapted from ContextParallel*)
#   - apply_sequence_parallel: registration logic (adapted from apply_context_parallel)
#
# Key differences from diffusers:
#   - Uses vLLM-Omni's SequenceParallelGroupCoordinator instead of DeviceMesh
#   - Uses sp_shard/sp_gather from sp_sharding.py instead of funcol operations
#   - Supports Ulysses + Ring hybrid parallelism
#
"""Sequence Parallelism hooks for non-intrusive SP support.

This module implements the hook-based mechanism for applying sequence parallelism
to models without modifying their forward() methods.

Usage:
    1. Define _sp_plan on your model class (corresponds to diffusers' _cp_plan)
    2. Call apply_sequence_parallel(model, config, plan) to enable SP
    3. Call remove_sequence_parallel(model, plan) to disable SP

The hooks automatically shard inputs before forward and gather outputs after,
based on the plan specification.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.sp_plan import (
    AnySequenceParallelInput,
    SequenceParallelConfig,
    SequenceParallelInput,
    SequenceParallelModelPlan,
    SequenceParallelOutput,
    SequenceParallelPartialInput,
)
from vllm_omni.diffusion.distributed.sp_sharding import sp_gather, sp_shard
from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook

logger = init_logger(__name__)

# Hook name templates for identifying SP hooks
_SP_INPUT_HOOK_TEMPLATE = "sp_input---{}"
_SP_OUTPUT_HOOK_TEMPLATE = "sp_output---{}"


@dataclass
class ModuleForwardMetadata:
    """Metadata for mapping forward() parameter names to args/kwargs positions.

    This caches the inspection of a module's forward signature to efficiently
    locate parameters by name in subsequent calls.
    """

    cached_parameter_indices: dict[str, int] | None = None
    _cls: type | None = None

    def _get_parameter_from_args_kwargs(
        self,
        identifier: str,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> tuple[Any, bool, int | None]:
        """Get a parameter value from args or kwargs by name.

        Args:
            identifier: The parameter name to look up.
            args: Positional arguments passed to forward.
            kwargs: Keyword arguments passed to forward.

        Returns:
            Tuple of (value, is_kwarg, index).
            - value: The parameter value (or None if not found)
            - is_kwarg: True if found in kwargs
            - index: Position in args if found there

        Raises:
            ValueError: If parameter not found in signature.
        """
        kwargs = kwargs or {}

        # First check kwargs
        if identifier in kwargs:
            return kwargs[identifier], True, None

        # Check cached indices
        if self.cached_parameter_indices is not None:
            index = self.cached_parameter_indices.get(identifier, None)
            if index is None:
                raise ValueError(f"Parameter '{identifier}' not found in cached indices.")
            if index < len(args):
                return args[index], False, index
            return None, False, index

        # Build cache from forward signature
        if self._cls is None:
            raise ValueError("Model class is not set for metadata.")

        parameters = list(inspect.signature(self._cls.forward).parameters.keys())
        parameters = parameters[1:]  # Skip `self`
        self.cached_parameter_indices = {param: i for i, param in enumerate(parameters)}

        if identifier not in self.cached_parameter_indices:
            raise ValueError(f"Parameter '{identifier}' not found in function signature.")

        index = self.cached_parameter_indices[identifier]

        if index >= len(args):
            return None, False, index

        return args[index], False, index


def _unwrap_module(module: nn.Module) -> nn.Module:
    """Unwrap a module from any wrappers to get the original class.

    Args:
        module: Potentially wrapped module.

    Returns:
        The unwrapped module.
    """
    # Handle common wrappers
    while hasattr(module, "_modules") and len(module._modules) == 1:
        inner = next(iter(module._modules.values()))
        if inner is not None:
            module = inner
        else:
            break
    return module


class SequenceParallelSplitHook(ModelHook):
    """Hook for splitting inputs before a module's forward pass.

    This hook is registered to modules that need their inputs sharded
    across sequence parallel ranks. It intercepts the forward call,
    shards specified inputs according to the plan, and passes the
    sharded inputs to the original forward.

    For split_output=True inputs, it shards the output instead.

    Supports both SequenceParallelInput (full split) and SequenceParallelPartialInput
    (partial split for text/image separation).

    Note: This corresponds to `ContextParallelSplitHook` in diffusers.
    """

    def __init__(
        self,
        metadata: dict[str | int, AnySequenceParallelInput | list[AnySequenceParallelInput]],
        config: SequenceParallelConfig,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.config = config
        self.module_forward_metadata: ModuleForwardMetadata | None = None
        # Cache for text lengths resolved from kwargs
        self._text_len_cache: dict[str, int] = {}

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        cls = _unwrap_module(module).__class__
        self.module_forward_metadata = ModuleForwardMetadata(_cls=cls)
        return module

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        """Shard inputs before forward."""
        args_list = list(args)
        # Clear text length cache for this forward pass
        self._text_len_cache.clear()

        for name, spm in self.metadata.items():
            # Skip if this is a split_output entry (handled in post_forward)
            if isinstance(spm, (SequenceParallelInput, SequenceParallelPartialInput)) and spm.split_output:
                continue

            # Get the parameter value
            input_val, is_kwarg, index = self.module_forward_metadata._get_parameter_from_args_kwargs(
                name, args_list, kwargs
            )

            if input_val is None:
                continue

            # Shard the input
            if isinstance(input_val, torch.Tensor):
                input_val = self._prepare_sp_input(input_val, spm, args_list, kwargs)
            elif isinstance(input_val, (list, tuple)):
                # Handle list/tuple of tensors with per-element config
                if not isinstance(spm, (list, tuple)):
                    raise ValueError(
                        f"Expected list/tuple of SequenceParallelInput for parameter '{name}' "
                        f"which is a list/tuple, but got {type(spm).__name__}"
                    )
                if len(input_val) != len(spm):
                    raise ValueError(f"Expected {len(spm)} elements for parameter '{name}', got {len(input_val)}")
                sharded_input_val = []
                for i, x in enumerate(input_val):
                    if torch.is_tensor(x) and not spm[i].split_output:
                        x = self._prepare_sp_input(x, spm[i], args_list, kwargs)
                    sharded_input_val.append(x)
                input_val = type(input_val)(sharded_input_val)
            else:
                raise ValueError(f"Unsupported input type for sharding: {type(input_val).__name__}")

            # Update args or kwargs
            if is_kwarg:
                kwargs[name] = input_val
            elif index is not None and index < len(args_list):
                args_list[index] = input_val
            else:
                raise ValueError(f"Failed to update parameter '{name}' after sharding.")

        # Store kwargs for post_forward to resolve text lengths
        self._last_kwargs = kwargs
        self._last_args = tuple(args_list)

        return tuple(args_list), kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Shard outputs for split_output=True entries."""
        is_tensor = isinstance(output, torch.Tensor)
        is_tensor_list = isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output)

        if not is_tensor and not is_tensor_list:
            # No tensor outputs to shard
            return output

        output_list = [output] if is_tensor else list(output)

        for index, spm in self.metadata.items():
            if not isinstance(index, int):
                continue
            if not isinstance(spm, (SequenceParallelInput, SequenceParallelPartialInput)) or not spm.split_output:
                continue
            if index >= len(output_list):
                raise ValueError(f"Index {index} out of bounds for output of length {len(output_list)}.")

            output_list[index] = self._prepare_sp_input(output_list[index], spm, self._last_args, self._last_kwargs)

        return output_list[0] if is_tensor else type(output)(output_list)

    def _resolve_text_len(
        self,
        sp_input: SequenceParallelPartialInput,
        args: tuple,
        kwargs: dict,
    ) -> int:
        """Resolve text length from the source specification."""
        source = sp_input.text_len_source

        if isinstance(source, int):
            return source

        # String source - look up from kwargs or args
        if source in self._text_len_cache:
            return self._text_len_cache[source]

        # Try to get from kwargs/args
        try:
            val, _, _ = self.module_forward_metadata._get_parameter_from_args_kwargs(source, args, kwargs)
            if val is None:
                raise ValueError(f"Parameter '{source}' is None, cannot determine text length.")
            if isinstance(val, torch.Tensor):
                # TODO: Currently assumes batch_size=1, where shape[0] is sequence length.
                # For batch inference support, this should be updated to handle
                # shape (batch_size, seq_len, ...) where text_len varies per sample.
                text_len = val.shape[0]
            elif isinstance(val, int):
                text_len = val
            else:
                raise ValueError(f"Cannot determine text length from '{source}' of type {type(val).__name__}")
            self._text_len_cache[source] = text_len
            return text_len
        except ValueError as e:
            raise ValueError(f"Failed to resolve text_len_source '{source}': {e}") from e

    def _prepare_sp_input(
        self,
        x: torch.Tensor,
        sp_input: AnySequenceParallelInput,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Shard a tensor according to the input specification."""
        kwargs = kwargs or {}

        if sp_input.expected_dims is not None and x.dim() != sp_input.expected_dims:
            logger.warning_once(f"Expected tensor with {sp_input.expected_dims} dims, got {x.dim()}. Skipping split.")
            return x

        if isinstance(sp_input, SequenceParallelInput):
            # Full split with optional auto-padding
            if sp_input.auto_pad:
                return self._shard_with_auto_pad(x, sp_input.split_dim)
            return sp_shard(x, sp_input.split_dim, validate=False)
        elif isinstance(sp_input, SequenceParallelPartialInput):
            # Partial split: keep text portion, split image portion
            text_len = self._resolve_text_len(sp_input, args, kwargs)
            dim = sp_input.split_dim

            # Split tensor into text and image portions
            text_part = x.narrow(dim, 0, text_len)
            image_part = x.narrow(dim, text_len, x.size(dim) - text_len)

            # Only shard the image portion
            image_part_sharded = sp_shard(image_part, dim, validate=False)

            # Concatenate back: [text_full, image_sharded]
            return torch.cat([text_part, image_part_sharded], dim=dim)
        else:
            raise ValueError(f"Unsupported input config type: {type(sp_input).__name__}")

    def _shard_with_auto_pad(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Shard tensor with automatic padding and attention mask creation.

        When sequence length is not divisible by SP world size, this method:
        1. Pads the tensor to make it divisible
        2. Creates an attention mask indicating valid vs padding positions
        3. Stores the mask and padding info in ForwardContext
        """
        from vllm_omni.diffusion.attention.selector import get_attn_backend
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_ring_parallel_world_size,
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )
        from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

        world_size = get_sequence_parallel_world_size()
        if world_size == 1:
            return x

        seq_len = x.size(dim)
        remainder = seq_len % world_size

        if remainder == 0:
            # No padding needed
            return sp_shard(x, dim, validate=False)

        # Check backend compatibility
        attn_backend = get_attn_backend(-1)
        if not attn_backend.supports_attention_mask:
            raise ValueError(
                f"Sequence length ({seq_len}) is not divisible by SP world size ({world_size}). "
                f"Cannot use {attn_backend.get_name()} which does not support attention_mask. "
                f"Please switch to SDPA or Ascend attention backend."
            )

        # Ring attention does not support attention_mask
        if get_ring_parallel_world_size() > 1:
            raise ValueError(
                f"Sequence length ({seq_len}) is not divisible by SP world size ({world_size}). "
                f"Cannot use Ring attention which does not support attention_mask. "
                f"Please switch to Ulysses SP only."
            )

        # Calculate padding
        pad_size = world_size - remainder
        padded_seq_len = seq_len + pad_size

        # Pad the tensor
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, padding], dim=dim)

        # Store padding info in forward context (only once, for primary tensor)
        # Attention layers will create masks dynamically using this info
        if is_forward_context_available():
            ctx = get_forward_context()
            # Only set if not already set (first auto_pad tensor wins)
            if ctx.sp_original_seq_len is None:
                ctx.sp_padding_size = pad_size
                ctx.sp_original_seq_len = seq_len
                logger.debug(
                    f"Auto-padded sequence from {seq_len} to {padded_seq_len} "
                    f"(pad_size={pad_size}, world_size={world_size}, dim={dim})"
                )

        # Shard the padded tensor
        rank = get_sequence_parallel_rank()
        return x_padded.chunk(world_size, dim=dim)[rank]


class SequenceParallelGatherHook(ModelHook):
    """Hook for gathering outputs after a module's forward pass.

    This hook is registered to modules that need their outputs gathered
    from all sequence parallel ranks. It intercepts the output and gathers
    it according to the plan specification.

    Note: This corresponds to `ContextParallelGatherHook` in diffusers.
    """

    def __init__(
        self,
        metadata: SequenceParallelOutput | list[SequenceParallelOutput],
        config: SequenceParallelConfig,
    ) -> None:
        super().__init__()
        if isinstance(metadata, SequenceParallelOutput):
            metadata = [metadata]
        self.metadata = metadata
        self.config = config

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        return module

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Gather outputs after forward and remove padding if applied."""
        from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

        is_tensor = isinstance(output, torch.Tensor)

        if is_tensor:
            output = [output]
        elif not (isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output)):
            raise ValueError(f"Expected tensor or list/tuple of tensors, got {type(output).__name__}")

        output = list(output)

        if len(output) != len(self.metadata):
            raise ValueError(f"Expected {len(self.metadata)} outputs, got {len(output)}.")

        # Check if padding was applied during split
        original_seq_len = None
        if is_forward_context_available():
            ctx = get_forward_context()
            original_seq_len = ctx.sp_original_seq_len

        for i, spm in enumerate(self.metadata):
            if spm is None:
                continue

            x = output[i]
            if spm.expected_dims is not None and x.dim() != spm.expected_dims:
                logger.warning_once(
                    f"Expected output tensor with {spm.expected_dims} dims, got {x.dim()}. Skipping gather."
                )
                continue

            # Gather from all ranks
            gathered = sp_gather(x, spm.gather_dim, validate=False)

            # Remove padding if it was applied
            if original_seq_len is not None and gathered.size(spm.gather_dim) > original_seq_len:
                gathered = gathered.narrow(spm.gather_dim, 0, original_seq_len)
                logger.debug(f"Removed padding: gathered shape {gathered.shape} (original_seq_len={original_seq_len})")

            output[i] = gathered

        return output[0] if is_tensor else type(output)(output)


def _get_submodule_by_name(model: nn.Module, name: str) -> nn.Module | list[nn.Module]:
    """Get a submodule by dotted name, supporting wildcards.

    Args:
        model: The root module.
        name: Dotted path to submodule. Use "*" to match all children
            of a ModuleList.

    Returns:
        The submodule or list of submodules if wildcard used.

    Raises:
        ValueError: If the path is invalid or module not found.
    """
    if name.count("*") > 1:
        raise ValueError("Wildcard '*' can only be used once in the name")
    return _find_submodule_by_name(model, name)


def _find_submodule_by_name(model: nn.Module, name: str) -> nn.Module | list[nn.Module]:
    """Recursive helper for _get_submodule_by_name."""
    if name == "":
        return model

    first_atom, remaining_name = name.split(".", 1) if "." in name else (name, "")

    if first_atom == "*":
        if not isinstance(model, nn.ModuleList):
            raise ValueError("Wildcard '*' can only be used with ModuleList")
        submodules = []
        for submodule in model:
            subsubmodules = _find_submodule_by_name(submodule, remaining_name)
            if not isinstance(subsubmodules, list):
                subsubmodules = [subsubmodules]
            submodules.extend(subsubmodules)
        return submodules
    else:
        if hasattr(model, first_atom):
            submodule = getattr(model, first_atom)
            return _find_submodule_by_name(submodule, remaining_name)
        else:
            raise ValueError(f"'{first_atom}' is not a submodule of '{model.__class__.__name__}'")


def apply_sequence_parallel(
    module: nn.Module,
    config: SequenceParallelConfig,
    plan: SequenceParallelModelPlan,
) -> None:
    """Apply sequence parallel hooks to a model according to the plan.

    This function registers hooks on the specified submodules to automatically
    shard inputs and gather outputs for sequence parallelism.

    Note: This corresponds to `apply_context_parallel` in diffusers.

    The complete SP flow is:
    1. Input sharding (SequenceParallelSplitHook): Split sequence across SP ranks
    2. Attention parallelism (handled by vLLM-Omni's Attention layer):
       - Ulysses: All-to-All over Q/K/V heads
       - Ring: K/V circulation in ring topology
       - Hybrid: Both (Ulysses handles head redistribution, Ring handles K/V)
    3. Output gathering (SequenceParallelGatherHook): Gather sequence from SP ranks

    Args:
        module: The model to apply SP to.
        config: The sequence parallel configuration.
        plan: Dictionary mapping module names to input/output specifications.

    Example:
        config = SequenceParallelConfig(ulysses_degree=2)
        plan = {
            "": {"hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3)},
            "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }
        apply_sequence_parallel(model, config, plan)

    Note:
        vLLM-Omni's Attention layer automatically handles the internal
        parallelism (Ulysses All-to-All or Ring attention) based on the
        forward_context configuration. This function only handles input/output
        sharding for the model as a whole.
    """
    logger.debug(
        f"Applying sequence parallel with config: ulysses={config.ulysses_degree}, "
        f"ring={config.ring_degree}, plan keys: {list(plan.keys())}"
    )

    for module_id, sp_model_plan in plan.items():
        submodule = _get_submodule_by_name(module, module_id)
        if not isinstance(submodule, list):
            submodule = [submodule]

        logger.debug(f"Applying SP hooks to '{module_id}' ({len(submodule)} module(s))")

        for m in submodule:
            if isinstance(sp_model_plan, dict):
                # Input specification
                hook = SequenceParallelSplitHook(sp_model_plan, config)
                hook_name = _SP_INPUT_HOOK_TEMPLATE.format(module_id)
            elif isinstance(sp_model_plan, (SequenceParallelOutput, list, tuple)):
                # Output specification
                if isinstance(sp_model_plan, SequenceParallelOutput):
                    sp_model_plan = [sp_model_plan]
                if not all(isinstance(x, SequenceParallelOutput) or x is None for x in sp_model_plan):
                    raise ValueError(f"Expected SequenceParallelOutput elements, got {sp_model_plan}")
                hook = SequenceParallelGatherHook(sp_model_plan, config)
                hook_name = _SP_OUTPUT_HOOK_TEMPLATE.format(module_id)
            else:
                raise ValueError(f"Unsupported plan type: {type(sp_model_plan).__name__}")

            registry = HookRegistry.get_or_create(m)
            registry.register_hook(hook_name, hook)


def remove_sequence_parallel(
    module: nn.Module,
    plan: SequenceParallelModelPlan,
) -> None:
    """Remove sequence parallel hooks from a model.

    Note: This corresponds to `remove_context_parallel` in diffusers.

    Args:
        module: The model to remove SP from.
        plan: The same plan used when applying SP.
    """
    for module_id, sp_model_plan in plan.items():
        submodule = _get_submodule_by_name(module, module_id)
        if not isinstance(submodule, list):
            submodule = [submodule]

        for m in submodule:
            registry = getattr(m, "_hook_registry", None)
            if registry is None:
                continue

            if isinstance(sp_model_plan, dict):
                hook_name = _SP_INPUT_HOOK_TEMPLATE.format(module_id)
            elif isinstance(sp_model_plan, (SequenceParallelOutput, list, tuple)):
                hook_name = _SP_OUTPUT_HOOK_TEMPLATE.format(module_id)
            else:
                continue

            registry.remove_hook(hook_name)


def enable_sequence_parallel_for_model(
    model: nn.Module,
    config: SequenceParallelConfig | None = None,
) -> None:
    """Enable sequence parallelism for a model using its _sp_plan.

    This is a convenience function that reads the model's _sp_plan attribute
    and applies sequence parallelism automatically.

    Note: This corresponds to `enable_context_parallel_for_model` in diffusers,
    but uses vLLM-Omni's _sp_plan instead of diffusers' _cp_plan.

    The function performs two main tasks:
    1. Applies _sp_plan hooks to shard inputs and gather outputs
    2. Ensures Attention layers are configured for the correct parallel mode
       (handled automatically by vLLM-Omni's forward_context mechanism)

    Args:
        model: The model to enable SP for. Must have a _sp_plan attribute.
        config: Optional config. If None, uses default based on current
            parallel state.

    Raises:
        ValueError: If model has no _sp_plan defined.

    Note:
        vLLM-Omni supports Ulysses + Ring hybrid parallelism:
        - ulysses_degree > 1: Uses All-to-All communication over Q/K/V heads
        - ring_degree > 1: Uses Ring attention with K/V passing
        - Both > 1: Hybrid mode (Ulysses handles head redistribution,
          Ring handles K/V circulation)
    """
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_ring_parallel_world_size,
        get_ulysses_parallel_world_size,
    )
    from vllm_omni.diffusion.distributed.sp_plan import get_sp_plan_from_model

    plan = get_sp_plan_from_model(model)
    if plan is None:
        raise ValueError(
            f"Model {model.__class__.__name__} has no _sp_plan defined. "
            f"Define _sp_plan as a class attribute or pass a plan explicitly."
        )

    if config is None:
        # Create config from current parallel state
        ulysses_degree = get_ulysses_parallel_world_size()
        ring_degree = get_ring_parallel_world_size()
        config = SequenceParallelConfig(
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
        )
        if ulysses_degree > 1 and ring_degree > 1:
            mode = "hybrid"
        elif ulysses_degree > 1:
            mode = "ulysses"
        else:
            mode = "ring"
        logger.info(
            f"Created SP config from parallel state: "
            f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, "
            f"mode={mode}"
        )

    apply_sequence_parallel(model, config, plan)
    logger.info(f"Enabled sequence parallelism for {model.__class__.__name__}")


def disable_sequence_parallel_for_model(model: nn.Module) -> None:
    """Disable sequence parallelism for a model.

    Note: This corresponds to `disable_context_parallel_for_model` in diffusers.

    Args:
        model: The model to disable SP for.
    """
    from vllm_omni.diffusion.distributed.sp_plan import get_sp_plan_from_model

    plan = get_sp_plan_from_model(model)
    if plan is not None:
        remove_sequence_parallel(model, plan)
