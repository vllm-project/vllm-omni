# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.platforms import current_platform

from vllm_omni.diffusion.lora.layers import (
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
    DiffusionRowParallelLinearWithLoRA,
)


def _select_moe_lora_wrapper_cls() -> type:
    """Select the vLLM or Ascend MoE LoRA wrapper for the current platform."""
    if current_platform.device_type == "npu":
        from vllm_ascend.lora.fused_moe import AscendFusedMoEWithLoRA

        return AscendFusedMoEWithLoRA
    return FusedMoEWithLoRA


def _move_moe_lora_buffers(instance: FusedMoEWithLoRA, device: torch.device) -> None:
    """Keep MoE LoRA sidecars resident while DLO moves base weights.

    Rebuild expert views after moving their source W13/W2 stacks.
    """
    for attr in (
        "w13_lora_a_stacked",
        "w13_lora_b_stacked",
        "w2_lora_a_stacked",
        "w2_lora_b_stacked",
    ):
        buffers = getattr(instance, attr)
        setattr(instance, attr, tuple(buffer.to(device=device) for buffer in buffers))
    instance.adapter_enabled = instance.adapter_enabled.to(device=device)

    lora_a_stacked: list[torch.Tensor] = []
    lora_b_stacked: list[torch.Tensor] = []
    for lora_id in range(instance.max_loras):
        for expert_id in range(instance.local_num_experts):
            w13_a_expert_id = 0 if instance.enable_moe_shared_loras else expert_id
            w2_b_expert_id = 0 if instance.enable_moe_shared_loras else expert_id
            lora_a_stacked.extend(
                (
                    instance.w13_lora_a_stacked[0][lora_id][w13_a_expert_id],
                    instance.w2_lora_a_stacked[0][lora_id][expert_id],
                )
            )
            lora_b_stacked.extend(
                (
                    instance.w13_lora_b_stacked[0][lora_id][expert_id],
                    instance.w2_lora_b_stacked[0][lora_id][w2_b_expert_id],
                )
            )
            if instance._w13_slices == 2:
                lora_a_stacked.append(instance.w13_lora_a_stacked[1][lora_id][w13_a_expert_id])
                lora_b_stacked.append(instance.w13_lora_b_stacked[1][lora_id][expert_id])
    instance.lora_a_stacked = lora_a_stacked
    instance.lora_b_stacked = lora_b_stacked


def _validate_moe_lora_token_capacity(hidden_states: torch.Tensor, max_tokens: int) -> None:
    num_tokens = hidden_states.shape[0]
    if num_tokens > max_tokens:
        raise ValueError(
            f"MoE LoRA token count {num_tokens} exceeds the configured capacity {max_tokens}. "
            "Increase max_num_batched_tokens before model initialization."
        )


def _match_target_modules(module_name: str, target_modules: list[str]) -> bool:
    """from vllm/lora/model_manager.py _match_target_modules, helper function"""
    import regex as re

    return any(
        re.match(rf".*\.{target_module}$", module_name) or target_module == module_name
        for target_module in target_modules
    )


def _expand_expected_modules_for_packed_layers(
    supported_modules: set[str],
    packed_modules_mapping: dict[str, list[str]] | None,
) -> set[str]:
    """Expand expected LoRA module suffixes for packed (fused) projections.

    Some diffusion models use packed projections like `to_qkv` or `w13`, while
    LoRA checkpoints are typically saved against the logical sub-projections
    (e.g. `to_q`/`to_k`/`to_v`, `w1`/`w3`). The packed layer name is present in
    `supported_modules`, but the sublayer names are not. Expanding the set
    ensures these sublayer keys are not dropped when loading a LoRA checkpoint.

    The packed→sublayer mapping is model-specific and is derived from each
    diffusion model's `stacked_params_mapping` (used by `load_weights()`), so
    new packed layers are added alongside the model implementation rather than
    hard-coded in the LoRA framework.
    """
    expanded = set(supported_modules)
    if not packed_modules_mapping:
        return expanded

    for packed_name, sub_names in packed_modules_mapping.items():
        if packed_name in supported_modules:
            expanded.update(sub_names)

    return expanded


def from_layer_diffusion(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list[str],
    model_config: PretrainedConfig | None = None,
    max_num_batched_tokens: int | None = None,
) -> nn.Module:
    """
    Diffusion-specific layer replacement. similar to vLLM's `from_layer`
    """
    # MoE wrappers own the whole runner and do not use dense packed-layer mapping.
    if isinstance(layer, MoERunner):
        if max_num_batched_tokens is None:
            raise RuntimeError("MoE LoRA token capacity was not configured")
        wrapper_cls = _select_moe_lora_wrapper_cls()
        instance = wrapper_cls(layer)
        buffer_device: list[torch.device | None] = [None]
        _suspended: list[int | None] = [None]
        original_create_lora_weights = instance.create_lora_weights
        original_set_lora = instance.set_lora
        original_reset_lora = instance.reset_lora

        def _set_lora(*args, **kwargs) -> None:
            _suspended[0] = None
            original_set_lora(*args, **kwargs)

        def _reset_lora(*args, **kwargs) -> None:
            _suspended[0] = None
            original_reset_lora(*args, **kwargs)

        # Saved enablement belongs to the current upload only. Rebinding,
        # resetting or reallocating invalidates it, just like dense wrappers.
        instance.set_lora = _set_lora  # type: ignore[method-assign]
        instance.reset_lora = _reset_lora  # type: ignore[method-assign]

        def _set_diffusion_lora_buffer_device(device: torch.device) -> None:
            buffer_device[0] = device
            _move_moe_lora_buffers(instance, device)

        def _create_lora_weights(max_loras, lora_config, model_config=None) -> None:
            _suspended[0] = None
            original_create_lora_weights(max_loras, lora_config, model_config)
            if buffer_device[0] is not None:
                _move_moe_lora_buffers(instance, buffer_device[0])

        instance._set_diffusion_lora_buffer_device = _set_diffusion_lora_buffer_device  # type: ignore[method-assign]
        instance.create_lora_weights = _create_lora_weights  # type: ignore[method-assign]
        instance.create_lora_weights(max_loras, lora_config, model_config)
        # Runner pre-hooks initialize token counts and NPU communication context.
        assert len(layer._forward_pre_hooks) > 0, (
            "MoERunner lost its forward pre-hooks after LoRA wrapping; "
            "ForwardContext.num_tokens / NPU moe_comm_method would be "
            "uninitialized."
        )

        # Call through nn.Module.__call__ to preserve those hooks.
        # Instance-assigned functions capture the wrapper instead of taking self.
        def _forward_via_base_call(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hidden_states is not None:
                _validate_moe_lora_token_capacity(hidden_states, max_num_batched_tokens)
            return instance.base_layer(*args, **kwargs)

        instance.forward = _forward_via_base_call  # type: ignore[method-assign]

        # Toggle the context-referenced enablement tensor in place, retaining weights.
        def _suspend_lora() -> None:
            if _suspended[0] is not None:
                return
            _suspended[0] = int(instance.adapter_enabled[0].item())
            instance.adapter_enabled[0] = 0

        def _resume_lora() -> None:
            prev = _suspended[0]
            if prev is None:
                return
            instance.adapter_enabled[0] = prev
            _suspended[0] = None

        instance.suspend_lora = _suspend_lora  # type: ignore[method-assign]
        instance.resume_lora = _resume_lora  # type: ignore[method-assign]
        return instance

    diffusion_lora_classes = [
        DiffusionMergedQKVParallelLinearWithLoRA,
        DiffusionQKVParallelLinearWithLoRA,
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionColumnParallelLinearWithLoRA,
        DiffusionRowParallelLinearWithLoRA,
        DiffusionReplicatedLinearWithLoRA,
    ]

    for lora_cls in diffusion_lora_classes:
        if lora_cls.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance = lora_cls(layer)  # type: ignore[arg-type]
            instance.create_lora_weights(max_loras, lora_config, model_config)
            return instance

    return layer
