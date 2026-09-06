# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared component-plan helpers for diffusion offload backends."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain
from operator import attrgetter
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.logger import init_logger

from .block_discovery import get_blocks_from_dit
from .config import DIT_COMPONENT, TEXT_ENCODER_COMPONENT
from .offload_plan import OffloadPlan
from .tensor_utils import set_tensor_storage

if TYPE_CHECKING:
    from .base import OffloadConfig
    from .module_collector import PipelineModules

logger = init_logger(__name__)


def encoder_component_type(name: str, plan: OffloadPlan | None) -> str | None:
    """Map a discovered encoder path to its public offload component."""
    declared = None if plan is None else plan.encoder_component_types.get(name)
    if declared is not None:
        if declared != TEXT_ENCODER_COMPONENT:
            raise ValueError(f"OffloadPlan maps encoder {name!r} to unknown component {declared!r}")
        return declared

    leaf_name = name.rsplit(".", 1)[-1]
    if leaf_name.startswith(TEXT_ENCODER_COMPONENT) or leaf_name.endswith(TEXT_ENCODER_COMPONENT):
        return TEXT_ENCODER_COMPONENT
    return None


def get_encoder_block_groups(
    module: nn.Module,
    name: str,
    plan: OffloadPlan | None,
    *,
    strict: bool = False,
) -> list[nn.ModuleList]:
    """Resolve the streamable block lists declared for one encoder."""
    if plan is None:
        return []

    # Some distributed models expose an unloaded stub on ranks where the
    # component never executes.  That is a valid local no-op even when the
    # component was selected explicitly.
    if getattr(module, "is_loaded", True) is False:
        return []

    groups: list[nn.ModuleList] = []
    for block_path in plan.encoder_block_attrs.get(name, ()):
        try:
            blocks = attrgetter(block_path)(module)
        except AttributeError:
            if strict:
                raise ValueError(f"Encoder offload path {name}.{block_path} was not found") from None
            logger.warning("Encoder offload path %s.%s was not found", name, block_path)
            continue
        if not isinstance(blocks, nn.ModuleList) or len(blocks) <= 1:
            if strict:
                raise ValueError(f"Encoder offload path {name}.{block_path} is not a streamable block list")
            logger.warning("Encoder offload path %s.%s is not a streamable block list", name, block_path)
            continue
        groups.append(blocks)
    return groups


def iter_streamable_dits(
    modules: PipelineModules,
    config: OffloadConfig,
    device: torch.device,
    plan: OffloadPlan | None,
) -> Iterator[tuple[str, nn.Module, list[str], list[nn.Module]]]:
    """Yield selected DiTs whose block metadata resolves successfully."""
    if not config.offloads(DIT_COMPONENT):
        return
    for name, module in zip(modules.dit_names, modules.dits):
        logger.info("Applying hooks on %s (%s)", name, module.__class__.__name__)
        planned_attrs = None if plan is None else plan.block_attrs.get(name)
        block_attrs, blocks = get_blocks_from_dit(module, planned_attrs)
        if blocks:
            yield name, module, block_attrs, blocks
            continue
        if config.components is not None:
            raise ValueError(f"Selected DiT {name!r} has no streamable layerwise-offload blocks")
        logger.warning("Target layers (blocks) not found. Skipping offloading on %s (%s)", name, type(module).__name__)
        module.to(device)


def move_non_block_state_to_device(
    module: nn.Module,
    block_groups: list[nn.ModuleList],
    device: torch.device,
) -> None:
    """Keep component state outside streamed block lists resident on device."""
    block_tensors = {
        id(tensor)
        for blocks in block_groups
        for block in blocks
        for tensor in chain(block.parameters(), block.buffers())
    }
    for tensor in chain(module.parameters(), module.buffers()):
        if id(tensor) in block_tensors:
            continue
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        if local.device != device:
            set_tensor_storage(tensor, local.to(device, non_blocking=True))


def set_encoder_layerwise_state(
    module: nn.Module,
    hooks: list[Any],
    block_groups: list[nn.ModuleList],
) -> None:
    """Publish the backend-neutral state used by encoder stage lifecycles."""
    module._omni_layerwise_hooks = hooks
    module._omni_layerwise_block_groups = block_groups
    module._omni_layerwise_enabled = True


def clear_encoder_layerwise_state(module: nn.Module) -> None:
    """Clear encoder layerwise state after its backend hooks are removed."""
    module._omni_layerwise_hooks = []
    module._omni_layerwise_block_groups = []
    module._omni_layerwise_enabled = False


def validate_on_demand_component(module: nn.Module, name: str) -> None:
    """Require the explicit lifecycle used by pipeline-managed components."""
    if not callable(getattr(module, "load_to_device", None)) or not callable(getattr(module, "offload_to_cpu", None)):
        raise ValueError(
            f"Component {name!r} declares on-demand offload but must implement load_to_device() and offload_to_cpu()"
        )


def prepare_component(
    module: nn.Module,
    name: str,
    *,
    device: torch.device,
    stage_on_demand: bool,
    blockwise: bool,
    staged_components: list[nn.Module],
) -> None:
    """Stage a selected component or keep its non-streamed form resident."""
    if stage_on_demand:
        validate_on_demand_component(module, name)
        getattr(module, "offload_to_cpu")()
        staged_components.append(module)
        logger.info("Prepared %s for pipeline-managed staged offload", name)
    elif not blockwise:
        module.to(device)


def prepare_pipeline_components(
    modules: PipelineModules,
    config: OffloadConfig,
    plan: OffloadPlan | None,
    *,
    device: torch.device,
    staged_components: list[nn.Module],
    enable_encoder_blocks: Callable[[nn.Module, str, OffloadPlan | None, bool], bool],
) -> None:
    """Apply the shared encoder/VAE/resident placement policy."""
    if config.components is not None and config.offloads(TEXT_ENCODER_COMPONENT):
        selected_encoder_names = [name for name in modules.encoder_names if config.offloads_encoder(name, plan)]
        if not selected_encoder_names:
            raise ValueError("No text encoder modules found for selected text_encoder offload")

    if plan is not None:
        for encoder, name in zip(modules.encoders, modules.encoder_names):
            if config.should_offload_encoder(name, plan) and name in plan.on_demand_component_paths:
                validate_on_demand_component(encoder, name)

    for encoder, name in zip(modules.encoders, modules.encoder_names):
        selected = config.should_offload_encoder(name, plan)
        stage_on_demand = bool(selected and plan is not None and name in plan.on_demand_component_paths)
        blockwise = selected and enable_encoder_blocks(encoder, name, plan, stage_on_demand)
        if stage_on_demand and not blockwise and config.uses_allgather(TEXT_ENCODER_COMPONENT):
            raise ValueError(
                f"Text encoder {name!r} cannot use AllGather without a model-declared streamable block plan"
            )
        if selected and config.components is not None and not (blockwise or stage_on_demand):
            raise ValueError(f"Selected text encoder {name!r} requires a model-declared streamable or on-demand plan")
        prepare_component(
            encoder,
            name,
            device=device,
            stage_on_demand=stage_on_demand,
            blockwise=blockwise,
            staged_components=staged_components,
        )

    for vae, name in zip(modules.vaes, modules.vae_names):
        legacy_staged = config.components is None and plan is not None and name in plan.on_demand_component_paths
        prepare_component(
            vae,
            name,
            device=device,
            stage_on_demand=legacy_staged,
            blockwise=False,
            staged_components=staged_components,
        )

    for name, module in zip(modules.resident_names, modules.resident_modules):
        module.to(device)
        logger.debug("Moved resident module %s to %s", name, device)

    if not config.offloads(DIT_COMPONENT):
        for dit in modules.dits:
            dit.to(device)
