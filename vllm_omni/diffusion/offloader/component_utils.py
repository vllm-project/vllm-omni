# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared component-plan helpers for diffusion offload backends."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import chain
from operator import attrgetter
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from vllm.logger import init_logger

from .config import TEXT_ENCODER_COMPONENT
from .offload_plan import OffloadPlan
from .tensor_utils import set_tensor_storage

if TYPE_CHECKING:
    from .plan_resolver import BlockStack, ResolvedComponent, ResolvedOffloadPlan

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
    resolved: ResolvedOffloadPlan,
    device: torch.device,
) -> Iterator[tuple[ResolvedComponent, BlockStack]]:
    """Yield selected DiTs that resolved to a streamable block stack."""
    for component in resolved.dits:
        if not component.selected:
            continue
        logger.info("Applying hooks on %s (%s)", component.path, component.module.__class__.__name__)
        if not component.stacks:
            # The resolver already warned; a legacy selection keeps the
            # unstreamable DiT resident instead of failing the run.
            component.module.to(device)
            continue
        yield component, component.stacks[0]


def move_non_block_state_to_device(
    module: nn.Module,
    block_groups: Sequence[Sequence[nn.Module]],
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
    block_groups: Sequence[Sequence[nn.Module]],
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
    resolved: ResolvedOffloadPlan,
    *,
    device: torch.device,
    staged_components: list[nn.Module],
    enable_encoder_blocks: Callable[[ResolvedComponent], bool],
) -> None:
    """Apply the shared encoder/VAE/resident placement policy."""
    for component in resolved.encoders:
        blockwise = bool(component.stacks) and enable_encoder_blocks(component)
        prepare_component(
            component.module,
            component.path,
            device=device,
            stage_on_demand=component.on_demand,
            blockwise=blockwise,
            staged_components=staged_components,
        )

    for component in resolved.vaes:
        prepare_component(
            component.module,
            component.path,
            device=device,
            stage_on_demand=component.on_demand,
            blockwise=False,
            staged_components=staged_components,
        )

    for component in resolved.residents:
        component.module.to(device)
        logger.debug("Moved resident module %s to %s", component.path, device)

    for component in resolved.dits:
        if not component.selected:
            component.module.to(device)
