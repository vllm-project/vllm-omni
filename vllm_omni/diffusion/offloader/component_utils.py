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
from .module_residency import BoundedAllocatorCache, PinnedModuleStager
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


def supports_layerwise_staging(module: nn.Module) -> bool:
    """Whether an on-demand component stages around hooked blocks itself."""
    return bool(getattr(module, "_omni_supports_layerwise_staging", False))


def non_block_modules(module: nn.Module, block_groups: Sequence[Sequence[nn.Module]]) -> list[nn.Module]:
    """Return the largest submodules of ``module`` that hold no streamed block."""
    block_ids = {id(block) for blocks in block_groups for block in blocks}
    found: list[nn.Module] = []

    def visit(current: nn.Module, name: str) -> None:
        if id(current) in block_ids:
            return
        if not any(id(child) in block_ids for child in current.modules()):
            found.append(current)
            return
        if any(tensor is not None for tensor in chain(current._parameters.values(), current._buffers.values())):
            # The stager works on whole modules, so a tensor owned directly by
            # a block's ancestor has no module of its own to stage it with.
            raise ValueError(
                f"{type(current).__name__} at {name or '<root>'} owns tensors next to streamed blocks; "
                "move them into a submodule to stage it around layerwise hooks"
            )
        for child_name, child in current.named_children():
            visit(child, f"{name}.{child_name}" if name else child_name)

    visit(module, "")
    return found


class LayerwiseAwareStager:
    """Stage a pipeline-managed component around layerwise hooks on its blocks.

    Without hooks the whole module is staged. Once a layerwise backend has
    hooked block lists on ``owner``, the hooks stream those blocks and only the
    remaining modules are staged. The choice is made at every load, so it
    follows what the backend actually selected rather than a global flag.
    Both stagers are built lazily: snapshotting the whole module before the
    hooks copy the blocks would pin those blocks on the host twice.
    """

    def __init__(
        self,
        owner: nn.Module,
        module: nn.Module,
        device: torch.device,
        *,
        pin_memory: bool,
        cache_retention: BoundedAllocatorCache | None = None,
    ) -> None:
        self.owner = owner
        self.module = module
        self.device = device
        self.pin_memory = pin_memory
        self.cache_retention = cache_retention
        self._whole: PinnedModuleStager | None = None
        self._non_block: PinnedModuleStager | None = None

    def _hooked(self) -> bool:
        return bool(getattr(self.owner, "_omni_layerwise_enabled", False))

    def _stager(self) -> PinnedModuleStager:
        if not self._hooked():
            if self._whole is None:
                self._whole = PinnedModuleStager(
                    self.module, self.device, pin_memory=self.pin_memory, cache_retention=self.cache_retention
                )
            return self._whole
        if self._whole is not None:
            # Hooks were installed after a whole-module load; they own the blocks now.
            self._whole.offload()
            self._whole = None
        if self._non_block is None:
            self._non_block = PinnedModuleStager(
                non_block_modules(self.module, self.owner._omni_layerwise_block_groups),
                self.device,
                pin_memory=self.pin_memory,
                cache_retention=self.cache_retention,
            )
        return self._non_block

    def load(self) -> None:
        self._stager().load()

    def offload(self) -> None:
        if self._hooked():
            for hook in self.owner._omni_layerwise_hooks:
                hook.offload_layer()
            stager = self._non_block
        else:
            stager = self._whole
        if stager is not None:
            stager.offload()


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
