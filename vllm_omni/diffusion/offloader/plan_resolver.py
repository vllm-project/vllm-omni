# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One topology resolver shared by every diffusion offload backend.

Model topology is declared through several mechanisms that grew independently:
:class:`~vllm_omni.diffusion.models.interface.SupportsComponentDiscovery` class
variables, the pipeline-level :class:`OffloadPlan`, per-DiT
``_layerwise_offload_blocks_attrs``, and a leaf-name rule for encoders. This
module runs them in one defined order and returns one backend-neutral artifact.

The resolver is pure: it moves no tensor, installs no hook, writes no module
attribute, and touches no process group. Everything it can reject is rejected
here, so plan-dependent failures happen before a backend mutates the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

from .block_discovery import get_blocks_from_dit
from .component_utils import get_encoder_block_groups, validate_on_demand_component
from .config import DIT_COMPONENT, TEXT_ENCODER_COMPONENT, OffloadStrategy
from .module_collector import ModuleDiscovery, PipelineModules
from .offload_plan import OffloadPlan, get_offload_plan

if TYPE_CHECKING:
    from .base import OffloadConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class BlockStack:
    """One ring of repeated blocks that a backend streams together.

    ``attrs`` names the owner attributes that contributed blocks; backends use
    it to skip those children when they place the non-streamed remainder. It is
    filled for DiT stacks only — encoder stacks are hooked as whole stacks and
    their remainder is placed by tensor identity.
    """

    blocks: tuple[nn.Module, ...]
    attrs: tuple[str, ...] = ()
    resident_head: int = 0

    @property
    def resident(self) -> tuple[nn.Module, ...]:
        """Leading blocks that stay on the device instead of streaming."""
        return self.blocks[: self.resident_head]

    @property
    def streaming(self) -> tuple[nn.Module, ...]:
        """Blocks a hook ring transfers on demand."""
        return self.blocks[self.resident_head :]


@dataclass(frozen=True)
class ResolvedComponent:
    """One pipeline component with its resolved, validated offload topology.

    ``selected`` means the active selector covers this component. ``on_demand``
    means the pipeline owns its residency through ``load_to_device`` /
    ``offload_to_cpu``; a VAE staged by a legacy model plan is ``on_demand``
    without being selectable through the public component grammar.
    """

    path: str
    module: nn.Module
    selected: bool
    stacks: tuple[BlockStack, ...] = ()
    on_demand: bool = False


@dataclass(frozen=True)
class ResolvedOffloadPlan:
    """Backend-neutral topology for one pipeline under one offload config."""

    dits: tuple[ResolvedComponent, ...]
    encoders: tuple[ResolvedComponent, ...]
    vaes: tuple[ResolvedComponent, ...]
    residents: tuple[ResolvedComponent, ...]
    # Raw discovery output and the model declaration remain reachable for the
    # loader-owned mmap path and for DLO's nested-submodule staging, which
    # still interpret topology themselves.
    modules: PipelineModules
    declaration: OffloadPlan | None = None

    @property
    def components(self) -> tuple[ResolvedComponent, ...]:
        return self.dits + self.encoders + self.vaes + self.residents


@cache
def _warn_legacy_discovery(pipeline_class: str) -> None:
    """Warn once per pipeline class that component discovery guessed.

    The key is the fully qualified class name so two same-named pipeline
    classes in different modules each get their own warning.
    """
    logger.warning(
        "%s does not declare SupportsComponentDiscovery; offload components were "
        "discovered by scanning well-known attribute names. Declare _dit_modules, "
        "_encoder_modules, and _vae_modules; the attribute scan is deprecated.",
        pipeline_class,
    )


def _resolve_dit_stacks(
    module: nn.Module,
    path: str,
    declaration: OffloadPlan | None,
    config: OffloadConfig,
    *,
    explicit: bool,
) -> tuple[BlockStack, ...]:
    """Resolve one DiT's streamed blocks and its device-resident head."""
    planned_attrs = None if declaration is None else declaration.block_attrs.get(path)
    attrs, blocks = get_blocks_from_dit(module, planned_attrs)
    if not blocks:
        if explicit:
            raise ValueError(f"Selected DiT {path!r} has no streamable layerwise-offload blocks")
        logger.warning(
            "Target layers (blocks) not found. Skipping offloading on %s (%s)",
            path,
            type(module).__name__,
        )
        return ()

    distributed = config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
    resident_head = 0
    if (
        distributed
        and config.dlo_resident_layers
        and declaration is not None
        and path in declaration.resident_dit_paths
    ):
        resident_head = min(config.dlo_resident_layers, len(blocks))

    if len(blocks) - resident_head == 1:
        if explicit and distributed:
            raise ValueError(
                f"Selected DiT {path!r} leaves only one streaming block after "
                f"resident_layers={resident_head}; choose a resident count that "
                "leaves zero or at least two streaming blocks"
            )
        if explicit:
            raise ValueError(f"Selected DiT {path!r} requires at least two streamable layerwise-offload blocks")
        if distributed:
            # One streamed block cannot form a prefetch ring; keeping it
            # resident preserves the model instead of skipping placement.
            logger.warning(
                "#Streaming target layers (blocks) <= 1. Keeping the final block resident on %s (%s)",
                path,
                type(module).__name__,
            )
            resident_head = len(blocks)
        else:
            logger.warning(
                "#Target layers (blocks) <= 1. Skipping offloading on %s (%s)",
                path,
                type(module).__name__,
            )
            return ()

    return (BlockStack(blocks=tuple(blocks), attrs=tuple(attrs), resident_head=resident_head),)


def _resolve_encoder_stacks(
    module: nn.Module,
    path: str,
    declaration: OffloadPlan | None,
    config: OffloadConfig,
    *,
    explicit: bool,
) -> tuple[BlockStack, ...]:
    """Resolve one encoder's streamable stacks and check transfer safety."""
    if declaration is None:
        return ()

    group_size = 1
    if config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE and config.uses_allgather(TEXT_ENCODER_COMPONENT):
        group_size = config.dp_size
    if group_size > 1 and path not in declaration.encoder_dlo_weight_replication:
        # Every rank must reject an unsafe group, including ranks whose local
        # encoder is an unloaded stub, so the check precedes block discovery.
        raise ValueError(
            f"Text encoder {path!r} cannot use DLO AllGather across the DiT offload group: "
            "its loader-produced weights are not declared replicated across that group. "
            "Set layer_options.text_encoder.weight_transfer='rank-local' in diffusion_offload_config "
            "for encoder-TP or rank-specific layouts."
        )

    groups = get_encoder_block_groups(module, path, declaration, strict=explicit or group_size > 1)
    return tuple(BlockStack(blocks=tuple(blocks)) for blocks in groups)


def resolve_offload_plan(pipeline: nn.Module, config: OffloadConfig) -> ResolvedOffloadPlan:
    """Resolve and validate one pipeline's offload topology.

    Raises ``ValueError`` for every topology that the requested configuration
    cannot serve, before the caller places a module or installs a hook.
    """
    modules = ModuleDiscovery.discover(pipeline)
    declaration = get_offload_plan(pipeline)
    if not isinstance(pipeline, SupportsComponentDiscovery) and (modules.dits or modules.encoders or modules.vaes):
        pipeline_class = type(pipeline)
        _warn_legacy_discovery(f"{pipeline_class.__module__}.{pipeline_class.__qualname__}")

    explicit = config.components is not None
    layerwise = config.strategy in (OffloadStrategy.LAYER_WISE, OffloadStrategy.DISTRIBUTED_LAYER_WISE)
    dit_selected = config.offloads(DIT_COMPONENT)

    if explicit:
        if config.strategy is OffloadStrategy.MODEL_LEVEL:
            # Model-level offload swaps the DiT against an encoder stage, so
            # both sides must exist regardless of which side was selected.
            if not modules.dits:
                raise ValueError("Component-selective model offload requires a DiT/transformer module")
            if not modules.encoders:
                raise ValueError("Component-selective model offload requires an encoder execution stage")
        elif dit_selected and not modules.dits:
            raise ValueError("No DiT/transformer modules found for selected DiT offload")
        if config.offloads(TEXT_ENCODER_COMPONENT) and not any(
            config.offloads_encoder(path, declaration) for path in modules.encoder_names
        ):
            raise ValueError("No text encoder modules found for selected text_encoder offload")

    if config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE and config.dlo_resident_layers:
        resident_paths = frozenset() if declaration is None else declaration.resident_dit_paths
        if not resident_paths.intersection(modules.dit_names):
            message = (
                f"resident_layers={config.dlo_resident_layers} was requested, but this model declares "
                "no matching resident_dit_paths"
            )
            if explicit:
                raise ValueError(message)
            logger.warning("%s; all blocks will be streamed.", message)

    dits: list[ResolvedComponent] = []
    encoders: list[ResolvedComponent] = []
    vaes: list[ResolvedComponent] = []
    residents: list[ResolvedComponent] = []

    for path, module in zip(modules.dit_names, modules.dits):
        stacks: tuple[BlockStack, ...] = ()
        if dit_selected and layerwise:
            stacks = _resolve_dit_stacks(module, path, declaration, config, explicit=explicit)
        dits.append(ResolvedComponent(path=path, module=module, selected=dit_selected, stacks=stacks))

    for path, module in zip(modules.encoder_names, modules.encoders):
        selected = config.should_offload_encoder(path, declaration)
        # Staged residency belongs to the layer backends; model-level offload
        # swaps whole modules through its own hooks.
        on_demand = bool(
            selected and layerwise and declaration is not None and path in declaration.on_demand_component_paths
        )
        if on_demand:
            validate_on_demand_component(module, path)
        stacks = ()
        if selected and layerwise:
            stacks = _resolve_encoder_stacks(module, path, declaration, config, explicit=explicit)
            if on_demand and not stacks and config.uses_allgather(TEXT_ENCODER_COMPONENT):
                raise ValueError(
                    f"Text encoder {path!r} cannot use AllGather without a model-declared streamable block plan"
                )
            if explicit and not (stacks or on_demand):
                raise ValueError(
                    f"Selected text encoder {path!r} requires a model-declared streamable or on-demand plan"
                )
        encoders.append(
            ResolvedComponent(
                path=path,
                module=module,
                selected=selected,
                stacks=stacks,
                on_demand=on_demand,
            )
        )

    for path, module in zip(modules.vae_names, modules.vaes):
        # VAEs are not part of the public selector. A model plan may still own
        # their residency, which the compatibility topology preserves.
        legacy_staged = (
            layerwise and not explicit and declaration is not None and path in declaration.on_demand_component_paths
        )
        if legacy_staged:
            validate_on_demand_component(module, path)
        vaes.append(ResolvedComponent(path=path, module=module, selected=False, on_demand=legacy_staged))

    for path, module in zip(modules.resident_names, modules.resident_modules):
        residents.append(ResolvedComponent(path=path, module=module, selected=False))

    resolved = ResolvedOffloadPlan(
        dits=tuple(dits),
        encoders=tuple(encoders),
        vaes=tuple(vaes),
        residents=tuple(residents),
        modules=modules,
        declaration=declaration,
    )
    _validate_unique_ownership(resolved)
    return resolved


def _validate_unique_ownership(resolved: ResolvedOffloadPlan) -> None:
    """Reject topologies where two components would hook the same blocks."""
    owner_by_block: dict[int, str] = {}
    for component in resolved.components:
        for block in (block for stack in component.stacks for block in stack.blocks):
            owner = owner_by_block.setdefault(id(block), component.path)
            if owner != component.path:
                raise ValueError(
                    f"Offload block is claimed by both {owner!r} and {component.path!r}; "
                    "declare one owner for each block list"
                )


__all__ = [
    "BlockStack",
    "ResolvedComponent",
    "ResolvedOffloadPlan",
    "resolve_offload_plan",
]
