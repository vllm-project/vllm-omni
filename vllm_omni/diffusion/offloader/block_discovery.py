# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block discovery for layerwise offload.

Shared between LayerWiseOffloadBackend and DistributedLayerwiseOffloadBackend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_blocks_attr_names(model: nn.Module) -> list[str]:
    """Get block attribute names from model class."""
    attrs: list[str] = getattr(model.__class__, "_layerwise_offload_blocks_attrs", [])

    if not attrs:
        old_attr = getattr(model.__class__, "_layerwise_offload_blocks_attr", None)
        if old_attr is not None:
            logger.warning(
                "'_layerwise_offload_blocks_attr' is deprecated, "
                "please use '_layerwise_offload_blocks_attrs' instead. "
                "Example: _layerwise_offload_blocks_attrs = ['blocks']"
            )
            attrs = [old_attr] if isinstance(old_attr, str) else list(old_attr)

    return attrs


def set_blocks_attr_names(model: nn.Module, names: list[str]) -> None:
    if not hasattr(model.__class__, "_layerwise_offload_blocks_attrs"):
        setattr(model.__class__, "_layerwise_offload_blocks_attrs", names)


def get_blocks_from_dit(model: nn.Module) -> tuple[list[str], list[nn.Module]]:
    """Retrieve blocks and attribute names from provided DiT model."""
    blocks_attr_names = get_blocks_attr_names(model)
    if not blocks_attr_names:
        logger.warning(
            f"No _layerwise_offload_blocks_attrs defined for {model.__class__.__name__}, "
            "skipping distributed layerwise offloading"
        )
        return [], []

    blocks: list[nn.Module] = []
    for name in blocks_attr_names:
        attr = getattr(model, name, None)
        if attr is None:
            raise AttributeError(
                f"Attribute '{name}' declared in _layerwise_offload_blocks_attrs "
                f"does not exist on model {model.__class__.__name__}"
            )
        try:
            attr_iter = iter(attr)
        except TypeError:
            if isinstance(attr, nn.Module):
                logger.warning(
                    "Attribute '%s' on %s is not iterable; treating it as one block.",
                    name,
                    model.__class__.__name__,
                )
                blocks.append(attr)
                continue

            logger.warning(
                "Attribute '%s' on %s is not iterable (got %s); skipping it.",
                name,
                model.__class__.__name__,
                type(attr).__name__,
            )
        else:
            blocks.extend(attr_iter)

    if not blocks:
        logger.warning(
            "No blocks found in %s for %s, skipping distributed layerwise offloading",
            blocks_attr_names,
            model.__class__.__name__,
        )
        return [], []

    return blocks_attr_names, blocks


@dataclass(frozen=True)
class ChunkOwnedBlock:
    """One repeated DiT block owned by the chunked FS offload engine.

    ``path`` is a stable, unique string id for the block. It is used as the
    pin-budget key by the chunk engine, so it must be identical on every rank
    and stable across the load -> enable() handoff. It is derived from the
    module's qualified name inside the root object passed to
    :func:`discover_chunk_owned_blocks`.
    """

    module: nn.Module
    path: str


@dataclass
class ChunkOwnership:
    """Ownership split between the chunk engine and FSDP.

    Every parameter must have exactly one owner. The blocks listed here are
    handed to the chunked FS offload engine and are therefore excluded from
    FSDP wrapping (see ``apply_hsdp_to_model(chunk_owned_blocks=...)``).

    Attributes:
        blocks: Chunk-owned repeated blocks, ordered by execution order.
        block_ids: ``id(module) -> stable 0-based block index``, same order.
    """

    blocks: list[ChunkOwnedBlock] = field(default_factory=list)
    block_ids: dict[int, int] = field(default_factory=dict)

    @property
    def modules(self) -> list[nn.Module]:
        return [entry.module for entry in self.blocks]

    def __len__(self) -> int:
        return len(self.blocks)

    def __bool__(self) -> bool:
        return bool(self.blocks)


def _qualified_name_map(root: nn.Module) -> dict[int, str]:
    """Map ``id(submodule) -> dotted qualified name`` within *root*."""
    return {id(module): name for name, module in root.named_modules()}


def discover_chunk_owned_blocks(pipeline_or_model: nn.Module) -> ChunkOwnership:
    """Resolve the repeated DiT blocks that the chunk engine owns.

    Accepts either a pipeline (components discovered via ``ModuleDiscovery``)
    or a bare DiT/transformer module. Blocks are collected from every declared
    DiT ring, in execution order, deduplicated, and assigned stable 0-based ids.

    Returns an empty :class:`ChunkOwnership` when the model declares no
    repeated-block attributes, which lets callers degrade to plain FSDP
    wrapping instead of failing.
    """
    name_map = _qualified_name_map(pipeline_or_model)

    # A bare DiT declares the block attrs itself; a pipeline does not.
    if get_blocks_attr_names(pipeline_or_model):
        dit_names: list[str] = [""]
        dits: list[nn.Module] = [pipeline_or_model]
    else:
        # Imported lazily and only on the pipeline path: module_collector
        # imports model-side helpers, and this module is imported by the
        # offload backends, so a top-level import would create a cycle.
        from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

        discovered = ModuleDiscovery.discover(pipeline_or_model)
        # Ownership covers every independently discovered DiT block ring.  A
        # model such as Cosmos3 exposes both an outer video transformer and a
        # nested language-model transformer; HSDP wraps only the outer module,
        # but the chunk engine streams both rings.  ``seen`` below deduplicates
        # aliases and descendants exposed more than once.
        dit_names = discovered.dit_names
        dits = discovered.dits

    ownership = ChunkOwnership()
    seen: set[int] = set()
    for dit_name, dit in zip(dit_names, dits):
        _, blocks = get_blocks_from_dit(dit)
        for block in blocks:
            key = id(block)
            if key in seen:
                # A pipeline may expose the same block twice (e.g. an alias
                # attribute). One owner means one entry.
                continue
            seen.add(key)
            qualified = name_map.get(key)
            if qualified is None:
                # Not reachable from the root we were handed; fall back to a
                # deterministic name so the pin-budget key stays stable.
                index = len(ownership.blocks)
                qualified = f"{dit_name}.blocks.{index}" if dit_name else f"blocks.{index}"
            ownership.block_ids[key] = len(ownership.blocks)
            ownership.blocks.append(ChunkOwnedBlock(module=block, path=qualified))

    return ownership
