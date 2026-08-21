# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block discovery for layerwise offload.

Shared between LayerWiseOffloadBackend and DistributedLayerwiseOffloadBackend.
"""

from __future__ import annotations

from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import PPMissingLayer

logger = init_logger(__name__)


def _is_pp_missing_layer(module: nn.Module) -> bool:
    return isinstance(module, PPMissingLayer)


def select_local_offload_blocks(model: nn.Module, modules: list[nn.Module]) -> list[nn.Module]:
    """Keep the blocks that actually run on this rank for the prefetch ring.

    Pipeline-parallel DiTs built with vLLM ``make_layers`` store a full-length
    ``ModuleList`` and fill non-local slots with ``PPMissingLayer``. The offload
    hook ring must cover only ``blocks[start_layer:end_layer]`` so the last
    local layer prefetches the first local layer of the next forward.
    """
    if not modules:
        return []

    start = getattr(model, "start_layer", None)
    end = getattr(model, "end_layer", None)
    n_executable = sum(1 for module in modules if not _is_pp_missing_layer(module))
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(modules)
        and (end - start) == n_executable
    ):
        sliced = modules[start:end]
        if sliced and not any(_is_pp_missing_layer(module) for module in sliced):
            if len(sliced) != len(modules):
                logger.info(
                    "Restricting layerwise offload ring to local PP slice [%d, %d) (%d blocks)",
                    start,
                    end,
                    len(sliced),
                )
            return list(sliced)

    return [module for module in modules if not _is_pp_missing_layer(module)]


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
    """Retrieve executable blocks and attribute names from a DiT model.

    Each declared block container is reduced to this rank's local PP slice
    before containers are concatenated, so the prefetch ring wraps inside a
    pipeline stage rather than across ``PPMissingLayer`` placeholders.
    """
    blocks_attr_names = get_blocks_attr_names(model)
    if not blocks_attr_names:
        logger.warning(
            f"No _layerwise_offload_blocks_attrs defined for {model.__class__.__name__}, skipping layerwise offloading"
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
            collected = list(iter(attr))
        except TypeError:
            if isinstance(attr, nn.Module):
                logger.warning(
                    "Attribute '%s' on %s is not iterable; treating it as one block.",
                    name,
                    model.__class__.__name__,
                )
                collected = [attr]
            else:
                logger.warning(
                    "Attribute '%s' on %s is not iterable (got %s); skipping it.",
                    name,
                    model.__class__.__name__,
                    type(attr).__name__,
                )
                continue
        blocks.extend(select_local_offload_blocks(model, collected))

    if not blocks:
        logger.warning(
            "No blocks found in %s for %s, skipping layerwise offloading",
            blocks_attr_names,
            model.__class__.__name__,
        )
        return [], []

    return blocks_attr_names, blocks
