# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block discovery for layerwise offload.

Shared between LayerWiseOffloadBackend and DistributedLayerwiseOffloadBackend.
"""

from __future__ import annotations

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
