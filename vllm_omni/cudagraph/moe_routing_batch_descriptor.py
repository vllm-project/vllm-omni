# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Extend vLLM ``BatchDescriptor`` with Mammoth MoE routing specialty (#9865).

vLLM keys CUDA graphs by ``BatchDescriptor``. Mask-keyed split graphs need a
routing field analogous to ``num_active_loras``. Until upstream grows the
field, Omni subclasses the live upstream dataclass and rebinds the hot import
sites.

Subclassing, rather than re-declaring the fields, keeps future upstream keys
(another LoRA-like dimension, a microbatch tag) flowing through construction,
``replace`` and hashing. A re-declaration would silently drop a new field and
the specialty key would stop matching the captured graphs, falling back to
BLEND without an obvious error.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from vllm.forward_context import BatchDescriptor as _UpstreamBatchDescriptor

from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import MoERoutingKey

_REBIND_ROOTS = ("vllm", "vllm_omni")


@dataclass(frozen=True)
class BatchDescriptor(_UpstreamBatchDescriptor):
    """Upstream ``BatchDescriptor`` plus an optional MoE routing specialty.

    Inherits every upstream field so new keys keep participating in graph
    dispatch. ``moe_routing_key`` defaults to BLEND, keeping #9764 dual-expert
    behaviour for callers that omit it.
    """

    moe_routing_key: int = int(MoERoutingKey.BLEND)


def patch_batch_descriptor_moe_routing_key() -> bool:
    """Install the extended BatchDescriptor. Idempotent.

    Rebinding is a sweep over already-imported ``vllm`` modules rather than a
    fixed list: any module that did ``from vllm.forward_context import
    BatchDescriptor`` before this ran holds its own reference, and a descriptor
    built from that stale class hashes without the routing key, so replay would
    silently stay on BLEND. Modules imported after the patch pull the patched
    class straight from ``vllm.forward_context``.
    """
    import sys

    import vllm.forward_context as forward_context

    current = forward_context.BatchDescriptor
    if getattr(current, "__omni_moe_routing_key__", False):
        return False

    BatchDescriptor.__omni_moe_routing_key__ = True  # type: ignore[attr-defined]
    forward_context.BatchDescriptor = BatchDescriptor

    rebound = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        # Also rebind Omni's own modules: several import BatchDescriptor at
        # module scope and would otherwise keep the pre-patch class.
        if not any(mod_name == root or mod_name.startswith(f"{root}.") for root in _REBIND_ROOTS):
            continue
        if getattr(mod, "BatchDescriptor", None) is current:
            setattr(mod, "BatchDescriptor", BatchDescriptor)
            rebound += 1

    # Startup assertion: the installed class must still carry the routing key.
    # If upstream ever ships or renames it, expect a hard failure here instead
    # of specialty graphs that capture and then never match on replay.
    installed_fields = {field.name for field in fields(forward_context.BatchDescriptor)}
    if "moe_routing_key" not in installed_fields:
        raise RuntimeError(
            "vllm_omni failed to install BatchDescriptor.moe_routing_key "
            f"(got fields {sorted(installed_fields)}); mask-keyed MoE CUDA graphs need it."
        )
    _log_rebind(rebound)
    return True


def _log_rebind(rebound: int) -> None:
    import logging

    logging.getLogger("vllm_omni.cudagraph").debug(
        "Installed BatchDescriptor.moe_routing_key; rebound %d already-imported vllm module(s).",
        rebound,
    )
