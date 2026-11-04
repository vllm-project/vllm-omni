# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Extend vLLM ``BatchDescriptor`` with Mammoth MoE routing specialty (#9865).

vLLM keys CUDA graphs by ``BatchDescriptor``. Mask-keyed split graphs need a
routing field analogous to ``num_active_loras``. Until upstream grows the
field, Omni replaces the dataclass early in ``vllm_omni.patch`` and rebinds
the hot import sites.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import MoERoutingKey


@dataclass(frozen=True)
class BatchDescriptor:
    """Omni BatchDescriptor with optional MoE routing specialty.

    Compatible with upstream fields; ``moe_routing_key`` defaults to BLEND so
    callers that omit it keep #9764 dual-expert behaviour.
    """

    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0
    moe_routing_key: int = int(MoERoutingKey.BLEND)


_REBIND_MODULES = (
    "vllm.forward_context",
    "vllm.v1.cudagraph_dispatcher",
    "vllm.compilation.cuda_graph",
    "vllm.compilation.breakable_cudagraph",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.cudagraph_utils",
    "vllm.v1.worker.gpu.model_runner",
)


def patch_batch_descriptor_moe_routing_key() -> bool:
    """Install the extended BatchDescriptor. Idempotent."""
    import vllm.forward_context as forward_context

    current = forward_context.BatchDescriptor
    if getattr(current, "__omni_moe_routing_key__", False):
        return False

    BatchDescriptor.__omni_moe_routing_key__ = True  # type: ignore[attr-defined]
    forward_context.BatchDescriptor = BatchDescriptor

    import sys

    for mod_name in _REBIND_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "BatchDescriptor"):
            setattr(mod, "BatchDescriptor", BatchDescriptor)
    return True
