# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Expand / dispatch CUDA-graph keys by MoE routing specialty (#9865)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import MoERoutingKey

# Capture all three specialties for FULL graphs when specialization is on.
_MOE_CAPTURE_KEYS: tuple[MoERoutingKey, ...] = (
    MoERoutingKey.BLEND,
    MoERoutingKey.ALL_UND,
    MoERoutingKey.ALL_GEN,
)


def expand_batch_descriptors_for_moe_routing(
    descs: Iterable[BatchDescriptor],
    *,
    enabled: bool,
) -> list[BatchDescriptor]:
    """Multiply capture descriptors by MoE routing keys when enabled."""
    base = list(descs)
    if not enabled:
        return [replace(d, moe_routing_key=int(MoERoutingKey.BLEND)) for d in base]
    expanded: list[BatchDescriptor] = []
    for desc in base:
        for key in _MOE_CAPTURE_KEYS:
            expanded.append(replace(desc, moe_routing_key=int(key)))
    return expanded


def with_moe_routing_key(
    desc: BatchDescriptor,
    key: MoERoutingKey | int,
) -> BatchDescriptor:
    return replace(desc, moe_routing_key=int(key))


def sort_capture_descs(descs: list[BatchDescriptor]) -> list[BatchDescriptor]:
    """Largest batches first; within a size, BLEND then ALL_UND then ALL_GEN."""
    return sorted(
        descs,
        key=lambda d: (
            d.num_tokens,
            getattr(d, "num_active_loras", 0),
            -getattr(d, "moe_routing_key", 0),
        ),
        reverse=True,
    )


def patch_cudagraph_dispatcher_moe_routing() -> bool:
    """Wrap ``CudagraphDispatcher`` key init / capture listing for MoE keys.

    Activation is gated by ``vllm_config.compilation_config`` carrying
    ``cudagraph_specialize_moe_routing=True`` (Omni may set this on Mammoth
    FULL presets). When the attr is absent/false, behaviour is unchanged
    aside from defaulting ``moe_routing_key=0`` on created descriptors.
    """
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    if getattr(CudagraphDispatcher.initialize_cudagraph_keys, "_omni_moe_routing", False):
        return False

    _orig_init_keys = CudagraphDispatcher.initialize_cudagraph_keys
    _orig_create = CudagraphDispatcher._create_padded_batch_descriptor
    _orig_get_capture = CudagraphDispatcher.get_capture_descs
    _orig_dispatch = CudagraphDispatcher.dispatch

    def _moe_enabled(self) -> bool:
        return bool(getattr(self.compilation_config, "cudagraph_specialize_moe_routing", False))

    def _create_padded_batch_descriptor(self, *args, moe_routing_key: int = 0, **kwargs):
        desc = _orig_create(self, *args, **kwargs)
        return replace(desc, moe_routing_key=int(moe_routing_key))

    def initialize_cudagraph_keys(self, cudagraph_mode, uniform_decode_query_len: int = 1):
        _orig_init_keys(self, cudagraph_mode, uniform_decode_query_len)
        if not _moe_enabled(self):
            return
        # Re-expand FULL keys with the three MoE specialties.
        full_keys = self.cudagraph_keys.get(CUDAGraphMode.FULL)
        if not full_keys:
            return
        # Collapse to unique base keys (ignore prior moe_routing_key).
        bases: dict[tuple, BatchDescriptor] = {}
        for desc in list(full_keys):
            base = replace(desc, moe_routing_key=int(MoERoutingKey.BLEND))
            bases[(base.num_tokens, base.num_reqs, base.uniform, base.has_lora, base.num_active_loras)] = base
        full_keys.clear()
        for desc in expand_batch_descriptors_for_moe_routing(bases.values(), enabled=True):
            full_keys.add(desc)

    def get_capture_descs(self):
        result = _orig_get_capture(self)
        if not _moe_enabled(self):
            return result
        out = []
        for mode, descs in result:
            out.append((mode, sort_capture_descs(list(descs))))
        return out

    def dispatch(self, num_tokens, uniform_decode=False, has_lora=False, num_active_loras=0, valid_modes=None, invalid_modes=None, moe_routing_key: int = 0):
        mode, desc = _orig_dispatch(
            self,
            num_tokens,
            uniform_decode=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
            valid_modes=valid_modes,
            invalid_modes=invalid_modes,
        )
        if mode == CUDAGraphMode.NONE or not _moe_enabled(self):
            return mode, replace(desc, moe_routing_key=int(MoERoutingKey.BLEND))
        keyed = replace(desc, moe_routing_key=int(moe_routing_key))
        if keyed in self.cudagraph_keys.get(mode, ()):
            return mode, keyed
        # Fall back to BLEND graph if the specialized key was not captured.
        blend = replace(desc, moe_routing_key=int(MoERoutingKey.BLEND))
        if blend in self.cudagraph_keys.get(mode, ()):
            return mode, blend
        return mode, keyed

    initialize_cudagraph_keys._omni_moe_routing = True  # type: ignore[attr-defined]
    CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
    CudagraphDispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys
    CudagraphDispatcher.get_capture_descs = get_capture_descs
    CudagraphDispatcher.dispatch = dispatch
    return True
