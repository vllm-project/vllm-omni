# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mask-keyed CUDA-graph specialty for MammothModa2 AR MoE (#9865).

#9764 records both und/gen experts under capture so host-side mask branches
cannot bake the wrong path. That dual-expert tax hits all-und / all-gen
batches that share the same FULL graph.

This module defines the routing key used to capture three specialized graphs
per batch size (similar spirit to ``cudagraph_specialize_lora``):

| Key | Capture records | Host selects when |
| --- | --- | --- |
| BLEND | both + ``torch.where`` | mixed (or unknown) |
| ALL_UND | ``und_expert`` only | mask all false |
| ALL_GEN | ``gen_expert`` only | mask all true |

On Model Runner V2, ``install_moe_routing_graphs`` adds the ALL_UND and
ALL_GEN captures to the runner's graph manager. Before each FULL replay the
manager classifies the unpadded token ids on the host and picks the graph.
Inside the graph, mixed still uses the #9764 blend path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from enum import IntEnum

import torch
from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CreateForwardFn,
    CudaGraphManager,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch


class MoERoutingKey(IntEnum):
    """CUDA-graph specialty for MammothModa2 MoE layers."""

    BLEND = 0
    ALL_UND = 1
    ALL_GEN = 2


# Active specialty while a graph is being captured. Replay does not consult
# this: the graph manager selects the already-recorded graph.
# Default BLEND preserves #9764 if capture forgets to enter a scope.
_moe_routing_key: ContextVar[MoERoutingKey] = ContextVar(
    "mammoth_moda2_moe_routing_key",
    default=MoERoutingKey.BLEND,
)


def get_moe_routing_key() -> MoERoutingKey:
    return _moe_routing_key.get()


@contextmanager
def moe_routing_key_scope(key: MoERoutingKey) -> Iterator[None]:
    """Temporarily force the MoE capture/replay specialty."""
    token = _moe_routing_key.set(key)
    try:
        yield
    finally:
        _moe_routing_key.reset(token)


def moe_routing_key_from_mask(gen_token_mask: torch.Tensor | None) -> MoERoutingKey:
    """Classify a batch mask for host-side graph dispatch (outside capture).

    Missing / empty masks fall back to BLEND so dispatch stays fail-closed
    toward the dual-expert graph rather than guessing ALL_UND.
    """
    if gen_token_mask is None or gen_token_mask.numel() == 0:
        return MoERoutingKey.BLEND
    flat = gen_token_mask.reshape(-1)
    # One host sync per step: sum() then compare on the host, instead of a
    # separate any().item() and all().item(). This sits on the FULL replay
    # path, so the extra DtoH round trip is worth folding away.
    num_gen = int(flat.sum().item())
    if num_gen == 0:
        return MoERoutingKey.ALL_UND
    if num_gen == flat.numel():
        return MoERoutingKey.ALL_GEN
    return MoERoutingKey.BLEND


def moe_routing_key_from_input_ids(
    input_ids: torch.Tensor | None,
    gen_vocab_start_index: int,
) -> MoERoutingKey:
    """Derive the dispatch key from raw token ids (runner path)."""
    if input_ids is None or input_ids.numel() == 0:
        return MoERoutingKey.BLEND
    return moe_routing_key_from_mask(input_ids >= gen_vocab_start_index)


# ---------------------------------------------------------------------------
# Model Runner V2 graph manager
# ---------------------------------------------------------------------------

_SPECIALTY_KEYS = (MoERoutingKey.ALL_UND, MoERoutingKey.ALL_GEN)


@dataclass(frozen=True)
class MoERoutingBatchExecutionDescriptor(BatchExecutionDescriptor):
    """A FULL graph captured with a single-expert MoE specialty.

    BLEND graphs keep the upstream descriptor, so upstream dispatch tables
    stay valid. This keyed subclass never compares equal to them.
    """

    moe_routing_key: int = int(MoERoutingKey.BLEND)


def _keyed(desc: BatchExecutionDescriptor, key: MoERoutingKey) -> MoERoutingBatchExecutionDescriptor:
    values = {f.name: getattr(desc, f.name) for f in fields(BatchExecutionDescriptor)}
    return MoERoutingBatchExecutionDescriptor(**values, moe_routing_key=int(key))


class _MoERoutingCaptureScope(CudaGraphManager):
    """Run each capture forward under its descriptor's MoE specialty.

    ``ModelCudaGraphManager.capture`` builds the forward functions and hands
    them to ``super().capture``. Sitting after it in the MRO lets this class
    wrap them without copying the upstream body.
    """

    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        def create_keyed_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            forward_fn = create_forward_fn(desc, warmup)
            key = MoERoutingKey(getattr(desc, "moe_routing_key", MoERoutingKey.BLEND))

            def keyed_forward_fn(cg_mode: CUDAGraphMode) -> None:
                with moe_routing_key_scope(key):
                    forward_fn(cg_mode)

            return keyed_forward_fn

        super().capture(create_keyed_forward_fn, progress_bar_desc, stream)


class MoERoutingCudaGraphManager(ModelCudaGraphManager, _MoERoutingCaptureScope):
    """``ModelCudaGraphManager`` that also captures single-expert FULL graphs."""

    gen_vocab_start_index: int
    _moe_routing_descs: dict[tuple[BatchExecutionDescriptor, MoERoutingKey], MoERoutingBatchExecutionDescriptor]

    def select_replay_graph(self, desc: BatchExecutionDescriptor, input_batch: InputBatch) -> BatchExecutionDescriptor:
        """Return the specialty graph for this batch, else ``desc``.

        Only the unpadded ids count: pad ids classify as understanding tokens
        and would turn an all-gen decode into a mixed batch. Reading the ids
        costs one device sync per FULL replay.
        """
        if desc.cg_mode != CUDAGraphMode.FULL:
            return desc
        input_ids = input_batch.input_ids[: input_batch.num_tokens]
        key = moe_routing_key_from_input_ids(input_ids, self.gen_vocab_start_index)
        if key is MoERoutingKey.BLEND:
            return desc
        return self._moe_routing_descs.get((desc, key), desc)


def install_moe_routing_graphs(manager: ModelCudaGraphManager, gen_vocab_start_index: int) -> None:
    """Add ALL_UND / ALL_GEN captures for every FULL descriptor of ``manager``.

    Runs right after the runner builds its graph manager and before capture,
    including the memory-profiling build, so the profiler counts every graph.
    """
    if type(manager) is not ModelCudaGraphManager or not current_platform.is_cuda_alike():
        raise RuntimeError(
            "cudagraph_specialize_moe_routing is enabled but "
            f"{type(manager).__name__} on {current_platform.device_name} does not implement MoE specialty "
            "capture/replay. Use a cudagraph_mode without FULL."
        )
    manager.__class__ = MoERoutingCudaGraphManager
    assert isinstance(manager, MoERoutingCudaGraphManager)
    manager.gen_vocab_start_index = int(gen_vocab_start_index)
    manager._moe_routing_descs = {}

    full_descs = manager._capture_descs.get(CUDAGraphMode.FULL)
    if not full_descs:
        return
    expanded: list[BatchExecutionDescriptor] = []
    for desc in full_descs:
        expanded.append(desc)
        # Microbatch threads do not inherit the capture scope's ContextVar.
        if desc.num_ubatches > 1:
            continue
        for key in _SPECIALTY_KEYS:
            keyed = _keyed(desc, key)
            manager._moe_routing_descs[(desc, key)] = keyed
            expanded.append(keyed)
    # Stable sort keeps BLEND first within a size; capture stays widest first.
    expanded.sort(key=lambda d: d.num_tokens, reverse=True)
    manager._capture_descs[CUDAGraphMode.FULL] = expanded
