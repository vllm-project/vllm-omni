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

Host ``sum().item()`` runs **outside** the captured region (graph dispatch).
Inside the graph, mixed still uses the #9764 blend path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum

import torch

from vllm_omni.cudagraph.moe_routing_compat import moe_routing_specialize_enabled


class MoERoutingKey(IntEnum):
    """CUDA-graph specialty for MammothModa2 MoE layers."""

    BLEND = 0
    ALL_UND = 1
    ALL_GEN = 2


# Active specialty while a graph is being captured. Replay does not consult
# this: the runner selects the already-recorded graph via BatchDescriptor.
# Default BLEND preserves #9764 if capture forgets to enter a scope.
_moe_routing_key: ContextVar[MoERoutingKey] = ContextVar(
    "mammoth_moda2_moe_routing_key",
    default=MoERoutingKey.BLEND,
)


def get_moe_routing_key() -> MoERoutingKey:
    return _moe_routing_key.get()


def set_moe_routing_key(key: MoERoutingKey) -> None:
    _moe_routing_key.set(key)


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


def routing_ids_for_replay(
    cpu_ids: torch.Tensor | None,
    gpu_ids: torch.Tensor | None,
    *,
    num_tokens: int,
    prev_sampled_token_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Scheduled token ids used to pick a specialty graph on replay.

    ``num_tokens`` is the unpadded scheduled count. CUDA-graph padding sits
    in the tail and is usually 0, which classifies as understanding. Including
    it turns an all-gen decode into a mixed batch, so the runner would keep
    replaying the blend graph.

    Async scheduling writes the newest decode ids only into the GPU buffer
    (from ``prev_sampled_token_ids``). The CPU buffer is stale for those
    positions; reading it would pick the wrong expert.
    """
    if num_tokens <= 0:
        return None

    def _slice(ids: torch.Tensor | None) -> torch.Tensor | None:
        if ids is None or ids.numel() < num_tokens:
            return None
        return ids[:num_tokens]

    if prev_sampled_token_ids is not None:
        gpu_slice = _slice(gpu_ids)
        if gpu_slice is not None:
            return gpu_slice
    cpu_slice = _slice(cpu_ids)
    if cpu_slice is not None:
        return cpu_slice
    return _slice(gpu_ids)


def capture_mask_for_routing_key(
    num_tokens: int,
    key: MoERoutingKey,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dummy ``gen_token_mask`` matching the specialty being captured."""
    if key is MoERoutingKey.ALL_GEN:
        return torch.ones(num_tokens, dtype=torch.bool, device=device)
    if key is MoERoutingKey.ALL_UND:
        return torch.zeros(num_tokens, dtype=torch.bool, device=device)
    # BLEND: mixed pattern so torch.where stays live in the recorded graph.
    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    if num_tokens > 0:
        mask[0] = True
    if num_tokens > 1:
        mask[1] = False
    if num_tokens > 2:
        mask[2:] = True
    return mask


# ---------------------------------------------------------------------------
# Runner-facing helpers (used by MammothGPUARModelRunner only)
# ---------------------------------------------------------------------------


def find_gen_vocab_start_index(runner: object) -> int | None:
    cursor = getattr(runner, "model", None)
    # Mammoth pipeline wraps AR as ``model.ar.language_model``.
    for _ in range(4):
        if cursor is None:
            return None
        gen_start = getattr(cursor, "gen_vocab_start_index", None)
        if gen_start is not None:
            return int(gen_start)
        cursor = getattr(cursor, "language_model", None) or getattr(cursor, "ar", None)
    return None


def _warn_missing_gen_vocab_start_index_once(runner: object) -> None:
    if getattr(runner, "_warned_missing_gen_vocab_start_index", False):
        return
    runner._warned_missing_gen_vocab_start_index = True  # type: ignore[attr-defined]
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "cudagraph_specialize_moe_routing is enabled but no gen_vocab_start_index "
        "was found on the model; MoE CUDA-graph replay will stay on the BLEND "
        "graph and the ALL_UND / ALL_GEN graphs will never be selected. Check "
        "the model wrapper path used by this stage."
    )


def resolve_moe_routing_key(
    runner: object,
    *,
    force_moe_routing_key: int | None = None,
    num_tokens: int | None = None,
    token_ids: torch.Tensor | None = None,
    token_ids_provided: bool = False,
) -> MoERoutingKey:
    if force_moe_routing_key is not None:
        return MoERoutingKey(int(force_moe_routing_key))
    if not moe_routing_specialize_enabled(runner):
        return MoERoutingKey.BLEND
    gen_start = find_gen_vocab_start_index(runner)
    if gen_start is None:
        # Specialize is on, so ALL_UND / ALL_GEN graphs are being captured
        # for a model we cannot classify. Warning once here keeps the
        # pre-fix failure mode (capture then never select) visible instead
        # of silent.
        _warn_missing_gen_vocab_start_index_once(runner)
        return MoERoutingKey.BLEND
    if token_ids_provided:
        return moe_routing_key_from_input_ids(token_ids, int(gen_start))
    if num_tokens is None:
        return MoERoutingKey.BLEND
    input_ids = runner.input_ids.gpu[:num_tokens]  # type: ignore[attr-defined]
    return moe_routing_key_from_input_ids(input_ids, int(gen_start))


def annotate_moe_routing_batch_descriptor(
    runner: object,
    batch_desc,
    *,
    num_tokens: int,
    force_moe_routing_key: int | None = None,
):
    """Pin the MoE specialty on the graph key used for capture."""
    from vllm_omni.cudagraph.moe_routing_dispatcher import with_moe_routing_key

    moe_key = resolve_moe_routing_key(
        runner,
        force_moe_routing_key=force_moe_routing_key,
        num_tokens=num_tokens,
    )
    return with_moe_routing_key(batch_desc, moe_key)


def select_moe_replay_batch_descriptor(runner: object, mode, batch_desc, num_tokens: int):
    """Choose the FULL specialty graph for this step's scheduled tokens.

    Capture records BLEND / ALL_UND / ALL_GEN. ``CudagraphDispatcher.dispatch``
    is invoked by upstream padding logic without a routing key, so the
    descriptor it returns is BLEND. Replay has to retarget that descriptor
    before ``set_forward_context`` or the specialized graphs never run.
    """
    from vllm.logger import init_logger

    from vllm_omni.cudagraph.moe_routing_dispatcher import apply_moe_routing_key

    if not moe_routing_specialize_enabled(runner):
        return batch_desc
    input_ids = getattr(runner, "input_ids", None)
    prev_sampled = getattr(getattr(runner, "input_batch", None), "prev_sampled_token_ids", None)
    token_ids = routing_ids_for_replay(
        None if input_ids is None else getattr(input_ids, "cpu", None),
        None if input_ids is None else getattr(input_ids, "gpu", None),
        num_tokens=num_tokens,
        prev_sampled_token_ids=prev_sampled,
    )
    requested = resolve_moe_routing_key(runner, token_ids=token_ids, token_ids_provided=True)
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    captured = () if dispatcher is None else dispatcher.cudagraph_keys.get(mode, ())
    selected = apply_moe_routing_key(mode, batch_desc, requested, captured, enabled=True)
    selected_key = int(getattr(selected, "moe_routing_key", int(MoERoutingKey.BLEND)))
    logger = init_logger(__name__)
    # Mixed und/gen traffic flips this every few steps, so only the first
    # selection is worth INFO. Later flips stay at DEBUG to keep the worker
    # log readable.
    if getattr(runner, "_last_moe_routing_replay_key", None) is not None:
        logger.debug(
            "Mammoth MoE CUDA-graph replay specialty %s (requested %s, num_tokens=%s, mode=%s)",
            MoERoutingKey(selected_key).name,
            requested.name,
            num_tokens,
            getattr(mode, "name", mode),
        )
    else:
        logger.info(
            "Mammoth MoE CUDA-graph replay specialty is %s (requested %s, num_tokens=%s, mode=%s); "
            "further changes log at DEBUG",
            MoERoutingKey(selected_key).name,
            requested.name,
            num_tokens,
            getattr(mode, "name", mode),
        )
    runner._last_moe_routing_replay_key = selected_key  # type: ignore[attr-defined]
    return selected


def warmup_and_capture_with_moe_routing(
    runner: object,
    desc,
    cudagraph_runtime_mode,
    profile_seq_lens: int | None = None,
    allow_microbatching: bool = False,
    num_warmups: int | None = None,
    profiler=None,
) -> None:
    """Capture one graph; honour ``desc.moe_routing_key`` when specializing (#9865)."""
    from contextlib import nullcontext

    import torch
    from vllm.config import CUDAGraphMode

    if profiler is None:
        profiler = nullcontext()
    if num_warmups is None:
        num_warmups = runner.compilation_config.cudagraph_num_of_warmups  # type: ignore[attr-defined]
    force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
    moe_key = int(getattr(desc, "moe_routing_key", MoERoutingKey.BLEND))
    # Always force the capture key on the Mammoth runner — BLEND included —
    # so the resolver cannot reclassify a uniform dummy buffer as ALL_UND/ALL_GEN.
    moe_kwargs = {"force_moe_routing_key": moe_key} if moe_routing_specialize_enabled(runner) else {}
    for _ in range(num_warmups):
        runner._dummy_run(  # type: ignore[attr-defined]
            desc.num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            force_attention=force_attention,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            profile_seq_lens=profile_seq_lens,
            **moe_kwargs,
        )
    if num_warmups > 0:
        torch.accelerator.synchronize()
    with (
        profiler,
        torch.profiler.record_function(f"capture_{desc.num_tokens}_{cudagraph_runtime_mode.name}_moe{moe_key}"),
        moe_routing_key_scope(MoERoutingKey(moe_key)),
    ):
        runner._dummy_run(  # type: ignore[attr-defined]
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            is_graph_capturing=True,
            profile_seq_lens=profile_seq_lens,
            **moe_kwargs,
        )
