# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request RNG handoff after an accepted prefill sample, before decode."""

from __future__ import annotations

import msgspec
import torch

from vllm_omni.engine.pd_continuation import PD_PREFILL_KEY, needs_pd_rng_state


class PDRNGState(msgspec.Struct):
    seed: int
    device_type: str
    state: bytes


class PDRNGHandoff(msgspec.Struct):
    ranks: list[PDRNGState]
    version: int = 1


def capture_pd_rng_states(requests, req_ids, valid_token_ids, *, tp_group, async_scheduling):
    """Called on all TP workers AFTER bookkeeping, including discard rollback.

    Invalid partial-prefill samples have empty token lists and must not export
    state. P is synchronous; D and ordinary requests incur no synchronization
    or collective. Each logical rank retains its own generator stream.
    """
    candidates = []
    for index, req_id in enumerate(req_ids):
        request = requests[req_id]
        params = request.sampling_params
        if params is None or not (params.extra_args or {}).get(PD_PREFILL_KEY) or not needs_pd_rng_state(params):
            continue
        if async_scheduling:
            raise RuntimeError("PD RNG export requires synchronous producer scheduling")
        tokens = valid_token_ids[index]
        if not tokens:
            continue
        if len(tokens) != 1 or len(request.output_token_ids) != 1:
            raise RuntimeError("PD RNG export requires exactly one accepted producer token")
        candidates.append((req_id, request.generator))
    if not candidates:
        return None

    # Propagate export failures through the same collective so a rank-local
    # get_state failure cannot leave the other ranks waiting at all_gather.
    local = {}
    error = None
    try:
        devices = {generator.device for _, generator in candidates if generator is not None}
        for device in devices:
            if device.type != "cpu":
                torch.get_device_module(device).synchronize(device)
        for req_id, generator in candidates:
            if generator is None:
                raise RuntimeError(f"PD producer {req_id} has no request generator")
            state = generator.get_state().detach().cpu().contiguous()
            local[req_id] = PDRNGState(int(generator.initial_seed()), generator.device.type, state.numpy().tobytes())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    # Lazy lookup avoids even requiring distributed state for non-PD requests.
    group = tp_group()
    packets = [(local, error)]
    if group.world_size > 1:
        packets = [None] * group.world_size
        torch.distributed.all_gather_object(packets, (local, error), group=group.cpu_group)
    for rank, (states, rank_error) in enumerate(packets):
        if rank_error is not None or states.keys() != local.keys():
            raise RuntimeError(f"PD RNG export failed on TP rank {rank}: {rank_error or 'request mismatch'}")
    result = {}
    for req_id, _ in candidates:
        ranks = [states[req_id] for states, _ in packets]
        result[req_id] = msgspec.msgpack.encode(PDRNGHandoff(ranks))
    return result


def restore_pd_rng_state(generator, payload, *, seed, tp_rank, tp_size, req_id):
    """Restore once when creating D's request, before inserting it in a batch.

    Seed is a consistency check, never a replacement for the producer's state.
    Reject incompatible/missing state instead of silently restarting a stream.
    """
    if generator is None or not payload:
        raise ValueError(f"PD request {req_id} is missing a generator or producer RNG state")
    # torch.manual_seed maps negative seeds to their unsigned 64-bit value.
    expected_seed = seed % (1 << 64)
    if generator.initial_seed() != expected_seed:
        raise ValueError("PD generator was initialized with a different seed")
    handoff = msgspec.msgpack.decode(payload, type=PDRNGHandoff)
    if handoff.version != 1 or len(handoff.ranks) != tp_size or not 0 <= tp_rank < tp_size:
        raise ValueError("PD RNG handoff version or TP layout mismatch")
    for state in handoff.ranks:
        if state.seed != expected_seed or state.device_type != generator.device.type or not state.state:
            raise ValueError("PD RNG seed or device type mismatch, or empty state")
    state = handoff.ranks[tp_rank]
    generator.set_state(torch.frombuffer(bytearray(state.state), dtype=torch.uint8))
    if generator.initial_seed() != expected_seed or generator.get_state().cpu().numpy().tobytes() != state.state:
        raise RuntimeError("PD generator did not restore the exact producer state")
