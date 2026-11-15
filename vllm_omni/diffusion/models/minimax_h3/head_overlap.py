# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-request dense MiniMax H3 head-group Ulysses experiment.

The scheduling follows sgl-project/sglang#43164 and #43361. Transport uses
vLLM-Omni's existing symmetric-memory mappings and CUDA copy-engine runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.a2a_permute import (
    _SYMM_WORKSPACES,
    _ce_runtime,
    _get_symm_buffer,
)
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope, stage_head_group_norm_rope

logger = init_logger(__name__)


@dataclass
class _State:
    transfer: torch.cuda.Stream
    received: list[torch.cuda.Event]
    computed: list[torch.cuda.Event]
    finished: torch.cuda.Event
    checked: set = field(default_factory=set)


_states = {}


@torch.compiler.disable
def run_head_overlap(attention, q, k, v, metadata, groups, direct, *, rope, q_weight, k_weight, eps):
    strategy = attention._get_active_parallel_strategy()
    if (
        strategy.name != "ulysses"
        or not strategy._ulysses_a2a_permute
        or attention._effective_kv_cache_dtype() not in (None, "float")
        or attention.is_paged_kv_active()
    ):
        return None
    if torch.cuda.is_current_stream_capturing() or strategy._sp_group.ring_world_size != 1:
        return None
    world = strategy._sp_group.ulysses_world_size
    group_name = strategy._ulysses_pg.group_name
    rows, heads, dim = q.shape
    if (
        q.shape != k.shape
        or q.shape != v.shape
        or heads % (world * groups)
        or (rope is not None and (dim != 128 or rope.shape[-1] != 96))
        or any(x.stride(2) != 1 or x.stride(1) != dim for x in (q, k, v))
    ):
        return None
    device = q.device
    main = torch.cuda.current_stream(device)
    state_key = (device, group_name, main.cuda_stream, groups, direct)
    state = _states.get(state_key)
    if state is None:
        state = _State(
            torch.cuda.Stream(device=device),
            [torch.cuda.Event() for _ in range(groups)],
            [torch.cuda.Event() for _ in range(groups)],
            torch.cuda.Event(),
        )
        _states[state_key] = state
        logger.info("H3 head overlap active: groups=%d direct=%s rows=%d heads=%d", groups, direct, rows, heads)

    # One request owns the shared buffers until the final main-stream wait.
    # Every rank executes the same sequence of barriers on its copy stream.
    part_heads = heads // (world * groups)
    part_width = part_heads * dim
    group_size = world * rows * part_width
    transfer_size = groups * group_size
    runtime = _ce_runtime()
    shared = _get_symm_buffer((4 * transfer_size,), q.dtype, device, group_name)
    handle = _SYMM_WORKSPACES[(device, group_name)].handle
    rank = handle.rank
    element = q.element_size()
    width = part_width * element
    copy = runtime.lib.cudaMemcpy2DAsync
    transfer = state.transfer

    staged = None
    if rope is not None:
        staged = q.new_empty((2, world * groups, rows, part_heads, dim))
        stage_head_group_norm_rope(q, q_weight, rope, eps, staged[0])
        stage_head_group_norm_rope(k, k_weight, rope, eps, staged[1])

    transfer.wait_stream(main)
    # The forward workspace can include data read by the previous call's FA.
    with torch.cuda.stream(transfer):
        handle.barrier()
    for group in range(groups):
        for peer in range(world):
            head = (peer * groups + group) * part_heads
            dest_offset = (group * group_size + rank * rows * part_width) * element
            dest = handle.buffer_ptrs[peer] + dest_offset
            for index, tensor in enumerate((q, k, v)):
                source_pitch = tensor.stride(0) * element
                source = tensor.data_ptr() + head * dim * element
                if staged is not None and index < 2:
                    source = (
                        staged.data_ptr()
                        + (index * transfer_size + (peer * groups + group) * rows * part_width) * element
                    )
                    source_pitch = width
                runtime.CUDART_CHECK(
                    copy(
                        dest + index * transfer_size * element,
                        width,
                        source,
                        source_pitch,
                        width,
                        rows,
                        4,
                        transfer.cuda_stream,
                    )
                )
        with torch.cuda.stream(transfer):
            handle.barrier()
            state.received[group].record(transfer)

    attended = []
    for group in range(groups):
        main.wait_event(state.received[group])
        offset = group * group_size
        parts = [
            shared.narrow(0, index * transfer_size + offset, group_size).view(1, world * rows, part_heads, dim)
            for index in range(3)
        ]
        attended.append(attention._run_local_attention(*parts, metadata))
        state.computed[group].record(main)
        transfer.wait_event(state.computed[group])
        tensor = attended[-1]
        for peer in range(world):
            local_head = rank * groups + group
            dest = handle.buffer_ptrs[peer] + 3 * transfer_size * element
            if direct:
                dest += local_head * width
                dest_pitch = heads * dim * element
            else:
                dest += (group * group_size + rank * rows * part_width) * element
                dest_pitch = width
            source = tensor.data_ptr() + peer * rows * tensor.stride(1) * element
            runtime.CUDART_CHECK(
                copy(dest, dest_pitch, source, tensor.stride(1) * element, width, rows, 4, transfer.cuda_stream)
            )
    with torch.cuda.stream(transfer):
        handle.barrier()
        state.finished.record(transfer)
    main.wait_event(state.finished)

    output = shared.narrow(0, 3 * transfer_size, transfer_size)
    if direct:
        output = output.view(rows, heads, dim).clone()
    else:
        output = output.view(groups, world, rows, part_heads, dim).permute(2, 1, 0, 3, 4).reshape(rows, heads, dim)
    if os.environ.get("OMNI_H3_OVERLAP_CHECK") == "1":
        signature = (rows, heads, dim, metadata.extra["valid_kv_length"])
        if signature not in state.checked:
            reference_q, reference_k = (
                (q, k) if rope is None else fused_qk_norm_rope(q, k, q_weight, k_weight, rope, eps)
            )
            expected = attention(reference_q.unsqueeze(0), reference_k.unsqueeze(0), v.unsqueeze(0), metadata).squeeze(
                0
            )
            match = torch.tensor(int(torch.equal(output, expected)), device=device)
            dist.all_reduce(match, op=dist.ReduceOp.MIN, group=strategy._ulysses_pg)
            if not match.item():
                raise RuntimeError("H3 head-group attention differs from full-head attention")
            state.checked.add(signature)
            logger.info("H3 head overlap matched full-head bytes: groups=%d shape=%s", groups, signature)
    return output
