# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-shape O producers queued before the first blocking RDMA exchange."""

import os
from functools import lru_cache

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
FLAG = "VLLM_OMNI_H3_O_PRODUCER_LOOKAHEAD"


@lru_cache(None)
def producer_stream(device):
    return torch.get_device_module().Stream(device=device, priority=-1)


def prepare(active, fine, coarse, tiles, state):
    from vllm_omni.diffusion.models.minimax_h3.ops.attention.chunks import produce

    if os.environ.get(FLAG) != "1" or "o_lookahead" in active:
        raise RuntimeError("This arm requires the fixed four-chunk O lookahead route")
    assert torch.get_device_module().current_stream(fine.device) == active["comm"]
    assert fine.dtype == coarse.dtype == torch.bfloat16
    assert tuple(fine.shape) == (1, 95936, 7, 128)
    assert tuple(coarse.shape) == (1, 1648, 7, 128)
    assert dist.get_world_size() == 8 and state.communicator is not None
    count = 2998
    shape = (1, 8 * (count + 270), 7, 128)
    entries = []
    for index in range(4):
        prototype = state.registered_shape_prototype(shape, dtype=fine.dtype, device=fine.device)
        out = state.output(prototype, op="gather_heads", slot=f"h3_o_chunk4_{index}")
        landing = state.communicator.input_buffer(out, shape)
        assert landing.is_contiguous() and out.is_contiguous()
        entries.append(dict(out=out, landing=landing, ready=None))
    # Concurrent writes must use separate registered source slots and must not
    # overlap any receive buffer consumed by gate/projection on the side stream.
    ranges = sorted(
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for e in entries for t in (e["landing"], e["out"])
    )
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])), "O registered ranges overlap"
    stream = producer_stream(fine.device)
    ticket = dict(stream=stream, entries=entries, consumed=0)
    active["o_lookahead"] = ticket  # Install before submitting anything: cleanup also covers exceptions.
    stream.wait_stream(active["comm"])  # fine and the original coarse join precede this event.
    for tensor in (fine, coarse, tiles):
        tensor.record_stream(stream)
    with torch.get_device_module().stream(stream):
        for index, e in enumerate(entries):
            e["landing"].record_stream(stream)
            with torch.cuda.nvtx.range(f"h3.oproducer.lookahead.produce.{index}"):
                produce(fine, coarse, tiles, e["landing"], index * count, count)
            ready = torch.get_device_module().Event(enable_timing=False)
            ready.record(stream)
            e["ready"] = ready
    return ticket


def consume(active, index):
    ticket = active["o_lookahead"]
    assert index == ticket["consumed"] and 0 <= index < 4
    entry = ticket["entries"][index]
    assert entry["ready"] is not None
    active["comm"].wait_event(entry["ready"])
    entry["landing"].record_stream(active["comm"])
    ticket["consumed"] += 1
    return entry["out"], entry["landing"]


def complete(active):
    if active["o_lookahead"]["consumed"] != 4:
        raise RuntimeError("O producer must be consumed once per output chunk")


def cleanup(active, caller):
    ticket = active.get("o_lookahead")
    if ticket is not None:
        caller.wait_stream(ticket["stream"])
        for entry in ticket["entries"]:
            entry["landing"].record_stream(caller)
