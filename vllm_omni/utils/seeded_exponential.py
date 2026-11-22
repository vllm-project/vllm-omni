# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched per-request ``exponential_(generator=...)`` that is bit-identical to torch.

Seeded sampling draws exponential noise one request at a time, which costs a
kernel launch and host overhead per request and step. This kernel reproduces
torch's CUDA distribution kernel exactly for float32 (the grid-stride
Philox4x32-10 ``curand_uniform4`` stream, the fast-math ``__logf`` transform
and the per-call Philox offset increment), so a batch draws in one launch while
every value and every generator state afterwards matches the per-request loop.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _generator_exponential_kernel(
    out_ptr,
    rows_ptr,
    seed_ptr,
    offset_ptr,
    numel,
    threads,
    HAS_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    elements = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = elements < numel
    seed = tl.load(seed_ptr + request).to(tl.uint64)
    offset = tl.load(offset_ptr + request).to(tl.uint64)
    # torch's grid-stride kernel: thread t draws curand_uniform4 once per grid
    # stride and element j = t + threads * (4 * call + component).
    thread = (elements % threads).to(tl.uint64)
    stride = (elements // threads).to(tl.uint64)
    counter = offset // 4 + stride // 4
    component = stride % 4
    c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    c2 = (thread & 0xFFFFFFFF).to(tl.uint32)
    c3 = (thread >> 32).to(tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, c2, c3, 10)
    bits = tl.where(component == 0, r0, tl.where(component == 1, r1, tl.where(component == 2, r2, r3)))
    # curand_uniform: (0, 1] from 32 bits.
    uniform = bits.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10
    # at::exponential with lambda 1 under fast math (at::log is __logf).
    log = tl.where(
        uniform >= 1.0 - 5.960464477539063e-08,
        -5.960464477539063e-08,
        tl.extra.cuda.libdevice.fast_logf(uniform),
    )
    row = tl.load(rows_ptr + request).to(tl.int64) if HAS_ROWS else request.to(tl.int64)
    tl.store(out_ptr + row * numel + elements, -log, mask=mask)


_POLICY_CACHE: dict[tuple[int, int], tuple[int, int]] = {}


def torch_exponential_policy(numel: int, device: torch.device) -> tuple[int, int]:
    """(grid threads, Philox offset increment) of torch's CUDA ``exponential_``."""
    key = (numel, device.index or 0)
    policy = _POLICY_CACHE.get(key)
    if policy is None:
        props = torch.cuda.get_device_properties(device)
        blocks = min(props.multi_processor_count * (props.max_threads_per_multi_processor // 256), -(-numel // 256))
        threads = blocks * 256
        policy = _POLICY_CACHE[key] = (threads, ((numel - 1) // (threads * 4) + 1) * 4)
    return policy


def fill_exponential_rows(out: torch.Tensor, generators: list, rows: list[int] | None = None) -> torch.Tensor:
    """Draw ``out[rows[i]].exponential_(generator=generators[i])`` for all ``i`` in one launch.

    ``out`` is a contiguous float32 CUDA matrix; ``rows`` defaults to
    ``range(len(generators))``. ``None`` generators use the default CUDA
    generator in order. Each generator is advanced exactly as torch would.
    """
    numel = int(out.shape[-1]) if rows is not None else int(out.numel()) // len(generators)
    device = out.device
    threads, increment = torch_exponential_policy(numel, device)
    default = None
    seeds, offsets = [], []
    for generator in generators:
        if generator is None:
            if default is None:
                default = torch.cuda.default_generators[device.index or 0]
            generator = default
        offset = generator.get_offset()
        generator.set_offset(offset + increment)
        seed = generator.initial_seed()
        seeds.append(seed - (1 << 64) if seed >= 1 << 63 else seed)
        offsets.append(offset)
    table = [seeds, offsets] if rows is None else [seeds, offsets, list(rows)]
    state = torch.tensor(table, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
    block = 1024
    _generator_exponential_kernel[(len(generators), triton.cdiv(numel, block))](
        out,
        state[2] if rows is not None else state[0],
        state[0],
        state[1],
        numel,
        threads,
        HAS_ROWS=rows is not None,
        BLOCK=block,
    )
    return out


def batched_seeded_exponential_supported(q: torch.Tensor, generators: dict) -> bool:
    """Whether ``fill_exponential_rows`` reproduces the per-row loop for ``q``."""
    return (
        len(generators) > 1
        and q.is_cuda
        and q.dtype == torch.float32
        and q.dim() == 2
        and q.is_contiguous()
        # A generator shared by two rows makes the loop order observable.
        and len({id(g) for g in generators.values()}) == len(generators)
    )
