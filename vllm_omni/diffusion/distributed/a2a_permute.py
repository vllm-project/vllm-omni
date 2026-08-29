# SPDX-License-Identifier: Apache-2.0
"""Fused permute-free Ulysses all-to-all over NCCL symmetric memory.

JIT-compiles the CUDA kernel from pytorch/pytorch#178230 (``all_to_all_permute``)
and exposes it as two *functional* custom ops:

    ulysses_qkv_fwd(x, group_name, world_size)  # (B, S/p, H, D) -> (B, S, H/p, D)
    ulysses_o_rev(y, group_name, world_size)     # (B, S, H/p, D) -> (B, S/p, H, D)

These replace the synchronous ``all_to_all_4D`` (permute + NCCL all_to_all_single)
used by Ulysses SP. All symmetric-memory bookkeeping (a shape-keyed, rendezvoused
buffer cache + the copy-in) lives *inside* the ops, so from torch.compile's point
of view each op is an opaque ``input -> output`` function: no graph break, no
visible buffer mutation, no Python control flow in the traced graph. A
``register_fake`` provides output metadata so Dynamo keeps the op in-graph.
"""

from __future__ import annotations

import glob
import os
import sysconfig
import threading

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group
from vllm.logger import init_logger

logger = init_logger(__name__)

_BUILD_LOCK = threading.Lock()
_BUILT = False
# (symm_shape, dtype, group_name) -> rendezvoused symmetric-memory buffer
_SYMM_CACHE: dict[tuple, torch.Tensor] = {}


def _ensure_built() -> None:
    """JIT-compile + load the kernel once (process-wide). Cached by torch."""
    global _BUILT
    if _BUILT:
        return
    with _BUILD_LOCK:
        if _BUILT:
            return
        from torch.utils.cpp_extension import load

        here = os.path.dirname(__file__)
        src = os.path.join(here, "csrc", "a2a_permute.cu")
        sp = sysconfig.get_paths()["purelib"]
        nccl_dir = os.path.join(sp, "nvidia", "nccl")
        nccl_inc = os.path.join(nccl_dir, "include")
        nccl_libs = glob.glob(os.path.join(nccl_dir, "lib", "libnccl.so*"))
        if not nccl_libs:
            raise RuntimeError("a2a_permute: could not locate nvidia-nccl libnccl.so")
        load(
            name="vllm_omni_a2a_permute",
            sources=[src],
            extra_include_paths=[nccl_inc],
            extra_cflags=["-DUSE_NCCL", "-DUSE_C10D_NCCL", "-O3"],
            extra_cuda_cflags=["-DUSE_NCCL", "-DUSE_C10D_NCCL", "-O3", "--expt-relaxed-constexpr"],
            extra_ldflags=[nccl_libs[0]],
            is_python_module=False,
            verbose=False,
        )
        symm_mem.set_backend("NCCL")
        logger.info("[a2a_permute] JIT kernel built and loaded; symm-mem backend=NCCL")
        _BUILT = True


def _get_symm_buffer(
    symm_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, group_name: str
) -> torch.Tensor:
    """Get (or lazily allocate + rendezvous) a symmetric-memory buffer.

    Rendezvous is a collective; on a cache miss every rank hits the same shape in
    lockstep (SP is symmetric), so they rendezvous together. A one-element
    all_reduce first guarantees the group's NCCL communicator exists (required
    before NCCL-symm rendezvous, else it segfaults).
    """
    key = (symm_shape, dtype, group_name)
    buf = _SYMM_CACHE.get(key)
    if buf is None:
        pg = _resolve_process_group(group_name)
        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=pg)
        torch.accelerator.synchronize(device)
        buf = symm_mem.empty(*symm_shape, dtype=dtype, device=device)
        symm_mem.rendezvous(buf, group_name)
        _SYMM_CACHE[key] = buf
    return buf


# ---------------------------------------------------------------------------
# Forward: (B, S_local, H, D) -> (B, S_global, H_local, D)
#   scatter heads (dim 2), gather sequence (dim 1). Matches all_to_all_4D(x,2,1).
# ---------------------------------------------------------------------------
@torch.library.custom_op("vllm_omni_a2a::ulysses_qkv_fwd", mutates_args=(), device_types="cuda")
def ulysses_qkv_fwd(x: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    _ensure_built()
    p = world_size
    B, s_local, H, D = x.shape
    Hl = H // p
    lc = Hl * D
    rows = B * s_local
    symm_in = _get_symm_buffer((rows, p, lc), x.dtype, x.device, group_name)
    # (B, S_local, H, D) -> (rows, p, lc); H is row-major so column block r = heads [r*Hl:(r+1)*Hl]
    symm_in.copy_(x.reshape(rows, p, lc))
    out = torch.empty(p, rows, lc, device=x.device, dtype=x.dtype)
    torch.ops.a2ap.all_to_all_permute(symm_in, out, 1, 0, group_name)
    # (p, rows, lc) -> (B, S_global, H_local, D), sequence ordered rank-major
    return out.reshape(p, B, s_local, Hl, D).permute(1, 0, 2, 3, 4).reshape(B, p * s_local, Hl, D).contiguous()


@ulysses_qkv_fwd.register_fake
def _(x: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    B, s_local, H, D = x.shape
    return x.new_empty(B, s_local * world_size, H // world_size, D)


# ---------------------------------------------------------------------------
# Reverse: (B, S_global, H_local, D) -> (B, S_local, H, D)
#   scatter sequence (dim 1), gather heads (dim 2). Matches all_to_all_4D(y,1,2).
# ---------------------------------------------------------------------------
@torch.library.custom_op("vllm_omni_a2a::ulysses_o_rev", mutates_args=(), device_types="cuda")
def ulysses_o_rev(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    _ensure_built()
    p = world_size
    B, s_global, Hl, D = y.shape
    s_local = s_global // p
    H = Hl * p
    cols = Hl * D
    rows = B * s_local
    symm_in = _get_symm_buffer((p, rows, cols), y.dtype, y.device, group_name)
    # (B, p*S_local, Hl, D) -> (p, B*S_local, cols): block r = sequence shard destined to rank r
    symm_in.copy_(y.reshape(B, p, s_local, Hl, D).permute(1, 0, 2, 3, 4).reshape(p, rows, cols))
    out = torch.empty(rows, p, cols, device=y.device, dtype=y.dtype)
    torch.ops.a2ap.all_to_all_permute(symm_in, out, 0, 1, group_name)
    # (rows, p, cols) -> (B, S_local, H, D), heads ordered rank-major
    return out.reshape(B, s_local, p, Hl, D).reshape(B, s_local, H, D).contiguous()


@ulysses_o_rev.register_fake
def _(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    B, s_global, Hl, D = y.shape
    return y.new_empty(B, s_global // world_size, Hl * world_size, D)
