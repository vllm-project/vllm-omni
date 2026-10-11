# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
import math
import os
import sysconfig
import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import _resolve_process_group
from vllm.logger import init_logger

logger = init_logger(__name__)

_BUILD_LOCK = threading.Lock()
_BUILT = False
_WORKSPACE_LOCK = threading.Lock()


@dataclass(slots=True)
class _SymmWorkspace:
    allocation: torch.Tensor
    handle: Any
    capacity_bytes: int
    stream_id: int


# A single grow-only byte workspace per device/process-group. Shape changes
# reuse a typed view into the workspace instead of retaining one allocation per
# historical request shape.
_SYMM_WORKSPACES: dict[tuple[torch.device, str], _SymmWorkspace] = {}


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
        site_packages = sysconfig.get_paths()["purelib"]
        nvidia_root = os.path.join(site_packages, "nvidia")
        include_paths = glob.glob(os.path.join(nvidia_root, "*", "include"))
        nccl_libs = glob.glob(os.path.join(nvidia_root, "nccl", "lib", "libnccl.so*"))
        # The runtime ships its own NCCL instead of the PyPI nvidia-nccl-cu13:
        # a bundled libnccl.so plus the nccl_device headers.
        # Without this fallback the build aborts with "could not locate nvidia-nccl libnccl.so",
        # which kills the whole lane at model init.
        if not nccl_libs or not any(os.path.isfile(os.path.join(path, "nccl.h")) for path in include_paths):
            for root in sorted(
                glob.glob(os.path.join(site_packages, "*nccl*")) + glob.glob(os.path.join(site_packages, "nvidia", "*"))
            ):
                include_paths += glob.glob(os.path.join(root, "include"))
                nccl_libs += glob.glob(os.path.join(root, "lib", "libnccl.so*"))
        if not nccl_libs:
            raise RuntimeError("a2a_permute: could not locate libnccl.so (checked the usual site-packages locations)")
        if not any(os.path.isfile(os.path.join(path, "nccl.h")) for path in include_paths):
            raise RuntimeError("a2a_permute: could not locate nccl.h")
        load(
            name="vllm_omni_a2a_permute",
            sources=[src],
            # PyTorch wheel-only installations keep headers such as cusparse.h
            # under nvidia/{cu13,cusparse}/include rather than CUDA_HOME.
            extra_include_paths=list(include_paths),
            extra_cflags=["-DUSE_NCCL", "-DUSE_C10D_NCCL", "-O3"],
            extra_cuda_cflags=["-DUSE_NCCL", "-DUSE_C10D_NCCL", "-O3", "--expt-relaxed-constexpr"],
            extra_ldflags=[nccl_libs[0]],
            is_python_module=False,
            verbose=False,
        )
        symm_mem.set_backend("NCCL")
        logger.info("[a2a_permute] JIT kernel built and loaded; symm-mem backend=NCCL")
        _BUILT = True


def ensure_a2a_permute_available() -> None:
    """Build the extension during worker/model initialization, not a request."""
    _ensure_built()


def _required_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


_SYMM_NAMES: dict[str, str] = {}


def _symm_group_name(group_name: str) -> str:
    """Return the name symmetric memory should use for this process group.

    Use the process group's OWN name. An earlier revision renamed it ("a2a_symm_36") and
    registered that with c10d, on the theory that the host-communicator registry is keyed by
    registered name. The renaming is itself what breaks
    it -- rendezvous(t, pg) and rendezvous(t, pg.group_name) both succeed, while the alias
    fails with "NCCL host communicator for group a2a_symm_... not found".
    """
    cached = _SYMM_NAMES.get(group_name)
    if cached is not None:
        return cached
    name = group_name
    try:
        symm_mem.enable_symm_mem_for_group(name)
        logger.info(
            "[a2a_permute] symm-mem group %r enabled=%s",
            name,
            symm_mem.is_symm_mem_enabled_for_group(name),
        )
    except Exception as exc:
        logger.warning("[a2a_permute] enable_symm_mem_for_group(%r) failed: %s: %s", name, type(exc).__name__, exc)
    _SYMM_NAMES[group_name] = name
    return name


_COMM_PTRS: dict[tuple[str, str], int] = {}


def _comm_ptr_for(group_name: str, device: torch.device) -> int:
    """Host ncclComm_t for this group/device as an int, for the CUDA op.

    The op cannot consult NCCLDevCommManager (see the comment in csrc/a2a_permute.cu), so
    the pointer is read from the process group's NCCL backend and handed down.
    """
    key = (str(device), str(group_name))
    cached = _COMM_PTRS.get(key)
    if cached:
        return cached
    pg = _resolve_process_group(group_name)
    backend = pg._get_backend(torch.device(device))
    ptr = int(backend._comm_ptr())
    if ptr == 0:
        raise RuntimeError(f"a2a_permute: no host NCCL comm for group {group_name!r} on {device}")
    _COMM_PTRS[key] = ptr
    logger.info("[a2a_permute] group %r host comm ptr=%d on %s", group_name, ptr, device)
    return ptr


def _get_symm_buffer(
    symm_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, group_name: str
) -> torch.Tensor:
    """Get a typed view from a grow-only symmetric-memory workspace.

    Allocation/growth and rendezvous are collective. Every rank reaches them in
    lockstep because sequence parallel execution is symmetric. Steady-state
    requests at or below the peak capacity take no synchronization path.
    """
    device = torch.device(device)
    key = (device, group_name)
    required_bytes = _required_nbytes(symm_shape, dtype)
    stream_id = torch.cuda.current_stream(device).cuda_stream
    with _WORKSPACE_LOCK:
        workspace = _SYMM_WORKSPACES.get(key)
        if workspace is not None and workspace.stream_id != stream_id:
            raise RuntimeError("a2a_permute workspace must be used from a single CUDA stream")
        if workspace is None or workspace.capacity_bytes < required_bytes:
            if torch.cuda.is_current_stream_capturing():
                capacity = 0 if workspace is None else workspace.capacity_bytes
                raise RuntimeError(
                    "a2a_permute: cannot grow the symmetric-memory workspace "
                    f"from {capacity} to {required_bytes} bytes during CUDA graph capture; "
                    "warm up the maximum request shape before capture"
                )
            if workspace is None:
                # Registered name (not the numeric pg id) -- see _symm_group_name.
                symm_name = _symm_group_name(group_name)
                pg = _resolve_process_group(symm_name)
                # NCCL symmetric-memory rendezvous requires the communicator to
                # exist before the first allocation.
                pg = _resolve_process_group(group_name)
                warm = torch.ones(1, device=device)
                dist.all_reduce(warm, group=pg)
            # Growth is rare and must not release the previous allocation while
            # an earlier kernel is still consuming it.
            torch.accelerator.synchronize(device)
            allocation = symm_mem.empty(required_bytes, dtype=torch.uint8, device=device)
            handle = symm_mem.rendezvous(allocation, symm_name)
            workspace = _SymmWorkspace(
                allocation=allocation,
                handle=handle,
                capacity_bytes=required_bytes,
                stream_id=stream_id,
            )
            _SYMM_WORKSPACES[key] = workspace
        return workspace.allocation[:required_bytes].view(dtype).view(symm_shape)


def clear_a2a_permute_workspaces() -> None:
    """Release cached symmetric-memory workspaces during worker shutdown."""
    with _WORKSPACE_LOCK:
        if not _SYMM_WORKSPACES:
            return
        devices = {workspace.allocation.device for workspace in _SYMM_WORKSPACES.values()}
        for device in devices:
            torch.accelerator.synchronize(device)
        count = len(_SYMM_WORKSPACES)
        _SYMM_WORKSPACES.clear()
    logger.info("[a2a_permute] Released %d symmetric-memory workspace(s)", count)


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
    # Fused QKV projections produce row-strided views. Stage complete rows with
    # the copy engine instead of materializing the view with TensorIterator.
    torch.ops.a2ap.copy_rows(x.view(rows, p, lc), symm_in)
    out = torch.empty(p, rows, lc, device=x.device, dtype=x.dtype)
    torch.ops.a2ap.all_to_all_permute(
        symm_in, out, 1, 0, _symm_group_name(group_name), _comm_ptr_for(group_name, x.device)
    )
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
    torch.ops.a2ap.all_to_all_permute(
        symm_in, out, 0, 1, _symm_group_name(group_name), _comm_ptr_for(group_name, y.device)
    )
    # (rows, p, cols) -> (B, S_local, H, D), heads ordered rank-major
    return out.reshape(B, s_local, p, Hl, D).reshape(B, s_local, H, D).contiguous()


@ulysses_o_rev.register_fake
def _(y: torch.Tensor, group_name: str, world_size: int) -> torch.Tensor:
    B, s_global, Hl, D = y.shape
    return y.new_empty(B, s_global // world_size, Hl * world_size, D)
