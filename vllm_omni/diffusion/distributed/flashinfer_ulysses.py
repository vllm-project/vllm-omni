# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional FlashInfer PCIe/RDMA transport for strict Ulysses attention.

Communicators own separate registered Q/K/V/gate/output slots. All ranks must
close them before destroying the process group. The default transport is
unchanged; no model layout or producer scheduling is owned by this module.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group
from vllm.logger import init_logger

logger = init_logger(__name__)
_BACKEND_ENV = "VLLM_OMNI_ULYSSES_A2A_BACKEND"
_MAX_BYTES_ENV = "VLLM_OMNI_FLASHINFER_ULYSSES_MAX_BYTES"
_REQUIRE_RDMA_ENV = "VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA"
_EXPECTED_BACKEND = "flashinfer-pcie"
_LOCK = threading.RLock()


def configured_backend() -> str:
    """Return the explicitly selected optimized Ulysses implementation."""
    return os.getenv(_BACKEND_ENV, "symmetric").strip().lower()


def is_flashinfer_pcie_enabled() -> bool:
    return configured_backend() == _EXPECTED_BACKEND


def _require_rdma() -> bool:
    value = os.getenv(_REQUIRE_RDMA_ENV, "1").strip().lower()
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{_REQUIRE_RDMA_ENV} must be 0/1/false/true, got {value!r}")
    return value in ("1", "true")


def ensure_flashinfer_pcie_available() -> None:
    """Validate the PR #4876 Python/API surface without compiling a kernel."""
    try:
        from flashinfer.comm import UlyssesCommunicator
        from flashinfer.comm.ulysses import missing_ulysses_pcie_dependencies
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "flashinfer-pcie requires FlashInfer PR #4876 (UlyssesCommunicator "
            "with backend='pcie'); prepend its pinned source checkout to PYTHONPATH"
        ) from exc

    missing = missing_ulysses_pcie_dependencies()
    if missing:
        raise RuntimeError("flashinfer-pcie JIT dependencies are missing: " + ", ".join(missing))
    methods = ["allocate_output", "input_buffer", "scatter_heads", "gather_heads", "close"]
    for method in methods:
        if not hasattr(UlyssesCommunicator, method):
            raise RuntimeError(f"FlashInfer UlyssesCommunicator lacks required method {method!r}")


def _configured_capacity(first_nbytes: int) -> int:
    raw = os.getenv(_MAX_BYTES_ENV)
    if raw is None or not raw.strip():
        return first_nbytes
    try:
        capacity = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_MAX_BYTES_ENV} must be a positive integer, got {raw!r}") from exc
    if capacity <= 0:
        raise ValueError(f"{_MAX_BYTES_ENV} must be positive, got {capacity}")
    return capacity


def _group_name(group: dist.ProcessGroup) -> str:
    name = getattr(group, "group_name", None)
    return str(name) if name is not None else f"process-group-{id(group)}"


@dataclass
class _PcieState:
    group: dist.ProcessGroup
    world_size: int
    device: torch.device
    capacity_bytes: int
    communicator: Any | None = None
    disabled_reason: str | None = None
    outputs: dict[tuple[str, str, torch.dtype], torch.Tensor] = field(default_factory=dict)

    def initialize(self, dtype: torch.dtype) -> bool:
        """Collectively construct once; a clean cold-start failure falls back."""
        if self.disabled_reason is not None:
            return False
        if self.communicator is not None:
            return True

        from flashinfer.comm import UlyssesCommunicator

        try:
            communicator = UlyssesCommunicator(
                self.group,
                max_bytes=self.capacity_bytes,
                dtype=dtype,
                backend="pcie",
                device=self.device,
            )
        except Exception as exc:  # FlashInfer coordinates init cleanup across ranks.
            if _require_rdma():
                raise RuntimeError("required FlashInfer all-RDMA Ulysses initialization failed") from exc
            self.disabled_reason = f"cold initialization failed ({type(exc).__name__}: {exc})"
            logger.warning("FlashInfer PCIe Ulysses unavailable; using NCCL: %s", self.disabled_reason)
            return False

        if _require_rdma() and communicator.transport != "rdma":
            observed = communicator.transport
            communicator.close()
            raise RuntimeError(
                f"required FlashInfer all-RDMA Ulysses route was not selected; observed transport={observed!r}"
            )

        self.communicator = communicator
        logger.info(
            "FlashInfer PCIe Ulysses armed: group=%s world=%d device=%s transport=%s capacity=%d",
            _group_name(self.group),
            self.world_size,
            self.device,
            communicator.transport,
            self.capacity_bytes,
        )
        return True

    def output(self, x: torch.Tensor, *, op: str, slot: str) -> torch.Tensor:
        assert self.communicator is not None
        key = (slot, op, x.dtype)
        out = self.outputs.get(key)
        if out is None:
            # allocate_output validates contiguity but never reads the values.
            prototype = x if x.is_contiguous() else torch.empty(x.shape, dtype=x.dtype, device=x.device)
            # PR #4876 permits a per-call element type on the PCIe backend.
            # Always state it explicitly: the communicator default is fixed by
            # the first collective, while callers may
            # use E4M3 Q/K/V and retain BF16 for the reverse-O exchange.
            out = self.communicator.allocate_output(
                prototype,
                op=op,
                dtype=x.dtype,
            )
            self.outputs[key] = out
            return out

        batch, seq, heads, head_dim = x.shape
        if op == "scatter_heads":
            shape = (batch, seq * self.world_size, heads // self.world_size, head_dim)
        elif op == "gather_heads":
            shape = (batch, seq // self.world_size, heads * self.world_size, head_dim)
        else:  # This adapter deliberately exposes only the two Ulysses transforms.
            raise ValueError(f"unsupported cached FlashInfer Ulysses op: {op!r}")

        # ``out`` may be a narrow view of a capacity-sized native allocation.
        # as_strided checks against the backing storage, so it can safely form
        # a different contiguous geometry without allocating or changing the
        # registered base pointer.
        strides = [1] * len(shape)
        for index in range(len(shape) - 2, -1, -1):
            strides[index] = strides[index + 1] * shape[index + 1]
        return out.as_strided(shape, tuple(strides))

    def scatter(self, x: torch.Tensor, *, slot: str) -> torch.Tensor:
        assert self.communicator is not None
        out = self.output(x, op="scatter_heads", slot=slot)
        if x.is_contiguous():
            source = x
        else:
            source = self.communicator.input_buffer(out, x.shape)
            source.copy_(x)
        return self.communicator.scatter_heads(source, out=out, dtype=source.dtype)

    def gather(self, x: torch.Tensor, *, slot: str) -> torch.Tensor:
        assert self.communicator is not None
        out = self.output(x, op="gather_heads", slot=slot)
        if x.is_contiguous():
            source = x
        else:
            source = self.communicator.input_buffer(out, x.shape)
            source.copy_(x)
        return self.communicator.gather_heads(source, out=out, dtype=source.dtype)

    def close(self) -> None:
        # Preserve the handle and registered storage if collective close fails.
        if self.communicator is not None:
            self.communicator.close()
        self.communicator = None
        self.outputs.clear()


_STATES: dict[tuple[str, torch.device], _PcieState] = {}


def _state_for(x: torch.Tensor, group: dist.ProcessGroup, world_size: int) -> _PcieState:
    actual_world_size = dist.get_world_size(group)
    if actual_world_size != world_size:
        raise RuntimeError(
            f"FlashInfer PCIe Ulysses world-size mismatch: caller={world_size}, process-group={actual_world_size}"
        )
    key = (_group_name(group), x.device)
    with _LOCK:
        state = _STATES.get(key)
        if state is None:
            state = _PcieState(
                group=group,
                world_size=world_size,
                device=x.device,
                capacity_bytes=_configured_capacity(x.nbytes),
            )
            _STATES[key] = state
        elif state.group is not group or state.world_size != world_size:
            raise RuntimeError("FlashInfer PCIe Ulysses process-group identity changed for a live cache entry")
        return state


def _nccl_scatter_heads(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> torch.Tensor:
    batch, local_seq, heads, head_dim = x.shape
    local_heads = heads // world_size
    send = x.reshape(batch, local_seq, world_size, local_heads, head_dim).transpose(0, 2).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.reshape(local_seq * world_size, batch, local_heads, head_dim)
        .transpose(0, 1)
        .contiguous()
        .reshape(batch, local_seq * world_size, local_heads, head_dim)
    )


def _nccl_gather_heads(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> torch.Tensor:
    batch, global_seq, local_heads, head_dim = x.shape
    local_seq = global_seq // world_size
    heads = local_heads * world_size
    send = (
        x.reshape(batch, world_size, local_seq, local_heads, head_dim)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(world_size, local_heads, local_seq, batch, head_dim)
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.reshape(heads, local_seq, batch, head_dim)
        .transpose(0, 2)
        .contiguous()
        .reshape(batch, local_seq, heads, head_dim)
    )


def _scatter_impl(
    x: torch.Tensor,
    group_name: str,
    world_size: int,
    slot: str,
    use_sync: bool,
) -> torch.Tensor:
    group = _resolve_process_group(group_name)
    if x.shape[0] != 1:
        if _require_rdma():
            raise ValueError("required FlashInfer RDMA Ulysses supports batch=1 only")
        out = _nccl_scatter_heads(x, group, world_size)
        if use_sync:
            torch.accelerator.synchronize(x.device)
        return out
    state = _state_for(x, group, world_size)
    if x.nbytes > state.capacity_bytes and _require_rdma():
        raise RuntimeError(
            "required FlashInfer all-RDMA Ulysses operand exceeds configured "
            f"capacity: operand={x.nbytes}, capacity={state.capacity_bytes}"
        )
    if x.nbytes <= state.capacity_bytes and state.initialize(x.dtype):
        output = state.scatter(x, slot=slot)
        if use_sync:
            torch.accelerator.synchronize(x.device)
        return output
    if x.nbytes > state.capacity_bytes:
        logger.warning_once(
            "FlashInfer PCIe Ulysses operand %d exceeds capacity %d; using NCCL for this geometry",
            x.nbytes,
            state.capacity_bytes,
        )
    out = _nccl_scatter_heads(x, group, world_size)
    if use_sync:
        torch.accelerator.synchronize(x.device)
    return out


def _gather_impl(
    x: torch.Tensor,
    group_name: str,
    world_size: int,
    use_sync: bool,
) -> torch.Tensor:
    group = _resolve_process_group(group_name)
    if x.shape[0] != 1:
        if _require_rdma():
            raise ValueError("required FlashInfer RDMA Ulysses supports batch=1 only")
        out = _nccl_gather_heads(x, group, world_size)
        if use_sync:
            torch.accelerator.synchronize(x.device)
        return out
    state = _state_for(x, group, world_size)
    if x.nbytes > state.capacity_bytes and _require_rdma():
        raise RuntimeError(
            "required FlashInfer all-RDMA Ulysses operand exceeds configured "
            f"capacity: operand={x.nbytes}, capacity={state.capacity_bytes}"
        )
    if x.nbytes <= state.capacity_bytes and state.initialize(x.dtype):
        output = state.gather(x, slot="o")
        if use_sync:
            torch.accelerator.synchronize(x.device)
        return output
    if x.nbytes > state.capacity_bytes:
        logger.warning_once(
            "FlashInfer PCIe Ulysses operand %d exceeds capacity %d; using NCCL for this geometry",
            x.nbytes,
            state.capacity_bytes,
        )
    out = _nccl_gather_heads(x, group, world_size)
    if use_sync:
        torch.accelerator.synchronize(x.device)
    return out


@torch.library.custom_op(
    "vllm_omni_flashinfer_ulysses::scatter_heads",
    mutates_args=(),
    device_types="cuda",
)
def flashinfer_ulysses_qkv_fwd(
    x: torch.Tensor,
    group_name: str,
    world_size: int,
    slot: str,
    use_sync: bool,
) -> torch.Tensor:
    """Opaque compiled-graph node for PCIe Ulysses Q/K/V exchange."""
    return _scatter_impl(x, group_name, world_size, slot, use_sync)


@flashinfer_ulysses_qkv_fwd.register_fake
def _(
    x: torch.Tensor,
    group_name: str,
    world_size: int,
    slot: str,
    use_sync: bool,
) -> torch.Tensor:
    batch, local_seq, heads, head_dim = x.shape
    return x.new_empty(batch, local_seq * world_size, heads // world_size, head_dim)


@torch.library.custom_op(
    "vllm_omni_flashinfer_ulysses::gather_heads",
    mutates_args=(),
    device_types="cuda",
)
def flashinfer_ulysses_o_rev(
    x: torch.Tensor,
    group_name: str,
    world_size: int,
    use_sync: bool,
) -> torch.Tensor:
    """Opaque compiled-graph node for inverse PCIe Ulysses output exchange."""
    return _gather_impl(x, group_name, world_size, use_sync)


@flashinfer_ulysses_o_rev.register_fake
def _(
    x: torch.Tensor,
    group_name: str,
    world_size: int,
    use_sync: bool,
) -> torch.Tensor:
    batch, global_seq, local_heads, head_dim = x.shape
    return x.new_empty(
        batch,
        global_seq // world_size,
        local_heads * world_size,
        head_dim,
    )


def clear_flashinfer_ulysses_communicators() -> None:
    """Collectively close all process-local communicators before PG teardown."""
    with _LOCK:
        states = list(_STATES.items())
    closed: list[tuple[tuple[str, torch.device], _PcieState]] = []
    errors: list[Exception] = []
    for key, state in states:
        try:
            state.close()
            closed.append((key, state))
        except Exception as exc:  # teardown errors must remain visible to shutdown logs
            errors.append(exc)
    with _LOCK:
        for key, state in closed:
            if _STATES.get(key) is state:
                del _STATES[key]
    if errors:
        raise RuntimeError(f"failed to close {len(errors)} FlashInfer Ulysses communicator(s): {errors}")
