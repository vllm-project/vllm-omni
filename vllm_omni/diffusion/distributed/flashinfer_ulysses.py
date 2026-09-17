# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in FlashInfer PCIe/RDMA transport for strict Ulysses attention.

This adapter intentionally sits behind the existing ``--ulysses-a2a-permute``
switch plus ``VLLM_OMNI_ULYSSES_A2A_BACKEND=flashinfer-pcie``.  The default
continues to be vLLM-Omni's symmetric-memory kernel.

Q/K producer-direct landing is a second, independent opt-in through
``VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT=1``. This keeps the existing
FlashInfer adapter path unchanged for isolated A/B measurements.

Attention kernels may independently write its fused tile-to-aligned output into the
registered reverse-Ulysses source buffer through
``VLLM_OMNI_FLASHINFER_ULYSSES_O_PRODUCER_DIRECT=1``. The communicator still
performs the identical gather-heads exchange; only its source staging copy is
removed.

FlashInfer's PCIe communicator owns registered output and RDMA landing buffers.
One communicator is shared by all attention layers in a process-group. Separate
registered slots keep Q/K/V and the inverse-attention result live at the same time. Non-contiguous projection views
are copied directly into the registered landing buffer, avoiding an intermediate
``contiguous()`` allocation and the transport's otherwise-required second device
copy.
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
_QK_PRODUCER_DIRECT_ENV = "VLLM_OMNI_FLASHINFER_ULYSSES_QK_PRODUCER_DIRECT"
_O_PRODUCER_DIRECT_ENV = "VLLM_OMNI_FLASHINFER_ULYSSES_O_PRODUCER_DIRECT"
_RELEASE_AFTER_DENOISE_ENV = "VLLM_OMNI_FLASHINFER_ULYSSES_RELEASE_AFTER_DENOISE"
_EXPECTED_BACKEND = "flashinfer-pcie"

_LOCK = threading.RLock()

_TensorIdentity = tuple[
    int,
    tuple[int, ...],
    tuple[int, ...],
    int,
    torch.dtype,
    torch.device,
]


def _tensor_identity(x: torch.Tensor) -> _TensorIdentity:
    """Describe the exact tensor view handed to the communicator."""
    return (
        x.data_ptr(),
        tuple(x.shape),
        tuple(x.stride()),
        x.storage_offset(),
        x.dtype,
        x.device,
    )


def configured_backend() -> str:
    """Return the explicitly selected optimized Ulysses implementation."""
    return os.getenv(_BACKEND_ENV, "symmetric").strip().lower()


def is_flashinfer_pcie_enabled() -> bool:
    return configured_backend() == _EXPECTED_BACKEND


def is_qk_producer_direct_enabled() -> bool:
    """Return whether a producer may write Q/K into registered source storage."""
    value = os.getenv(_QK_PRODUCER_DIRECT_ENV, "0").strip().lower()
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{_QK_PRODUCER_DIRECT_ENV} must be 0/1/false/true, got {value!r}")
    return is_flashinfer_pcie_enabled() and value in ("1", "true")


def is_o_producer_direct_enabled() -> bool:
    """Return whether Attention kernels may target the registered reverse-O source."""
    value = os.getenv(_O_PRODUCER_DIRECT_ENV, "0").strip().lower()
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{_O_PRODUCER_DIRECT_ENV} must be 0/1/false/true, got {value!r}")
    return is_flashinfer_pcie_enabled() and value in ("1", "true")


def release_after_denoise_enabled() -> bool:
    """Whether registered PCIe/RDMA windows are scoped to one denoise.

    Re-registering the windows adds cold work to the next request, so the
    memory-saving lifecycle remains explicitly disabled by default.
    """
    value = os.getenv(_RELEASE_AFTER_DENOISE_ENV, "0").strip().lower()
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{_RELEASE_AFTER_DENOISE_ENV} must be 0/1/false/true, got {value!r}")
    return is_flashinfer_pcie_enabled() and value in ("1", "true")


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
    # ``release_after_denoise`` returns the registered raw-CUDA allocations
    # before VAE decode, but the VAE then leaves its much larger scratch arena
    # in PyTorch's caching allocator.  FlashInfer's next registered windows use
    # native cudaMalloc and cannot reuse those cached segments.  Remember that
    # lifecycle transition so only the next cold re-arm returns inactive VAE
    # cache to CUDA; the first request and every hot layer call stay untouched.
    rearm_pending: bool = False
    producer_direct_logged: bool = False
    producer_direct_generation: int = 0
    producer_direct_expected: dict[str, _TensorIdentity] = field(
        default_factory=dict,
        repr=False,
    )
    producer_direct_exchanged: set[str] = field(default_factory=set, repr=False)
    producer_direct_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )
    o_producer_direct_logged: bool = False
    o_producer_direct_expected: _TensorIdentity | None = field(
        default=None,
        repr=False,
    )
    o_producer_direct_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )
    # One registered allocation per logical Q/K/V/O slot.  FlashInfer sizes
    # each allocation (and its RDMA landing buffer) at ``capacity_bytes`` even
    # though allocate_output returns a view for the first call's geometry.
    # Reuse that storage for smaller/later geometries instead of registering
    # another capacity-sized pair for every token-refiner/main-DiT shape.
    outputs: dict[tuple[str, str, torch.dtype], torch.Tensor] = field(default_factory=dict)
    input_landing_identities: dict[tuple[str, str, torch.dtype], _TensorIdentity] = field(
        default_factory=dict, repr=False
    )

    def initialize(self, dtype: torch.dtype) -> bool:
        """Collectively construct once; a clean cold-start failure falls back."""
        if self.disabled_reason is not None:
            return False
        if self.communicator is not None:
            return True

        if self.rearm_pending:
            # Every rank reaches initialize through the same strict-Ulysses
            # operation.  Drain its local work and release only *inactive*
            # allocator segments before FlashInfer asks CUDA for raw registered
            # storage.  Live model, compiled-region and current-layer tensors
            # are unaffected by empty_cache().  Vote before entering the
            # communicator constructor so a rank-local allocator failure cannot
            # strand peers in its collective setup protocol.
            reclaim_error: BaseException | None = None
            try:
                torch.accelerator.synchronize(self.device)
                torch.accelerator.empty_cache()
            except BaseException as exc:
                reclaim_error = exc
            reclaim_failures = self._collective_sum(int(reclaim_error is not None))
            if reclaim_failures:
                raise RuntimeError(
                    "FlashInfer PCIe Ulysses re-arm could not reclaim the "
                    f"post-VAE allocator cache on {reclaim_failures}/{self.world_size} ranks"
                ) from reclaim_error
            logger.info(
                "FlashInfer PCIe Ulysses re-arm reclaimed post-VAE allocator cache: rank=%d group=%s device=%s",
                dist.get_rank(self.group),
                _group_name(self.group),
                self.device,
            )

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
        self.rearm_pending = False
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

    def registered_shape_prototype(
        self,
        shape: tuple[int, int, int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a shape-only view over an existing capacity-sized slot.

        FlashInfer's PCIe ``allocate_output`` validates the input geometry and
        element type but never reads its values. After Q/K/V have armed the
        communicator, one of their registered native outputs therefore
        provides a safe untyped backing store for a reverse-O *prototype*
        view. The view may intentionally use a different dtype (FP8 QKV ->
        BF16 O); this avoids allocating a transient full-size CUDA tensor
        solely to register the Attention kernels O or O-bundle slot.
        """
        if len(shape) != 4 or any(int(dim) <= 0 for dim in shape):
            raise ValueError(f"registered shape prototype requires positive 4-D shape, got {shape}")
        required_bytes = int(torch.Size(shape).numel()) * dtype.itemsize
        if required_bytes > self.capacity_bytes:
            raise RuntimeError(
                "registered shape prototype exceeds communicator capacity: "
                f"operand={required_bytes}, capacity={self.capacity_bytes}"
            )
        strides = [1] * len(shape)
        for index in range(len(shape) - 2, -1, -1):
            strides[index] = strides[index + 1] * shape[index + 1]
        # Prefer an exact typed view for the ordinary BF16-only path.  If the
        # forward collective used FP8, reinterpret only the metadata over the
        # same untyped registered storage.  ``allocate_output`` never reads
        # this prototype, and the newly allocated O slot remains separately
        # typed and registered by FlashInfer.
        candidates = sorted(
            self.outputs.values(),
            key=lambda candidate: candidate.dtype != dtype,
        )
        for candidate in candidates:
            if candidate.device != device:
                continue
            storage = candidate.untyped_storage()
            if candidate.storage_offset() != 0 or candidate.data_ptr() != storage.data_ptr():
                continue
            if storage.nbytes() < required_bytes:
                continue
            if candidate.dtype == dtype:
                return candidate.as_strided(shape, tuple(strides))
            prototype = torch.empty(0, dtype=dtype, device=device)
            return prototype.set_(storage, 0, shape, tuple(strides))
        raise RuntimeError(
            "no existing registered Ulysses output has capacity for the "
            f"shape-only prototype {shape}; Q/K/V must be armed first"
        )

    def scatter(self, x: torch.Tensor, *, slot: str) -> torch.Tensor:
        assert self.communicator is not None
        out = self.output(x, op="scatter_heads", slot=slot)
        if x.is_contiguous():
            source = x
        else:
            # Directly materialize a strided Q/K/V projection view in the
            # registered source buffer. Passing this exact pointer suppresses
            # FlashInfer's ordinary source -> landing staging copy.
            source = self.communicator.input_buffer(out, x.shape)
            source.copy_(x)
        result = self.communicator.scatter_heads(
            source,
            out=out,
            dtype=source.dtype,
        )
        if slot == "g":
            logger.info_once(
                "FlashInfer PCIe Ulysses VSA gate exchange active: rank=%d group=%s shape=%s dtype=%s",
                dist.get_rank(self.group),
                _group_name(self.group),
                tuple(source.shape),
                source.dtype,
                # The default vLLM "local" scope logs only local rank zero.
                # Qualification needs evidence that every SP rank completed
                # this separate registered-RDMA collective.
                scope="process",
            )
        # This branch becomes a single false boolean check after the first
        # confirmed Q/K pair. Keep all bookkeeping out of the steady-state
        # transport path while requiring the collective call to return first.
        landing_key = (slot, "scatter_heads", source.dtype)
        landed_directly = self.input_landing_identities.get(landing_key) == _tensor_identity(source)
        if slot in ("q", "k", "v") and landed_directly and not self.producer_direct_logged:
            self.mark_producer_direct_exchange(source, slot=slot)
        return result

    def input_landing(
        self,
        x: torch.Tensor,
        *,
        slot: str,
        op: str = "scatter_heads",
    ) -> torch.Tensor:
        """Return the registered source storage paired with ``slot`` output.

        An eager producer may write this view directly. A later
        ``scatter(..., slot=slot)`` resolves the same output allocation, and
        FlashInfer recognizes the exact landing pointer and skips staging.
        """
        assert self.communicator is not None
        out = self.output(x, op=op, slot=slot)
        landing = self.communicator.input_buffer(out, x.shape)
        if landing.shape != x.shape:
            raise RuntimeError(
                "FlashInfer PCIe input landing shape mismatch: "
                f"expected={tuple(x.shape)}, observed={tuple(landing.shape)}"
            )
        if landing.dtype != x.dtype or landing.device != x.device:
            raise RuntimeError(
                "FlashInfer PCIe input landing dtype/device mismatch: "
                f"expected=({x.dtype}, {x.device}), "
                f"observed=({landing.dtype}, {landing.device})"
            )
        if not landing.is_contiguous():
            raise RuntimeError("FlashInfer PCIe input landing must be contiguous")
        if landing.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
            raise RuntimeError("FlashInfer PCIe input and output buffers unexpectedly alias")
        self.input_landing_identities[(slot, op, x.dtype)] = _tensor_identity(landing)
        return landing

    def arm_o_producer_direct(self, landing: torch.Tensor) -> None:
        """Bind the next reverse-O exchange to one exact registered view."""
        landing_key = ("o", "gather_heads", landing.dtype)
        identity = _tensor_identity(landing)
        if self.input_landing_identities.get(landing_key) != identity:
            raise RuntimeError("FlashInfer reverse-O producer landing is not registered for the gather-heads slot")
        with self.o_producer_direct_lock:
            self.o_producer_direct_expected = identity

    def arm_qk_producer_direct(
        self,
        q_landing: torch.Tensor,
        k_landing: torch.Tensor,
    ) -> None:
        """Atomically bind one Q/K producer generation to exact landing views."""
        if self.producer_direct_logged:
            return
        with self.producer_direct_lock:
            if self.producer_direct_logged:
                return
            self.producer_direct_generation += 1
            self.producer_direct_expected = {
                "q": _tensor_identity(q_landing),
                "k": _tensor_identity(k_landing),
            }
            self.producer_direct_exchanged.clear()

    def mark_producer_direct_exchange(
        self,
        source: torch.Tensor,
        *,
        slot: str,
    ) -> None:
        """Record a direct scatter only after its communicator call succeeds."""
        if self.producer_direct_logged or slot not in self.producer_direct_expected:
            return
        with self.producer_direct_lock:
            if self.producer_direct_logged:
                return
            if self.producer_direct_expected.get(slot) != _tensor_identity(source):
                return
            self.producer_direct_exchanged.add(slot)
            expected_slots = set(self.producer_direct_expected)
            if self.producer_direct_exchanged != expected_slots:
                return

            logger.info(
                "FlashInfer PCIe Ulysses %s producer-direct exchange active: rank=%d group=%s shape=%s dtype=%s",
                "/".join(name.upper() for name in ("q", "k", "v") if name in expected_slots),
                dist.get_rank(self.group),
                _group_name(self.group),
                tuple(source.shape),
                source.dtype,
            )
            self.producer_direct_logged = True

    def gather(self, x: torch.Tensor, *, slot: str) -> torch.Tensor:
        assert self.communicator is not None
        out = self.output(x, op="gather_heads", slot=slot)
        input_identity = _tensor_identity(x)
        with self.o_producer_direct_lock:
            expected = self.o_producer_direct_expected if slot == "o" else None
        producer_direct = expected is not None and expected == input_identity
        if x.is_contiguous():
            source = x
        else:
            source = self.communicator.input_buffer(out, x.shape)
            source.copy_(x)
        result = self.communicator.gather_heads(
            source,
            out=out,
            dtype=source.dtype,
        )
        if expected is not None:
            with self.o_producer_direct_lock:
                # Clear only the generation consumed by this exchange. A
                # failed communicator call leaves it armed for diagnostics or
                # an explicit retry.
                if self.o_producer_direct_expected == expected:
                    self.o_producer_direct_expected = None
                if producer_direct and not self.o_producer_direct_logged:
                    logger.info(
                        "FlashInfer PCIe Ulysses O producer-direct exchange active: rank=%d group=%s shape=%s dtype=%s",
                        dist.get_rank(self.group),
                        _group_name(self.group),
                        tuple(x.shape),
                        x.dtype,
                    )
                    self.o_producer_direct_logged = True
            if not producer_direct:
                logger.warning_once(
                    "FlashInfer PCIe Ulysses O producer-direct exchange fell back: "
                    "attention output did not match the armed registered landing"
                )
        return result

    def close(self) -> None:
        communicator = self.communicator
        if communicator is not None:
            # FlashInfer keeps a failed collective close in CLOSING and
            # requires every rank to retry that same communicator.  Retain the
            # handle and every registered tensor until teardown succeeds.
            communicator.close()
        self.communicator = None
        self.outputs.clear()
        self.input_landing_identities.clear()
        with self.producer_direct_lock:
            self.producer_direct_logged = False
            self.producer_direct_generation = 0
            self.producer_direct_expected.clear()
            self.producer_direct_exchanged.clear()
        with self.o_producer_direct_lock:
            self.o_producer_direct_logged = False
            self.o_producer_direct_expected = None

    def _collective_sum(self, value: int) -> int:
        """Return a group-wide lifecycle vote on the existing CUDA group."""
        vote = torch.tensor([value], dtype=torch.int32, device=self.device)
        dist.all_reduce(vote, op=dist.ReduceOp.SUM, group=self.group)
        return int(vote.item())

    def release_after_denoise(self) -> bool:
        """Collectively release registered windows between execution stages.

        The state object remains cached, so the next strict-Ulysses operation
        lazily re-enters :meth:`initialize`. Every rank votes before draining
        CUDA work and before reporting close success, turning an accidental
        asymmetric lifecycle into an explicit failure instead of silent reuse.
        """
        if not release_after_denoise_enabled():
            return False

        armed_count = self._collective_sum(int(self.communicator is not None))
        if armed_count == 0:
            return False
        if armed_count != self.world_size:
            raise RuntimeError(
                f"FlashInfer PCIe Ulysses lifecycle is rank-asymmetric: {armed_count}/{self.world_size} ranks are armed"
            )

        sync_error: BaseException | None = None
        try:
            torch.accelerator.synchronize(self.device)
        except BaseException as exc:  # every rank must vote before failing
            sync_error = exc
        sync_failures = self._collective_sum(int(sync_error is not None))
        if sync_failures:
            raise RuntimeError(
                f"FlashInfer PCIe Ulysses release could not drain CUDA work on {sync_failures}/{self.world_size} ranks"
            ) from sync_error

        close_error: BaseException | None = None
        try:
            self.close()
            # Registered allocations are now ordinary cached PyTorch storage.
            # Return it to the allocator before the much larger video VAE tail.
            torch.accelerator.empty_cache()
        except BaseException as exc:  # report one collective verdict
            close_error = exc
        close_failures = self._collective_sum(int(close_error is not None))
        if close_failures:
            raise RuntimeError(
                f"FlashInfer PCIe Ulysses release failed on {close_failures}/{self.world_size} ranks"
            ) from close_error

        # VAE decode follows immediately and repopulates PyTorch's cache.  The
        # next request must return those inactive segments before native RDMA
        # output registration; doing it lazily avoids adding another sync/cache
        # flush to this request's decode/output critical path.
        self.rearm_pending = True

        logger.info(
            "FlashInfer PCIe Ulysses released after denoise; next request will re-arm "
            "(capacity %.2f GiB per registered slot)",
            self.capacity_bytes / 2**30,
        )
        return True


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
    state = _state_for(x, group, world_size)
    if x.nbytes > state.capacity_bytes and _require_rdma():
        raise RuntimeError(
            "required FlashInfer all-RDMA Ulysses operand exceeds configured "
            f"capacity: operand={x.nbytes}, capacity={state.capacity_bytes}"
        )
    if x.nbytes <= state.capacity_bytes and state.initialize(x.dtype):
        return state.scatter(x, slot=slot)
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


def flashinfer_ulysses_qk_input_landings(
    q: torch.Tensor,
    k: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Acquire Q/K registered source buffers from an eager graph island.

    ``None`` means the optional PCIe backend declined its cold start and the
    caller should produce ordinary Q/K tensors for the NCCL fallback. With
    ``VLLM_OMNI_FLASHINFER_ULYSSES_REQUIRE_RDMA=1`` every inability to use the
    all-RDMA route remains a hard error.
    """
    if torch.compiler.is_compiling():
        raise RuntimeError("FlashInfer input landing buffers must be acquired outside torch.compile")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"q and k must be [batch, sequence, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape != k.shape:
        raise ValueError(f"producer-direct Q/K requires equal shapes, got {q.shape} and {k.shape}")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("producer-direct Q/K must have the same dtype and device")
    if q.device.type != "cuda":
        raise ValueError(f"FlashInfer PCIe input landing requires CUDA tensors, got {q.device}")
    with torch.get_device_module().device(q.device):
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        raise RuntimeError("FlashInfer PCIe/RDMA input landing does not support CUDA graph capture")

    state = _state_for(q, group, world_size)
    if q.nbytes > state.capacity_bytes:
        if _require_rdma():
            raise RuntimeError(
                "required FlashInfer all-RDMA Ulysses Q/K exceeds configured "
                f"capacity: operand={q.nbytes}, capacity={state.capacity_bytes}"
            )
        return None
    if not state.initialize(q.dtype):
        return None
    communicator = state.communicator
    assert communicator is not None
    if communicator.transport not in ("hybrid", "rdma"):
        # All-P2P reads its caller tensor directly and therefore has no
        # registered source landing to target. Producing ordinary contiguous
        # Q/K is already the zero-staging path for that transport.
        return None
    q_op = "scatter_heads"
    q_slot = "q"
    landings = (
        state.input_landing(q, slot=q_slot, op=q_op),
        state.input_landing(k, slot="k"),
    )
    state.arm_qk_producer_direct(*landings)
    return landings


def _typed_contiguous_shape_prototype(
    source: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build allocation metadata over source storage without a full temporary."""
    storage = source.untyped_storage()
    byte_offset = source.storage_offset() * source.element_size()
    if byte_offset < 0 or byte_offset % dtype.itemsize:
        raise RuntimeError("source storage cannot back the requested typed prototype")
    required_bytes = source.numel() * dtype.itemsize
    if byte_offset + required_bytes > storage.nbytes():
        raise RuntimeError(
            "source storage is too small for the requested typed prototype: "
            f"offset={byte_offset}, required={required_bytes}, "
            f"storage={storage.nbytes()}"
        )
    shape = tuple(int(dim) for dim in source.shape)
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    prototype = torch.empty(0, dtype=dtype, device=source.device)
    return prototype.set_(
        storage,
        byte_offset // dtype.itemsize,
        shape,
        tuple(strides),
    )


def flashinfer_ulysses_o_input_landing(
    prototype: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
    *,
    input_shape: tuple[int, int, int, int] | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Acquire the registered source for the next reverse-Ulysses exchange.

    The caller supplies the post-Ulysses attention-output geometry
    ``[B, S_global, H_local, D]``. ``input_shape`` may expand only its sequence
    dimension for an owner-piggyback payload. ``output_dtype`` may differ from
    the query prototype for low-precision forward transport; an already
    registered capacity-sized Q/K/V output then backs a typed, shape-only
    prototype without a full CUDA allocation. A returned tensor is the exact storage that must be passed to
    :func:`flashinfer_ulysses_o_rev`; otherwise FlashInfer stages the source
    before the gather-heads collective.
    """
    if not is_o_producer_direct_enabled():
        return None
    if torch.compiler.is_compiling():
        raise RuntimeError("FlashInfer reverse-O input landing must be acquired outside torch.compile")
    if prototype.ndim != 4:
        raise ValueError(
            "reverse-O producer-direct prototype must be "
            f"[batch, sequence, heads, head_dim], got {tuple(prototype.shape)}"
        )
    if prototype.device.type != "cuda":
        raise ValueError(f"FlashInfer reverse-O input landing requires a CUDA prototype, got {prototype.device}")
    requested_dtype = prototype.dtype if output_dtype is None else output_dtype
    if requested_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"FlashInfer reverse-O producer-direct supports fp16/bf16, got {requested_dtype}")
    if prototype.requires_grad:
        raise ValueError("FlashInfer reverse-O producer-direct is inference-only")
    if not prototype.is_contiguous():
        raise ValueError("FlashInfer reverse-O prototype must be contiguous")
    if world_size <= 1:
        raise ValueError(f"FlashInfer reverse-O producer-direct requires world_size > 1, got {world_size}")
    if any(int(dim) <= 0 for dim in prototype.shape):
        raise ValueError(f"FlashInfer reverse-O prototype dimensions must be positive, got {tuple(prototype.shape)}")
    if prototype.shape[1] % world_size:
        raise ValueError(
            "FlashInfer reverse-O global sequence must divide the Ulysses world: "
            f"sequence={prototype.shape[1]}, world={world_size}"
        )
    landing_shape = tuple(int(dim) for dim in (input_shape or tuple(prototype.shape)))
    if len(landing_shape) != 4 or any(dim <= 0 for dim in landing_shape):
        raise ValueError(
            f"reverse-O producer-direct input_shape must contain four positive dimensions, got {landing_shape}"
        )
    if (
        landing_shape[0] != prototype.shape[0]
        or landing_shape[2] != prototype.shape[2]
        or landing_shape[3] != prototype.shape[3]
    ):
        raise ValueError(
            "reverse-O expanded landing may change only the sequence dimension: "
            f"prototype={tuple(prototype.shape)}, requested={landing_shape}"
        )
    if landing_shape[1] % world_size:
        raise ValueError(
            "FlashInfer reverse-O landing sequence must divide the Ulysses world: "
            f"sequence={landing_shape[1]}, world={world_size}"
        )
    with torch.get_device_module().device(prototype.device):
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        raise RuntimeError("FlashInfer PCIe/RDMA reverse-O landing does not support CUDA graph capture")

    state = _state_for(prototype, group, world_size)
    landing_nbytes = int(torch.Size(landing_shape).numel()) * requested_dtype.itemsize
    if landing_nbytes > state.capacity_bytes:
        if _require_rdma():
            raise RuntimeError(
                "required FlashInfer all-RDMA Ulysses reverse-O exceeds "
                f"configured capacity: operand={landing_nbytes}, "
                f"capacity={state.capacity_bytes}"
            )
        logger.warning_once(
            "FlashInfer PCIe Ulysses O producer-direct exchange fell back: operand %d exceeds capacity %d",
            landing_nbytes,
            state.capacity_bytes,
        )
        return None
    if not state.initialize(requested_dtype):
        logger.warning_once(
            "FlashInfer PCIe Ulysses O producer-direct exchange fell back: communicator initialization declined"
        )
        return None
    communicator = state.communicator
    assert communicator is not None
    if communicator.transport not in ("hybrid", "rdma"):
        # All-P2P reads the producer tensor directly and therefore has no
        # staging copy for this optimization to remove.
        logger.warning_once(
            "FlashInfer PCIe Ulysses O producer-direct exchange fell back: "
            "transport %s has no registered source landing",
            communicator.transport,
        )
        return None

    landing_prototype = prototype
    if landing_shape != tuple(prototype.shape) or requested_dtype != prototype.dtype:
        landing_prototype = state.registered_shape_prototype(
            landing_shape,
            dtype=requested_dtype,
            device=prototype.device,
        )
    landing = state.input_landing(
        landing_prototype,
        slot="o",
        op="gather_heads",
    )
    state.arm_o_producer_direct(landing)
    return landing


def _gather_impl(
    x: torch.Tensor,
    group_name: str,
    world_size: int,
    use_sync: bool,
) -> torch.Tensor:
    group = _resolve_process_group(group_name)
    state = _state_for(x, group, world_size)
    if x.nbytes > state.capacity_bytes and _require_rdma():
        raise RuntimeError(
            "required FlashInfer all-RDMA Ulysses operand exceeds configured "
            f"capacity: operand={x.nbytes}, capacity={state.capacity_bytes}"
        )
    if x.nbytes <= state.capacity_bytes and state.initialize(x.dtype):
        return state.gather(x, slot="o")
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


def release_flashinfer_ulysses_after_denoise() -> bool:
    """Release every process-local FlashInfer Ulysses state, if opted in.

    Call this on every rank at the request/step boundary. Keeping
    the states in ``_STATES`` preserves their process-group and capacity
    contracts while allowing the communicator itself to be initialized lazily
    by the next request.
    """
    if not release_after_denoise_enabled():
        return False
    with _LOCK:
        # Match collective order across ranks even if unrelated Python object
        # construction happened in a different order in each worker.
        states = [
            state
            for _key, state in sorted(
                _STATES.items(),
                key=lambda item: (item[0][0], str(item[0][1])),
            )
        ]
    released = False
    for state in states:
        released = state.release_after_denoise() or released
    return released


__all__ = [
    "clear_flashinfer_ulysses_communicators",
    "configured_backend",
    "ensure_flashinfer_pcie_available",
    "flashinfer_ulysses_o_input_landing",
    "flashinfer_ulysses_o_rev",
    "flashinfer_ulysses_qk_input_landings",
    "flashinfer_ulysses_qkv_fwd",
    "is_flashinfer_pcie_enabled",
    "is_o_producer_direct_enabled",
    "is_qk_producer_direct_enabled",
    "release_after_denoise_enabled",
    "release_flashinfer_ulysses_after_denoise",
]
