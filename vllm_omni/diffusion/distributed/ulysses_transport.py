# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transport selection for the strict Ulysses all-to-all (``ulysses_a2a_backend``).

Follows vLLM's all-reduce selection (``CudaCommunicator`` with custom / symmetric-memory all-reduce): at init each
transport checks its eligibility on the Ulysses group (CUDA, one node, NVLink between every GPU pair, peer access,
compute capability and world size present in the measured tables, its library importable) and is disabled with a
logged reason otherwise. Each call then dispatches by message size through a fixed chain, using the first transport
whose gate passes: copy-engine overlap -> copy engine -> NCCL. Explicit backends skip the checks.
"""

from __future__ import annotations

import socket
import uuid
import zlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed import ulysses_transport_tables as tables
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

NCCL = "nccl"
COPY_ENGINE = "copy_engine"
COPY_ENGINE_OVERLAP = "copy_engine_overlap"
DISPATCH_ORDER = (COPY_ENGINE_OVERLAP, COPY_ENGINE, NCCL)

# Correctness gate for the head-chunked overlap, like vLLM's batch-invariant kernel list: attention kernels whose
# per-head output does not depend on how many heads one call covers, as (backend name, implementation).
PER_HEAD_INVARIANT_ATTENTION = frozenset({("FLASHINFER_ATTN", "cute-dsl")})


@dataclass(frozen=True)
class UlyssesTopology:
    """What the Ulysses group supports, identical on every rank."""

    world_size: int
    cuda: bool
    single_node: bool = False
    nvlink: bool = False
    """Every GPU pair in the group is connected by NVLink (NVML P2P NVLink status)."""
    p2p: bool = False
    """Every peer this process can see accepts CUDA peer access."""
    capability: str = ""
    """Lowest compute capability in the group, e.g. ``"10.0"``."""
    symmetric_memory: bool = False


@dataclass(frozen=True)
class UlyssesTransportPlan:
    """Per-layer dispatch for the strict Ulysses exchange.

    A ``None`` gate means the transport is disabled; otherwise it is the minimum message size for using it.
    """

    copy_engine_min_bytes: int | None = None
    overlap_chunks: int = 0
    """Head chunks per rank for the copy-engine overlap (0: disabled)."""
    overlap_min_chunk_bytes: int = 0

    @property
    def symmetric_memory(self) -> bool:
        return self.copy_engine_min_bytes is not None or self.overlap_chunks > 0

    def use_overlap(self, chunk_copy_bytes: int) -> bool:
        return self.overlap_chunks > 0 and chunk_copy_bytes >= self.overlap_min_chunk_bytes

    def exchange(self, q_bytes: int) -> str:
        """Transport for a Q/K/V exchange whose Q holds ``q_bytes`` on this rank (the output exchange follows it)."""
        if self.copy_engine_min_bytes is not None and q_bytes >= self.copy_engine_min_bytes:
            return COPY_ENGINE
        return NCCL


NCCL_PLAN = UlyssesTransportPlan()


def overlap_chunk_count(heads_per_rank: int | None, requested: int = 0) -> int:
    """``requested`` if set, else two heads per chunk (the measured best); 0 when that does not divide the heads."""
    if heads_per_rank is None:
        return requested  # head count unknown until the call; the layer checks divisibility
    if heads_per_rank <= 1:
        return 0
    chunks = requested or heads_per_rank // 2
    return chunks if chunks > 1 and heads_per_rank % chunks == 0 else 0


def _symmetric_memory_available() -> bool:
    try:
        import torch.distributed._symmetric_memory  # noqa: F401
    except ImportError as error:
        logger.debug("torch symmetric memory unavailable: %s", error)
        return False
    return True


def _probe_device() -> torch.device:
    return torch.device("cuda", torch.accelerator.current_device_index())


def _nvlink_all_pairs(uuids: list[str]) -> bool:
    """NVML NVLink P2P status for every pair, as in vLLM's ``NvmlCudaPlatform.is_fully_connected`` (by UUID, which
    stays valid whatever devices each process sees)."""
    from vllm.utils.import_utils import import_pynvml

    # Containers can expose CUDA compute without NVML; report no NVLink so every rank still reaches the agreement
    # all-reduce and falls back together. Shut NVML down only after a successful init.
    try:
        nvml = import_pynvml()
    except (ImportError, OSError) as error:
        logger.debug("pynvml unavailable, treating NVLink as absent: %s", error)
        return False
    try:
        nvml.nvmlInit()
    except nvml.NVMLError as error:
        logger.debug("NVML init failed, treating NVLink as absent: %s", error)
        return False
    try:
        handles = [nvml.nvmlDeviceGetHandleByUUID(gpu) for gpu in uuids]
        for i, a in enumerate(handles):
            for b in handles[i + 1 :]:
                if nvml.nvmlDeviceGetP2PStatus(a, b, nvml.NVML_P2P_CAPS_INDEX_NVLINK) != nvml.NVML_P2P_STATUS_OK:
                    return False
        return True
    except nvml.NVMLError:
        return False
    finally:
        nvml.nvmlShutdown()


def _peer_access(device: torch.device, uuids: list[str]) -> bool:
    """CUDA peer access from this device to every peer visible in this process (others are covered by NVML)."""
    visible = {}
    for index in range(torch.accelerator.device_count()):
        visible[f"GPU-{uuid.UUID(str(torch.cuda.get_device_properties(index).uuid))}"] = index
    local = device.index
    return all(
        visible[gpu] == local or torch.cuda.can_device_access_peer(local, visible[gpu])
        for gpu in uuids
        if gpu in visible
    )


_TOPOLOGIES: dict[str, UlyssesTopology] = {}


def probe_ulysses_topology(group: dist.ProcessGroup, world_size: int) -> UlyssesTopology:
    """Collective on ``group``: every rank must call it, and all get the same answer (cached per group)."""
    key = group.group_name
    cached = _TOPOLOGIES.get(key)
    if cached is not None:
        return cached
    if not (current_omni_platform.is_cuda() and torch.cuda.is_available()):
        # Other platforms never leave the default transport, so no collective is needed.
        topology = UlyssesTopology(world_size, cuda=False)
        _TOPOLOGIES[key] = topology
        return topology

    device = _probe_device()
    props = torch.cuda.get_device_properties(device)
    host = zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF
    # Exchange host names and GPU UUIDs (16 bytes) so every rank can check every pair. The Ulysses group has no CPU
    # group, so this runs on the device group with explicit device tensors.
    local = torch.tensor([host, *uuid.UUID(str(props.uuid)).bytes], dtype=torch.int64, device=device)
    gathered = torch.empty(world_size * local.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    rows = gathered.view(world_size, -1).tolist()
    single_node = len({row[0] for row in rows}) == 1
    uuids = [f"GPU-{uuid.UUID(bytes=bytes(row[1:]))}" for row in rows]
    nvlink = single_node and _nvlink_all_pairs(uuids)
    p2p = single_node and _peer_access(device, uuids)
    features = torch.tensor(
        [
            int(single_node),
            int(nvlink),
            int(p2p),
            props.major * 10 + props.minor,
            int(_symmetric_memory_available()),
        ],
        dtype=torch.int64,
        device=device,
    )
    # Ranks act on the weakest answer so they never pick different transports.
    dist.all_reduce(features, op=dist.ReduceOp.MIN, group=group)
    single_node, nvlink, p2p, sm, symmetric_memory = features.tolist()
    topology = UlyssesTopology(
        world_size,
        cuda=True,
        single_node=bool(single_node),
        nvlink=bool(nvlink),
        p2p=bool(p2p),
        capability=f"{sm // 10}.{sm % 10}",
        symmetric_memory=bool(symmetric_memory),
    )
    _TOPOLOGIES[key] = topology
    return topology


def _shared_disabled_reason(topology: UlyssesTopology, pure_strict: bool) -> str | None:
    if not topology.cuda:
        return "not CUDA"
    if not pure_strict:
        return "not strict pure Ulysses"
    if not topology.single_node:
        return "group spans nodes"
    if not topology.nvlink:
        return "GPUs not fully connected by NVLink"
    if not topology.p2p:
        return "CUDA peer access unavailable"
    return None


def _table_entry(table: dict[str, dict[int, int]], topology: UlyssesTopology) -> int | None:
    return table.get(topology.capability, {}).get(topology.world_size)


def eligible_transports(topology: UlyssesTopology, *, pure_strict: bool) -> dict[str, str | None]:
    """Disabled reason per symmetric-memory transport (``None``: eligible), from the topology and measured tables."""
    shared = _shared_disabled_reason(topology, pure_strict)
    reasons: dict[str, str | None] = {}
    measured = f"not measured for sm{topology.capability} x {topology.world_size} GPUs"
    for name, table in (
        (COPY_ENGINE_OVERLAP, tables.COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES),
        (COPY_ENGINE, tables.COPY_ENGINE_MIN_BYTES),
    ):
        if shared is not None:
            reasons[name] = shared
        elif not topology.symmetric_memory:
            reasons[name] = "no symmetric memory"
        elif _table_entry(table, topology) is None:
            reasons[name] = measured
        else:
            reasons[name] = None
    return reasons


def auto_plan(
    topology: UlyssesTopology,
    *,
    pure_strict: bool,
    heads_per_rank: int | None,
    per_head_invariant: bool,
    overlap_chunks: int = 0,
) -> UlyssesTransportPlan:
    """The ``"auto"`` plan for one attention layer: every eligible transport with its measured gate."""
    reasons = eligible_transports(topology, pure_strict=pure_strict)
    chunks = 0
    if reasons[COPY_ENGINE_OVERLAP] is None and reasons[COPY_ENGINE] is None and per_head_invariant:
        chunks = overlap_chunk_count(heads_per_rank, overlap_chunks)
    return UlyssesTransportPlan(
        copy_engine_min_bytes=(
            _table_entry(tables.COPY_ENGINE_MIN_BYTES, topology) if reasons[COPY_ENGINE] is None else None
        ),
        overlap_chunks=chunks,
        overlap_min_chunk_bytes=(_table_entry(tables.COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES, topology) or 0)
        if chunks
        else 0,
    )


def explicit_plan(backend: str, *, heads_per_rank: int | None, overlap_chunks: int = 0) -> UlyssesTransportPlan:
    """An explicitly configured backend, used for every eligible call without size gates."""
    if backend == NCCL:
        return NCCL_PLAN
    if backend == COPY_ENGINE_OVERLAP:
        return UlyssesTransportPlan(
            copy_engine_min_bytes=0, overlap_chunks=overlap_chunk_count(heads_per_rank, overlap_chunks)
        )
    if backend == COPY_ENGINE:
        return UlyssesTransportPlan(copy_engine_min_bytes=0)
    raise ValueError(f"unknown ulysses_a2a_backend {backend!r}")


_LOGGED: set[tuple[str, str]] = set()


def _log_once(level: str, message: str, *args: object) -> None:
    """Log ``message % args`` once per distinct rendered line (e.g. once per group)."""
    key = (level, message % args)
    if key not in _LOGGED:
        _LOGGED.add(key)
        getattr(logger, level)(message, *args)


def resolve_ulysses_transports(
    *,
    requested: str,
    group: dist.ProcessGroup,
    world_size: int,
    pure_strict: bool,
    num_heads: int | None,
    attention_backend: str | None,
    attention_impl: str | None,
    overlap_chunks: int = 0,
) -> UlyssesTransportPlan:
    """The plan one attention layer uses; ``requested`` is the configured ``ulysses_a2a_backend``."""
    heads_per_rank = num_heads // world_size if num_heads and num_heads % world_size == 0 else None
    group_name = getattr(group, "group_name", "ulysses")
    if requested != "auto":
        _log_once("info", "Using Ulysses a2a transport %r for group %r (explicit)", requested, group_name)
        return explicit_plan(requested, heads_per_rank=heads_per_rank, overlap_chunks=overlap_chunks)

    topology = probe_ulysses_topology(group, world_size)
    reasons = eligible_transports(topology, pure_strict=pure_strict)
    invariant = (attention_backend, attention_impl) in PER_HEAD_INVARIANT_ATTENTION
    if reasons[COPY_ENGINE_OVERLAP] is None and not invariant:
        attention = f"{attention_backend}({attention_impl})" if attention_impl else attention_backend
        _log_once(
            "info",
            "Ulysses a2a transport %r skipped for %s attention on group %r: not per-head invariant",
            COPY_ENGINE_OVERLAP,
            attention,
            group_name,
        )
    for name, reason in reasons.items():
        if reason is not None and topology.cuda:
            _log_once("warning", "Ulysses a2a transport %r disabled on group %r: %s", name, group_name, reason)
    plan = auto_plan(
        topology,
        pure_strict=pure_strict,
        heads_per_rank=heads_per_rank,
        per_head_invariant=invariant,
        overlap_chunks=overlap_chunks,
    )
    enabled = [
        name
        for name, on in (
            (COPY_ENGINE_OVERLAP, plan.overlap_chunks > 0),
            (COPY_ENGINE, plan.copy_engine_min_bytes is not None),
            (NCCL, True),
        )
        if on
    ]
    _log_once(
        "info",
        "Using %s Ulysses a2a transports (in dispatch order) for group %r out of potential: %s",
        enabled,
        group_name,
        list(DISPATCH_ORDER),
    )
    return plan


def require_default_ulysses_transport(parallel_config, model: str, error: type[Exception] = ValueError) -> None:
    """For models whose sequence parallelism supports only the process-group all-to-all.

    Rejects explicit symmetric-memory transports and pins ``"auto"`` to ``"nccl"`` so the topology probe never
    selects one for them.
    """
    backend = getattr(parallel_config, "ulysses_a2a_backend", "auto")
    if getattr(parallel_config, "ulysses_a2a_permute", False) or backend not in ("auto", NCCL):
        raise error(f"{model} supports only ulysses_a2a_backend='nccl' (got {backend!r}).")
    if backend == "auto":
        parallel_config.ulysses_a2a_backend = NCCL
