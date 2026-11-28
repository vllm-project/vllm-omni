# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transport for the strict Ulysses all-to-all (``ulysses_a2a_backend``).

The copy engine is faster than NCCL, so ``"auto"`` uses it wherever it is supported, with vLLM's custom all-reduce
rules (``CustomAllreduce``): NVIDIA CUDA, the whole group on one node, full NVLink connectivity for more than two GPUs
(two GPUs may use PCIe P2P), P2P access between the GPUs and torch symmetric memory; plus a strict pure Ulysses layer
and no CUDA graphs. The group is probed once and the ranks agree on the weakest answer, so they never pick different
transports.
"""

from __future__ import annotations

import socket
import zlib

import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

AUTO = "auto"
NCCL = "nccl"
COPY_ENGINE = "copy_engine"

# Per Ulysses group (name, size, members): (why the copy engine is unsupported or None, what the probe found).
_GROUPS: dict[tuple[str, int, tuple[int, ...]], tuple[str | None, str]] = {}
_LOGGED: set[str] = set()


def _is_nvidia_cuda() -> bool:
    return current_omni_platform.is_cuda() and torch.cuda.is_available() and torch.version.hip is None


def _symmetric_memory_available() -> bool:
    try:
        import torch.distributed._symmetric_memory  # noqa: F401
    except ImportError:
        return False
    return True


def _probe_device() -> torch.device:
    return torch.device("cuda", torch.accelerator.current_device_index())


def _fully_connected(physical_device_ids: list[int]) -> bool:
    """vLLM's NVLink check; an NVML failure (e.g. a container without NVML) counts as not fully connected."""
    try:
        return current_platform.is_fully_connected(physical_device_ids)
    except Exception as error:
        logger.debug("NVLink detection failed, assuming no NVLink: %s", error)
        return False


def _can_p2p(device: torch.device, physical_device_ids: list[int]) -> bool:
    """The driver's P2P report from this GPU to every peer, as vLLM trusts it with ``VLLM_SKIP_P2P_CHECK``; a peer
    this process cannot see counts as no P2P."""
    visible = {
        current_platform.visible_device_id_to_physical_device_id(index): index
        for index in range(torch.accelerator.device_count())
    }
    return all(
        physical in visible
        and (visible[physical] == device.index or torch.cuda.can_device_access_peer(device.index, visible[physical]))
        for physical in physical_device_ids
    )


def _probe(group: dist.ProcessGroup, world_size: int) -> tuple[str | None, str]:
    """Collective on ``group``: every rank must call it, and all get the same answer."""
    device = _probe_device()
    host = zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF
    physical = current_platform.visible_device_id_to_physical_device_id(device.index)
    # The Ulysses group has no CPU group, so the exchange uses device tensors.
    local = torch.tensor([host, physical], dtype=torch.int64, device=device)
    gathered = torch.empty(world_size * 2, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    hosts, physical_ids = gathered.view(world_size, 2).T.tolist()
    same_node = len(set(hosts)) == 1
    features = torch.tensor(
        [
            same_node,
            same_node and _fully_connected(physical_ids),
            same_node and _can_p2p(device, physical_ids),
            _symmetric_memory_available(),
        ],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(features, op=dist.ReduceOp.MIN, group=group)
    same_node, nvlink, p2p, symmetric_memory = features.tolist()
    found = f"{world_size} GPUs fully NVLink-connected" if nvlink else f"{world_size} GPUs without full NVLink"
    if not same_node:
        return "group spans nodes", f"{world_size} GPUs on several nodes"
    if world_size > 2 and not nvlink:
        return "copy engine needs full NVLink for more than two PCIe GPUs", found
    if not p2p:
        return "GPU P2P access unavailable", found
    if not symmetric_memory:
        return "torch symmetric memory unavailable", found
    return None, found if nvlink else f"{world_size} GPUs with PCIe P2P"


def copy_engine_unsupported_reason(
    group: dist.ProcessGroup, world_size: int, *, cuda_graphs: bool = False, strict_pure: bool = True
) -> str | None:
    """Why the copy engine cannot run this Ulysses exchange here, or ``None`` when it can."""
    if not _is_nvidia_cuda():
        return "not NVIDIA CUDA"
    if cuda_graphs:
        return "CUDA-graphed denoising (symmetric-memory workspaces are bound to one stream)"
    if not strict_pure:
        return "layer is not strict pure Ulysses"
    return _group_probe(group, world_size)[0]


def _group_probe(group: dist.ProcessGroup, world_size: int) -> tuple[str | None, str]:
    # Group names can be reused after a process group is destroyed, so the key includes the members.
    key = (group.group_name, world_size, tuple(dist.get_process_group_ranks(group)))
    if key not in _GROUPS:
        _GROUPS[key] = _probe(group, world_size)
    return _GROUPS[key]


def select_transport(
    requested: str,
    group: dist.ProcessGroup,
    world_size: int,
    *,
    cuda_graphs: bool,
    strict_pure: bool,
    overlap_chunks: int = 0,
) -> str:
    """The copy engine is faster: use it if it is supported here, otherwise NCCL.

    ``"auto"`` falls back to NCCL and says why once per group; an explicit ``"copy_engine"`` that cannot run raises.
    """
    if requested == NCCL:
        return NCCL
    if requested == COPY_ENGINE:
        if not _is_nvidia_cuda():
            return COPY_ENGINE  # other platforms validate their own symmetric memory
        # Layers outside strict pure Ulysses keep the strategy's own per-call fallback.
        reason = copy_engine_unsupported_reason(group, world_size, cuda_graphs=cuda_graphs)
        if reason is not None:
            raise ValueError(f"ulysses_a2a_backend='copy_engine' is not supported here: {reason}.")
        _log_once(f"Using 'copy_engine' Ulysses all-to-all for group {group.group_name!r} (explicit)")
        return COPY_ENGINE
    reason = copy_engine_unsupported_reason(group, world_size, cuda_graphs=cuda_graphs, strict_pure=strict_pure)
    if reason is None:
        found = _group_probe(group, world_size)[1]
        _log_once(f"Using 'copy_engine' Ulysses all-to-all for group {group.group_name!r} (auto: {found})")
        return COPY_ENGINE
    ignored = "; ulysses_overlap_chunks ignored" if overlap_chunks else ""
    _log_once(f"Using 'nccl' Ulysses all-to-all for group {group.group_name!r} (auto: {reason}{ignored})")
    return NCCL


def _log_once(line: str) -> None:
    if line not in _LOGGED:
        _LOGGED.add(line)
        logger.info(line)


def clear_ulysses_transport_state() -> None:
    """Forget probed groups and logged decisions, e.g. after the process groups are torn down."""
    _GROUPS.clear()
    _LOGGED.clear()


def require_default_ulysses_transport(parallel_config, model: str, error: type[Exception] = ValueError) -> None:
    """For models whose sequence parallelism does not support the copy engine: ``"auto"`` becomes ``"nccl"``, and an
    explicit ``"copy_engine"`` is rejected."""
    backend = getattr(parallel_config, "ulysses_a2a_backend", AUTO)
    if backend not in (AUTO, NCCL):
        raise error(f"{model} supports only ulysses_a2a_backend='nccl' (got {backend!r}).")
    parallel_config.ulysses_a2a_backend = NCCL
