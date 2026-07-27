# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified OmniConnector and KV cache transfer management."""

import enum
import json
import struct
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import KVPrefetchJob
from vllm_omni.platforms import current_omni_platform

from .factory import OmniConnectorFactory
from .utils.config import TRANSFER_ENGINE_CONNECTOR_NAMES, ConnectorSpec
from .utils.env import expand_env_int
from .utils.initialization import KV_RANK_PORT_STRIDE
from .utils.kv_utils import (
    KVTPTopology,
    build_rank_aware_recv_keys,
    build_rank_aware_send_keys,
    get_kv_target_ranks,
    get_local_tp_rank,
    get_omni_replica_id,
    get_tp_world_size,
    kv_zmq_port,
    merge_received_rank_shards,
    normalize_layer_kv,
    slice_layer_blocks,
    slice_received_rank_shard,
)
from .utils.serialization import OmniSerializer

logger = init_logger(__name__)

LayerKV = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class KVPrefetchConsumeError(RuntimeError):
    """Payload consumed from connector but post-get processing failed — sync retry impossible."""


class ReceiveRole(enum.Enum):
    """How a rank obtains KV: LOCAL pulls its own shard, LEADER pulls then
    distributes, FOLLOWER receives via collective."""

    LOCAL = "local"
    LEADER = "leader"
    FOLLOWER = "follower"


@dataclass(frozen=True)
class _TransferTopoConfig:
    """Snapshot of the current rank's parallel topology for KV receive routing."""

    role: ReceiveRole
    tp_active: bool
    cfg_size: int
    cfg_rank: int
    cfg_group: Any
    sp_size: int
    sp_rank: int
    sp_group: Any
    world: Any

    @property
    def is_leader(self) -> bool:
        return self.role == ReceiveRole.LEADER

    @property
    def is_follower(self) -> bool:
        return self.role == ReceiveRole.FOLLOWER

    @property
    def is_local(self) -> bool:
        return self.role == ReceiveRole.LOCAL

    @property
    def cfg_active(self) -> bool:
        return self.cfg_size > 1

    @property
    def sp_active(self) -> bool:
        return self.sp_size > 1


# Placeholder for the heavy primary KV in the side-payload dict; receiver swaps in the rebuilt object from the blob.
_KV_PLACEHOLDER = "__kv_placeholder__"

_SAFE_TORCH_DTYPES = {
    name: dtype
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex64",
        "complex128",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    if isinstance((dtype := getattr(torch, name, None)), torch.dtype)
}


@dataclass
class OmniKVCacheConfig:
    """Configuration for OmniKVTransferManager."""

    connector_config: dict[str, Any] | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    stage_id: str | int | None = None
    engine_input_source: list[str | int] | None = None
    need_recv_cache: bool = False
    need_send_cache: bool = False
    recv_timeout: float = 30.0
    from_tp: int = 1
    to_tp: int = 1
    enable_kv_async_prefetch: bool = False
    kv_prefetch_min_free_mem_ratio: float = 0.0


@dataclass
class KVCacheTransferData:
    """Container for KV cache transfer data."""

    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def _build_tensors_desc(self, *, cpu: bool) -> tuple[list[dict[str, Any]], list, int, torch.device | None]:
        """Iterate layer blocks and build tensor descriptors + data chunks.

        Returns ``(tensors_desc, chunks, total_bytes, device)``.
        *chunks* contains ``bytes`` when *cpu* is True, flat uint8 GPU tensors otherwise.
        """
        tensors_desc: list[dict[str, Any]] = []
        chunks: list = []
        data_offset = 0
        device = None

        for cache_name in ("key_cache", "value_cache"):
            for layer_idx, tensor in enumerate(self.layer_blocks.get(cache_name, [])):
                if tensor is None:
                    tensors_desc.append({"n": f"{cache_name}_{layer_idx}", "x": True})
                    continue
                t = tensor.detach().contiguous()
                if cpu:
                    t = t.cpu()
                elif device is None and getattr(t.device, "type", "cpu") != "cpu":
                    device = t.device
                nbytes = t.numel() * t.element_size()
                tensors_desc.append(
                    {
                        "n": f"{cache_name}_{layer_idx}",
                        "i": layer_idx,
                        "d": str(t.dtype).removeprefix("torch."),
                        "s": list(t.shape),
                        "o": data_offset,
                        "b": nbytes,
                    }
                )
                chunks.append(t.view(torch.uint8).numpy().tobytes() if cpu else t.view(torch.uint8).flatten())
                data_offset += nbytes

        return tensors_desc, chunks, data_offset, device

    def _build_header_bytes(self, tensors_desc: list[dict[str, Any]]) -> bytes:
        header = json.dumps(
            {
                "rid": self.request_id,
                "bids": self.block_ids,
                "meta": self.metadata,
                "td": tensors_desc,
                "nl": len(self.layer_blocks.get("key_cache", [])),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return struct.pack(">I", len(header)) + header

    def to_bytes(self) -> bytes:
        """Convert to compact binary format for fast transfer."""
        tensors_desc, chunks, _, _ = self._build_tensors_desc(cpu=True)
        return b"".join([self._build_header_bytes(tensors_desc)] + chunks)

    def to_gpu_tensor(self) -> torch.Tensor:
        """Convert to a packed device tensor for raw-data connectors."""
        tensors_desc, chunks, data_offset, device = self._build_tensors_desc(cpu=False)
        if device is None:
            raise RuntimeError("No device tensors found, use to_bytes() instead")
        header_prefix = self._build_header_bytes(tensors_desc)
        output = torch.empty(len(header_prefix) + data_offset, dtype=torch.uint8, device=device)
        output[: len(header_prefix)].copy_(torch.frombuffer(bytearray(header_prefix), dtype=torch.uint8))
        pos = len(header_prefix)
        for t_flat in chunks:
            n = t_flat.numel()
            output[pos : pos + n].copy_(t_flat)
            pos += n
        return output

    @staticmethod
    def _load_header_from_memoryview(raw_mv: memoryview) -> tuple[dict[str, Any], memoryview]:
        if len(raw_mv) < 4:
            raise ValueError("Corrupted KV payload: missing 4-byte header length")

        header_len = struct.unpack(">I", raw_mv[:4])[0]
        if header_len > len(raw_mv) - 4:
            raise ValueError(f"Corrupted KV payload: header_len={header_len} exceeds buffer size={len(raw_mv)}")

        return json.loads(bytes(raw_mv[4 : 4 + header_len])), raw_mv[4 + header_len :]

    @staticmethod
    def _load_header_from_tensor(tensor: torch.Tensor) -> tuple[dict[str, Any], int]:
        if tensor.dtype != torch.uint8 or tensor.dim() != 1:
            raise ValueError("Packed device KV payload must be a 1-D uint8 tensor")

        total_bytes = int(tensor.numel())
        if total_bytes < 4:
            raise ValueError("Corrupted KV payload: missing 4-byte header length")

        header_len = struct.unpack(">I", tensor[:4].cpu().numpy().tobytes())[0]
        if header_len > total_bytes - 4:
            raise ValueError(f"Corrupted KV payload: header_len={header_len} exceeds buffer size={total_bytes}")

        header_bytes = tensor[4 : 4 + header_len].cpu().numpy().tobytes()
        return json.loads(header_bytes), 4 + header_len

    @staticmethod
    def _validate_tensor_span(name: str, info: dict[str, Any], tensor_data_bytes: int) -> tuple[int, int]:
        offset = info["o"]
        nbytes = info["b"]
        if offset < 0 or nbytes < 0 or offset + nbytes > tensor_data_bytes:
            raise ValueError(
                f"Corrupted KV payload tensor span for {name}: "
                f"offset={offset}, bytes={nbytes}, tensor_data_bytes={tensor_data_bytes}"
            )
        return offset, nbytes

    @staticmethod
    def _resolve_torch_dtype(dtype_name: Any) -> torch.dtype:
        torch_dtype = _SAFE_TORCH_DTYPES.get(str(dtype_name))
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype in KV payload: {dtype_name}")
        return torch_dtype

    @staticmethod
    def _resolve_layer_idx(info: dict[str, Any], num_layers: int) -> int:
        layer_idx = info.get("i")
        if layer_idx is None:
            name = info.get("n")
            if isinstance(name, str) and name.startswith("key_cache_"):
                layer_idx = int(name.removeprefix("key_cache_"))
            elif isinstance(name, str) and name.startswith("value_cache_"):
                layer_idx = int(name.removeprefix("value_cache_"))
            else:
                raise ValueError(f"Invalid KV tensor name in payload: {name}")

        if not isinstance(layer_idx, int):
            raise ValueError(f"Invalid layer index in KV payload: {layer_idx}")
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Invalid layer index in KV payload: {layer_idx} (num_layers={num_layers})")
        return layer_idx

    @staticmethod
    def _populate_caches(header: dict[str, Any], get_tensor: callable) -> dict[str, Any]:
        """Shared deserialization loop for both CPU and GPU paths."""
        num_layers = header["nl"]
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for info in header["td"]:
            if info.get("x"):
                continue
            name: str = info["n"]
            torch_dtype = KVCacheTransferData._resolve_torch_dtype(info["d"])
            t = get_tensor(info).view(torch_dtype).reshape(info["s"])
            layer_idx = KVCacheTransferData._resolve_layer_idx(info, num_layers)
            if name.startswith("key_cache_"):
                key_cache[layer_idx] = t
            elif name.startswith("value_cache_"):
                value_cache[layer_idx] = t

        return {
            "request_id": header["rid"],
            "layer_blocks": {"key_cache": key_cache, "value_cache": value_cache},
            "block_ids": header["bids"],
            "metadata": header["meta"],
        }

    @staticmethod
    def from_bytes(raw: "bytes | bytearray | memoryview") -> dict[str, Any]:
        """Reconstruct KV cache data from the packed bytes format."""
        raw_mv = memoryview(raw) if not isinstance(raw, memoryview) else raw
        header, tensor_data_mv = KVCacheTransferData._load_header_from_memoryview(raw_mv)
        data_len = len(tensor_data_mv)

        def _get(info: dict) -> torch.Tensor:
            offset, nbytes = KVCacheTransferData._validate_tensor_span(info["n"], info, data_len)
            return torch.frombuffer(tensor_data_mv, dtype=torch.uint8, offset=offset, count=nbytes)

        return KVCacheTransferData._populate_caches(header, _get)

    @staticmethod
    def from_bytes_device(tensor: torch.Tensor) -> dict[str, Any]:
        """Reconstruct KV cache data from a packed device tensor."""
        header, data_start = KVCacheTransferData._load_header_from_tensor(tensor)
        data_len = int(tensor.numel()) - data_start

        def _get(info: dict) -> torch.Tensor:
            offset, nbytes = KVCacheTransferData._validate_tensor_span(info["n"], info, data_len)
            return tensor[data_start + offset : data_start + offset + nbytes].clone()

        return KVCacheTransferData._populate_caches(header, _get)

    @staticmethod
    def from_bytes_gpu(tensor: torch.Tensor) -> dict[str, Any]:
        """Compatibility alias for callers using the old GPU-specific name."""
        return KVCacheTransferData.from_bytes_device(tensor)


class OmniKVTransferManager:
    """Unified management for OmniConnector and KV cache transfer.

    This class encapsulates all KV cache related operations:
    - Connector initialization and lazy creation
    - KV cache extraction from GPU blocks
    - KV cache transfer with retry logic
    - KV cache receiving with timeout
    """

    def __init__(self, config: OmniKVCacheConfig, *, async_prefetch: bool = False):
        self.config = config
        self._connector = None

        # Pre-calculate send stages (from_stage, to_stage)
        self.send_stages = (
            (str(config.from_stage), str(config.to_stage)) if config.from_stage and config.to_stage else (None, None)
        )

        # Pre-calculate receive stages (from_stage, to_stage)
        recv_from = config.from_stage
        if config.engine_input_source:
            recv_from = config.engine_input_source[0]
        elif isinstance(config.stage_id, int):
            recv_from = config.stage_id - 1

        self.recv_stages = (
            (str(recv_from), str(config.stage_id))
            if recv_from is not None and config.stage_id is not None
            else (None, None)
        )

        local_rank = get_local_tp_rank()

        if config.from_tp <= 1 and config.to_tp <= 1:
            detected_tp = get_tp_world_size()
            from_tp = detected_tp
            to_tp = detected_tp
        else:
            from_tp = config.from_tp
            to_tp = config.to_tp

        self._tp_topo = KVTPTopology(source_tp_size=from_tp, target_tp_size=to_tp, local_rank=local_rank)

        # Injectable hooks (compatible with PR #2677 OmniConnectorModelRunnerMixin).
        self.kv_send_key_builder: Callable | None = None
        self.kv_recv_key_builder: Callable | None = None
        self.kv_payload_merger: Callable | None = None
        self.kv_payload_slicer: Callable | None = None

        # Base sender endpoint (rank-0 host/port) stored during
        # update_sender_info().  Used by the receive path to construct
        # per-rank metadata for heterogeneous TP without querying a registry.
        self._sender_base_host: str | None = None
        self._sender_base_zmq_port: int | None = None

        # Prefetch
        self._async_prefetch = async_prefetch
        self._prefetch_min_free_mem_ratio: float = max(0.0, config.kv_prefetch_min_free_mem_ratio)
        # Single-worker: serial mode ensures at most one outstanding prefetch
        # and avoids stream-creation races on _bg_copy_stream.
        self._prefetch_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-prefetch") if async_prefetch else None
        )
        self._prefetch_futures: dict[str, Any] = {}
        self._bg_copy_stream: current_omni_platform.Stream | None = None

        self._topo_config: _TransferTopoConfig | None = None

        if config.need_send_cache and config.connector_config:
            try:
                _ = self.connector
                logger.info("Sender connector eagerly initialized")
            except Exception as e:
                logger.warning("Failed to eagerly initialize sender connector: %s", e)

    @property
    def topo_config(self) -> _TransferTopoConfig:
        """Lazily build the parallel topology snapshot on first access."""
        if self._topo_config is None:
            self._topo_config = self._build_topo_config()
        return self._topo_config

    # ------------------------------------------------------------------ #
    #  Factory helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _create(cls, cfg: dict | None, *, async_prefetch: bool = False) -> "OmniKVTransferManager":
        """Create manager from raw config dict."""
        if not cfg or not isinstance(cfg, dict):
            return cls(OmniKVCacheConfig(), async_prefetch=async_prefetch)

        rank_mapping = cfg.get("rank_mapping", {})
        if not isinstance(rank_mapping, dict):
            rank_mapping = {}

        return cls(
            OmniKVCacheConfig(
                connector_config=cfg.get("connector_config"),
                from_stage=cfg.get("omni_from_stage"),
                to_stage=cfg.get("omni_to_stage"),
                stage_id=cfg.get("stage_id"),
                engine_input_source=cfg.get("engine_input_source", []),
                need_recv_cache=cfg.get("need_recv_cache", False),
                need_send_cache=cfg.get("need_send_cache", False),
                recv_timeout=cfg.get("recv_timeout", 30.0),
                from_tp=int(rank_mapping.get("from_tp", 1)),
                to_tp=int(rank_mapping.get("to_tp", 1)),
                enable_kv_async_prefetch=async_prefetch,
                kv_prefetch_min_free_mem_ratio=cfg.get("kv_prefetch_min_free_mem_ratio", 0.0),
            ),
            async_prefetch=async_prefetch,
        )

    @staticmethod
    def _receiver_pool_on_device(omni_kv: Any) -> bool:
        """True when the receiver connector pool is on a device (non-CPU)."""
        if not isinstance(omni_kv, dict):
            return False
        cc = omni_kv.get("connector_config")
        if not isinstance(cc, dict):
            return False
        return str(cc.get("memory_pool_device", "cpu")).lower() != "cpu"

    @classmethod
    def _resolve_async_prefetch(cls, omni_kv: Any, *, has_companion: bool) -> bool:
        """Effective prefetch flag: requested by config and not force-disabled."""
        if not isinstance(omni_kv, dict) or not bool(omni_kv.get("enable_kv_async_prefetch", False)):
            return False
        return not (has_companion or cls._receiver_pool_on_device(omni_kv))

    @classmethod
    def from_od_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from model or OmniDiffusion config."""
        omni_kv = getattr(config, "omni_kv_config", None)
        has_companion = getattr(config, "cfg_kv_collect_func", None) is not None
        async_prefetch = cls._resolve_async_prefetch(omni_kv, has_companion=has_companion)
        return cls._create(omni_kv, async_prefetch=async_prefetch)

    from_model_config = from_od_config

    @classmethod
    def from_vllm_config(cls, vllm_config: Any, model_config: Any) -> "OmniKVTransferManager":
        """Create from vllm config with fallback to kv_transfer_config."""
        # Primary: omni_kv_config on model_config — same shape as from_od_config.
        if isinstance(getattr(model_config, "omni_kv_config", None), dict):
            return cls.from_od_config(model_config)

        connector_cfg = cls._connector_cfg_from_kv_transfer(getattr(vllm_config, "kv_transfer_config", None))
        if connector_cfg is not None:
            return cls._create({"connector_config": connector_cfg})
        return cls(OmniKVCacheConfig())

    @staticmethod
    def _connector_cfg_from_kv_transfer(kv_cfg: Any) -> dict | None:
        """Extract the omni connector config dict nested in a kv_transfer_config."""
        if not kv_cfg:
            return None
        direct = getattr(kv_cfg, "omni_connector_config", None)
        if isinstance(direct, dict) and direct:
            return direct
        extra = getattr(kv_cfg, "kv_connector_extra_config", None)
        if isinstance(extra, dict):
            omni = extra.get("omni_connector_config")
            if isinstance(omni, dict) and omni:
                return omni
        return None

    @property
    def connector(self):
        """Lazy initialization of connector."""
        # If a previous initialization attempt failed, don't retry on every access.
        if self._connector is False:
            return None

        if self._connector is None:
            cfg = self.config.connector_config
            if cfg and (c_type := cfg.get("type")):
                try:
                    c_extra = {k: v for k, v in cfg.items() if k != "type"}
                    if c_type in TRANSFER_ENGINE_CONNECTOR_NAMES:
                        base_port = expand_env_int(c_extra.get("zmq_port", 50051), "zmq_port")
                        c_extra["from_stage"] = (
                            str(self.config.from_stage) if self.config.from_stage is not None else "0"
                        )
                        c_extra["to_stage"] = str(self.config.to_stage) if self.config.to_stage is not None else "1"

                        try:
                            stage_int = int(self.config.from_stage) if self.config.from_stage is not None else 0
                        except (TypeError, ValueError):
                            stage_int = 0
                        replica_id = get_omni_replica_id()
                        zmq_port = kv_zmq_port(
                            base_port,
                            stage_int,
                            self._tp_topo.local_rank,
                            replica_id=replica_id,
                        )

                        if self.config.need_send_cache:
                            c_extra["role"] = "sender"
                            c_extra["zmq_port"] = zmq_port
                        elif self.config.need_recv_cache:
                            c_extra["role"] = "receiver"
                            # Receiver-side sender endpoints are request-scoped.
                            # They are attached by the orchestrator as
                            # kv_sender_info and applied in update_sender_info().
                            # Do not derive sender_host from this receiver's
                            # local YAML host; on multi-node runs that points at
                            # the wrong process. Explicit sender_* YAML values
                            # are still preserved for standalone connector use.

                    logger.info(
                        "Initializing OmniConnector type=%s role=%s",
                        c_type,
                        c_extra.get("role", "N/A"),
                    )
                    self._connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
                except Exception:
                    logger.exception("Failed to initialize OmniConnector")
                    self._connector = False

        return self._connector if self._connector else None

    get_connector = property(lambda self: self.connector)

    def _build_topo_config(self) -> _TransferTopoConfig:
        """Build the parallel topology snapshot (called once at init)."""
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_cfg_group,
            get_classifier_free_guidance_rank,
            get_classifier_free_guidance_world_size,
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
            get_world_group,
        )

        try:
            world = get_world_group()
            world_size = world.world_size
            world_rank = world.rank_in_group
        except Exception:
            logger.exception("World group unavailable; defaulting to LOCAL")
            return _TransferTopoConfig(
                role=ReceiveRole.LOCAL,
                tp_active=False,
                cfg_size=1,
                cfg_rank=0,
                cfg_group=None,
                sp_size=1,
                sp_rank=0,
                sp_group=None,
                world=None,
            )

        if world_size <= 1:
            return _TransferTopoConfig(
                role=ReceiveRole.LOCAL,
                tp_active=False,
                cfg_size=1,
                cfg_rank=0,
                cfg_group=None,
                sp_size=1,
                sp_rank=0,
                sp_group=None,
                world=world,
            )

        tp_active = self._tp_topo.source_tp_size > 1 or self._tp_topo.target_tp_size > 1

        # CFG/SP groups may legitimately be uninitialized on pure-TP or
        # world-only distributed paths; treat that as "not active" (size 1)
        # rather than raising, so those paths can still take the local/world
        # receive route.
        try:
            cfg_size = get_classifier_free_guidance_world_size()
            cfg_rank = get_classifier_free_guidance_rank()
            cfg_group = get_cfg_group()
        except Exception:
            cfg_size, cfg_rank, cfg_group = 1, 0, None  # CFG-parallel not enabled

        try:
            sp_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            sp_group = get_sp_group()
        except Exception:
            sp_size, sp_rank, sp_group = 1, 0, None  # SP not enabled

        if tp_active and cfg_size <= 1 and sp_size <= 1:
            role = ReceiveRole.LOCAL
        elif tp_active and (cfg_size > 1 or sp_size > 1):
            role = ReceiveRole.LEADER if (cfg_rank == 0 and sp_rank == 0) else ReceiveRole.FOLLOWER
        else:
            role = ReceiveRole.LEADER if world_rank == 0 else ReceiveRole.FOLLOWER

        return _TransferTopoConfig(
            role=role,
            tp_active=tp_active,
            cfg_size=cfg_size,
            cfg_rank=cfg_rank,
            cfg_group=cfg_group,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            world=world,
        )

    def _resolve_sender_info(
        self, sender_info: dict[str, Any], sender_stage_id: str | int | None = None
    ) -> dict[str, Any] | None:
        if not sender_info:
            return None

        if "host" in sender_info:
            return sender_info

        if not isinstance(sender_info, dict):
            return None

        preferred_keys: list[str | int] = []
        if sender_stage_id is None:
            recv_from, _ = self.recv_stages
            sender_stage_id = recv_from

        if sender_stage_id is not None:
            preferred_keys.append(sender_stage_id)
            preferred_keys.append(str(sender_stage_id))
            try:
                preferred_keys.append(int(sender_stage_id))
            except (TypeError, ValueError):
                pass

        for key in dict.fromkeys(preferred_keys):
            info = sender_info.get(key)
            if isinstance(info, dict) and "host" in info:
                return info

        candidates = [info for info in sender_info.values() if isinstance(info, dict) and "host" in info]
        if len(candidates) == 1:
            return candidates[0]

        if candidates:
            logger.warning(
                "Ambiguous sender_info for sender_stage_id=%s: "
                "expected caller to resolve a single sender entry, got %s",
                sender_stage_id,
                sender_info,
            )
        return None

    def _slice_transfer_data_for_target(self, kv_data: KVCacheTransferData, target_rank: int) -> KVCacheTransferData:
        """Pre-slice sender payload for one target rank when sender TP < receiver TP."""
        topo = self._tp_topo
        ratio = topo.target_tp_size // topo.source_tp_size
        offset_in_sender = target_rank % ratio
        metadata = dict(kv_data.metadata) if isinstance(kv_data.metadata, dict) else {}
        metadata["tp_head_slice"] = {
            "applied": True,
            "side": "sender",
            "target_rank": target_rank,
            "source_rank": topo.local_rank,
            "from_tp": topo.source_tp_size,
            "to_tp": topo.target_tp_size,
            "offset_in_shard": offset_in_sender,
            "num_slices": ratio,
        }
        return KVCacheTransferData(
            request_id=kv_data.request_id,
            layer_blocks=slice_layer_blocks(kv_data.layer_blocks, offset_in_sender, ratio),
            block_ids=list(kv_data.block_ids),
            metadata=metadata,
        )

    def _serialize_transfer_payload(self, kv_data: KVCacheTransferData) -> torch.Tensor | bytes | dict[str, Any]:
        """Serialize KV transfer data using the connector's fastest supported path."""
        if getattr(self.connector, "supports_raw_data", False):
            try:
                return kv_data.to_gpu_tensor()
            except Exception:
                pass
        try:
            return kv_data.to_bytes()
        except Exception:
            return kv_data.to_dict()

    @staticmethod
    def _collect_request_kv_payload(req: Any) -> dict[str, object]:
        """Collect request-side KV objects for object broadcast."""
        kv_payload: dict[str, object] = {}
        for attr in ("past_key_values", "kv_metadata"):
            val = getattr(req, attr, None)
            if val is not None:
                kv_payload[attr] = val

        if hasattr(req, "sampling_params") and req.sampling_params is not None:
            for key in list(vars(req.sampling_params).keys()):
                if key in ("past_key_values", "kv_metadata") or (
                    key.startswith("cfg_")
                    and (
                        key.endswith("_past_key_values")
                        or key.endswith("_kv_metadata")
                        or key
                        in (
                            "cfg_kv_request_ids",
                            "cfg_active_branch",
                            "cfg_branch_roles",
                            "cfg_branch_past_key_values",
                            "cfg_branch_kv_metadata",
                        )
                    )
                ):
                    val = getattr(req.sampling_params, key, None)
                    if val is not None:
                        kv_payload[f"sp.{key}"] = val

        return kv_payload

    @staticmethod
    def _apply_request_kv_payload(
        req: Any,
        kv_payload: dict[str, object],
        target_device: torch.device | None = None,
    ) -> None:
        """Apply a broadcast KV payload back onto a request object."""
        for attr in ("past_key_values", "kv_metadata"):
            val = kv_payload.get(attr)
            if val is not None:
                if target_device is not None:
                    val = _move_to_device(val, target_device)
                setattr(req, attr, val)

        if hasattr(req, "sampling_params") and req.sampling_params is not None:
            for key, val in kv_payload.items():
                if key.startswith("sp."):
                    if target_device is not None:
                        val = _move_to_device(val, target_device)
                    setattr(req.sampling_params, key[3:], val)

    @staticmethod
    def _discover_cfg_branch_roles(req: Any) -> list[str]:
        """Discover CFG branch roles in a stable order."""
        sampling_params = getattr(req, "sampling_params", None)
        if sampling_params is None:
            return []

        roles: list[str] = []
        branch_map = getattr(sampling_params, "cfg_branch_past_key_values", None) or {}
        for preferred_role in ("cfg_text", "cfg_img"):
            if (
                preferred_role in branch_map
                or getattr(sampling_params, f"{preferred_role}_past_key_values", None) is not None
            ):
                roles.append(preferred_role)

        for role in branch_map.keys():
            if role not in roles and branch_map.get(role) is not None:
                roles.append(role)

        for key in vars(sampling_params).keys():
            if not (key.startswith("cfg_") and key.endswith("_past_key_values")):
                continue
            role = key.removesuffix("_past_key_values")
            if role in ("cfg_branch",) or role in roles:
                continue
            if getattr(sampling_params, key, None) is not None:
                roles.append(role)

        return roles

    @classmethod
    def _build_cfg_rank_local_payloads(cls, req: Any, cfg_size: int) -> list[dict[str, object] | None]:
        """Build per-cfg-rank payloads so each rank receives only its branch KV."""
        full_payload = cls._collect_request_kv_payload(req)
        payloads: list[dict[str, object] | None] = []

        main_payload = {
            key: value
            for key, value in full_payload.items()
            if key in ("past_key_values", "kv_metadata", "sp.past_key_values", "sp.kv_metadata")
        }
        branch_roles = cls._discover_cfg_branch_roles(req)
        if branch_roles:
            main_payload["sp.cfg_branch_roles"] = list(branch_roles)
            main_payload["sp.cfg_active_branch"] = None
        payloads.append(main_payload or None)

        sampling_params = getattr(req, "sampling_params", None)
        branch_map = getattr(sampling_params, "cfg_branch_past_key_values", None) or {}
        branch_metadata_map = getattr(sampling_params, "cfg_branch_kv_metadata", None) or {}

        for role in branch_roles:
            if sampling_params is None:
                payloads.append(None)
                continue

            branch_kv = branch_map.get(role)
            if branch_kv is None:
                branch_kv = getattr(sampling_params, f"{role}_past_key_values", None)
            branch_metadata = branch_metadata_map.get(role)
            if branch_metadata is None:
                branch_metadata = getattr(sampling_params, f"{role}_kv_metadata", None)
            if branch_kv is None:
                payloads.append(None)
                continue

            local_payload = dict(main_payload)
            local_payload["sp.cfg_active_branch"] = role
            local_payload["sp.cfg_branch_roles"] = list(branch_roles)
            local_payload["sp.cfg_branch_past_key_values"] = {role: branch_kv}
            local_payload[f"sp.{role}_past_key_values"] = branch_kv
            if branch_metadata is not None:
                local_payload["sp.cfg_branch_kv_metadata"] = {role: branch_metadata}
                local_payload[f"sp.{role}_kv_metadata"] = branch_metadata

            payloads.append(local_payload)

        padding = None
        if not branch_roles and cfg_size == 2 and main_payload:
            # AR didn't split KV by branch; cfg follower needs the same
            # positive KV for shared-prefix reuse. Only safe for cfg_size == 2:
            # for cfg_size > 2 every rank must get its own KV, which we can't
            # assign here, so padding stays None and the runner raises.
            # Shallow dict copy: shares PKV tensors with main_payload. Safe
            # only because the send path (cfg_scatter -> pack -> to_gpu_tensor)
            # reads tensors to serialize, never mutates in place. Any async/
            # KV-pool path that mutates req.past_key_values mid-scatter must
            # clone tensors here instead.
            padding = dict(main_payload)
        while len(payloads) < cfg_size:
            payloads.append(padding)

        return payloads[:cfg_size]

    def update_sender_info(self, sender_info: dict[str, Any], sender_stage_id: str | int | None = None) -> None:
        """Update receiver-side sender info before loading remote KV cache.

        The orchestrator always reports rank-0's ZMQ port.  When TP > 1 the
        receiver must offset the port so that each TP rank connects to the
        corresponding sender rank's port.

        The base host/port are also stored so that the receive path can
        construct per-rank metadata for heterogeneous TP scenarios.
        """
        if not self.config.need_recv_cache:
            return

        actual_info = self._resolve_sender_info(sender_info, sender_stage_id=sender_stage_id)
        if not actual_info or "host" not in actual_info:
            logger.warning("Invalid sender_info format: %s", sender_info)
            return

        sender_host = actual_info.get("host")
        base_zmq_port = actual_info.get("zmq_port")

        # Store base sender info for per-rank metadata construction.
        self._sender_base_host = sender_host
        if base_zmq_port is not None:
            self._sender_base_zmq_port = int(base_zmq_port)

        # --- Default sender: offset to match this receiver's corresponding sender rank ---
        zmq_port = base_zmq_port
        if zmq_port is not None and self._tp_topo.local_rank > 0:
            zmq_port = int(zmq_port) + self._tp_topo.local_rank * KV_RANK_PORT_STRIDE

        if self.config.connector_config:
            self.config.connector_config["sender_host"] = sender_host
            self.config.connector_config["sender_zmq_port"] = zmq_port

        if self._connector and hasattr(self._connector, "update_sender_info"):
            try:
                self._connector.update_sender_info(sender_host, zmq_port)
            except Exception:
                if hasattr(self._connector, "sender_host"):
                    self._connector.sender_host = sender_host
                if hasattr(self._connector, "sender_zmq_port"):
                    self._connector.sender_zmq_port = zmq_port

        logger.info(
            "Sender info updated: host=%s, base_port=%s, adjusted_port=%s (local_rank=%s)",
            sender_host,
            base_zmq_port,
            zmq_port,
            self._tp_topo.local_rank,
        )

    def handle_finished_requests_kv_transfer(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[LayerKV],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Handle KV cache transfer for finished requests.

        This method extracts KV cache from GPU blocks and transfers them
        to the downstream stage via the connector.

        Args:
            finished_reqs: Dict mapping request_id to {block_ids, seq_len}
            kv_caches: List of KV cache (tensor or tuple) per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
            request_id_resolver: Optional function to resolve global request ID

        Returns:
            List of request IDs that were processed
        """
        if not finished_reqs:
            return []

        if not self.config.need_send_cache:
            return list(finished_reqs.keys())

        if not self.connector:
            logger.warning("No connector available, skipping KV transfer but freeing resources")
            return list(finished_reqs.keys())

        logger.debug(f"Processing KV transfer for {len(finished_reqs)} requests")

        extracted_ids = []
        for req_id, data in finished_reqs.items():
            try:
                seq_len = data.get("seq_len", 0)
                block_ids = data.get("block_ids", [])
                if not block_ids:
                    logger.warning(f"Request {req_id} has no block IDs, skipping")
                    continue

                custom_metadata = data.get("custom_metadata")

                # Extract KV cache from GPU blocks and keep it on-device when
                # possible so raw-data connectors can use the fast path.
                kv_data = self._extract_kv_cache(
                    req_id, block_ids, seq_len, kv_caches, block_size, cache_dtype, custom_metadata
                )
                if kv_data:
                    # Resolve global request ID if available
                    transfer_req_id = request_id_resolver(req_id) if request_id_resolver else req_id

                    # Transfer to downstream stage via connector
                    self._transfer_kv_cache(kv_data, transfer_req_id)

            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                extracted_ids.append(req_id)

        return extracted_ids

    def _extract_kv_cache(
        self,
        req_id: str,
        block_ids: list[int],
        seq_len: int,
        kv_caches: list[LayerKV],
        block_size: int,
        cache_dtype: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> KVCacheTransferData | None:
        """Extract KV cache from GPU blocks for a single request.

        Args:
            req_id: Request identifier
            block_ids: List of block IDs to extract
            seq_len: Sequence length
            kv_caches: List of KV cache (tensor or tuple) per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
            custom_metadata: Optional custom metadata to include

        Note: If key/value block counts differ, extraction uses only the overlapping
        block range. Extra key/value blocks are ignored, so returned KV may be partial.

        Returns:
            KVCacheTransferData if extraction successful, None otherwise
        """
        num_layers = len(kv_caches)
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for layer_idx, layer_kv in enumerate(kv_caches):
            kv_pair = normalize_layer_kv(layer_kv, req_id=req_id, layer_idx=layer_idx)
            if kv_pair is None:
                continue
            key_blocks, value_blocks = kv_pair

            if key_blocks.shape[0] != value_blocks.shape[0]:
                logger.warning(
                    f"Layer {layer_idx} for request {req_id} has mismatched KV block counts: "
                    f"key={key_blocks.shape[0]}, value={value_blocks.shape[0]}; using shared range"
                )

            # Validate block IDs - shape: [num_blocks, block_size, n_heads, head_dim]
            max_block = min(key_blocks.shape[0], value_blocks.shape[0]) - 1
            valid_ids = [bid for bid in block_ids if 0 <= bid <= max_block]
            if not valid_ids:
                continue

            # Extract and reshape: [n_blocks, block_size, n_heads, head_dim]
            # -> [seq_len, n_heads, head_dim]
            selected_k = key_blocks[valid_ids]
            selected_v = value_blocks[valid_ids]
            flat_k = selected_k.flatten(0, 1)
            flat_v = selected_v.flatten(0, 1)
            if seq_len < flat_k.shape[0]:
                flat_k = flat_k[:seq_len]
                flat_v = flat_v[:seq_len]

            key_cache[layer_idx] = flat_k.detach().contiguous()
            value_cache[layer_idx] = flat_v.detach().contiguous()

        if not any(k is not None for k in key_cache):
            return None

        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=block_ids,
            metadata={
                "block_size": block_size,
                "num_layers": num_layers,
                "dtype": str(cache_dtype),
                "seq_len": seq_len,
                **(custom_metadata or {}),
            },
        )

    def _transfer_kv_cache(self, kv_data: KVCacheTransferData, transfer_req_id: str) -> None:
        """Transfer KV cache data to downstream stage via OmniConnector.

        Args:
            kv_data: The extracted KV cache data
            transfer_req_id: The request ID to use for transfer
        """
        from_stage, to_stage = self.send_stages
        if not from_stage or not to_stage:
            raise ValueError("Transfer stages (omni_from_stage, omni_to_stage) not configured")

        kv_data.request_id = transfer_req_id
        serialization_start = time.perf_counter()
        topo = self._tp_topo
        send_keys = build_rank_aware_send_keys(
            transfer_req_id, from_stage, to_stage, topo, hook=self.kv_send_key_builder
        )
        sender_slice_active = (
            topo.source_tp_size < topo.target_tp_size and len(send_keys) > 1 and not callable(self.kv_send_key_builder)
        )
        per_key_payloads: list[tuple[str, torch.Tensor | bytes | dict[str, Any]]] = []

        if sender_slice_active:
            target_ranks = get_kv_target_ranks(topo)
            if len(target_ranks) != len(send_keys):
                logger.warning(
                    "Skip sender-side KV slicing because target rank count does not match send key count: "
                    "target_ranks=%s send_keys=%s",
                    len(target_ranks),
                    len(send_keys),
                )
                sender_slice_active = False
            else:
                for put_key, target_rank in zip(send_keys, target_ranks, strict=False):
                    sliced_kv_data = self._slice_transfer_data_for_target(kv_data, target_rank)
                    per_key_payloads.append((put_key, self._serialize_transfer_payload(sliced_kv_data)))

        if not per_key_payloads:
            transfer_data = self._serialize_transfer_payload(kv_data)
            per_key_payloads = [(put_key, transfer_data) for put_key in send_keys]

        serialization_ms = (time.perf_counter() - serialization_start) * 1000
        logger.info("KV cache serialized for %s in %.1f ms", transfer_req_id, serialization_ms)

        transfer_start = time.perf_counter()
        total_size = 0
        all_succeeded = True
        for put_key, transfer_data in per_key_payloads:
            success, size, _ = self._transfer_with_retry(from_stage, to_stage, put_key, transfer_data)
            total_size += size
            all_succeeded = all_succeeded and success

        elapsed = time.perf_counter() - transfer_start

        if all_succeeded:
            mbps = (total_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
            logger.info(
                "KV transfer OK: %s, %s bytes across %s key(s), %.3fs, %.1f MB/s",
                transfer_req_id,
                total_size,
                len(send_keys),
                elapsed,
                mbps,
            )
        else:
            logger.error(f"KV transfer FAILED: {transfer_req_id}")

    def _transfer_with_retry(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: "dict[str, Any] | bytes | torch.Tensor",
        max_retries: int = 3,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Transfer data with retry and exponential backoff.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier
            put_key: Pre-built connector key (rank-aware when TP > 1)
            data: Data to transfer
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, size, metadata)
        """
        for attempt in range(max_retries):
            try:
                success, size, metadata = self.connector.put(
                    from_stage=from_stage, to_stage=to_stage, put_key=put_key, data=data
                )
                if success:
                    return success, size, metadata
                logger.warning(f"Transfer attempt {attempt + 1} failed for {put_key}")
            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception: {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))

        return False, 0, None

    def _resolve_sender_base(self, sender_info: dict[str, Any] | None) -> tuple[str | None, int | None]:
        """Resolve (host, base_zmq_port) from sender_info."""
        actual = self._resolve_sender_info(sender_info, sender_stage_id=self.recv_stages[0])
        if not actual or "host" not in actual:
            return None, None
        port = actual.get("zmq_port")
        return actual.get("host"), (int(port) if port is not None else None)

    def _has_enough_prefetch_device_mem(self, device: torch.device | None) -> bool:
        """False when free-device-memory fraction is below the prefetch threshold.

        CUDA/NPU only; other platforms report unconstrained (throttle skipped).
        """
        ratio = self._prefetch_min_free_mem_ratio
        if ratio <= 0.0 or device is None:
            return True
        try:
            free, total = current_omni_platform.get_device_memory(device)
        except Exception:
            return True
        return total <= 0 or (free / total) >= ratio

    @staticmethod
    def _release_pool_buffers(buffers: list[Any]) -> None:
        """Release all pool buffers in *buffers* and clear the list."""
        if not buffers:
            return
        for buf in buffers:
            try:
                buf.release()
            except Exception:
                logger.exception("Failed to release KV pool buffer")
        buffers.clear()

    def start_prefetch(self, kv_prefetch_job: KVPrefetchJob | None, target_device: torch.device | None = None) -> None:
        """Kick off a background KV load (non-blocking). No-op unless prefetch enabled."""
        if not (self._async_prefetch and self.config.need_recv_cache) or not kv_prefetch_job:
            return
        # Followers receive via collective distribute; bg pull would consume the owner's payload.
        if self.topo_config.is_follower:
            return
        rid = kv_prefetch_job.get("request_id")
        if not rid:
            return
        # Memory pressure → skip prefetch; sync receive handles it.
        if not self._has_enough_prefetch_device_mem(target_device):
            logger.warning(
                "Skip KV prefetch for %s: device free mem below %.2f", rid, self._prefetch_min_free_mem_ratio
            )
            return
        # Serial mode: at most one outstanding prefetch; drop any leftover request's future.
        for stale_rid in [k for k in self._prefetch_futures if k != rid]:
            self._discard_future(stale_rid)
        if rid in self._prefetch_futures:
            return
        sender_info = kv_prefetch_job.get("kv_sender_info")
        if not sender_info:
            # No explicit endpoint → bg receive would target wrong sender under multi-replica.
            logger.debug("Skip KV prefetch for %s: stub has no kv_sender_info", rid)
            return
        try:
            self._prefetch_futures[rid] = self._prefetch_executor.submit(
                self._prefetch_payload, rid, sender_info, target_device
            )
        except Exception:
            logger.exception("Failed to submit KV prefetch for %s", rid)

    def _prefetch_payload(
        self,
        request_id: str,
        sender_info: dict[str, Any] | None,
        target_device: torch.device | None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Bg-thread body: get + deserialize + H2D on the dedicated ``_bg_copy_stream``.

        Raises on failure (payload may be consumed → no sync retry).
        """
        try:
            on_device = target_device is not None and target_device.type != "cpu"
            if on_device:
                # bg thread doesn't inherit the main thread's current device.
                torch.accelerator.set_device_index(target_device.index)
                if self._bg_copy_stream is None:
                    assert self._prefetch_executor._max_workers == 1
                    self._bg_copy_stream = current_omni_platform.Stream()
                with current_omni_platform.stream(self._bg_copy_stream):
                    data, size = self.receive_kv_cache_for_request(
                        request_id, target_device=target_device, sender_info=sender_info
                    )
            else:
                data, size = self.receive_kv_cache_for_request(
                    request_id, target_device=target_device, sender_info=sender_info
                )
        except Exception:
            logger.exception("KV prefetch payload failed for %s (payload may be lost)", request_id)
            raise

        if data is None:
            return None, 0
        return data, size

    def consume_prefetched_kv(self, req: Any) -> tuple[dict[str, Any] | None, int]:
        """Consume a prefetched KV payload; (None, 0) on miss/recoverable failure.

        Raises ``KVPrefetchConsumeError`` when consumed but post-get failed (no fallback).
        """
        request_id = self._resolve_request_id(req)
        if not request_id:
            return None, 0

        fut = self._prefetch_futures.pop(request_id, None)
        # Serial mode: any other request still in the table is an orphan — drop it.
        for stale_rid in list(self._prefetch_futures):
            self._discard_future(stale_rid)

        if fut is None:
            return None, 0

        try:
            return fut.result()
        except KVPrefetchConsumeError:
            logger.exception("KV load failed for %s (payload consumed, cannot retry)", request_id)
            raise
        except Exception:
            logger.exception("KV load failed for %s; falling back to sync receive", request_id)
            return None, 0

    def _discard_future(self, request_id: str) -> None:
        """Cancel an unstarted prefetch or attach a callback to drop a running one."""
        fut = self._prefetch_futures.pop(request_id, None)
        if fut is None:
            return
        if not fut.cancel():
            fut.add_done_callback(_drop_prefetch_result)

    def shutdown_prefetch(self) -> None:
        """Cancel pending prefetches and stop the executor (call on teardown)."""
        for rid in list(self._prefetch_futures):
            self._discard_future(rid)
        if self._prefetch_executor is not None:
            try:
                self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                logger.exception("Failed to shut down KV prefetch executor")
            self._prefetch_executor = None

    @staticmethod
    def _record_stream_for_prefetched(data: dict[str, Any]) -> None:
        """``record_stream(current_stream)`` on GPU tensors in *data*.

        Protects prefetch-thread tensors from allocator reuse on LEADER collective paths.
        """
        if not isinstance(data, dict) or "layer_blocks" not in data:
            return
        if current_omni_platform.get_device_count() < 1:
            return
        current_stream = current_omni_platform.current_stream()
        layer_blocks = data["layer_blocks"]
        for cache_list in (layer_blocks.get("key_cache", []), layer_blocks.get("value_cache", [])):
            for tensor in cache_list:
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                    tensor.record_stream(current_stream)

    @torch.inference_mode()
    def receive_kv_cache_for_request(
        self,
        request_id: str,
        target_device: torch.device | None = None,
        *,
        sender_info: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for *request_id*; returns (data dict, size) or (None, 0).

        ``sender_info`` overrides instance ``_sender_base_*`` (per-call endpoint).
        """
        if not self.config.need_recv_cache:
            logger.debug("Skip receiving KV cache for %s (need_recv_cache=False)", request_id)
            return None, 0

        if not self.connector:
            logger.warning("No connector available for receiving KV cache")
            return None, 0

        from_stage, to_stage = self.recv_stages
        if not from_stage or not to_stage:
            logger.warning("Receive stages not configured")
            return None, 0

        # Skip during warmup dummy run — no sender is available.
        if OmniDiffusionRequest.is_dummy_run_request_id(request_id):
            logger.info("Skip receiving KV cache for dummy warmup request")
            return None, 0

        timeout = self.config.recv_timeout
        start_time = time.time()
        poll_interval = 0.01
        max_poll_interval = 0.5

        topo = self._tp_topo
        recv_key_pairs = build_rank_aware_recv_keys(
            request_id, from_stage, to_stage, topo, hook=self.kv_recv_key_builder
        )
        pending_pairs = list(recv_key_pairs)
        received_payloads: dict[str, tuple[dict[str, Any], int]] = {}
        deferred_memory: list[Any] = []

        # Per-call sender_info takes precedence over instance fields.
        if sender_info is not None:
            base_host, base_port = self._resolve_sender_base(sender_info)
        else:
            base_host, base_port = self._sender_base_host, self._sender_base_zmq_port

        logger.info(
            "Wait for KV cache for request %s from stage %s to %s via %s key(s)...",
            request_id,
            from_stage,
            to_stage,
            len(recv_key_pairs),
        )

        try:
            while True:
                link_start = time.perf_counter()
                for get_key, from_rank in list(pending_pairs):
                    rank_metadata: dict[str, Any] | None = None
                    build_meta = from_rank is not None or sender_info is not None
                    if build_meta and base_host and base_port is not None:
                        rank_metadata = {
                            "source_host": base_host,
                            "source_port": base_port + (from_rank or 0) * KV_RANK_PORT_STRIDE,
                        }
                    elif sender_info is not None:
                        logger.error(
                            "KV receive for %s: unresolved per-call sender_info (host=%s, port=%s); aborting",
                            request_id,
                            base_host,
                            base_port,
                        )
                        return None, 0

                    result = self.connector.get(
                        from_stage=from_stage,
                        to_stage=to_stage,
                        get_key=get_key,
                        metadata=rank_metadata,
                    )
                    if not result:
                        continue

                    raw_data, size = result

                    if hasattr(raw_data, "tensor") and hasattr(raw_data, "release"):
                        try:
                            buf_tensor = raw_data.tensor
                            if getattr(buf_tensor.device, "type", "cpu") != "cpu":
                                data = KVCacheTransferData.from_bytes_device(buf_tensor)
                                raw_data.release()
                            else:
                                data = KVCacheTransferData.from_bytes(memoryview(buf_tensor.numpy()))
                                deferred_memory.append(raw_data)
                        except Exception as e:
                            logger.error("Failed to deserialize KV cache from ManagedBuffer: %s", e)
                            raw_data.release()
                            return None, 0
                    elif isinstance(raw_data, (bytes, bytearray)):
                        data = KVCacheTransferData.from_bytes(raw_data)
                    elif isinstance(raw_data, torch.Tensor) and raw_data.dtype == torch.uint8 and raw_data.dim() == 1:
                        if getattr(raw_data.device, "type", "cpu") != "cpu":
                            data = KVCacheTransferData.from_bytes_device(raw_data)
                        else:
                            data = KVCacheTransferData.from_bytes(raw_data.numpy().tobytes())
                    else:
                        data = raw_data

                    received_payloads[get_key] = (data, size)
                    pending_pairs.remove((get_key, from_rank))

                if not pending_pairs and received_payloads:
                    elapsed = time.time() - start_time
                    link_ms = (time.perf_counter() - link_start) * 1000
                    ordered_payloads = [received_payloads[key][0] for key, _ in recv_key_pairs]
                    total_size = sum(received_payloads[key][1] for key, _ in recv_key_pairs)

                    if len(ordered_payloads) == 1:
                        data = ordered_payloads[0]
                    else:
                        data = merge_received_rank_shards(ordered_payloads, merger=self.kv_payload_merger)
                    data = slice_received_rank_shard(data, topo, slicer=self.kv_payload_slicer)

                    needs_clone = bool(deferred_memory)
                    try:
                        if isinstance(data, dict) and "layer_blocks" in data:
                            layer_blocks = data["layer_blocks"]
                            cache_lists = [
                                layer_blocks.get("key_cache", []),
                                layer_blocks.get("value_cache", []),
                            ]
                            for cache_list in cache_lists:
                                for i, tensor in enumerate(cache_list):
                                    if not isinstance(tensor, torch.Tensor):
                                        continue
                                    if target_device is not None and tensor.device != target_device:
                                        cache_list[i] = tensor.to(target_device).contiguous()
                                    elif needs_clone:
                                        cache_list[i] = tensor.clone()
                    except Exception as exc:
                        logger.exception("Failed to detach/move KV cache tensors for %s", request_id)
                        raise KVPrefetchConsumeError(
                            f"Post-get processing failed for {request_id} (payload already consumed)"
                        ) from exc

                    logger.info(
                        "Successfully received KV cache for %s, %s bytes across %s key(s), wait=%.3fs, link=%.1fms",
                        request_id,
                        total_size,
                        len(recv_key_pairs),
                        elapsed,
                        link_ms,
                    )
                    return data, total_size

                if time.time() - start_time > timeout:
                    logger.error(f"Timeout waiting for KV cache for request {request_id} after {timeout}s")
                    return None, 0

                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 2, max_poll_interval)

        except KVPrefetchConsumeError:
            raise
        except Exception:
            logger.exception("Error receiving KV cache for %s", request_id)
            return None, 0
        finally:
            self._release_pool_buffers(deferred_memory)

    def apply_kv_cache_to_request(self, req: Any, data: dict[str, Any]) -> None:
        """Apply received KV cache data to a request object.

        Args:
            req: The request object to apply KV cache to
            data: The received KV cache data dictionary
        """
        if isinstance(data, dict) and "layer_blocks" in data:
            layer_blocks = data["layer_blocks"]
            from types import SimpleNamespace

            kv_obj = SimpleNamespace(**layer_blocks)
            req.past_key_values = kv_obj

            # [Omni] Also attach to sampling_params for BagelPipeline compatibility
            # BagelPipeline checks req.sampling_params.past_key_values
            if hasattr(req, "sampling_params") and req.sampling_params is not None:
                req.sampling_params.past_key_values = kv_obj

        if "metadata" in data:
            req.kv_metadata = data["metadata"]
            if hasattr(req, "sampling_params") and req.sampling_params is not None:
                req.sampling_params.kv_metadata = data["metadata"]

    @staticmethod
    def _resolve_request_id(req: Any) -> str | None:
        """Resolve the logical request ID used for KV transfer lookups."""
        return getattr(req, "request_id", None)

    # Legacy compatibility method
    def receive_kv_cache(self, req: Any, target_device: torch.device | None = None) -> bool:
        """Receive KV cache and populate request object (legacy interface).

        Args:
            req: Request object with request_id attribute
            target_device: Optional device to move tensors to

        Returns:
            True if successful, False otherwise
        """
        kv_sender_info = getattr(req, "kv_sender_info", None)
        if kv_sender_info:
            self.update_sender_info(kv_sender_info, sender_stage_id=self.recv_stages[0])

        request_id = self._resolve_request_id(req)
        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        data, size = self.receive_kv_cache_for_request(request_id, target_device)
        if data:
            self.apply_kv_cache_to_request(req, data)
            return True
        return False

    def receive_multi_kv_cache(
        self,
        req: Any,
        cfg_kv_collect_func: Callable | None = None,
        target_device: torch.device | None = None,
    ) -> bool:
        """Receive primary KV cache and optional CFG companion KV caches.

        First receives the primary KV cache (existing logic). Then, if the
        request carries cfg_kv_request_ids and a model-specific
        cfg_kv_collect_func is provided, calls it to fetch and attach the
        companion KV caches to sampling_params.

        Args:
            req: Request object with request_id and sampling_params.
            cfg_kv_collect_func: Model-specific function for collecting
                CFG KV caches. Signature:
                (request_id, cfg_request_ids, kv_transfer_manager, target_device)
                -> dict[str, Any]
            target_device: Device to move tensors to.

        Returns:
            True if primary KV cache was received successfully.
        """
        primary_ok = self.receive_kv_cache(req, target_device)

        cfg_ids = getattr(getattr(req, "sampling_params", None), "cfg_kv_request_ids", None)
        if cfg_ids and cfg_kv_collect_func:
            request_id = self._resolve_request_id(req)
            try:
                cfg_kvs = cfg_kv_collect_func(
                    request_id,
                    cfg_ids,
                    self,
                    target_device,
                )
                if cfg_kvs and hasattr(req, "sampling_params") and req.sampling_params is not None:
                    for key, value in cfg_kvs.items():
                        setattr(req.sampling_params, key, value)
                    logger.info("Applied CFG KV caches: %s", list(cfg_kvs.keys()))
            except Exception:
                logger.exception("Failed to collect CFG KV caches for %s", request_id)

        return primary_ok

    def _extract_primary_kv_obj(self, kv_payload: dict[str, Any]) -> Any | None:
        """Return the primary KV object (``SimpleNamespace(key_cache, value_cache)``)."""
        for key in ("past_key_values", "sp.past_key_values"):
            obj = kv_payload.get(key)
            if obj is not None and hasattr(obj, "key_cache"):
                return obj
        return None

    def _pack_kv_payload(self, kv_payload: dict[str, Any] | None, device: torch.device) -> torch.Tensor | None:
        """Pack a KV payload into one uint8 tensor on device
        Layout: ``[4B side_len][msgpack side_bytes][gpu blob]``.
        Returns None on failure."""
        if not kv_payload:
            return None
        primary = self._extract_primary_kv_obj(kv_payload)
        if primary is None:
            return None

        key_cache = _move_to_device(getattr(primary, "key_cache", []) or [], device)
        value_cache = _move_to_device(getattr(primary, "value_cache", []) or [], device)
        if not any(isinstance(t, torch.Tensor) for t in key_cache):
            return None
        td = KVCacheTransferData(
            request_id="",
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=[],
            metadata={},
        )
        try:
            blob = td.to_gpu_tensor()
        except Exception:
            logger.exception("D2D: failed to pack primary KV; falling back to object transport")
            return None

        # Side-payload: replace the heavy primary with a sentinel, msgpack the rest.
        side = dict(kv_payload)
        for key in ("past_key_values", "sp.past_key_values"):
            if side.get(key) is primary:
                side[key] = _KV_PLACEHOLDER

        try:
            side_bytes = OmniSerializer.serialize(side)
        except Exception:
            logger.exception("D2D: failed to serialize side-payload; falling back to object transport")
            return None
        side_len = len(side_bytes)
        side_tensor = torch.frombuffer(bytearray(side_bytes), dtype=torch.uint8).to(device)

        # [4B side_len][side_bytes][blob]
        header = torch.tensor([side_len], dtype=torch.int32, device=device).view(torch.uint8)
        combined = torch.cat([header, side_tensor, blob])
        return combined

    def _unpack_kv_payload(self, combined: torch.Tensor) -> dict[str, Any] | None:
        """Inverse of :meth:`_pack_kv_payload`."""
        if combined is None or combined.numel() == 0:
            return None

        offset = 4
        side_len = int(combined[:offset].cpu().view(torch.int32).item())
        if side_len <= 0 and combined.numel() > offset:
            # Corrupt/zeroed blob (sender-side failure) → treat as miss; raising
            # would crash only the receiver and desync the group.
            logger.warning(
                "D2D: corrupt KV payload (side_len=%d, total_bytes=%d, likely sender-side failure); treating as miss",
                side_len,
                combined.numel(),
            )
            return None
        side_bytes = combined[offset : offset + side_len].cpu().numpy().tobytes()
        offset += side_len
        side: dict[str, Any] = OmniSerializer.deserialize(side_bytes)

        # Rebuild primary KV from the trailing blob and restore it in place.
        primary: Any = None
        blob = combined[offset:]
        if blob.numel() > 0:
            from types import SimpleNamespace

            data = KVCacheTransferData.from_bytes_device(blob)
            lb = data["layer_blocks"]
            primary = SimpleNamespace(key_cache=lb.get("key_cache"), value_cache=lb.get("value_cache"))

        for key in ("past_key_values", "sp.past_key_values"):
            if side.get(key) == _KV_PLACEHOLDER:
                side[key] = primary
        return side

    def _broadcast_kv_payload(
        self, group: Any, kv_payload: dict[str, Any] | None, device: torch.device, src: int = 0
    ) -> dict[str, Any] | None:
        """Broadcast KV payload via packed tensor; falls back to ``broadcast_object`` on packing failure.

        The packed-tensor path uses a two-phase collective (size then data).
        Both phases must always be called together — a rank entering only
        one phase will deadlock the group.
        """
        is_src = group.rank_in_group == src
        if is_src:
            combined = self._pack_kv_payload(kv_payload, device)
            if combined is None:
                group.broadcast_object(kv_payload, src=src)  # packing failed -> object
                return kv_payload
            group.broadcast_object(int(combined.numel()), src=src)
            group.broadcast(combined, src=src)
            return kv_payload

        size_obj = group.broadcast_object(None, src=src)
        if size_obj is None or not isinstance(size_obj, int):
            return size_obj  # src fell back to object
        combined = torch.empty(size_obj, dtype=torch.uint8, device=device)
        group.broadcast(combined, src=src)
        return self._unpack_kv_payload(combined)

    def _send_kv_payload(self, group: Any, kv_payload: dict[str, Any] | None, dst: int, device: torch.device) -> None:
        """Point-to-point send; falls back to ``send_object`` on packing failure."""
        combined = self._pack_kv_payload(kv_payload, device)
        if combined is None:
            group.send_object(kv_payload, dst)
            return
        size = int(combined.numel())
        group.send_object(size, dst)
        try:
            group.send(combined, dst)
        except Exception:
            logger.exception("D2D: send(combined) to rank %d failed; sending zero sentinel", dst)
            combined.zero_()
            group.send(combined, dst)
            raise

    def _recv_kv_payload(self, group: Any, src: int, device: torch.device) -> dict[str, Any] | None:
        """Inverse of :meth:`_send_kv_payload`."""
        size_or_obj = group.recv_object(src)
        if not isinstance(size_or_obj, int):
            return size_or_obj
        combined = group.recv(torch.Size([size_or_obj]), torch.uint8, src)
        combined = combined.to(device)
        return self._unpack_kv_payload(combined)

    def receive_multi_kv_cache_distributed(
        self,
        req: Any,
        cfg_kv_collect_func: Callable | None = None,
        target_device: torch.device | None = None,
    ) -> bool:
        """Synchronous receive: fetch → distribute → apply (FOLLOWER skips fetch)."""
        received = False
        if not self.topo_config.is_follower:
            received = self.receive_multi_kv_cache(req, cfg_kv_collect_func, target_device)
        kv_payload = self.distribute_kv_cache(req, target_device, received=received)
        if kv_payload is not None:
            self._apply_request_kv_payload(req, kv_payload, target_device)
        return received or kv_payload is not None

    def consume_and_distribute_kv_cache(self, req: Any, target_device: torch.device | None = None) -> bool:
        """Consume prefetched KV → apply → distribute; sync-receive on miss (no retry on consume-error)."""
        received = False
        payload_consumed = False
        if self._async_prefetch and not self.topo_config.is_follower:
            try:
                data, _ = self.consume_prefetched_kv(req)
                if data is not None:
                    self._record_stream_for_prefetched(data)
                    self.apply_kv_cache_to_request(req, data)
                    received = True
            except KVPrefetchConsumeError:
                logger.error(
                    "KV prefetch consumed payload for %s but post-get failed; request cannot recover",
                    self._resolve_request_id(req),
                )
                payload_consumed = True
        # Only fall back to sync receive on miss, not when payload was
        # already consumed from the connector (sync receive would block
        # until timeout as the data is gone).
        if not received and not self.topo_config.is_follower and not payload_consumed:
            logger.debug("KV prefetch miss for %s; falling back to sync receive", self._resolve_request_id(req))
            received = self.receive_multi_kv_cache(req, None, target_device)
        kv_payload = self.distribute_kv_cache(req, target_device, received=received)
        if kv_payload is not None:
            self._apply_request_kv_payload(req, kv_payload, target_device)
        return received or kv_payload is not None

    def _cfg_scatter(
        self,
        group: Any,
        payloads: list[Any],
        cfg_size: int,
        device: torch.device | None,
        request_id: str | None = None,
    ) -> None:
        """Scatter one payload per CFG follower."""
        for dst in range(1, cfg_size):
            try:
                self._send_kv_payload(group, payloads[dst], dst, device)
            except Exception:
                logger.warning("CFG scatter to rank %d failed for %s; that rank runs without KV", dst, request_id)
                continue

    def distribute_kv_cache(
        self, req: Any, target_device: torch.device | None = None, *, received: bool = False
    ) -> dict[str, Any] | None:
        """Distribute KV LEADER→FOLLOWER. No-op for LOCAL/pure TP."""
        pt = self.topo_config

        # LOCAL: no distribution.
        if pt.is_local:
            return None

        device = target_device if target_device is not None else torch.device("cpu")
        # TP + CFG/SP
        if pt.tp_active and (pt.cfg_active or pt.sp_active):
            kv_payload: dict[str, object] | None = None
            if pt.is_leader:
                if pt.cfg_active:
                    payloads = (
                        self._build_cfg_rank_local_payloads(req, pt.cfg_size) if received else [None] * pt.cfg_size
                    )
                    kv_payload = payloads[0]
                    self._cfg_scatter(pt.cfg_group, payloads, pt.cfg_size, device, self._resolve_request_id(req))
                elif pt.sp_active:
                    kv_payload = self._collect_request_kv_payload(req) if received else None
                if pt.sp_active and pt.sp_group is not None:
                    kv_payload = self._broadcast_kv_payload(pt.sp_group, kv_payload, device, src=0)
            elif pt.is_follower:
                if pt.cfg_active and pt.sp_rank == 0:
                    kv_payload = self._recv_kv_payload(pt.cfg_group, 0, device)
                if pt.sp_active and pt.sp_group is not None:
                    kv_payload = self._broadcast_kv_payload(pt.sp_group, kv_payload, device, src=0)
            else:
                logger.error("distribute_kv_cache: unexpected role %s in TP+CFG/SP path", pt.role)
                return None
            return kv_payload or None

        # TP inactive, world broadcast
        kv_payload = None
        if pt.is_leader and received:
            kv_payload = self._collect_request_kv_payload(req)
        kv_payload = self._broadcast_kv_payload(pt.world, kv_payload, device, src=0)
        return kv_payload or None


def _drop_prefetch_result(fut: Any) -> None:
    try:
        if not fut.cancelled():
            fut.result()
    except Exception:
        pass


def _move_to_device(obj: object, device: torch.device) -> object:
    """Recursively move tensors inside a KV cache object to *device*."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device).contiguous() if obj.device != device else obj
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            setattr(obj, k, _move_to_device(v, device))
        return obj
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    return obj
