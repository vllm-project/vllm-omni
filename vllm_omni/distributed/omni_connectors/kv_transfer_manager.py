# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified OmniConnector and KV cache transfer management."""

import json
import struct
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec
from .utils.initialization import KV_TRANSFER_PORT_OFFSET
from .utils.kv_utils import normalize_layer_kv

logger = init_logger(__name__)

LayerKV = torch.Tensor | tuple[torch.Tensor, torch.Tensor]

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

    def to_bytes(self) -> bytes:
        """Convert to compact binary format for fast transfer."""
        tensors_desc: list[dict[str, Any]] = []
        tensor_bufs: list[bytes] = []
        data_offset = 0

        for cache_name in ("key_cache", "value_cache"):
            cache_list = self.layer_blocks.get(cache_name, [])
            for layer_idx, tensor in enumerate(cache_list):
                if tensor is None:
                    tensors_desc.append({"n": f"{cache_name}_{layer_idx}", "x": True})
                    continue

                t = tensor.detach().cpu().contiguous()
                dtype_str = str(t.dtype).removeprefix("torch.")
                raw = t.view(torch.uint8).numpy().tobytes()
                tensors_desc.append(
                    {
                        "n": f"{cache_name}_{layer_idx}",
                        "i": layer_idx,
                        "d": dtype_str,
                        "s": list(t.shape),
                        "o": data_offset,
                        "b": len(raw),
                    }
                )
                tensor_bufs.append(raw)
                data_offset += len(raw)

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
        return b"".join([struct.pack(">I", len(header)), header] + tensor_bufs)

    def to_gpu_tensor(self) -> torch.Tensor:
        """Convert to a packed GPU tensor for raw-data connectors."""
        tensors_desc: list[dict[str, Any]] = []
        gpu_tensors: list[torch.Tensor] = []
        data_offset = 0
        device = None

        for cache_name in ("key_cache", "value_cache"):
            cache_list = self.layer_blocks.get(cache_name, [])
            for layer_idx, tensor in enumerate(cache_list):
                if tensor is None:
                    tensors_desc.append({"n": f"{cache_name}_{layer_idx}", "x": True})
                    continue

                t = tensor.detach().contiguous()
                if device is None and t.is_cuda:
                    device = t.device
                dtype_str = str(t.dtype).removeprefix("torch.")
                nbytes = t.numel() * t.element_size()
                tensors_desc.append(
                    {
                        "n": f"{cache_name}_{layer_idx}",
                        "i": layer_idx,
                        "d": dtype_str,
                        "s": list(t.shape),
                        "o": data_offset,
                        "b": nbytes,
                    }
                )
                gpu_tensors.append(t.view(torch.uint8).flatten())
                data_offset += nbytes

        if device is None:
            raise RuntimeError("No CUDA tensors found, use to_bytes() instead")

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

        header_prefix = struct.pack(">I", len(header)) + header
        total_size = len(header_prefix) + data_offset
        output = torch.empty(total_size, dtype=torch.uint8, device=device)
        header_tensor = torch.frombuffer(bytearray(header_prefix), dtype=torch.uint8)
        output[: len(header_prefix)].copy_(header_tensor)

        pos = len(header_prefix)
        for t_flat in gpu_tensors:
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
    def _load_header_from_tensor(gpu_tensor: torch.Tensor) -> tuple[dict[str, Any], int]:
        if gpu_tensor.dtype != torch.uint8 or gpu_tensor.dim() != 1:
            raise ValueError("Packed GPU KV payload must be a 1-D uint8 tensor")

        total_bytes = int(gpu_tensor.numel())
        if total_bytes < 4:
            raise ValueError("Corrupted KV payload: missing 4-byte header length")

        header_len = struct.unpack(">I", gpu_tensor[:4].cpu().numpy().tobytes())[0]
        if header_len > total_bytes - 4:
            raise ValueError(f"Corrupted KV payload: header_len={header_len} exceeds buffer size={total_bytes}")

        header_bytes = gpu_tensor[4 : 4 + header_len].cpu().numpy().tobytes()
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
    def from_bytes(raw: "bytes | bytearray | memoryview") -> dict[str, Any]:
        """Reconstruct KV cache data from the packed bytes format."""
        raw_mv = memoryview(raw) if not isinstance(raw, memoryview) else raw
        header, tensor_data_mv = KVCacheTransferData._load_header_from_memoryview(raw_mv)

        num_layers = header["nl"]
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for info in header["td"]:
            if info.get("x"):
                continue

            name: str = info["n"]
            torch_dtype = KVCacheTransferData._resolve_torch_dtype(info["d"])
            offset, nbytes = KVCacheTransferData._validate_tensor_span(name, info, len(tensor_data_mv))
            t = (
                torch.frombuffer(
                    tensor_data_mv,
                    dtype=torch.uint8,
                    offset=offset,
                    count=nbytes,
                )
                .view(torch_dtype)
                .reshape(info["s"])
            )
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
    def from_bytes_gpu(gpu_tensor: torch.Tensor) -> dict[str, Any]:
        """Reconstruct KV cache data from a packed GPU tensor."""
        header, data_start = KVCacheTransferData._load_header_from_tensor(gpu_tensor)

        num_layers = header["nl"]
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers
        tensor_data_bytes = int(gpu_tensor.numel()) - data_start

        for info in header["td"]:
            if info.get("x"):
                continue

            name: str = info["n"]
            torch_dtype = KVCacheTransferData._resolve_torch_dtype(info["d"])
            offset, nbytes = KVCacheTransferData._validate_tensor_span(name, info, tensor_data_bytes)
            t = gpu_tensor[data_start + offset : data_start + offset + nbytes].clone()
            t = t.view(torch_dtype).reshape(info["s"])
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


class OmniKVTransferManager:
    """Unified management for OmniConnector and KV cache transfer.

    This class encapsulates all KV cache related operations:
    - Connector initialization and lazy creation
    - KV cache extraction from GPU blocks
    - KV cache transfer with retry logic
    - KV cache receiving with timeout
    """

    def __init__(self, config: OmniKVCacheConfig):
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

        if config.need_send_cache and config.connector_config:
            try:
                _ = self.connector
                logger.info("Sender connector eagerly initialized")
            except Exception as e:
                logger.warning("Failed to eagerly initialize sender connector: %s", e)

    @classmethod
    def _create(cls, cfg: dict | None) -> "OmniKVTransferManager":
        """Create manager from raw config dict."""
        if not cfg or not isinstance(cfg, dict):
            return cls(OmniKVCacheConfig())
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
            )
        )

    @classmethod
    def from_model_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from model config (for AR model runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_od_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from OmniDiffusion config (for diffusion runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any, model_config: Any) -> "OmniKVTransferManager":
        """Create from vllm config with fallback to kv_transfer_config."""
        # Primary: omni_kv_config from model_config
        omni_kv = getattr(model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            return cls._create(omni_kv)

        # Fallback: check kv_transfer_config
        kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
        if kv_cfg:
            direct = getattr(kv_cfg, "omni_connector_config", None)
            if isinstance(direct, dict) and direct:
                return cls._create({"connector_config": direct})
            extra = getattr(kv_cfg, "kv_connector_extra_config", None)
            if isinstance(extra, dict):
                omni = extra.get("omni_connector_config")
                if isinstance(omni, dict) and omni:
                    return cls._create({"connector_config": omni})

        return cls(OmniKVCacheConfig())

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
                    if c_type == "MooncakeTransferEngineConnector":
                        base_port = c_extra.get("zmq_port", 50051)
                        c_extra["from_stage"] = (
                            str(self.config.from_stage) if self.config.from_stage is not None else "0"
                        )
                        c_extra["to_stage"] = str(self.config.to_stage) if self.config.to_stage is not None else "1"

                        if self.config.need_send_cache:
                            c_extra["role"] = "sender"
                            from_stage = self.config.from_stage
                            if from_stage is not None:
                                try:
                                    c_extra["zmq_port"] = base_port + KV_TRANSFER_PORT_OFFSET + int(from_stage)
                                except (TypeError, ValueError):
                                    c_extra["zmq_port"] = base_port + KV_TRANSFER_PORT_OFFSET
                        elif self.config.need_recv_cache:
                            c_extra["role"] = "receiver"
                            from_stage = self.config.from_stage
                            sender_port = base_port + KV_TRANSFER_PORT_OFFSET
                            if from_stage is not None:
                                try:
                                    sender_port = base_port + KV_TRANSFER_PORT_OFFSET + int(from_stage)
                                except (TypeError, ValueError):
                                    pass
                            c_extra.setdefault("sender_host", c_extra.get("host", "127.0.0.1"))
                            c_extra.setdefault("sender_zmq_port", sender_port)

                    logger.info(
                        "Initializing OmniConnector (purpose=kv_transfer) with config: %s, role: %s",
                        cfg,
                        c_extra.get("role", "N/A"),
                    )
                    self._connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
                except Exception as e:
                    logger.error(f"Failed to initialize OmniConnector: {e}")
                    import traceback

                    traceback.print_exc()
                    # Cache failure sentinel to avoid repeated initialization attempts in hot paths.
                    self._connector = False

        return self._connector if self._connector else None

    def get_connector(self):
        """Get connector (compatibility wrapper for existing code)."""
        return self.connector

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

    @staticmethod
    def _clone_received_payload_tensors(data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict) or "layer_blocks" not in data:
            return data

        layer_blocks = data["layer_blocks"]
        for cache_name in ("key_cache", "value_cache"):
            cache_list = layer_blocks.get(cache_name, [])
            for idx, tensor in enumerate(cache_list):
                if isinstance(tensor, torch.Tensor):
                    cache_list[idx] = tensor.clone()
        return data

    def update_sender_info(self, sender_info: dict[str, Any], sender_stage_id: str | int | None = None) -> None:
        """Update receiver-side sender info before loading remote KV cache."""
        if not self.config.need_recv_cache:
            return

        actual_info = self._resolve_sender_info(sender_info, sender_stage_id=sender_stage_id)
        if not actual_info or "host" not in actual_info:
            logger.warning("Invalid sender_info format: %s", sender_info)
            return

        if self.config.connector_config:
            self.config.connector_config["sender_host"] = actual_info.get("host")
            self.config.connector_config["sender_zmq_port"] = actual_info.get("zmq_port")

        if self._connector and hasattr(self._connector, "update_sender_info"):
            try:
                self._connector.update_sender_info(actual_info.get("host"), actual_info.get("zmq_port"))
            except Exception:
                if hasattr(self._connector, "sender_host"):
                    self._connector.sender_host = actual_info.get("host")
                if hasattr(self._connector, "sender_zmq_port"):
                    self._connector.sender_zmq_port = actual_info.get("zmq_port")

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
        transfer_data: torch.Tensor | bytes | dict[str, Any]
        supports_raw = getattr(self.connector, "supports_raw_data", False)

        try:
            if supports_raw:
                transfer_data = kv_data.to_gpu_tensor()
            else:
                raise RuntimeError("Connector does not support raw tensor")
        except Exception:
            try:
                transfer_data = kv_data.to_bytes()
            except Exception:
                data_dict = kv_data.to_dict()
                data_dict["request_id"] = transfer_req_id
                transfer_data = data_dict

        serialization_ms = (time.perf_counter() - serialization_start) * 1000
        logger.info("KV cache serialized for %s in %.1f ms", transfer_req_id, serialization_ms)

        transfer_start = time.perf_counter()
        success, size, _ = self._transfer_with_retry(from_stage, to_stage, f"kv_cache_{transfer_req_id}", transfer_data)
        elapsed = time.perf_counter() - transfer_start

        if success:
            mbps = (size / 1024 / 1024) / elapsed if elapsed > 0 else 0
            logger.info(
                "KV transfer OK: %s, %s bytes, %.3fs, %.1f MB/s",
                transfer_req_id,
                size,
                elapsed,
                mbps,
            )
        else:
            logger.error(f"KV transfer FAILED: {transfer_req_id}")

    def _transfer_with_retry(
        self,
        from_stage: str,
        to_stage: str,
        request_id: str,
        data: "dict[str, Any] | bytes | torch.Tensor",
        max_retries: int = 3,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Transfer data with retry and exponential backoff.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier
            request_id: Request identifier for the key
            data: Data to transfer
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, size, metadata)
        """
        for attempt in range(max_retries):
            try:
                # Build the full key for connector
                full_request_id = f"omni_{from_stage}_to_{to_stage}_{request_id}"
                success, size, metadata = self.connector.put(
                    from_stage=from_stage, to_stage=to_stage, put_key=full_request_id, data=data
                )
                if success:
                    return success, size, metadata
                logger.warning(f"Transfer attempt {attempt + 1} failed for {request_id}")
            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception: {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))

        return False, 0, None

    @torch.inference_mode()
    def receive_kv_cache_for_request(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a specific request.

        This implements the receiving logic from gpu_diffusion_model_runner.py.

        Args:
            request_id: The request ID to receive KV cache for
            target_device: Optional device to move tensors to

        Returns:
            Tuple of (data dict, size) if successful, (None, 0) otherwise
        """
        if not self.connector:
            logger.warning("No connector available for receiving KV cache")
            return None, 0

        from_stage, to_stage = self.recv_stages
        if not from_stage or not to_stage:
            logger.warning("Receive stages not configured")
            return None, 0

        # Check if we should receive KV cache based on config
        if not self.config.need_recv_cache:
            logger.info(f"Skip receiving KV cache for {request_id} (need_recv_cache=False)")
            return None, 0

        timeout = self.config.recv_timeout
        start_time = time.time()
        poll_interval = 0.01
        max_poll_interval = 0.5

        logger.info(f"Wait for KV cache for request {request_id} from stage {from_stage} to {to_stage}...")

        try:
            while True:
                # Build the full key for connector
                full_request_id = f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}"
                link_start = time.perf_counter()
                result = self.connector.get(
                    from_stage=from_stage,
                    to_stage=to_stage,
                    get_key=full_request_id,
                )
                if result:
                    raw_data, size = result
                    elapsed = time.time() - start_time
                    link_ms = (time.perf_counter() - link_start) * 1000
                    managed_buffer = None

                    if hasattr(raw_data, "tensor") and hasattr(raw_data, "release"):
                        managed_buffer = raw_data
                        try:
                            buf_tensor = raw_data.tensor
                            if buf_tensor.is_cuda:
                                data = KVCacheTransferData.from_bytes_gpu(buf_tensor)
                                raw_data.release()
                                managed_buffer = None
                            else:
                                data = KVCacheTransferData.from_bytes(memoryview(buf_tensor.numpy()))
                                data = self._clone_received_payload_tensors(data)
                                raw_data.release()
                                managed_buffer = None
                        except Exception as e:
                            logger.error("Failed to deserialize KV cache from ManagedBuffer: %s", e)
                            if managed_buffer is not None:
                                raw_data.release()
                            return None, 0
                    elif isinstance(raw_data, (bytes, bytearray)):
                        data = KVCacheTransferData.from_bytes(raw_data)
                    elif isinstance(raw_data, torch.Tensor) and raw_data.dtype == torch.uint8 and raw_data.dim() == 1:
                        data = KVCacheTransferData.from_bytes(raw_data.cpu().numpy().tobytes())
                    else:
                        data = raw_data

                    try:
                        if isinstance(data, dict) and "layer_blocks" in data:
                            layer_blocks = data["layer_blocks"]
                            for cache_list in [
                                layer_blocks.get("key_cache", []),
                                layer_blocks.get("value_cache", []),
                            ]:
                                for i, tensor in enumerate(cache_list):
                                    if not isinstance(tensor, torch.Tensor):
                                        continue
                                    if target_device is not None and tensor.device != target_device:
                                        cache_list[i] = tensor.to(target_device).contiguous()
                    finally:
                        if managed_buffer is not None:
                            managed_buffer.release()

                    logger.info(
                        "Successfully received KV cache for %s, %s bytes, wait=%.3fs, link=%.1fms",
                        request_id,
                        size,
                        elapsed,
                        link_ms,
                    )
                    return data, size

                if time.time() - start_time > timeout:
                    logger.error(f"Timeout waiting for KV cache for request {request_id} after {timeout}s")
                    return None, 0

                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 2, max_poll_interval)

        except Exception as e:
            logger.error(f"Error receiving KV cache for {request_id}: {e}")
            import traceback

            traceback.print_exc()
            return None, 0

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
        request_id = getattr(req, "request_id", None)
        if request_id:
            return request_id
        if hasattr(req, "request_ids") and req.request_ids:
            return req.request_ids[0]
        return None

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

    def receive_multi_kv_cache_distributed(
        self,
        req: Any,
        cfg_kv_collect_func: Callable | None = None,
        target_device: torch.device | None = None,
    ) -> bool:
        """Broadcast-aware wrapper around :meth:`receive_multi_kv_cache`.

        SharedMemory connector is single-reader: once rank 0 consumes the
        segment it is deleted.  For multi-GPU stages (e.g. sequence-parallel)
        only rank 0 receives; the result is then broadcast to every other
        rank via the world process-group.

        For single-worker stages this is equivalent to calling
        :meth:`receive_multi_kv_cache` directly.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        world = get_world_group()

        if world.world_size <= 1:
            return self.receive_multi_kv_cache(req, cfg_kv_collect_func, target_device)

        # --- rank 0: receive to CPU (needed for pickle-based broadcast) ---
        if world.rank_in_group == 0:
            self.receive_multi_kv_cache(req, cfg_kv_collect_func, torch.device("cpu"))

            kv_payload: dict[str, object] = {}
            for attr in ("past_key_values", "kv_metadata"):
                val = getattr(req, attr, None)
                if val is not None:
                    kv_payload[attr] = val

            if hasattr(req, "sampling_params") and req.sampling_params is not None:
                for key in list(vars(req.sampling_params).keys()):
                    if (key.startswith("cfg_") and key.endswith("_past_key_values")) or key in (
                        "past_key_values",
                        "kv_metadata",
                    ):
                        val = getattr(req.sampling_params, key, None)
                        if val is not None:
                            kv_payload[f"sp.{key}"] = val

            payload_list = [kv_payload]
            # Use broadcast_object_list (pickle-based) instead of broadcast_tensor_dict
            # because the KV cache is a heterogeneous nested structure (NaiveCache objects
            # with metadata + tensors), not a flat tensor dict.  This runs once before
            # the denoising loop so the serialization cost is negligible.
            torch.distributed.broadcast_object_list(payload_list, src=world.ranks[0], group=world.cpu_group)
            kv_payload = payload_list[0]
        else:
            payload_list: list[dict[str, object] | None] = [None]
            torch.distributed.broadcast_object_list(payload_list, src=world.ranks[0], group=world.cpu_group)
            kv_payload = payload_list[0]

        # --- apply on ALL ranks (rank 0 also needs CPU→GPU move) ---
        if not kv_payload:
            return False

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

        return True


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
