# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV cache transfer payload representation and validation."""

import json
import struct
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

KV_PAYLOAD_CONTRACT_KEY = "_kv_payload_contract"
KV_PAYLOAD_CONTRACT_VERSION = 1

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
class KVCacheTransferData:
    """Container for KV cache transfer data."""

    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    @staticmethod
    def validate_payload_contract(
        data: object,
        expected_request_id: str,
        *,
        require_contract: bool = False,
    ) -> None:
        """Validate a versioned, complete KV payload before model injection.

        Payloads without a contract predate this protocol and remain accepted
        for rolling-upgrade compatibility.
        """
        if not isinstance(data, dict):
            raise ValueError(f"KV payload must be a dictionary, got {type(data).__name__}")
        metadata = data.get("metadata")
        contract = metadata.get(KV_PAYLOAD_CONTRACT_KEY) if isinstance(metadata, dict) else None
        if contract is None:
            if require_contract:
                raise ValueError("KV payload contract was removed during rank merge or slicing")
            return
        assert isinstance(metadata, dict)
        if contract != KV_PAYLOAD_CONTRACT_VERSION:
            raise ValueError(f"Unsupported KV payload contract version: {contract}")
        if data.get("request_id") != expected_request_id:
            raise ValueError(
                f"KV payload request ID mismatch: expected {expected_request_id!r}, got {data.get('request_id')!r}"
            )

        num_layers = metadata.get("num_layers")
        seq_len = metadata.get("seq_len")
        block_size = metadata.get("block_size")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
            raise ValueError(f"Invalid KV payload num_layers: {num_layers!r}")
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            raise ValueError(f"Invalid KV payload seq_len: {seq_len!r}")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError(f"Invalid KV payload block_size: {block_size!r}")

        block_ids = data.get("block_ids")
        required_blocks = (seq_len + block_size - 1) // block_size
        if not isinstance(block_ids, list) or len(block_ids) < required_blocks:
            actual_blocks = len(block_ids) if isinstance(block_ids, list) else type(block_ids).__name__
            raise ValueError(f"Incomplete KV payload block table: need {required_blocks} blocks, got {actual_blocks}")

        layer_blocks = data.get("layer_blocks")
        if not isinstance(layer_blocks, dict):
            raise ValueError("KV payload is missing layer_blocks")
        key_cache = layer_blocks.get("key_cache")
        value_cache = layer_blocks.get("value_cache")
        if not isinstance(key_cache, list) or not isinstance(value_cache, list):
            raise ValueError("KV payload key_cache and value_cache must be lists")
        if len(key_cache) != num_layers or len(value_cache) != num_layers:
            raise ValueError(
                f"Incomplete KV payload layers: expected {num_layers}, "
                f"got key={len(key_cache)} value={len(value_cache)}"
            )

        for layer_idx, (key, value) in enumerate(zip(key_cache, value_cache, strict=True)):
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise ValueError(f"KV payload layer {layer_idx} is missing key or value tensor")
            if key.shape != value.shape:
                raise ValueError(
                    f"KV payload layer {layer_idx} has mismatched K/V shapes: "
                    f"key={tuple(key.shape)} value={tuple(value.shape)}"
                )
            if key.ndim < 1 or key.shape[0] != seq_len:
                raise ValueError(
                    f"KV payload layer {layer_idx} has invalid token dimension: "
                    f"expected {seq_len}, got {tuple(key.shape)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def _build_tensors_desc(self, *, cpu: bool) -> tuple[list[dict[str, Any]], list, int, torch.device | None]:
        """Build tensor descriptors and payload chunks."""
        tensors_desc: list[dict[str, Any]] = []
        chunks: list = []
        data_offset = 0
        device = None

        for cache_name in ("key_cache", "value_cache"):
            for layer_idx, tensor in enumerate(self.layer_blocks.get(cache_name, [])):
                if tensor is None:
                    tensors_desc.append({"n": f"{cache_name}_{layer_idx}", "x": True})
                    continue
                tensor = tensor.detach().contiguous()
                if cpu:
                    tensor = tensor.cpu()
                elif device is None and getattr(tensor.device, "type", "cpu") != "cpu":
                    device = tensor.device
                nbytes = tensor.numel() * tensor.element_size()
                tensors_desc.append(
                    {
                        "n": f"{cache_name}_{layer_idx}",
                        "i": layer_idx,
                        "d": str(tensor.dtype).removeprefix("torch."),
                        "s": list(tensor.shape),
                        "o": data_offset,
                        "b": nbytes,
                    }
                )
                chunks.append(tensor.view(torch.uint8).numpy().tobytes() if cpu else tensor.view(torch.uint8).flatten())
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
        for tensor in chunks:
            num_bytes = tensor.numel()
            output[pos : pos + num_bytes].copy_(tensor)
            pos += num_bytes
        return output

    @staticmethod
    def _load_header_from_memoryview(raw: memoryview) -> tuple[dict[str, Any], memoryview]:
        if len(raw) < 4:
            raise ValueError("Corrupted KV payload: missing 4-byte header length")

        header_len = struct.unpack(">I", raw[:4])[0]
        if header_len > len(raw) - 4:
            raise ValueError(f"Corrupted KV payload: header_len={header_len} exceeds buffer size={len(raw)}")

        return json.loads(bytes(raw[4 : 4 + header_len])), raw[4 + header_len :]

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
    def _populate_caches(
        header: dict[str, Any],
        get_tensor: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, Any]:
        """Shared deserialization loop for both CPU and GPU paths."""
        num_layers = header["nl"]
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for info in header["td"]:
            if info.get("x"):
                continue
            name: str = info["n"]
            torch_dtype = KVCacheTransferData._resolve_torch_dtype(info["d"])
            tensor = get_tensor(info).view(torch_dtype).reshape(info["s"])
            layer_idx = KVCacheTransferData._resolve_layer_idx(info, num_layers)
            if name.startswith("key_cache_"):
                key_cache[layer_idx] = tensor
            elif name.startswith("value_cache_"):
                value_cache[layer_idx] = tensor

        return {
            "request_id": header["rid"],
            "layer_blocks": {"key_cache": key_cache, "value_cache": value_cache},
            "block_ids": header["bids"],
            "metadata": header["meta"],
        }

    @staticmethod
    def from_bytes(raw: bytes | bytearray | memoryview) -> dict[str, Any]:
        """Reconstruct KV cache data from the packed bytes format."""
        raw_view = memoryview(raw) if not isinstance(raw, memoryview) else raw
        header, tensor_data = KVCacheTransferData._load_header_from_memoryview(raw_view)
        data_len = len(tensor_data)

        def _get(info: dict) -> torch.Tensor:
            offset, nbytes = KVCacheTransferData._validate_tensor_span(info["n"], info, data_len)
            return torch.frombuffer(tensor_data, dtype=torch.uint8, offset=offset, count=nbytes)

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
