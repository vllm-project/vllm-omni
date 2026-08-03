# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni stage-transfer packet helpers.

Packs large stage tensors into one contiguous raw-data buffer plus a small
versioned sidecar for reconstruction. Connectors that do not support raw data
fall back to the legacy single-dict put/get path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

PACKET_VERSION = 1
MODE_ASYNC_CHUNK = "async_chunk"
MODE_NON_ASYNC_FULL_PAYLOAD = "non_async_full_payload"
PAYLOAD_KIND_THINKER_TO_TALKER_FULL = "thinker_to_talker_full_payload"
PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL = "talker_to_code2wav_full_payload"
EDGE_THINKER_TO_TALKER = "thinker_to_talker"
EDGE_TALKER_TO_CODE2WAV = "talker_to_code2wav"

# Tensor field names for thinker→talker full payload (dot-separated paths).
_THINKER_TO_TALKER_TENSOR_PATHS: tuple[str, ...] = (
    "embed.prefill",
    "embed.tts_bos",
    "embed.tts_eos",
    "embed.tts_pad",
    "hidden_states.output",
)

_TALKER_TO_CODE2WAV_TENSOR_PATHS: tuple[str, ...] = ("codes.audio",)


# Byte alignment for each tensor within the packed buffer. Must be >= the
# largest tensor element size (8 bytes covers float64/int64) so that slicing
# the packed uint8 buffer and calling ``.view(dtype)`` is always valid.
_TENSOR_ALIGNMENT = 8


def _align_up(n: int, alignment: int = _TENSOR_ALIGNMENT) -> int:
    return (n + alignment - 1) // alignment * alignment


def build_full_payload_base_key(external_req_id: str, from_stage: int, chunk_id: int) -> str:
    """Legacy-compatible connector key used for the sidecar put/get."""
    return f"{external_req_id}_{from_stage}_{chunk_id}"


def build_packed_tensor_key(base_key: str) -> str:
    """Connector key for the single packed tensor buffer that accompanies a sidecar."""
    return f"{base_key}@packed"


def is_packet_sidecar(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and data.get("packet_version") == PACKET_VERSION
        and isinstance(data.get("tensor_entries"), list)
    )


def should_use_qwen3_packet_path(
    *,
    async_chunk: bool,
    supports_raw_data: bool,
    model_arch: str | None,
    from_stage_id: int,
    to_stage_id: int,
    transfer_mode: str | None = None,
) -> bool:
    if not supports_raw_data:
        return False
    if model_arch != "Qwen3OmniMoeForConditionalGeneration":
        return False
    if (from_stage_id, to_stage_id) not in {(0, 1), (1, 2)}:
        return False

    mode = transfer_mode or MODE_ASYNC_CHUNK
    if mode == MODE_ASYNC_CHUNK:
        return bool(async_chunk)
    if mode == MODE_NON_ASYNC_FULL_PAYLOAD:
        return not bool(async_chunk)
    return False


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _set_nested(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _resolve_edge(from_stage_id: int, to_stage_id: int) -> tuple[str, str, tuple[str, ...]]:
    """Return ``(payload_kind, edge_id, tensor_paths)`` for a stage edge."""
    if from_stage_id == 1 and to_stage_id == 2:
        return (
            PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL,
            EDGE_TALKER_TO_CODE2WAV,
            _TALKER_TO_CODE2WAV_TENSOR_PATHS,
        )
    return (
        PAYLOAD_KIND_THINKER_TO_TALKER_FULL,
        EDGE_THINKER_TO_TALKER,
        _THINKER_TO_TALKER_TENSOR_PATHS,
    )


def _pack_tensors(
    payload: dict[str, Any],
    tensor_paths: tuple[str, ...],
) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
    """Pack the payload's tensor fields into one contiguous uint8 buffer.

    Returns ``(packed_uint8_1d, entries)`` where ``entries`` is the layout
    table (``name``/``dtype``/``shape``/``offset``/``nbytes``) needed to slice
    the buffer back apart on the receiver. Returns ``(None, [])`` when the
    payload carries no tensor fields (e.g. a finish sentinel).

    Each tensor is copied once into its aligned slot; there is no per-element
    serialization. The single buffer is transferred in one connector put(),
    collapsing the previous one-put-per-tensor pattern.
    """
    regions: list[tuple[int, torch.Tensor]] = []
    entries: list[dict[str, Any]] = []
    offset = 0
    device: torch.device | None = None
    for path in tensor_paths:
        tensor = _coerce_tensor_payload_value(_get_nested(payload, path))
        if tensor is None:
            continue
        if device is None:
            device = tensor.device
        # ``tensor`` is already detached + contiguous (see _coerce). Reinterpret
        # its bytes as a flat uint8 view for a straight memcpy into the buffer.
        flat_u8 = tensor.reshape(-1).view(torch.uint8)
        nbytes = int(flat_u8.numel())
        entries.append(
            {
                "name": path,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        regions.append((offset, flat_u8))
        offset = _align_up(offset + nbytes)

    if not entries:
        return None, []

    packed = torch.empty(offset, dtype=torch.uint8, device=device)
    for region_offset, flat_u8 in regions:
        packed[region_offset : region_offset + flat_u8.numel()].copy_(flat_u8)
    return packed, entries


def _coerce_tensor_payload_value(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().contiguous()
    if isinstance(value, list):
        if not value:
            return None
        try:
            tensor = torch.as_tensor(value)
        except Exception:
            return None
        if tensor.numel() == 0:
            return None
        return tensor.detach().contiguous()
    return None


def _extract_sidecar_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "ids" in payload:
        metadata["ids"] = payload["ids"]
    if "next_stage_prompt_len" in payload:
        metadata["next_stage_prompt_len"] = payload["next_stage_prompt_len"]
    if "speaker" in payload:
        metadata["speaker"] = payload["speaker"]
    if "language" in payload:
        metadata["language"] = payload["language"]
    if "left_context_size" in payload:
        metadata["left_context_size"] = payload["left_context_size"]
    meta = payload.get("meta")
    if isinstance(meta, dict):
        sidecar_meta: dict[str, Any] = dict(meta)
        finished = sidecar_meta.get("finished")
        if isinstance(finished, torch.Tensor):
            sidecar_meta["finished"] = bool(finished.item())
        metadata["meta"] = sidecar_meta
    return metadata


def payload_has_packet_tensors(payload: dict[str, Any]) -> bool:
    """True when the payload carries at least one non-empty packet tensor field."""
    for path in _THINKER_TO_TALKER_TENSOR_PATHS + _TALKER_TO_CODE2WAV_TENSOR_PATHS:
        value = _get_nested(payload, path)
        if _coerce_tensor_payload_value(value) is not None:
            return True
    return False


def pack_qwen3_full_payload(
    payload: dict[str, Any],
    *,
    request_id: str,
    external_req_id: str,
    from_stage_id: int,
    to_stage_id: int,
    chunk_id: int,
    mode: str = MODE_ASYNC_CHUNK,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Pack a Qwen3 full payload into one contiguous tensor buffer + sidecar.

    Returns ``(packed_buffer, sidecar)``. ``packed_buffer`` is a single
    contiguous uint8 tensor holding every tensor field back-to-back (or
    ``None`` when the payload carries no tensors, e.g. a finish sentinel).
    ``sidecar`` carries the layout table plus scalar metadata needed to slice
    the buffer apart and rebuild the nested payload on the receiver.
    """
    base_key = build_full_payload_base_key(external_req_id, from_stage_id, chunk_id)
    payload_kind, edge_id, tensor_paths = _resolve_edge(from_stage_id, to_stage_id)
    packed, tensor_entries = _pack_tensors(payload, tensor_paths)

    sidecar = {
        "packet_version": PACKET_VERSION,
        "request_id": request_id,
        "external_req_id": external_req_id,
        "source_stage_id": from_stage_id,
        "target_stage_id": to_stage_id,
        "edge_id": edge_id,
        "mode": mode,
        "payload_kind": payload_kind,
        "sequence_id": chunk_id,
        "is_terminal": True,
        "is_empty": packed is None,
        "sidecar_put_key": base_key,
        "packed_key": build_packed_tensor_key(base_key),
        "packed_nbytes": int(packed.numel()) if packed is not None else 0,
        "tensor_entries": tensor_entries,
        "metadata": _extract_sidecar_metadata(payload),
    }
    return packed, sidecar


# Back-compat aliases for older call sites / tests.
should_use_thinker_to_talker_packet_path = should_use_qwen3_packet_path
split_qwen3_full_payload = pack_qwen3_full_payload
split_thinker_to_talker_full_payload = pack_qwen3_full_payload


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str.startswith("torch."):
        return getattr(torch, dtype_str.split(".", 1)[1])
    return getattr(torch, dtype_str)


def _packed_u8_from_connector_result(data: Any, *, to_cpu: bool = False) -> torch.Tensor:
    """Return a standalone 1D uint8 tensor from a connector ``get()`` result.

    The tensor is kept on the device the connector delivered it on (e.g. the
    RDMA pool's GPU); device placement is the consumer's concern, so we avoid a
    forced host round-trip that would defeat GPU-direct RDMA. For a
    ``ManagedBuffer`` the bytes are copied out (on that same device) before the
    buffer is released, so reconstructed views never alias pool memory the
    receive path may recycle. Pass ``to_cpu=True`` only when a specific
    consumer genuinely needs host tensors.
    """
    managed_buffer_cls = None
    try:
        from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
            ManagedBuffer,
        )

        managed_buffer_cls = ManagedBuffer
    except Exception:
        pass

    if managed_buffer_cls is not None and isinstance(data, managed_buffer_cls):
        try:
            buf = data.tensor.detach()  # 1D uint8 view into the pool
            if to_cpu and buf.device.type != "cpu":
                return buf.cpu()
            return buf.clone()
        finally:
            data.release()

    if isinstance(data, torch.Tensor):
        buf = data.detach().reshape(-1)
        if buf.dtype != torch.uint8:
            buf = buf.view(torch.uint8)
        return buf.cpu() if to_cpu else buf

    raise TypeError(f"Unsupported connector tensor payload type: {type(data)!r}")


def _unpack_tensors(packed_u8: torch.Tensor, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Slice a packed uint8 buffer back into a nested payload of tensors.

    Each slice is a zero-copy view over ``packed_u8``; ``packed_u8`` must
    already be a standalone tensor (see ``_packed_u8_from_connector_result``).
    """
    payload: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        offset = int(entry["offset"])
        nbytes = int(entry["nbytes"])
        dtype = _parse_dtype(str(entry["dtype"]))
        shape = tuple(int(x) for x in entry["shape"])
        region = packed_u8[offset : offset + nbytes]
        _set_nested(payload, str(name), region.view(dtype).reshape(shape))
    return payload


def reconstruct_qwen3_full_payload(
    sidecar: dict[str, Any],
    get_packed: Callable[[str], Any],
    *,
    to_cpu: bool = False,
) -> dict[str, Any]:
    """Rebuild the semantic payload dict from a sidecar and one packed fetch.

    ``get_packed`` is invoked at most once, with the packed-buffer key, and
    returns the connector's raw result (a ``ManagedBuffer`` or ``torch.Tensor``).
    Reconstructed tensors keep the connector's delivery device by default; pass
    ``to_cpu=True`` only when a consumer requires host tensors.
    """
    if sidecar.get("packet_version") != PACKET_VERSION:
        raise ValueError(f"Unsupported packet_version: {sidecar.get('packet_version')!r}")
    payload_kind = sidecar.get("payload_kind")
    if payload_kind not in {
        PAYLOAD_KIND_THINKER_TO_TALKER_FULL,
        PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL,
    }:
        raise ValueError(f"Unsupported payload_kind: {payload_kind!r}")

    entries = sidecar.get("tensor_entries") or []
    payload: dict[str, Any] = {}
    if entries:
        packed_key = sidecar.get("packed_key")
        if not packed_key:
            packed_key = build_packed_tensor_key(str(sidecar.get("sidecar_put_key", "")))
        raw = get_packed(str(packed_key))
        if raw is None:
            raise KeyError(f"Missing packed tensor buffer for key={packed_key!r}")
        packed_u8 = _packed_u8_from_connector_result(raw, to_cpu=to_cpu)
        payload = _unpack_tensors(packed_u8, entries)

    metadata = sidecar.get("metadata")
    if isinstance(metadata, dict):
        for key in ("ids", "speaker", "language"):
            if key in metadata:
                payload[key] = metadata[key]
        if "left_context_size" in metadata:
            payload["left_context_size"] = metadata["left_context_size"]
        meta = metadata.get("meta")
        if isinstance(meta, dict):
            payload_meta = dict(meta)
            if "finished" in payload_meta and not isinstance(payload_meta["finished"], torch.Tensor):
                payload_meta["finished"] = torch.tensor(bool(payload_meta["finished"]), dtype=torch.bool)
        else:
            payload_meta = {}
        if "next_stage_prompt_len" in metadata:
            # Prefer nested metadata contract to avoid legacy flat-key warnings.
            payload_meta.setdefault("next_stage_prompt_len", metadata["next_stage_prompt_len"])
            payload["next_stage_prompt_len"] = metadata["next_stage_prompt_len"]
        if payload_meta:
            payload["meta"] = payload_meta

    if payload_kind == PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL:
        audio_codes = _get_nested(payload, "codes.audio")
        if isinstance(audio_codes, torch.Tensor):
            payload["code_predictor_codes"] = audio_codes.reshape(-1).tolist()

    return payload


def reconstruct_thinker_to_talker_full_payload(
    sidecar: dict[str, Any],
    get_packed: Callable[[str], Any],
    *,
    to_cpu: bool = False,
) -> dict[str, Any]:
    return reconstruct_qwen3_full_payload(sidecar, get_packed, to_cpu=to_cpu)
