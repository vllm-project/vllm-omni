# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni stage transfer packet helpers (async_chunk thinker→talker MVP).

Splits large thinker→talker tensors into connector raw-data puts and a small
versioned sidecar for reconstruction. Legacy single-dict put/get remains the fallback.
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

_TALKER_TO_CODE2WAV_TENSOR_PATHS: tuple[str, ...] = (
    "codes.audio",
)


def build_full_payload_base_key(external_req_id: str, from_stage: int, chunk_id: int) -> str:
    """Legacy-compatible connector key used for the sidecar put/get."""
    return f"{external_req_id}_{from_stage}_{chunk_id}"


def build_tensor_transfer_key(base_key: str, entry_name: str) -> str:
    return f"{base_key}@tensor/{entry_name}"


def is_packet_sidecar(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and data.get("packet_version") == PACKET_VERSION
        and isinstance(data.get("tensor_entries"), list)
    )


def should_use_thinker_to_talker_packet_path(
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


def _tensor_entry_descriptor(name: str, tensor: torch.Tensor, transfer_key: str) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": tensor.device.type,
        "layout": "contiguous",
        "transfer_key": transfer_key,
    }


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


def split_qwen3_full_payload(
    payload: dict[str, Any],
    *,
    request_id: str,
    external_req_id: str,
    from_stage_id: int,
    to_stage_id: int,
    chunk_id: int,
    mode: str = MODE_ASYNC_CHUNK,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    """Split a Qwen3 full payload into tensor puts and a sidecar."""
    base_key = build_full_payload_base_key(external_req_id, from_stage_id, chunk_id)
    tensor_puts: list[tuple[str, torch.Tensor]] = []
    tensor_entries: list[dict[str, Any]] = []
    payload_kind = PAYLOAD_KIND_THINKER_TO_TALKER_FULL
    edge_id = EDGE_THINKER_TO_TALKER
    tensor_paths = _THINKER_TO_TALKER_TENSOR_PATHS
    if from_stage_id == 1 and to_stage_id == 2:
        payload_kind = PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL
        edge_id = EDGE_TALKER_TO_CODE2WAV
        tensor_paths = _TALKER_TO_CODE2WAV_TENSOR_PATHS

    for path in tensor_paths:
        value = _get_nested(payload, path)
        tensor = _coerce_tensor_payload_value(value)
        if tensor is None:
            continue
        transfer_key = build_tensor_transfer_key(base_key, path)
        tensor_puts.append((transfer_key, tensor))
        tensor_entries.append(_tensor_entry_descriptor(path, tensor, transfer_key))

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
        "is_empty": len(tensor_entries) == 0,
        "sidecar_put_key": base_key,
        "tensor_entries": tensor_entries,
        "metadata": _extract_sidecar_metadata(payload),
    }
    return tensor_puts, sidecar


def split_thinker_to_talker_full_payload(
    payload: dict[str, Any],
    *,
    request_id: str,
    external_req_id: str,
    from_stage_id: int,
    to_stage_id: int,
    chunk_id: int,
    mode: str = MODE_ASYNC_CHUNK,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    return split_qwen3_full_payload(
        payload,
        request_id=request_id,
        external_req_id=external_req_id,
        from_stage_id=from_stage_id,
        to_stage_id=to_stage_id,
        chunk_id=chunk_id,
        mode=mode,
    )


def _parse_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str.startswith("torch."):
        return getattr(torch, dtype_str.split(".", 1)[1])
    return getattr(torch, dtype_str)


def _tensor_from_connector_result(
    data: Any,
    entry: dict[str, Any],
    *,
    to_cpu: bool = True,
) -> torch.Tensor:
    dtype = _parse_dtype(str(entry["dtype"]))
    shape = tuple(int(x) for x in entry["shape"])

    managed_buffer_cls = None
    try:
        from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
            ManagedBuffer,
        )

        managed_buffer_cls = ManagedBuffer
    except ImportError:
        pass

    if managed_buffer_cls is not None and isinstance(data, managed_buffer_cls):
        try:
            tensor = data.as_tensor(dtype, shape)
            if to_cpu:
                tensor = tensor.detach().cpu().clone()
            else:
                tensor = tensor.detach().clone()
            return tensor
        finally:
            data.release()

    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype).reshape(shape)
        if to_cpu:
            return tensor.detach().cpu()
        return tensor.detach()

    raise TypeError(f"Unsupported connector tensor payload type: {type(data)!r}")


def reconstruct_qwen3_full_payload(
    sidecar: dict[str, Any],
    get_tensor: Callable[[str], Any],
    *,
    to_cpu: bool = True,
) -> dict[str, Any]:
    """Rebuild the semantic payload dict from a sidecar and tensor fetches."""
    if sidecar.get("packet_version") != PACKET_VERSION:
        raise ValueError(f"Unsupported packet_version: {sidecar.get('packet_version')!r}")
    payload_kind = sidecar.get("payload_kind")
    if payload_kind not in {
        PAYLOAD_KIND_THINKER_TO_TALKER_FULL,
        PAYLOAD_KIND_TALKER_TO_CODE2WAV_FULL,
    }:
        raise ValueError(f"Unsupported payload_kind: {payload_kind!r}")

    payload: dict[str, Any] = {}
    for entry in sidecar.get("tensor_entries", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        transfer_key = entry.get("transfer_key")
        if not name or not transfer_key:
            continue
        raw = get_tensor(str(transfer_key))
        if raw is None:
            raise KeyError(f"Missing tensor entry for transfer_key={transfer_key!r}")
        _set_nested(payload, str(name), _tensor_from_connector_result(raw, entry, to_cpu=to_cpu))

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
    get_tensor: Callable[[str], Any],
    *,
    to_cpu: bool = True,
) -> dict[str, Any]:
    return reconstruct_qwen3_full_payload(sidecar, get_tensor, to_cpu=to_cpu)
