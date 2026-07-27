from typing import Any

import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import DRAINABLE_MODALITIES

CHUNK_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "audio_text_total_chars",
        "duplex_epoch",
        "duplex_turn_id",
        "llm_output_text_utf8",
        "tts_is_last_chunk",
        "turn_end",
    }
)


def _is_chunk_metadata_key(key: str) -> bool:
    if key.startswith("meta."):
        return key.split(".", 1)[1] in CHUNK_METADATA_KEYS
    return key in CHUNK_METADATA_KEYS


def replace_snapshot_keys(
    accumulated: MultimodalPayload,
    incoming: MultimodalPayload,
) -> None:
    """Replace per-chunk metadata with the latest values."""
    for key in (*incoming.tensors, *incoming.metadata):
        if _is_chunk_metadata_key(key):
            accumulated.tensors.pop(key, None)
            accumulated.metadata.pop(key, None)


def drain_delta_payload(payload: MultimodalPayload) -> None:
    """Remove client-facing delta data while retaining request-level state."""
    for modality_key in DRAINABLE_MODALITIES:
        key = str(modality_key)
        payload.tensors.pop(key, None)
        payload.metadata.pop(key, None)

    for metadata_key in CHUNK_METADATA_KEYS:
        flat_key = f"meta.{metadata_key}"
        payload.tensors.pop(flat_key, None)
        payload.metadata.pop(flat_key, None)

    meta = payload.metadata.get("meta")
    if isinstance(meta, dict):
        filtered = {key: value for key, value in meta.items() if key not in CHUNK_METADATA_KEYS}
        if filtered:
            payload.metadata["meta"] = filtered
        else:
            payload.metadata.pop("meta", None)


def _payload_meta_value(payload: MultimodalPayload, key: str) -> Any:
    flat_key = f"meta.{key}"
    if flat_key in payload.tensors:
        return payload.tensors[flat_key]
    if flat_key in payload.metadata:
        return payload.metadata[flat_key]
    meta = payload.metadata.get("meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def _last_scalar_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool | int):
        return int(value)
    if isinstance(value, list | tuple):
        return _last_scalar_int(value[-1]) if value else None
    if isinstance(value, torch.Tensor):
        try:
            if value.numel() == 0:
                return None
            return int(value.detach().cpu().reshape(-1)[-1].item())
        except (RuntimeError, TypeError, ValueError):
            return None
    return None


def is_non_final_delta_audio_chunk(payload: MultimodalPayload, mm_type: str | None) -> bool:
    """Return whether an audio delta explicitly declares more chunks."""
    if str(mm_type or "").lower() != "audio" and "audio" not in payload:
        return False
    return _last_scalar_int(_payload_meta_value(payload, "tts_is_last_chunk")) == 0
