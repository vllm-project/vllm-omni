# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import DRAINABLE_MODALITIES

CHUNK_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "audio_text_total_chars",
        "duplex_epoch",
        "duplex_turn_id",
        "finished",
        "full_payload_snapshot",
        "llm_output_text_utf8",
        "segment_end",
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


# NemotronVoiceChat talker snapshots for the client-facing accumulator.
#
# Each talker wake re-ships the whole reply so far instead of a per-step delta:
# ``codes.audio`` is the cumulative [frames, Q] code stack and the meta counters
# are that wake's values. Appending them as deltas stores the sum of every
# prefix seen (storage grows with stream length and the emitted tensor repeats
# historical rows), so a wake carrying the talker's marker replaces its own
# previous snapshot values, leaving storage proportional to the current prefix.
# The marker keeps this scoped to Nemotron: other producers emitting genuine
# per-chunk deltas under the same keys keep accumulating.
#
# Mirrors ``_FULL_PAYLOAD_REPLACE_KEYS`` in
# ``vllm_omni/model_executor/stage_input_processors/nemotron_voicechat.py``;
# duplicated because the client-side output process never imports the model's
# worker-side stage input processor.
_NEMOTRON_SNAPSHOT_MARKER = "nvc_logical_prompt_len"
_NEMOTRON_SNAPSHOT_KEYS = frozenset(
    {
        "codes.audio",
        # Talker emits these per wake as tensors (talker.py _emit / preprocess
        # meta), so they would otherwise concatenate across wakes.
        "meta.nvc_tts_step",
        "meta.nvc_logical_prompt_len",
        "meta.codec_streaming",
    }
)


def _has_meta_marker(payload: MultimodalPayload, marker: str) -> bool:
    """Whether *payload* carries *marker* in nested or flattened form."""
    if marker in payload or f"meta.{marker}" in payload:
        return True
    meta = payload.metadata.get("meta")
    return isinstance(meta, dict) and marker in meta


def replace_nemotron_snapshot_keys(
    accumulated: MultimodalPayload,
    incoming: MultimodalPayload,
) -> None:
    """Drop prior Nemotron snapshot values for keys present in *incoming*.

    Only payloads carrying the talker's marker take REPLACE semantics, and keys
    absent from *incoming* are left untouched so a metadata-only wake does not
    erase content accumulated by an earlier one.
    """
    if not _has_meta_marker(incoming, _NEMOTRON_SNAPSHOT_MARKER):
        return
    for key in _NEMOTRON_SNAPSHOT_KEYS:
        if key not in incoming:
            continue
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
