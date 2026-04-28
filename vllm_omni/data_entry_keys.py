"""Structured payload types for inter-stage communication.

Adding a new model?
~~~~~~~~~~~~~~~~~~~
Every key you put into the inter-stage payload (``additional_information``,
``multimodal_output``, ``pooling_output``) **must** use the nested
``OmniPayload`` TypedDict structure.  For each category, every known
qualifier is an explicit field so misspellings are caught statically.

Categories
    hidden_states  – intermediate / output hidden-state tensors
    embed          – embedding tensors (prefill, decode, special tokens)
    ids            – token-ID sequences
    codes          – codec / audio code tensors
    meta           – scalar metadata, control flags, shapes

This module provides:
- Structured ``TypedDict`` types for static type checking (``OmniPayload``)
- ``serialize_payload`` / ``deserialize_payload`` for transport across
  process boundaries via ``AdditionalInformationPayload``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload

# ── Structured payload types ──
# These are TypedDicts (plain dicts at runtime, zero overhead) that give
# static type checking and IDE autocomplete for inter-stage payloads.
# Every field is optional (total=False) because each stage only populates
# the subset it needs.


class HiddenStates(TypedDict, total=False):
    output: torch.Tensor
    trailing_text: torch.Tensor
    last: torch.Tensor
    layers: dict[int, torch.Tensor]


class Embeddings(TypedDict, total=False):
    prefill: torch.Tensor
    decode: torch.Tensor
    cached_decode: torch.Tensor
    tts_bos: torch.Tensor
    tts_eos: torch.Tensor
    tts_pad: torch.Tensor
    tts_pad_projected: torch.Tensor
    voice: torch.Tensor
    speech_feat: torch.Tensor
    thinker_reply: torch.Tensor


class Codes(TypedDict, total=False):
    audio: torch.Tensor
    ref: torch.Tensor


class Ids(TypedDict, total=False):
    all: list[int]
    prompt: list[int]
    output: list[int]
    speech_token: list[int]
    prior_image: list[int]


class OmniPayloadMeta(TypedDict, total=False):
    finished: torch.Tensor
    left_context_size: int
    override_keys: list[tuple[str, str]]
    num_processed_tokens: int
    next_stage_prompt_len: int
    ar_width: int
    eol_token_id: int
    visual_token_start_id: int
    visual_token_end_id: int
    gen_token_mask: torch.Tensor
    omni_task: list[str]
    height: int
    width: int
    decode_flag: bool
    codec_streaming: bool
    ref_code_len: int
    talker_prefill_offset: int


class OmniPayload(TypedDict, total=False):
    hidden_states: HiddenStates
    embed: Embeddings
    ids: Ids
    codes: Codes
    meta: OmniPayloadMeta
    latent: torch.Tensor
    generated_len: int
    model_outputs: list[torch.Tensor]
    mtp_inputs: tuple[torch.Tensor, torch.Tensor]
    speaker: Any
    language: Any
    request_id: str


# ── Keys whose values are nested dicts (TypedDict sub-categories) ──
_NESTED_KEYS = frozenset({"hidden_states", "embed", "ids", "codes", "meta"})

# Sub-TypedDict for each nested category, used by runtime validation.
_NESTED_SCHEMAS: dict[str, type] = {
    "hidden_states": HiddenStates,
    "embed": Embeddings,
    "ids": Ids,
    "codes": Codes,
    "meta": OmniPayloadMeta,
}

_ROOT_KEYS: frozenset[str] = frozenset(OmniPayload.__annotations__.keys())


def assert_payload(payload: dict[str, Any], *, context: str = "payload") -> None:
    """Validate ``payload`` matches the ``OmniPayload`` nested schema.

    TypedDict is a static-only contract in Python; this helper closes the
    loop at runtime by rejecting:
      * non-dict payloads
      * top-level keys not declared on ``OmniPayload``
      * nested-category values that aren't dicts
      * sub-keys not declared on the matching nested TypedDict

    Call at producer/consumer boundaries when a schema violation should
    crash the pipeline instead of silently degrading audio quality.
    """
    assert isinstance(payload, dict), f"{context}: expected dict, got {type(payload).__name__}"
    extra_top = set(payload) - _ROOT_KEYS
    assert not extra_top, f"{context}: unknown top-level keys {sorted(extra_top)!r}"
    for nested_key, schema in _NESTED_SCHEMAS.items():
        if nested_key not in payload:
            continue
        sub = payload[nested_key]
        assert isinstance(sub, dict), f"{context}: payload[{nested_key!r}] must be dict, got {type(sub).__name__}"
        known_sub = frozenset(schema.__annotations__.keys())
        extra_sub = set(sub) - known_sub
        assert not extra_sub, f"{context}: payload[{nested_key!r}] unknown sub-keys {sorted(extra_sub)!r}"


def flatten_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested ``OmniPayload`` to dotted keys.

    Nested sub-dicts under ``_NESTED_KEYS`` are expanded:
    ``{"codes": {"audio": tensor}}`` → ``{"codes.audio": tensor}``.
    ``hidden_states["layers"]`` is expanded to ``hidden_states.layer_N``.
    Top-level values are kept as-is.
    """
    if not payload:
        return {}
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _NESTED_KEYS and isinstance(value, dict):
            for qual, val in value.items():
                if qual == "layers" and key == "hidden_states" and isinstance(val, dict):
                    for layer_idx, tensor in val.items():
                        flat[f"hidden_states.layer_{layer_idx}"] = tensor
                else:
                    flat[f"{key}.{qual}"] = val
        else:
            flat[key] = value
    return flat


def unflatten_payload(flat: dict[str, Any]) -> dict[str, Any]:
    """Unflatten dotted keys back to nested dicts.

    Reverse of :func:`flatten_payload`.
    ``hidden_states.layer_N`` keys are collected into ``hidden_states.layers``.
    """
    result: dict[str, Any] = {}
    for key, value in flat.items():
        if "." in key:
            type_key, qualifier = key.split(".", 1)
            sub = result.setdefault(type_key, {})
            if type_key == "hidden_states" and qualifier.startswith("layer_"):
                layers = sub.setdefault("layers", {})
                layer_idx = int(qualifier[len("layer_") :])
                layers[layer_idx] = value
            else:
                sub[qualifier] = value
        else:
            result[key] = value
    return result


# ── dtype helpers ──
_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def _dtype_to_name(dtype: torch.dtype) -> str:
    return _DTYPE_TO_NAME.get(dtype, str(dtype).replace("torch.", ""))


def _serialize_tensor(t: torch.Tensor) -> AdditionalInformationEntry:
    from vllm_omni.engine import AdditionalInformationEntry

    t_cpu = t.detach().to("cpu").contiguous()
    return AdditionalInformationEntry(
        tensor_data=t_cpu.numpy().tobytes(),
        tensor_shape=list(t_cpu.shape),
        tensor_dtype=_dtype_to_name(t_cpu.dtype),
    )


def _deserialize_tensor(entry: AdditionalInformationEntry) -> torch.Tensor:
    dt = np.dtype(entry.tensor_dtype or "float32")
    arr = np.frombuffer(entry.tensor_data, dtype=dt)  # type: ignore[arg-type]
    arr = arr.reshape(entry.tensor_shape)
    return torch.from_numpy(arr.copy())


def serialize_payload(
    payload: OmniPayload,
) -> AdditionalInformationPayload | None:
    """Serialize an ``OmniPayload`` for EngineCore transport.

    Uses :func:`flatten_payload` to produce dotted keys, then converts
    each value to an ``AdditionalInformationEntry``.
    """
    from vllm_omni.engine import (
        AdditionalInformationEntry,
        AdditionalInformationPayload,
    )

    flat = flatten_payload(payload)
    entries: dict[str, AdditionalInformationEntry] = {}

    for key, value in flat.items():
        if isinstance(value, torch.Tensor):
            entries[key] = _serialize_tensor(value)
        elif isinstance(value, list):
            entries[key] = AdditionalInformationEntry(list_data=value)
        elif value is not None:
            entries[key] = AdditionalInformationEntry(scalar_data=value)

    return AdditionalInformationPayload(entries=entries) if entries else None


def deserialize_payload(
    wire: AdditionalInformationPayload,
) -> OmniPayload:
    """Deserialize an ``AdditionalInformationPayload`` back to ``OmniPayload``.

    Decodes entries to tensors/lists, then uses :func:`unflatten_payload`
    to reconstruct the nested structure.
    """
    flat: dict[str, Any] = {}

    for key, entry in wire.entries.items():
        if entry.tensor_data is not None:
            flat[key] = _deserialize_tensor(entry)
        elif entry.list_data is not None:
            flat[key] = entry.list_data
        elif entry.scalar_data is not None:
            flat[key] = entry.scalar_data

    return unflatten_payload(flat)  # type: ignore[return-value]
