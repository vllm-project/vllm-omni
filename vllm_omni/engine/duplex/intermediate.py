# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.contracts import DuplexFence

_TRANSPORT_TENSOR_MARKER = "__tensor__"


class DuplexIntermediateBuffer(TypedDict, total=False):
    """Structured keys carried through ``model_intermediate_buffer``.

    The buffer remains a dict for scheduler and msgspec compatibility, but
    duplex-specific producers and consumers should use the helpers in this
    module instead of scattering nested string keys across serving, runner, and
    model code.
    """

    request_id: str
    global_request_id: list[str]
    prompt_token_ids: list[int]
    llm_output_token_ids: list[int]
    llm_output_text: list[str]
    stream_output: bool
    native_duplex: bool
    ids: dict[str, object]
    hidden_states: dict[str, object]
    codes: dict[str, object]
    meta: dict[str, object]
    duplex: dict[str, object]
    omni_payload: object
    waveform: object
    mel_spec: object


def build_duplex_append_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: Mapping[str, object],
    runtime_config: Mapping[str, object],
    seq: int,
    turn_seq: int,
    payload: object,
    final: bool,
    prompt_token_ids: list[int],
    model_fields: Mapping[str, object],
) -> dict[str, object]:
    """Wrap model-selected tokens and payload in the shared append envelope.

    Models own token selection and extra worker fields; the framework owns
    request identity, sequencing and shallow config snapshots.
    """
    return {
        "prompt_token_ids": prompt_token_ids,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                **model_fields,
                "data_plane": True,
                "fence": fence,
                "session_id": fence.session_id,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "turn_seq": turn_seq,
                "mode": "append_audio_chunk",
                "payload": payload,
                "final": final,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "scheduler_token_budget": len(prompt_token_ids),
            },
        },
    }


def drop_static_append_configs(prompt: dict[str, object]) -> dict[str, object]:
    """Remove the session-sized config snapshots from a later append.

    The model-intermediate buffer is deliberately mutable while a submission
    is being prepared.  Keeping this helper at the transport seam makes the
    optimization explicit and leaves the prompt/token calculation (which may
    need the full config) in the model plugin unchanged.
    """
    intermediate = prompt.get("model_intermediate_buffer")
    duplex = intermediate.get("duplex") if isinstance(intermediate, dict) else None
    if isinstance(duplex, dict):
        duplex.pop("session_config", None)
        duplex.pop("runtime_config", None)
    return prompt


def build_duplex_intermediate_buffer(
    *,
    request_id: str,
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    output_text: str | None = None,
    stream_output: bool = False,
    native_duplex: bool = False,
) -> DuplexIntermediateBuffer:
    buffer: DuplexIntermediateBuffer = {
        "global_request_id": [str(request_id)],
        "ids": {},
    }
    if prompt_token_ids is not None:
        prompt_ids = [int(token_id) for token_id in prompt_token_ids]
        buffer["prompt_token_ids"] = prompt_ids
        buffer["ids"]["prompt"] = prompt_ids
    if output_token_ids is not None:
        output_ids = [int(token_id) for token_id in output_token_ids]
        buffer["llm_output_token_ids"] = output_ids
        buffer["ids"]["output"] = output_ids
    if output_text is not None:
        buffer["llm_output_text"] = [output_text]
    if stream_output:
        buffer["stream_output"] = True
    if native_duplex:
        buffer["native_duplex"] = True
    return buffer


def pack_transport_tensor(value: object) -> object:
    """Pack a tensor as raw bytes so it survives ``model_intermediate_buffer`` transport.

    The buffer is ``dict[str, Any]``, so a raw tensor would decode as vLLM's
    ``(dtype, shape, aux-index)`` tuple, and a ``tolist()`` round trip costs
    ~100x a memcpy on the orchestrator thread. Non-tensors pass through.
    """
    if not isinstance(value, torch.Tensor):
        return value
    tensor = value.detach().to("cpu").contiguous()
    return {
        _TRANSPORT_TENSOR_MARKER: True,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
    }


def unpack_transport_tensor(value: object) -> object:
    """Inverse of :func:`pack_transport_tensor`; other values pass through."""
    if not isinstance(value, Mapping) or value.get(_TRANSPORT_TENSOR_MARKER) is not True:
        return value
    dtype = getattr(torch, str(value["dtype"]))
    shape = [int(dim) for dim in value["shape"]]
    data = value["data"]
    if not data:
        return torch.empty(shape, dtype=dtype)
    # Copy into a writable buffer: msgpack hands back immutable bytes.
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def set_ref_audio(buffer: DuplexIntermediateBuffer, waveform: object, sample_rate_hz: int) -> None:
    buffer.setdefault("codes", {})["ref"] = pack_transport_tensor(waveform)
    buffer.setdefault("meta", {})["ref_audio_sr"] = int(sample_rate_hz)


def set_tts_handoff(buffer: DuplexIntermediateBuffer, token_ids: object | None, hidden_states: object | None) -> None:
    """Store the AR-to-TTS handoff used by the full-duplex stage bridge."""
    if token_ids is not None:
        buffer.setdefault("ids", {})["tts"] = token_ids
    if hidden_states is not None:
        buffer.setdefault("hidden_states", {})["tts"] = pack_transport_tensor(hidden_states)


def get_tts_handoff(info: dict[str, object]) -> tuple[object | None, object | None]:
    """Read the canonical handoff, including the legacy flat aliases."""
    ids_info = info.get("ids")
    hidden_info = info.get("hidden_states")
    token_ids = ids_info.get("tts") if isinstance(ids_info, dict) else None
    hidden_states = hidden_info.get("tts") if isinstance(hidden_info, dict) else None
    return (
        info.get("tts_token_ids") if token_ids is None else token_ids,
        unpack_transport_tensor(info.get("tts_hidden_states") if hidden_states is None else hidden_states),
    )


def get_stream_request_key(info: dict[str, object]) -> str:
    key = info.get("global_request_id") or info.get("request_id") or info.get("_omni_req_id")
    if isinstance(key, (list, tuple)):
        key = key[0] if key else None
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    if key is None:
        raise ValueError(
            "Duplex streaming handoff requires a stable request id; "
            "expected global_request_id, request_id, or _omni_req_id."
        )
    return str(key)
