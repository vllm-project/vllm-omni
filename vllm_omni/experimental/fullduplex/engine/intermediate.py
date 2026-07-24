from __future__ import annotations

from typing import TypedDict


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


def set_ref_audio(buffer: dict[str, object], waveform: object, sample_rate_hz: int) -> None:
    buffer.setdefault("codes", {})["ref"] = waveform
    buffer.setdefault("meta", {})["ref_audio_sr"] = int(sample_rate_hz)


def set_tts_handoff(buffer: dict[str, object], token_ids: object | None, hidden_states: object | None) -> None:
    """Store the generic AR-stage token/hidden-state handoff consumed by TTS."""
    if token_ids is not None:
        buffer.setdefault("ids", {})["tts"] = token_ids
    if hidden_states is not None:
        buffer.setdefault("hidden_states", {})["tts"] = hidden_states


def get_tts_handoff(info: dict[str, object]) -> tuple[object | None, object | None]:
    """Read the generic AR-stage to TTS handoff, including legacy flat keys."""
    ids_info = info.get("ids")
    hidden_info = info.get("hidden_states")
    tts_token_ids = ids_info.get("tts") if isinstance(ids_info, dict) else None
    tts_hidden_states = hidden_info.get("tts") if isinstance(hidden_info, dict) else None
    if tts_token_ids is None:
        tts_token_ids = info.get("tts_token_ids")
    if tts_hidden_states is None:
        tts_hidden_states = info.get("tts_hidden_states")
    return tts_token_ids, tts_hidden_states


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
