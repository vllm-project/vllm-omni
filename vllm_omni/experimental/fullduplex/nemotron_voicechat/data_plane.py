# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Output projection for Nemotron VoiceChat's independent channels."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.engine.contracts import (
    duplex_resource_request_belongs_to_session,
)
from vllm_omni.experimental.fullduplex.output import (
    get_duplex_output_decision,
)

logger = init_logger(__name__)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


@dataclass(frozen=True, slots=True)
class NemotronVoiceChatDataPlaneContext:
    epoch: int = 0
    turn_id: int = 0
    auto_responds: bool = True
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ("text", "audio")


@dataclass(slots=True)
class _RequestState:
    text_frames: int = 0
    audio_frames: int = 0
    pending_speech_end_frames: list[int] = field(default_factory=list)
    function_active: bool = False
    function_tokens: list[int] = field(default_factory=list)
    function_call_id: str | None = None
    terminal: bool = False


def _coerce_ints(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list | tuple):
        return []
    values: list[int] = []
    for item in value:
        if isinstance(item, torch.Tensor):
            if item.numel() != 1:
                continue
            item = item.item()
        elif isinstance(item, np.generic):
            item = item.item()
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return values


def _unwrap(output: object) -> tuple[object, object | None, int | None]:
    stage_id = getattr(output, "stage_id", None)
    inner = getattr(output, "request_output", None)
    if inner is not None and inner is not output:
        output = inner
    outputs = getattr(output, "outputs", None)
    completion = outputs[0] if isinstance(outputs, list) and outputs else None
    if stage_id is None:
        stage_id = getattr(output, "stage_id", None)
    return output, completion, int(stage_id) if isinstance(stage_id, int) else None


def _multimodal(output: object, completion: object | None) -> dict[str, object]:
    decision = get_duplex_output_decision(output)
    metadata = getattr(decision, "metadata", None)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    for candidate in (
        getattr(output, "multimodal_output", None),
        getattr(completion, "multimodal_output", None) if completion is not None else None,
    ):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _audio_value(metadata: Mapping[str, object]) -> object | None:
    value = next((metadata[key] for key in ("audio", "model_outputs", "latent") if key in metadata), None)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _sample_rate(metadata: Mapping[str, object]) -> int:
    value = metadata.get("sr", metadata.get("sample_rate_hz", 22050))
    if isinstance(value, list) and value:
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value) if isinstance(value, int | float) else 22050


def _audio_samples(audio: object | None) -> int:
    if audio is None:
        return 0
    if isinstance(audio, torch.Tensor):
        return int(audio.numel())
    try:
        return int(np.asarray(audio, dtype=np.float32).size)
    except (TypeError, ValueError):
        return 0


def _speech_end_event(request_id: str) -> dict[str, object]:
    return {
        "stage_role": "tts",
        "is_listen": False,
        "data_plane_request_id": request_id,
        "text": "",
        "end_of_turn": True,
    }


class NemotronVoiceChatDataPlaneSession:
    """Join frame-locked text/function outputs with Stage-2 audio."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestState] = {}
        self._tokenizer = None
        self._tokenizer_ref: str | None = None
        self._special_ids: dict[str, int] | None = None

    def configure_runtime(self, runtime_config: Mapping[str, object], *, tokenizer: Any | None = None) -> None:
        """Install the serving-resolved tokenizer contract as the sole truth."""
        try:
            special_ids = {
                "bos": int(runtime_config["nvc_text_bos_id"]),
                "eos": int(runtime_config["nvc_text_eos_id"]),
                "pad": int(runtime_config["nvc_text_pad_id"]),
                "sotc": int(runtime_config["nvc_function_sotc_id"]),
                "eotc": int(runtime_config["nvc_function_eotc_id"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Nemotron VoiceChat data plane requires resolved special-token ids") from exc
        tokenizer_ref = runtime_config.get("nvc_tokenizer_ref")
        if not isinstance(tokenizer_ref, str) or not tokenizer_ref:
            raise ValueError("Nemotron VoiceChat data plane requires nvc_tokenizer_ref")
        self._special_ids = special_ids
        self._tokenizer_ref = tokenizer_ref
        self._tokenizer = tokenizer

    def begin_request(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestState()).terminal = False

    def is_terminal(self, request_id: str | None) -> bool:
        return bool(request_id and self._requests.get(request_id) and self._requests[request_id].terminal)

    def mark_terminal(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestState()).terminal = True

    def close_stream(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id:
            self._requests.pop(active_request_id, None)
        for request_id in tuple(self._requests):
            if duplex_resource_request_belongs_to_session(request_id, session_id):
                self._requests.pop(request_id, None)

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            if self._tokenizer_ref is None:
                raise RuntimeError("Nemotron VoiceChat data plane was used before runtime configuration")
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_ref, trust_remote_code=False)
        return self._tokenizer

    def _ids(self) -> dict[str, int]:
        if self._special_ids is None:
            raise RuntimeError("Nemotron VoiceChat data plane was used before runtime configuration")
        return self._special_ids

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return str(self._load_tokenizer().decode(token_ids, skip_special_tokens=True))

    def project(
        self,
        result: object,
        *,
        context: NemotronVoiceChatDataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        for output in outputs:
            yield from self.project_output(output, context=context)

    def project_output(
        self,
        output: object,
        *,
        context: NemotronVoiceChatDataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        context = context or NemotronVoiceChatDataPlaneContext()
        outer = output
        output, completion, stage_id = _unwrap(output)
        request_id = getattr(output, "request_id", None) or getattr(outer, "request_id", None)
        request_id = str(request_id) if request_id is not None else ""
        state = self._requests.setdefault(request_id, _RequestState())
        metadata = _multimodal(outer, completion)
        modalities = {modality.lower() for modality in context.modalities}
        wants_text = "text" in modalities
        wants_audio = "audio" in modalities
        if stage_id == 0 or "nvc_text_token_ids" in metadata:
            ids = self._ids()
            text_ids = _coerce_ints(metadata.get("nvc_text_token_ids"))
            if not text_ids and completion is not None:
                text_ids = _coerce_ints(
                    getattr(completion, "token_ids", None) or getattr(completion, "cumulative_token_ids", None)
                )
            for token_id in text_ids[-1:]:
                state.text_frames += 1
                if token_id == ids["eos"]:
                    if state.audio_frames >= state.text_frames:
                        yield _speech_end_event(request_id)
                    else:
                        state.pending_speech_end_frames.append(state.text_frames)
                elif token_id == ids["pad"]:
                    yield {
                        "stage_role": "llm",
                        "is_listen": True,
                        "model_listen": True,
                        "listen_source": "model_listen",
                        "data_plane_request_id": request_id,
                        "end_of_turn": False,
                    }
                elif token_id != ids["bos"] and wants_text:
                    text = self._decode([token_id])
                    if text:
                        yield {
                            "stage_role": "llm",
                            "is_listen": False,
                            "data_plane_request_id": request_id,
                            "text": text,
                            "end_of_turn": False,
                        }

            for function_id in _coerce_ints(metadata.get("nvc_function_token"))[-1:]:
                yield from self._project_function_token(
                    function_id,
                    state=state,
                    request_id=request_id,
                    ids=ids,
                )
            return

        audio = _audio_value(metadata)
        sample_rate = _sample_rate(metadata)
        sample_count = _audio_samples(audio)
        encoded = (
            self._encode_audio(audio, sample_rate, context.response_format, context.speed)
            if wants_audio and audio is not None
            else None
        )
        audio_chunk_complete = sample_count > 0
        if audio_chunk_complete:
            # A native-duplex Stage-2 segment can contain one or more 80 ms
            # codec frames.  Count them against the frame-locked Stage-0 text
            # channel. The long-lived codec request deliberately keeps
            # ``finished=False`` across scheduler wakes; a non-empty decoded
            # chunk, not request lifetime, is the audio-coverage signal.
            duration_ms = sample_count * 1000 / max(1, sample_rate)
            state.audio_frames += max(1, round(duration_ms / 80))
        end_of_turn = bool(state.pending_speech_end_frames) and state.audio_frames >= state.pending_speech_end_frames[0]
        if encoded:
            yield {
                "stage_role": "tts",
                "is_listen": False,
                "data_plane_request_id": request_id,
                "text": "",
                "audio_data": encoded,
                "audio_format": context.response_format,
                "sample_rate_hz": sample_rate,
                "audio_duration_ms": round(sample_count * 1000 / max(1, sample_rate)),
                "audio_text_mark": True,
                "end_of_turn": end_of_turn,
            }
        elif end_of_turn:
            yield _speech_end_event(request_id)
        if end_of_turn:
            state.pending_speech_end_frames.pop(0)

    def _project_function_token(
        self,
        token_id: int,
        *,
        state: _RequestState,
        request_id: str,
        ids: Mapping[str, int],
    ) -> Iterator[dict[str, object]]:
        if token_id == ids["sotc"]:
            state.function_active = True
            state.function_tokens.clear()
            state.function_call_id = f"call_{uuid4().hex}"
            return
        if token_id == ids["eotc"] and state.function_active:
            state.function_active = False
            raw = self._decode(state.function_tokens)
            state.function_tokens.clear()
            try:
                calls = self._parse_calls(raw)
            except ValueError as exc:
                state.function_call_id = None
                yield {
                    "stage_role": "function",
                    "data_plane_request_id": request_id,
                    "error_code": "nemotron_function_call_parse_error",
                    "error": str(exc),
                }
                return
            for index, call in enumerate(calls):
                yield {
                    "stage_role": "function",
                    "data_plane_request_id": request_id,
                    "function_call": True,
                    "call_id": (
                        state.function_call_id if index == 0 and state.function_call_id else f"call_{uuid4().hex}"
                    ),
                    "name": str(call["name"]),
                    "arguments": (
                        call.get("arguments")
                        if isinstance(call.get("arguments"), str)
                        else json.dumps(call.get("arguments", {}), separators=(",", ":"))
                    ),
                }
            state.function_call_id = None
            return
        if state.function_active and token_id != ids["pad"]:
            state.function_tokens.append(token_id)

    @staticmethod
    def _parse_calls(value: str) -> list[dict[str, object]]:
        text = value.strip()
        if "<TOOLCALL>" in text:
            text = text.split("<TOOLCALL>", 1)[1]
        if "</TOOLCALL>" in text:
            text = text.split("</TOOLCALL>", 1)[0]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Nemotron VoiceChat produced malformed function-call JSON") from exc
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Nemotron VoiceChat function-call payload must be a non-empty object or list")
        calls: list[dict[str, object]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("Nemotron VoiceChat function-call entries must be objects")
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Nemotron VoiceChat function-call entries require a non-empty name")
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    decoded_arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("Nemotron VoiceChat function-call arguments contain malformed JSON") from exc
                if not isinstance(decoded_arguments, dict):
                    raise ValueError("Nemotron VoiceChat function-call arguments must encode a JSON object")
            elif not isinstance(arguments, dict):
                raise ValueError("Nemotron VoiceChat function-call arguments must be a JSON object")
            calls.append({"name": name.strip(), "arguments": arguments})
        return calls


__all__ = [
    "NemotronVoiceChatDataPlaneContext",
    "NemotronVoiceChatDataPlaneSession",
]
