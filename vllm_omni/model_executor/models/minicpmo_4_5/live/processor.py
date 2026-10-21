# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 ``LiveSessionProcessor``.

Renders the official duplex layout (MiniCPM-o-Demo ``DuplexCapability``:
``prepare`` and ``streaming_prefill``) onto one resumable request::

    <|im_start|>system\\n{prompt}\\n<|audio_start|>[reference voice audio]
    [\\n\\nprevious: {text}]<|audio_end|><|im_end|>
    <unit>[audio x10]{model output up to a chunk terminator}</unit>
    <unit>[audio x10]...

The reference voice (the checkpoint's ``assets/HT_ref_audio.wav``) goes
through ``multi_modal_data`` once, in the first prompt (``live_duplex_unit``
drops the chat format's audio markers, which the prompt writes itself).

Unit audio follows the official streaming chunking: the checkpoint's
streaming mel processor in ``exact`` mode, the stream left-padded to the
1035 ms first chunk, then 1000 ms per unit. Each unit's mel features ride in
the streaming update's ``model_intermediate_buffer`` (``minicpmo_live_audio``)
and the Thinker encodes them with its stateful streaming audio encoder
(``live/encoder.py``). ``MiniCPMOLiveLogitsProcessor`` applies the official
greedy decode rules.
"""

from __future__ import annotations

import copy
import functools
import threading
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.live.audio import resample
from vllm_omni.entrypoints.openai.live.processor import LiveSessionProcessor, ProcessorContext
from vllm_omni.entrypoints.openai.live.protocol import LiveProtocolError
from vllm_omni.entrypoints.openai.live.session import (
    AssistantItem,
    LiveSessionState,
    NativeUnitItem,
    SystemItem,
    UserTextItem,
)

_AUDIO_PLACEHOLDER = "(<audio>./</audio>)"
_DEFAULT_SYSTEM_PROMPT = "Streaming Omni Conversation."
_PREVIOUS_MARKER = "\n\nprevious: "
# Newest tokens of dropped-unit text kept as "previous" (DuplexWindowConfig.
# context_previous_max_tokens); unbounded, it would fill the context.
_PREVIOUS_MAX_TOKENS = 500
_REFERENCE_AUDIO = "assets/HT_ref_audio.wav"
_SAMPLE_RATE = 16_000
# Official streaming mode (DuplexCapability.prepare -> init_streaming_processor).
_CHUNK_MS = 1000
_FIRST_CHUNK_MS = 1035
_CNN_REDUNDANCY_MS = 20
# Pooled audio embeddings per unit.
AUDIO_TOKENS_PER_UNIT = 10
# Unit audio rows in the prompt; the Thinker's Live branch overwrites them.
# <|tts_pad|> is never fed otherwise (the decoder always forbids it).
UNIT_AUDIO_PLACEHOLDER = "<|tts_pad|>"

_PROCESSOR_LOAD_LOCK = threading.Lock()

logger = init_logger(__name__)


@functools.cache
def _model_dir(model_path: str) -> Path:
    model_dir = Path(model_path)
    if model_dir.is_dir():
        return model_dir
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return Path(hf_api().snapshot_download(model_path, allow_patterns=[_REFERENCE_AUDIO]))


@functools.cache
def _reference_audio(model_path: str) -> np.ndarray:
    """The checkpoint's reference voice clip, mono at 16 kHz."""
    import soundfile

    samples, rate = soundfile.read(_model_dir(model_path) / _REFERENCE_AUDIO, dtype="float32", always_2d=True)
    return resample(samples.mean(axis=1), rate, _SAMPLE_RATE)


@functools.cache
def _hf_processor(model_path: str) -> Any:
    """The checkpoint's remote-code processor (streaming mel processor included)."""
    from transformers import AutoImageProcessor, AutoProcessor

    original_register = AutoImageProcessor.register

    def register(config_class, *args, **kwargs):
        # The checkpoint's auto_map already resolves this class; its legacy
        # string registration breaks on some Transformers versions.
        if config_class == "MiniCPMVImageProcessor":
            return None
        return original_register(config_class, *args, **kwargs)

    with _PROCESSOR_LOAD_LOCK:
        AutoImageProcessor.register = staticmethod(register)
        try:
            return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        finally:
            AutoImageProcessor.register = staticmethod(original_register)


class MiniCPMO45LiveSessionProcessor(LiveSessionProcessor):
    input_sample_rate: ClassVar[int] = _SAMPLE_RATE
    output_sample_rate: ClassVar[int] = 24_000
    # Official streaming_generate max_new_speak_tokens_per_chunk.
    max_unit_tokens: ClassVar[int] = 20

    @classmethod
    def preload(cls, model_path: str) -> None:
        _reference_audio(model_path)
        _hf_processor(model_path)

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        tok = context.raw_tokenizer

        def token_id(token: str) -> int:
            value = tok.convert_tokens_to_ids(token)
            if value is None or value == getattr(tok, "unk_token_id", None):
                raise ValueError(f"MiniCPM-o tokenizer has no {token}")
            return int(value)

        self.unit_id = token_id("<unit>")
        self.unit_end_id = token_id("</unit>")
        self.listen_id = token_id("<|listen|>")
        self.chunk_eos_id = token_id("<|chunk_eos|>")
        self.chunk_tts_eos_id = token_id("<|chunk_tts_eos|>")
        self.turn_eos_id = token_id("<|turn_eos|>")
        self.audio_row_id = token_id(UNIT_AUDIO_PLACEHOLDER)
        self.stop_ids = (self.listen_id, self.chunk_eos_id, self.chunk_tts_eos_id)
        self._placeholder_ids = list(tok.encode(_AUDIO_PLACEHOLDER, add_special_tokens=False))

        model_path = context.model_path
        self.reference_audio = _reference_audio(model_path)
        base = _hf_processor(model_path)
        reference_text = base.get_audio_placeholder(self.reference_audio.size, chunk_input=False)
        self._reference_tokens = len(tok.encode(reference_text, add_special_tokens=False))
        # One streaming mel state per session; set_streaming_mode builds a
        # fresh StreamingMelProcessorExact on this shallow copy.
        self._mel = copy.copy(base)
        self._mel.set_streaming_mode(
            mode="exact",
            chunk_ms=_CHUNK_MS,
            first_chunk_ms=_FIRST_CHUNK_MS,
            cnn_redundancy_ms=_CNN_REDUNDANCY_MS,
            enable_sliding_window=True,
            slide_trigger_seconds=30.0,
            slide_stride_seconds=10.0,
        )
        self._mel.reset_streaming()

        # Input stream: the official first chunk is the first 1000 ms of
        # audio left-padded to 1035 ms.
        pad = (_FIRST_CHUNK_MS - _CHUNK_MS) * _SAMPLE_RATE // 1000
        self._samples: list[np.ndarray] = [np.zeros(pad, dtype=np.float32)]
        self._buffered = pad
        self._pad = pad
        self._stream_start_ms: float | None = None
        self._consumed_input = 0
        self._unit_index = 0
        self._update_seq = 0
        self.session_key = ""
        # Text of units dropped by trimming, carried as "previous" context.
        self.previous_text = ""

    # ---- configuration -----------------------------------------------------------

    def resolve_voice(self, voice: Any) -> str | None:
        voice_id = super().resolve_voice(voice)
        if voice_id not in (None, "default"):
            raise LiveProtocolError(
                "invalid_value",
                "MiniCPM-o 4.5 serves its default voice only; use 'default'.",
                param="audio.output.voice",
            )
        return voice_id

    def configure(self, state: LiveSessionState) -> None:
        self.session_key = state.id

    def sampling_params_list(self, base: Any) -> list[Any]:
        params = super().sampling_params_list(base)
        if len(params) > 1 and hasattr(params[1], "clone"):
            # The Talker's per-unit codec budget comes with each handoff
            # (llm2tts); the deploy YAML's offline min_tokens would pad every
            # unit. Greedy codec sampling keeps sessions deterministic.
            params[1] = params[1].clone()
            params[1].min_tokens = 0
            params[1].temperature = 0.0
            params[1].top_p = 1.0
            params[1].top_k = -1
        return params

    def prefix_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        # A resumable request keeps its first chunk's max_tokens for every
        # later segment (vLLM updates only the sampling params), so the prefix
        # uses the unit budget and ends on its forced <|listen|> stop token.
        params = self.stage0_sampling_params(state, base)
        params.extra_args = {**params.extra_args, "minicpmo_live": {"session": self.session_key, "prefix": True}}
        return params

    def stage0_sampling_params(self, state: LiveSessionState, base: Any) -> Any:
        params = super().stage0_sampling_params(state, base)
        params.stop_token_ids = list(dict.fromkeys([*(params.stop_token_ids or []), *self.stop_ids]))
        params.max_tokens = self.max_unit_tokens
        params.temperature = 0.0
        # The Live logits processor applies the official windowed penalty.
        params.repetition_penalty = 1.0
        # The unit about to be submitted is the newest one in the history.
        units = state.units
        params.extra_args = {
            **(params.extra_args or {}),
            "minicpmo_live": {
                "session": self.session_key,
                "unit_index": int(units[-1].data["index"]) if units else 0,
                "max_unit_tokens": self.max_unit_tokens,
            },
        }
        return params

    # ---- input units -------------------------------------------------------------

    def push_audio(self, samples: np.ndarray, start_ms: float) -> None:
        if self._stream_start_ms is None:
            self._stream_start_ms = start_ms
        if samples.size:
            self._samples.append(np.asarray(samples, dtype=np.float32))
            self._buffered += samples.size

    def next_unit(self) -> NativeUnitItem | None:
        need = int(self._mel.get_streaming_chunk_size())
        if self._buffered < need:
            return None
        joined = np.concatenate(self._samples)
        chunk, rest = joined[:need], joined[need:]
        self._samples = [rest] if rest.size else []
        self._buffered = rest.size
        batch = self._mel.process_audio_streaming(chunk, reset=False, return_batch_feature=True)
        features = np.ascontiguousarray(np.asarray(batch["audio_features"], dtype=np.float32)[0])
        frames = int(np.asarray(batch["audio_feature_lens"][0]).reshape(-1)[0])
        index = self._unit_index
        self._unit_index += 1
        # Session time covered by the unit's real (unpadded) input samples.
        input_samples = need - (self._pad if index == 0 else 0)
        start_ms = (self._stream_start_ms or 0.0) + self._consumed_input * 1000.0 / _SAMPLE_RATE
        self._consumed_input += input_samples
        end_ms = (self._stream_start_ms or 0.0) + self._consumed_input * 1000.0 / _SAMPLE_RATE
        return NativeUnitItem(
            audio=chunk,
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            data={
                "index": index,
                "features": features.tobytes(),
                "mel_bins": int(features.shape[0]),
                "frames": frames,
                "prefix_extra_frames": 0 if index == 0 else 2,
                "suffix_extra_frames": 2,
            },
        )

    # ---- rendering -------------------------------------------------------------

    def _system_ids(self, state: LiveSessionState) -> list[int]:
        developer: list[str] = []
        history: list[str] = []
        for item in state.history:
            if isinstance(item, SystemItem):
                developer.append(item.text)
            elif isinstance(item, UserTextItem):
                history.append(f"user: {item.text}")
            elif isinstance(item, AssistantItem):
                history.append(f"assistant: {item.text}")
        prompt = "\n\n".join(developer) or _DEFAULT_SYSTEM_PROMPT
        previous = " ".join(part for part in (*history, self.previous_text) if part)
        encode = self.context.raw_tokenizer.encode
        prefix = f"<|im_start|>system\n{prompt}\n<|audio_start|>"
        suffix = (_PREVIOUS_MARKER + previous if previous else "") + "<|audio_end|><|im_end|>"
        return [
            *encode(prefix, add_special_tokens=False),
            *self._placeholder_ids,
            *encode(suffix, add_special_tokens=False),
        ]

    def _unit_rows(self) -> list[int]:
        return [self.unit_id, *([self.audio_row_id] * AUDIO_TOKENS_PER_UNIT)]

    def _audio_buffer(self, units: list[NativeUnitItem]) -> dict[str, Any]:
        self._update_seq += 1
        return {
            "minicpmo_live_audio": {
                "seq": self._update_seq,
                "placeholder_id": self.audio_row_id,
                "units": [dict(unit.data) for unit in units],
            }
        }

    def render_prefix(self, state: LiveSessionState) -> dict[str, Any]:
        """The system prompt and reference voice. Its one sampled token (a
        forced <|listen|>) is never fed: the first unit opens with <unit>."""
        return {
            "prompt_token_ids": self._system_ids(state),
            "multi_modal_data": {"audio": (self.reference_audio, _SAMPLE_RATE)},
            "mm_processor_kwargs": {"live_duplex_unit": True},
            # The orchestrator hands the request's first prompt to llm2tts on
            # every stage-0 output: this marks the request as Live and carries
            # the unit grammar's token ids. A top-level key, not a
            # model_intermediate_buffer entry, so no engine receives it (the
            # async-chunk prewarm copies the first prompt's buffer downstream).
            "minicpmo_live": {
                "listen": self.listen_id,
                "chunk_eos": self.chunk_eos_id,
                "chunk_tts_eos": self.chunk_tts_eos_id,
                "turn_eos": self.turn_eos_id,
            },
        }

    def render_unit(
        self,
        state: LiveSessionState,
        unit: NativeUnitItem,
        *,
        first: bool,
        last_sampled_token: int | None,
    ) -> dict[str, Any]:
        if first:
            ids = self._unit_rows()
        else:
            # The previous segment's terminator was sampled but never
            # computed; feed it, close the unit, open the next one.
            head = [last_sampled_token] if last_sampled_token is not None else []
            ids = [*head, self.unit_end_id, *self._unit_rows()]
        return {"prompt_token_ids": ids, "model_intermediate_buffer": self._audio_buffer([unit])}

    def render_resume(self, state: LiveSessionState) -> dict[str, Any]:
        units = state.units
        ids: list[int] = []
        for index, unit in enumerate(units):
            ids += self._unit_rows()
            if index < len(units) - 1:
                ids += [*unit.token_ids, self.unit_end_id]
        return {"prompt_token_ids": ids, "model_intermediate_buffer": self._audio_buffer(units)}

    # ---- context -----------------------------------------------------------------

    def prefix_tokens(self, state: LiveSessionState) -> int:
        return len(self._system_ids(state)) - len(self._placeholder_ids) + self._reference_tokens

    def unit_tokens(self, unit: NativeUnitItem) -> int:
        return 2 + AUDIO_TOKENS_PER_UNIT + len(unit.token_ids)

    def on_units_dropped(self, state: LiveSessionState, units: list[NativeUnitItem]) -> None:
        text = "".join(self.decode(unit.token_ids) for unit in units).strip()
        if text:
            previous = self.context.raw_tokenizer.encode(
                f"{self.previous_text} {text}".strip(), add_special_tokens=False
            )
            self.previous_text = self.decode(previous[-_PREVIOUS_MAX_TOKENS:]).strip()
        logger.debug(
            "Live session %s: %d dropped units carried as %d chars of previous text (%d chars total)",
            state.id,
            len(units),
            len(text),
            len(self.previous_text),
        )

    # ---- output ------------------------------------------------------------------

    def spoken_text(self, token_ids: list[int], text: str) -> str:
        return self.decode(token_ids)
