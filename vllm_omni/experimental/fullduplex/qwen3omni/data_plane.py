# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Projection of Qwen3-Omni stage outputs into Realtime data-plane events.

Implements ``RuntimeDataPlane``
(``vllm_omni/experimental/fullduplex/openai/runtime_adapter.py:93-104``).

Structural difference from MiniCPM-o 4.5, and the reason this is not a port
of ``minicpmo45/data_plane.py``: MiniCPM's ``project_output`` assumes ONE
stage output carries text, audio, and the turn decision together. Qwen3-Omni
splits these across independently-scheduled stages -- text from the thinker
(stage 0), audio from code2wav (stage 2) -- arriving at different times. This
module therefore correlates per-stage outputs rather than unpacking a single
fused one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

logger = init_logger(__name__)

#: ``(audio, sample_rate_hz, response_format, speed) -> base64 | None``
EncodeAudio = Callable[[object, int, str, float | None], str | None]

#: Qwen3-Omni's code2wav vocoder output rate (differs from the 16 kHz input).
OUTPUT_SAMPLE_RATE_HZ = 24000

_THINKER_STAGE_ID = 0
_CODE2WAV_STAGE_ID = 2


@dataclass
class Qwen3OmniDataPlaneContext:
    """Serving state needed to project one Qwen3-Omni data-plane output.

    Constructed with all eight keyword arguments by
    ``runtime_bridge.py:752-761``; the generic layer never reads it back, so
    the shape is private to this package.
    """

    epoch: int = 0
    turn_id: int = 0
    active_response_turn_id: int | None = None
    active_response_id: str | None = None
    auto_responds: bool = False
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ()


@dataclass
class _RequestState:
    """Per-request correlation state across the three stages."""

    terminal: bool = False
    #: Cumulative text already sent, for delta computation.
    sent_text: str = ""
    #: Samples of the cumulative waveform already sent to the client.
    audio_offset: int = 0
    seen_stage_ids: set[int] = field(default_factory=set)


def _field(output: object, name: str, default: object = None) -> object:
    """Read a field from a stage output.

    ``collect_registered_outputs`` yields ``OmniRequestOutput`` *objects*
    (``request_client.py:183``), not mappings. Reading them as dicts silently
    discards every output, so both shapes are accepted here.
    """
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _audio_length(audio: object) -> int:
    """Sample count of a waveform, tolerating list / ndarray / tensor.

    Code2Wav returns a tensor shaped ``[1, samples]``, so ``len()`` yields the
    batch dimension -- always 1 -- rather than the sample count. Prefer the
    last axis.
    """
    shape = getattr(audio, "shape", None)
    if shape is not None and len(shape):
        return int(shape[-1])
    try:
        return len(audio)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _audio_tail(audio: object, offset: int) -> object:
    """Samples after ``offset``, slicing the sample axis, not the batch axis."""
    if offset <= 0:
        return audio
    shape = getattr(audio, "shape", None)
    if shape is not None and len(shape) > 1:
        return audio[..., offset:]  # type: ignore[index]
    return audio[offset:]  # type: ignore[index]


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class Qwen3OmniDataPlaneSession:
    """Projects Qwen3-Omni stage outputs into Realtime event dicts."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestState] = {}

    # ---- RuntimeDataPlane protocol ----------------------------------------

    def begin_request(self, request_id: str) -> None:
        # Preserve both cursors across turns.
        #
        # Duplex reuses one request id for the whole session, and *both*
        # downstream texts accumulate across it: Code2Wav's waveform and the
        # thinker's generated text. Resetting the audio cursor made every turn
        # resend all previously spoken audio; resetting `sent_text` did the
        # same to the transcript, so each turn's first delta replayed the whole
        # conversation. That is not cosmetic -- a client watching the
        # transcript for `<tool_call>` re-runs every tool it has ever seen,
        # which is an infinite loop, since dispatching one produces the next
        # turn.
        existing = self._requests.get(request_id)
        state = _RequestState()
        if existing is not None:
            state.audio_offset = existing.audio_offset
            state.sent_text = existing.sent_text
            state.terminal = False
        self._requests[request_id] = state

    def is_terminal(self, request_id: str | None) -> bool:
        if request_id is None:
            return False
        state = self._requests.get(request_id)
        return bool(state and state.terminal)

    def mark_terminal(self, request_id: str) -> None:
        state = self._requests.get(request_id)
        if state is not None:
            state.terminal = True

    def close_stream(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self._requests.pop(active_request_id, None)
        # Request ids are session-prefixed; drop anything still outstanding.
        for request_id in [key for key in self._requests if key.startswith(session_id)]:
            self._requests.pop(request_id, None)

    def project(
        self,
        result: object,
        *,
        context: object | None = None,
    ) -> Iterable[dict[str, object]]:
        """Fan one engine result out into zero or more Realtime events."""
        if not isinstance(result, Mapping):
            return
        ctx = context if isinstance(context, Qwen3OmniDataPlaneContext) else Qwen3OmniDataPlaneContext()
        outputs = result.get("data_plane_outputs")
        logger.info(
            "[qwen3omni-dp] project: result_keys=%s outputs=%s types=%s",
            sorted(result.keys())[:8],
            None if not isinstance(outputs, list) else len(outputs),
            None if not isinstance(outputs, list) else [type(o).__name__ for o in outputs[:3]],
        )
        if not isinstance(outputs, list):
            return
        for output in outputs:
            if output is not None:
                yield from self._project_output(output, ctx)

    # ---- projection -------------------------------------------------------

    def _project_output(
        self,
        output: object,
        ctx: Qwen3OmniDataPlaneContext,
    ) -> Iterable[dict[str, object]]:
        request_id = _field(output, "request_id")
        request_id = request_id if isinstance(request_id, str) else None
        state = self._requests.get(request_id) if request_id else None
        if state is not None and state.terminal:
            return

        stage_id = _coerce_int(_field(output, "stage_id"))
        logger.info(
            "[qwen3omni-dp] output: stage=%s finished=%s rid=%s text=%r",
            stage_id,
            _field(output, "finished"),
            request_id,
            self._cumulative_text(output)[:60],
        )
        metadata = _field(output, "multimodal_output")
        metadata = metadata if isinstance(metadata, Mapping) else {}

        # Drop outputs from a superseded epoch/turn. The fence travels with
        # the append (runtime.py build_duplex_data_plane_prompt) and comes
        # back on the output metadata.
        output_epoch = _coerce_int(metadata.get("duplex_epoch"))
        if output_epoch is not None and output_epoch != ctx.epoch:
            return
        model_turn_id = _coerce_int(metadata.get("duplex_turn_id"))

        if state is not None and stage_id is not None:
            state.seen_stage_ids.add(stage_id)

        finished = bool(_field(output, "finished"))

        if stage_id == _THINKER_STAGE_ID:
            event = self._project_thinker(output, state, model_turn_id, finished)
            if event is not None:
                yield event
            return

        if stage_id == _CODE2WAV_STAGE_ID:
            event = self._project_code2wav(output, state, ctx, model_turn_id, finished)
            if event is not None:
                yield event
            return

        # Stage 1 (talker) emits codec tokens consumed by code2wav, not
        # client-visible content. Announce the handoff so the bridge knows a
        # response is coming without reserving one yet
        # (runtime_bridge.py:818-824).
        if stage_id is not None and not finished:
            yield {
                "data_plane_request_id": request_id,
                "requires_stage_handoff": True,
                "stage_role": "talker",
                "model_turn_id": model_turn_id,
            }

    def _project_thinker(
        self,
        output: object,
        state: _RequestState | None,
        model_turn_id: int | None,
        finished: bool,
    ) -> dict[str, object] | None:
        """Emit the incremental text delta from the thinker."""
        text = self._cumulative_text(output)
        delta = text
        if state is not None:
            if text.startswith(state.sent_text):
                delta = text[len(state.sent_text) :]
            state.sent_text = text
        if not delta and not finished:
            return None
        return {
            "data_plane_request_id": _field(output, "request_id"),
            "stage_role": "llm",
            "text": delta,
            "end_of_turn": finished,
            "model_turn_id": model_turn_id,
            "runtime_impl": "qwen3_omni_native_duplex",
            "owned_runtime": True,
        }

    def _project_code2wav(
        self,
        output: object,
        state: _RequestState | None,
        ctx: Qwen3OmniDataPlaneContext,
        model_turn_id: int | None,
        finished: bool,
    ) -> dict[str, object] | None:
        """Emit newly-produced audio from the vocoder stage."""
        if "audio" not in ctx.modalities:
            return None
        metadata = _field(output, "multimodal_output")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        audio = metadata.get("audio")
        if audio is None:
            return None
        sample_rate_hz = _coerce_int(metadata.get("sr")) or OUTPUT_SAMPLE_RATE_HZ

        # Code2Wav emits the waveform cumulatively: every output carries all
        # audio generated so far, not just the newest frames. Measured sizes
        # for one reply: 14250, 110250, 206250, 217770 bytes. Forwarding each
        # in full makes the client replay the reply from the start over and
        # over -- heard as "I. I. I. I'm doing great".
        #
        # Slice the raw samples, not the encoded string: base64 packs 3 bytes
        # into 4 characters, so cutting the encoded form at an arbitrary
        # offset corrupts it.
        if state is not None:
            total = _audio_length(audio)
            if total <= state.audio_offset:
                return None
            audio = _audio_tail(audio, state.audio_offset)
            state.audio_offset = total

        encoded = self._encode_audio(audio, sample_rate_hz, ctx.response_format, ctx.speed)
        if not encoded:
            return None
        return {
            "data_plane_request_id": _field(output, "request_id"),
            "stage_role": "tts",
            "audio_data": encoded,
            "audio_format": ctx.response_format,
            "sample_rate_hz": sample_rate_hz,
            "end_of_turn": finished,
            "model_turn_id": model_turn_id,
            "runtime_impl": "qwen3_omni_native_duplex",
            "owned_runtime": True,
        }

    @staticmethod
    def _cumulative_text(output: object) -> str:
        # Stage 0 hands the orchestrator a raw vllm ``RequestOutput`` whose
        # completions live on ``.outputs``; wrapped stages nest it under
        # ``.request_output``. Reading only the wrapped form yields empty text
        # for every thinker output.
        request_output = _field(output, "request_output") or output
        completions = getattr(request_output, "outputs", None)
        if completions:
            text = getattr(completions[0], "text", None)
            if isinstance(text, str):
                return text
        text = _field(output, "text")
        return text if isinstance(text, str) else ""


__all__ = [
    "OUTPUT_SAMPLE_RATE_HZ",
    "Qwen3OmniDataPlaneContext",
    "Qwen3OmniDataPlaneSession",
    "Qwen3OmniDuplexPolicy",
]
