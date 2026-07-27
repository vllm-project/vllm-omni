from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.engine.contracts import (
    duplex_resource_request_belongs_to_session,
)
from vllm_omni.experimental.fullduplex.output import get_duplex_output_decision

logger = init_logger(__name__)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


@dataclass(frozen=True, slots=True)
class MiniCPMO45DataPlaneContext:
    """Serving state needed to project one MiniCPM data-plane output."""

    epoch: int = 0
    turn_id: int = 0
    active_response_turn_id: int | None = None
    active_response_id: str | None = None
    auto_responds: bool = False
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ()


@dataclass(slots=True)
class _TurnState:
    sent_segment_text: str = ""
    has_text: bool = False
    tts_eos_done: bool = False
    turn_eos_done: bool = False


@dataclass(slots=True)
class _RequestState:
    audio_offset: int = 0
    uses_segment_text_metadata: bool = False
    pending_audio_without_text: list[dict[str, object]] = field(default_factory=list)
    terminal: bool = False
    turns: dict[int | None, _TurnState] = field(default_factory=dict)

    def turn(self, turn_id: int | None) -> _TurnState:
        return self.turns.setdefault(turn_id, _TurnState())


class MiniCPMO45DataPlaneSession:
    """MiniCPM output projector and request/turn cursor owner.

    The scheduler request owns cumulative Stage1 audio, while model turns own
    transcript and terminal cursors. Keeping both levels in one object avoids
    leaking MiniCPM-specific state into the generic Realtime handler.
    """

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestState] = {}

    def begin_request(self, request_id: str) -> None:
        state = self._requests.setdefault(request_id, _RequestState())
        state.terminal = False
        # Preserve turn-scoped cursors for resumable requests. The legacy path
        # only reset request-scoped terminal markers on append.
        request_turn = state.turns.get(None)
        if request_turn is not None:
            request_turn.tts_eos_done = False
            request_turn.turn_eos_done = False

    def is_terminal(self, request_id: str | None) -> bool:
        if request_id is None:
            return False
        state = self._requests.get(request_id)
        return state is not None and state.terminal

    def mark_terminal(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestState()).terminal = True

    def close_stream(self, request_id: str) -> None:
        """Release non-resumable request cursors after its drain finishes."""
        state = self._requests.get(request_id)
        if state is None:
            return
        state.audio_offset = 0
        state.turns.pop(None, None)

    def close_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self.close_request(active_request_id)
        for request_id in list(self._requests):
            if self.request_belongs_to_session(request_id, session_id):
                self.close_request(request_id)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._requests

    def has_pending_audio(self, request_id: str) -> bool:
        state = self._requests.get(request_id)
        return bool(state and state.pending_audio_without_text)

    def audio_offset(self, request_id: str | None) -> int | None:
        if request_id is None:
            return None
        state = self._requests.get(request_id)
        return state.audio_offset if state is not None and state.audio_offset > 0 else None

    def project(
        self,
        result: object,
        *,
        context: MiniCPMO45DataPlaneContext | None = None,
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
        context: MiniCPMO45DataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        context = context or MiniCPMO45DataPlaneContext()
        stage_metrics = _output_stage_metrics(output)

        def runtime_result(**values: object) -> dict[str, object]:
            result = _runtime_result(**values)
            if stage_metrics is not None:
                result["stage_metrics"] = stage_metrics
            return result

        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = None
        request_state = self._requests.setdefault(request_id, _RequestState()) if request_id is not None else None

        outputs = getattr(output, "outputs", None)
        completion = outputs[0] if isinstance(outputs, list) and outputs else None
        text = getattr(completion, "text", "") if completion is not None else ""
        direct_decision = get_duplex_output_decision(output)
        direct_metadata = getattr(direct_decision, "metadata", None)
        if isinstance(direct_metadata, Mapping):
            mm_output = direct_metadata
        else:
            mm_output = getattr(output, "multimodal_output", None)
        if not isinstance(mm_output, Mapping):
            mm_output = getattr(completion, "multimodal_output", {}) if completion is not None else {}
        if not mm_output:
            inner_output = getattr(output, "request_output", None)
            if inner_output is not None and inner_output is not output:
                inner_mm_output = getattr(inner_output, "multimodal_output", None)
                if isinstance(inner_mm_output, Mapping) and inner_mm_output:
                    mm_output = inner_mm_output
                else:
                    inner_outputs = getattr(inner_output, "outputs", None)
                    inner_completion = inner_outputs[0] if isinstance(inner_outputs, list) and inner_outputs else None
                    inner_mm_output = (
                        getattr(inner_completion, "multimodal_output", None) if inner_completion is not None else None
                    )
                    if completion is None and inner_completion is not None:
                        completion = inner_completion
                        text = getattr(inner_completion, "text", "") or text
                    if isinstance(inner_mm_output, Mapping):
                        mm_output = inner_mm_output
        mm_output = dict(mm_output) if isinstance(mm_output, Mapping) else {}

        output_turn_id = output_turn_id_from_metadata(mm_output)
        output_epoch = output_epoch_from_metadata(mm_output)
        expected_turn_id = context.active_response_turn_id
        stale_turn = expected_turn_id is not None and output_turn_id is not None and output_turn_id < expected_turn_id
        if expected_turn_id is None and output_turn_id is not None:
            stale_turn = output_turn_id < context.turn_id
        stale_epoch = output_epoch is not None and output_epoch != context.epoch
        if context.auto_responds and (stale_turn or stale_epoch):
            return

        mm_text = _llm_output_text(mm_output)
        if mm_text:
            text = mm_text
            if request_state is not None:
                request_state.uses_segment_text_metadata = True
        elif request_state is not None and request_state.uses_segment_text_metadata:
            text = ""

        finished = bool(getattr(output, "finished", False))
        tts_is_last_chunk = _bool_metadata(mm_output, ("tts_is_last_chunk",), default=False)
        token_ids = _completion_token_ids(completion)
        native_decision = _native_decision(completion, mm_output, token_ids=token_ids, finished=finished)
        if native_decision == "listen":
            listen_result = runtime_result(
                stage_role="llm",
                is_listen=True,
                model_listen=True,
                listen_source="model_listen",
                data_plane_request_id=request_id,
                end_of_turn=False,
            )
            if output_turn_id is not None:
                listen_result["model_turn_id"] = output_turn_id
            yield listen_result
            return

        raw_audio = next((mm_output[key] for key in ("audio", "model_outputs", "latent") if key in mm_output), None)
        raw_audio_samples = _audio_num_samples(raw_audio)
        offset_before = self.audio_offset(request_id)
        audio_chunks = list(
            self._encode_audio_chunks_with_duration(
                mm_output,
                request_id=request_id,
                response_format=context.response_format,
                speed=context.speed,
            )
        )
        stage_turn_end = _bool_metadata(mm_output, ("turn_end", "end_of_turn"), default=False)
        terminal_turn_state = request_state.turn(output_turn_id) if request_state is not None else None
        stage_tts_eos = (
            context.auto_responds
            and 151645 in token_ids
            and not audio_chunks
            and raw_audio_samples is not None
            and (raw_audio_samples == 0 or raw_audio_samples == offset_before)
            and (terminal_turn_state is None or not terminal_turn_state.tts_eos_done)
        )
        tts_segment_end = bool(tts_is_last_chunk or stage_tts_eos) and (
            terminal_turn_state is None or not terminal_turn_state.tts_eos_done
        )
        if tts_segment_end and terminal_turn_state is not None:
            terminal_turn_state.tts_eos_done = True
        stage_turn_end_new = bool(stage_turn_end) and (
            terminal_turn_state is None or not terminal_turn_state.turn_eos_done
        )
        if stage_turn_end_new and terminal_turn_state is not None:
            terminal_turn_state.turn_eos_done = True
        unit_end_of_turn = stage_turn_end_new or (finished and not context.auto_responds)

        text_turn_id = output_turn_id if output_turn_id is not None else context.turn_id
        text_turn_state = request_state.turn(text_turn_id) if request_state is not None else None
        if audio_chunks:
            delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id)
            last_idx = len(audio_chunks) - 1
            sample_rate_hz = _sample_rate_hz(mm_output)
            audio_text_marks = _audio_text_marks(mm_output)
            fallback_marks = _fallback_audio_text_marks(audio_chunks, delta_text)
            audio_results: list[dict[str, object]] = []
            for idx, (audio, duration_ms) in enumerate(audio_chunks):
                native_result = runtime_result(
                    stage_role="tts",
                    is_listen=False,
                    data_plane_request_id=request_id,
                    text=delta_text if idx == 0 else "",
                    audio_data=audio,
                    audio_format=context.response_format,
                    audio_duration_ms=duration_ms,
                    audio_text_mark=idx == last_idx,
                    sample_rate_hz=sample_rate_hz,
                    end_of_turn=unit_end_of_turn and idx == last_idx,
                    abort_data_plane_request=tts_segment_end and idx == last_idx,
                )
                if output_turn_id is not None:
                    native_result["model_turn_id"] = output_turn_id
                if audio_text_marks and idx == last_idx:
                    native_result["audio_text_marks"] = audio_text_marks
                    native_result["audio_text_marks_are_cumulative"] = True
                elif idx < len(fallback_marks) and fallback_marks[idx]:
                    native_result["audio_text_marks"] = fallback_marks[idx]
                    native_result["audio_text_marks_are_cumulative"] = True
                audio_results.append(native_result)

            if context.auto_responds:
                if delta_text and text_turn_state is not None:
                    text_turn_state.has_text = True
                future_model_turn = (
                    context.active_response_turn_id is not None
                    and output_turn_id is not None
                    and output_turn_id > context.active_response_turn_id
                )
                response_turn_bound = context.active_response_id is not None and (
                    output_turn_id is None
                    or context.active_response_turn_id is None
                    or context.active_response_turn_id == output_turn_id
                )
                turn_has_text = text_turn_state is not None and text_turn_state.has_text
                if not future_model_turn and not response_turn_bound and not turn_has_text:
                    if request_state is not None:
                        if tts_segment_end:
                            request_state.pending_audio_without_text.clear()
                        else:
                            request_state.pending_audio_without_text.extend(audio_results)
                    if tts_segment_end:
                        terminal_result = dict(audio_results[-1])
                        terminal_result.update(
                            audio_data="",
                            audio_duration_ms=0,
                            audio_text_mark=False,
                            end_of_turn=True,
                        )
                        yield terminal_result
                    return
                if request_state is not None and request_state.pending_audio_without_text:
                    pending = request_state.pending_audio_without_text
                    request_state.pending_audio_without_text = []
                    yield from pending
            yield from audio_results
            return

        if context.auto_responds and request_state is not None and isinstance(text, str) and text:
            pending_audio = request_state.pending_audio_without_text
            request_state.pending_audio_without_text = []
            if pending_audio:
                delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id)
                if delta_text:
                    pending_audio[0]["text"] = delta_text
                    if text_turn_state is not None:
                        text_turn_state.has_text = True
                    total_duration_ms = sum(
                        max(0, int(result.get("audio_duration_ms", 0) or 0)) for result in pending_audio
                    )
                    if total_duration_ms > 0 and not pending_audio[-1].get("audio_text_marks"):
                        pending_audio[-1]["audio_text_marks"] = [
                            {"text_chars": len(delta_text), "audio_end_ms": total_duration_ms}
                        ]
                        pending_audio[-1]["audio_text_marks_are_cumulative"] = True
                    if unit_end_of_turn:
                        pending_audio[-1]["end_of_turn"] = True
                    if tts_segment_end:
                        pending_audio[-1]["abort_data_plane_request"] = True
                    yield from pending_audio
                    return
                request_state.pending_audio_without_text = pending_audio

        if tts_segment_end:
            if unit_end_of_turn and request_state is not None and request_state.pending_audio_without_text:
                request_state.pending_audio_without_text[-1]["end_of_turn"] = True
                request_state.pending_audio_without_text[-1]["abort_data_plane_request"] = True
            terminal_result = runtime_result(
                stage_role="tts",
                is_listen=False,
                data_plane_request_id=request_id,
                text="",
                audio_data="",
                audio_format=context.response_format,
                audio_text_mark=False,
                end_of_turn=unit_end_of_turn,
                abort_data_plane_request=True,
            )
            if output_turn_id is not None:
                terminal_result["model_turn_id"] = output_turn_id
            yield terminal_result
            return

        if context.active_response_id is not None and unit_end_of_turn:
            terminal_result = runtime_result(
                stage_role="tts",
                is_listen=False,
                data_plane_request_id=request_id,
                text="",
                audio_data="",
                audio_format=context.response_format,
                audio_text_mark=False,
                end_of_turn=True,
            )
            if output_turn_id is not None:
                terminal_result["model_turn_id"] = output_turn_id
            yield terminal_result
            return

        if request_id is not None and context.auto_responds and unit_end_of_turn and context.active_response_id is None:
            # The legacy projector only removed the request-scoped pending
            # buffer here when no explicit model-turn key was present.
            if request_state is not None and output_turn_id is None:
                request_state.pending_audio_without_text.clear()
            terminal_result = runtime_result(
                stage_role="tts",
                is_listen=False,
                data_plane_request_id=request_id,
                text="",
                audio_data="",
                audio_format=context.response_format,
                audio_text_mark=False,
                end_of_turn=True,
                abort_data_plane_request=True,
            )
            if output_turn_id is not None:
                terminal_result["model_turn_id"] = output_turn_id
            yield terminal_result
            return

        if (
            finished
            and context.auto_responds
            and context.active_response_id is not None
            and request_id is not None
            and not unit_end_of_turn
        ):
            listen_result = runtime_result(
                stage_role="llm",
                is_listen=True,
                model_listen=False,
                listen_source="auto_response_segment_complete",
                reason="auto_response_segment_complete",
                data_plane_request_id=request_id,
                end_of_turn=False,
            )
            if output_turn_id is not None:
                listen_result["model_turn_id"] = output_turn_id
            yield listen_result
            return

        if not text:
            if context.auto_responds:
                return
            if finished:
                listen_result = runtime_result(
                    stage_role="llm",
                    is_listen=True,
                    model_listen=False,
                    listen_source="data_plane_finished_without_output",
                    reason="data_plane_finished_without_output",
                    data_plane_request_id=request_id,
                    end_of_turn=False,
                )
                if output_turn_id is not None:
                    listen_result["model_turn_id"] = output_turn_id
                yield listen_result
            return
        if context.auto_responds:
            return
        if "audio" in context.modalities:
            yield runtime_result(
                stage_role="tts",
                error_code="runtime_data_plane_text_without_audio",
                error="MiniCPM-o native duplex data-plane produced text without audio.",
                data_plane_request_id=request_id,
            )
            return
        yield runtime_result(
            stage_role="llm",
            is_listen=False,
            data_plane_request_id=request_id,
            text=text if isinstance(text, str) else "",
            audio_data="",
            end_of_turn=unit_end_of_turn,
        )

    def segment_text_delta(self, request_id: str | None, text: object, *, turn_id: int | None = None) -> str:
        if not isinstance(text, str) or not text:
            return ""
        if request_id is None:
            return text
        turn_state = self._requests.setdefault(request_id, _RequestState()).turn(turn_id)
        sent_text = turn_state.sent_segment_text
        if not sent_text:
            delta_text = text
        elif text == sent_text:
            delta_text = ""
        elif text.startswith(sent_text):
            delta_text = text[len(sent_text) :]
        else:
            # The producer sends text for the current thinker segment, not a
            # cumulative turn snapshot. Distinct adjacent segments need not
            # have a prefix relationship and must both remain visible.
            delta_text = text
        turn_state.sent_segment_text = text
        return delta_text

    def slice_cumulative_audio(self, request_id: str | None, audio_data: object) -> object:
        if request_id is None:
            return audio_data
        num_samples = _audio_num_samples(audio_data)
        if num_samples is None or num_samples <= 0:
            return audio_data
        state = self._requests.setdefault(request_id, _RequestState())
        prev_samples = state.audio_offset
        if prev_samples <= 0:
            state.audio_offset = num_samples
            return audio_data
        if num_samples == prev_samples:
            return None
        if num_samples < prev_samples:
            state.audio_offset = num_samples
            return audio_data
        state.audio_offset = num_samples
        try:
            import torch

            if isinstance(audio_data, torch.Tensor):
                return audio_data.reshape(-1)[prev_samples:].contiguous()
            return np.asarray(audio_data, dtype=np.float32).reshape(-1)[prev_samples:]
        except Exception:
            logger.exception("Failed to slice cumulative duplex audio output")
            return audio_data

    def _encode_audio_chunks_with_duration(
        self,
        mm_output: dict[str, object],
        *,
        request_id: str | None,
        response_format: str,
        speed: float | None,
    ) -> Iterator[tuple[str, int]]:
        sample_rate_hz = _sample_rate_hz(mm_output)
        audio_data = next((mm_output[key] for key in ("audio", "model_outputs", "latent") if key in mm_output), None)
        if isinstance(audio_data, list):
            for value in audio_data:
                encoded = self._encode_audio(value, sample_rate_hz, response_format, speed)
                if encoded:
                    duration_ms = int((_audio_num_samples(value) or 0) * 1000 / max(1, sample_rate_hz))
                    yield encoded, duration_ms
            return
        sliced = self.slice_cumulative_audio(request_id, audio_data)
        encoded = self._encode_audio(sliced, sample_rate_hz, response_format, speed)
        if encoded:
            duration_ms = int((_audio_num_samples(sliced) or 0) * 1000 / max(1, sample_rate_hz))
            yield encoded, duration_ms

    @staticmethod
    def request_belongs_to_session(request_id: str, session_id: str) -> bool:
        return (
            duplex_resource_request_belongs_to_session(request_id, session_id)
            or request_id.startswith(f"duplex-{session_id}-")
            or request_id.startswith(f"chatcmpl-duplex-{session_id}-")
        )


def coerce_int(value: object) -> int | None:
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1)
            if value.numel() == 0:
                return None
            value = value[0].item()
        except Exception:
            return None
    elif hasattr(value, "reshape") and hasattr(value, "size"):
        try:
            value = value.reshape(-1)
            if int(value.size) == 0:
                return None
            item = value[0]
            value = item.item() if hasattr(item, "item") else item
        except Exception:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def payload_turn_id(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    return coerce_int(payload.get("duplex_turn_id"))


def output_turn_id_from_metadata(mm_output: dict[str, object]) -> int | None:
    return _first_metadata_int(mm_output, "duplex_turn_id", "turn_id")


def output_epoch_from_metadata(mm_output: dict[str, object]) -> int | None:
    return _first_metadata_int(mm_output, "duplex_epoch", "epoch")


def _first_metadata_int(mm_output: dict[str, object], *names: str) -> int | None:
    candidates: list[object] = []
    meta = mm_output.get("meta")
    for name in names:
        candidates.extend((mm_output.get(name), mm_output.get(f"meta.{name}")))
        if isinstance(meta, dict):
            candidates.append(meta.get(name))
    for value in candidates:
        result = coerce_int(value)
        if result is not None:
            return result
    return None


def _runtime_result(**values: object) -> dict[str, object]:
    return {
        "supported": True,
        **values,
        "uses_model_runner_scheduler": True,
        "runner_kv_backed": True,
        "runtime_impl": "scheduler_data_plane",
        "owned_runtime": False,
    }


def _output_stage_metrics(output: object) -> dict[str, dict[str, object]] | None:
    metrics = getattr(output, "metrics", None)
    if not isinstance(metrics, Mapping):
        return None
    stage_metrics = metrics.get("stage_metrics")
    if not isinstance(stage_metrics, Mapping):
        return None
    snapshot = {
        str(stage_id): dict(values) for stage_id, values in stage_metrics.items() if isinstance(values, Mapping)
    }
    return snapshot or None


def _native_decision(
    completion: object,
    mm_output: dict[str, object],
    *,
    token_ids: list[int],
    finished: bool,
) -> str | None:
    if not finished:
        return None
    if mm_output.get("duplex_native_decision") == "listen" or mm_output.get("model_listen") is True:
        return "listen"
    listen_id = _special_token_ids(mm_output).get("listen_token_id")
    if listen_id is None:
        return None
    stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
    if coerce_int(stop_reason) == listen_id:
        return "listen"
    return "listen" if token_ids and token_ids[-1] == listen_id else None


def _special_token_ids(mm_output: dict[str, object]) -> dict[str, int]:
    sources: list[object] = []
    raw_special = mm_output.get("special_token_ids")
    if isinstance(raw_special, dict):
        sources.append(raw_special)
    raw_meta = mm_output.get("meta")
    if isinstance(raw_meta, dict):
        sources.append(raw_meta)
    sources.append(
        {
            key.removeprefix("meta."): value
            for key, value in mm_output.items()
            if isinstance(key, str) and key.startswith("meta.")
        }
    )
    out: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if not isinstance(key, str):
                continue
            token_id = coerce_int(value)
            if token_id is not None and token_id >= 0:
                out[key] = token_id
    return out


def _completion_token_ids(completion: object) -> list[int]:
    if completion is None:
        return []
    for candidate in (
        getattr(completion, "token_ids", None),
        getattr(completion, "cumulative_token_ids", None),
    ):
        token_ids = _coerce_int_list(candidate)
        if token_ids:
            return token_ids
    return []


def _coerce_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1).tolist()
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [item for raw in value if (item := coerce_int(raw)) is not None]


def _audio_num_samples(audio_data: object) -> int | None:
    try:
        import torch

        if isinstance(audio_data, torch.Tensor):
            return int(audio_data.numel())
        return int(np.asarray(audio_data, dtype=np.float32).size)
    except Exception:
        return None


def _bool_metadata(
    mm_output: dict[str, object],
    names: tuple[str, ...],
    *,
    default: bool,
) -> bool:
    def coerce(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return bool(value.reshape(-1)[-1].item()) if value.numel() else None
        except Exception:
            pass
        if isinstance(value, np.ndarray):
            return bool(value.reshape(-1)[-1].item()) if value.size else None
        if isinstance(value, np.generic):
            return bool(value.item())
        if isinstance(value, (list, tuple)):
            return coerce(value[-1]) if value else None
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    meta = mm_output.get("meta")
    for name in names:
        for key in (name, f"meta.{name}"):
            result = coerce(mm_output.get(key))
            if result is not None:
                return result
        if isinstance(meta, dict):
            result = coerce(meta.get(name))
            if result is not None:
                return result
    return default


def _sample_rate_hz(mm_output: dict[str, object]) -> int:
    sr_raw = mm_output.get("sr")
    if sr_raw is None:
        sr_raw = mm_output.get("sample_rate_hz", mm_output.get("sample_rate"))
    meta = mm_output.get("meta")
    if sr_raw is None and isinstance(meta, dict):
        sr_raw = meta.get("sr") or meta.get("sample_rate_hz") or meta.get("sample_rate")
    if sr_raw is None:
        sr_raw = mm_output.get("meta.sr") or mm_output.get("meta.sample_rate_hz") or mm_output.get("meta.sample_rate")
    if hasattr(sr_raw, "item"):
        try:
            return int(sr_raw.item())
        except Exception:
            return 24000
    return int(sr_raw) if isinstance(sr_raw, (int, float)) else 24000


def _audio_text_marks(mm_output: dict[str, object]) -> list[dict[str, object]]:
    names = ("audio_text_marks", "text_audio_marks", "audio_text_alignment", "alignment_marks")
    candidates = [mm_output.get(name) for name in names]
    meta = mm_output.get("meta")
    if isinstance(meta, dict):
        candidates.extend(meta.get(name) for name in names)
    candidates.extend(
        value
        for key, value in mm_output.items()
        if isinstance(key, str) and key.startswith("meta.") and key.rsplit(".", 1)[-1] in names
    )
    for raw in candidates:
        if not isinstance(raw, list):
            continue
        marks: list[dict[str, object]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text_chars = item.get("text_chars")
            audio_end_ms = item.get("audio_end_ms", item.get("audio_ms"))
            if isinstance(text_chars, (int, float)) and isinstance(audio_end_ms, (int, float)):
                marks.append({"text_chars": max(0, int(text_chars)), "audio_end_ms": max(0, int(audio_end_ms))})
        if marks:
            return marks
    return []


def _llm_output_text(mm_output: dict[str, object]) -> str:
    candidates: list[object] = [
        mm_output.get("llm_output_text"),
        mm_output.get("text"),
        mm_output.get("llm_output_text_utf8"),
        mm_output.get("meta.llm_output_text_utf8"),
    ]
    meta = mm_output.get("meta")
    if isinstance(meta, dict):
        candidates.extend((meta.get("llm_output_text"), meta.get("text"), meta.get("llm_output_text_utf8")))
    candidates.extend(mm_output.get(key) for key in ("meta.llm_output_text", "meta.text"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
        decoded = _decode_text_tensor(value)
        if decoded:
            return decoded
        if isinstance(value, list):
            text_chunks = [item for item in value if isinstance(item, str)]
            if text_chunks:
                return "".join(text_chunks)
    return ""


def _decode_text_tensor(value: object) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, np.ndarray):
            raw = value.astype(np.uint8, copy=False).reshape(-1).tobytes()
        elif hasattr(value, "detach"):
            raw = value.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1).tobytes()
        else:
            return ""
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _fallback_audio_text_marks(
    audio_chunks: list[tuple[str, int]],
    delta_text: str,
) -> list[list[dict[str, int]] | None]:
    if not delta_text:
        return []
    total_duration_ms = sum(max(0, int(duration_ms)) for _, duration_ms in audio_chunks)
    cumulative_duration_ms = 0
    marks: list[list[dict[str, int]] | None] = []
    for _, duration_ms in audio_chunks:
        cumulative_duration_ms += max(0, int(duration_ms))
        if total_duration_ms <= 0:
            marks.append(None)
            continue
        text_chars = int(len(delta_text) * min(1.0, cumulative_duration_ms / float(total_duration_ms)))
        marks.append([{"text_chars": max(0, text_chars), "audio_end_ms": max(0, cumulative_duration_ms)}])
    return marks
