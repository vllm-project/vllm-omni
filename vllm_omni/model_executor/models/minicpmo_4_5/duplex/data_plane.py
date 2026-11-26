# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from vllm.logger import init_logger

from vllm_omni.engine.duplex.contracts import (
    duplex_resource_request_belongs_to_session,
)
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, DuplexDataPlaneContext
from vllm_omni.model_executor.common.request_outputs import coerce_int
from vllm_omni.model_executor.common.request_outputs import coerce_int_list as _coerce_int_list
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
from vllm_omni.model_executor.models.minicpmo_4_5.handoff import TEXT_WITHOUT_AUDIO_KEY
from vllm_omni.outputs.duplex import get_duplex_output_decision

logger = init_logger(__name__)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


@dataclass(slots=True)
class _TurnState:
    sent_segment_text: str = ""
    last_text_chunk: tuple[int, int] | None = None
    has_text: bool = False
    tts_eos_done: bool = False
    last_tts_chunk: tuple[int, int] | None = None
    turn_eos_done: bool = False


_MAX_GAP_STEP_IDS = 1024


@dataclass(slots=True)
class _EOSCursor:
    """TTS EOS scan state for one stage; downstream stages reuse the Stage0 request id."""

    scanned_tokens: int = 0
    #: Step ids scanned while cumulative snapshots were incomplete; dropped past ``_MAX_GAP_STEP_IDS``.
    gap_step_ids: list[int] = field(default_factory=list)
    #: Turn whose scanned EOS has not ended a segment yet; only kept for a known turn.
    pending_turn: int | None = None


@dataclass(slots=True)
class _RequestState:
    audio_offset: int = 0
    #: A snapshot below ``audio_offset`` that has not been accepted as a restart
    #: yet. See :meth:`MiniCPMO45DataPlaneSession.slice_cumulative_audio`.
    audio_restart_candidate: int | None = None
    uses_segment_text_metadata: bool = False
    pending_audio_without_text: list[dict[str, object]] = field(default_factory=list)
    terminal: bool = False
    turns: dict[int | None, _TurnState] = field(default_factory=dict)
    stage0_consumed_tokens: int = 0
    stage0_decoded_tokens: int = 0
    stage0_decoder: DecodeStream | None = None
    #: A skipped stale snapshot ended inside a unit; drop the rest of it up to its terminator.
    stage0_skip_open_unit: bool = False
    eos_cursors: dict[int | None, _EOSCursor] = field(default_factory=dict)

    def eos_cursor(self, stage_id: int | None) -> _EOSCursor:
        return self.eos_cursors.setdefault(stage_id, _EOSCursor())

    def turn(self, turn_id: int | None) -> _TurnState:
        return self.turns.setdefault(turn_id, _TurnState())


class MiniCPMO45DataPlaneSession(DuplexDataPlane):
    """MiniCPM output projector and request/turn cursor owner.

    The scheduler request owns cumulative Stage1 audio, while model turns own
    transcript and terminal cursors. Keeping both levels in one object avoids
    leaking MiniCPM-specific state into the generic Realtime handler.
    """

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestState] = {}
        self.thinker_only = False
        self.tokenizer: Tokenizer | None = None

    def begin_request(self, request_id: str) -> None:
        state = self._requests.setdefault(request_id, _RequestState())
        state.terminal = False
        # A resumable append starts a new Talker segment, even when it
        # continues the same model turn. Re-arm only the segment boundary;
        # turn EOS remains deduplicated until the model advances the turn.
        for turn_state in state.turns.values():
            turn_state.tts_eos_done = False

    def is_terminal(self, request_id: str | None) -> bool:
        if request_id is None:
            return False
        state = self._requests.get(request_id)
        return state is not None and state.terminal

    def mark_terminal(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestState()).terminal = True

    def close_stream(self, request_id: str) -> None:
        """Reset stream-local audio cursors; Stage0 decoding stays cumulative per request."""
        state = self._requests.get(request_id)
        if state is None:
            return
        state.audio_offset = 0
        state.turns.pop(None, None)
        for cursor in state.eos_cursors.values():
            cursor.pending_turn = None

    def close_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self.close_request(active_request_id)
        for request_id in list(self._requests):
            if self.request_belongs_to_session(request_id, session_id):
                self.close_request(request_id)

    def audio_offset(self, request_id: str | None) -> int | None:
        if request_id is None:
            return None
        state = self._requests.get(request_id)
        return state.audio_offset if state is not None and state.audio_offset > 0 else None

    def project(
        self,
        result: object,
        *,
        context: DuplexDataPlaneContext | None = None,
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
        context: DuplexDataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        context = context or DuplexDataPlaneContext()
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
        text_input = coerce_int(mm_output.get("duplex_text_input")) == 1
        if text_input and context.model_turn_id is not None:
            output_turn_id = context.model_turn_id
        output_epoch = output_epoch_from_metadata(mm_output)
        expected_turn_id = context.active_response_turn_id
        stale_turn = expected_turn_id is not None and output_turn_id is not None and output_turn_id < expected_turn_id
        if expected_turn_id is None and output_turn_id is not None:
            stale_turn = output_turn_id < context.turn_id
        stale_epoch = output_epoch is not None and output_epoch != context.epoch
        # Advance the cumulative EOS cursor before skipping stale output, so a later
        # empty step does not rescan the skipped tokens.
        eos_cursor = None
        if request_state is not None and not self.thinker_only:
            eos_cursor = request_state.eos_cursor(coerce_int(getattr(output, "stage_id", None)))
        eos_scan_ids = [] if self.thinker_only else self._unscanned_token_ids(eos_cursor, completion)
        stale = context.auto_responds and (stale_turn or stale_epoch)
        if self.thinker_only:
            # No TTS stage reports segment or turn ends, and the latent is not audio.
            if request_state is not None:
                projected = self._project_stage0_text(
                    request_id=request_id,
                    request_state=request_state,
                    completion=completion,
                    mm_output=mm_output,
                    finished=bool(getattr(output, "finished", False)),
                    text_input=text_input,
                    context=context,
                    runtime_result=runtime_result,
                )
                if not stale:
                    yield from projected
                else:
                    # Consume skipped tokens so their text and boundaries do not reach the next reply.
                    for _ in projected:
                        pass
                    snapshot_len, _ = _stage0_snapshot_ids(completion, request_state.stage0_consumed_tokens)
                    if snapshot_len > request_state.stage0_consumed_tokens:
                        request_state.stage0_consumed_tokens = snapshot_len
                        request_state.stage0_skip_open_unit = True
            return
        if stale:
            return

        mm_text = _llm_output_text(mm_output)
        if mm_text:
            text = mm_text
            if request_state is not None:
                request_state.uses_segment_text_metadata = True
        elif request_state is not None and request_state.uses_segment_text_metadata:
            text = ""

        finished = bool(getattr(output, "finished", False))
        tts_is_last_chunk = is_last_tts_chunk(mm_output)
        token_ids = _completion_token_ids(completion)
        native_decision = _native_decision(completion, mm_output, token_ids=token_ids, finished=finished)
        if native_decision == "listen":
            listen_result = runtime_result(
                stage_role="llm",
                is_listen=True,
                model_listen=mm_output.get("model_listen") is not False,
                listen_source=mm_output.get("listen_source") or "model_listen",
                data_plane_request_id=request_id,
                end_of_turn=False,
                continue_with_partial_input=text_input,
            )
            if output_turn_id is not None:
                listen_result["model_turn_id"] = output_turn_id
            yield listen_result
            return

        if text_input:
            # Stage0 projection only records continuation ownership. Do not
            # encode its hidden states or emit its text before Talker output.
            yield runtime_result(
                requires_stage_handoff=True,
                data_plane_request_id=request_id,
                model_turn_id=output_turn_id,
                continue_with_partial_input=True,
            )
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
        audio_cache_epoch = _first_metadata_int(mm_output, "duplex_audio_cache_epoch")
        audio_chunk_seq = _first_metadata_int(mm_output, "duplex_audio_chunk_seq")
        chunk_key = (
            (audio_cache_epoch, audio_chunk_seq)
            if audio_cache_epoch is not None and audio_chunk_seq is not None
            else None
        )
        # Input submissions can run ahead of TTS output. Re-arming a boolean
        # in begin_request cannot distinguish two queued segment endings, or
        # an old ending replayed after a new input. Use the producer's output
        # identity when present; retain the legacy path for token-only output.
        new_tts_boundary = terminal_turn_state is None or not terminal_turn_state.tts_eos_done
        if terminal_turn_state is not None and chunk_key is not None:
            new_tts_boundary = (
                terminal_turn_state.last_tts_chunk is None or chunk_key > terminal_turn_state.last_tts_chunk
            )
        scanned_eos = 151645 in eos_scan_ids
        pending_eos = (
            eos_cursor is not None and output_turn_id is not None and eos_cursor.pending_turn == output_turn_id
        )
        stage_tts_eos = (
            context.auto_responds
            and (scanned_eos or pending_eos)
            and not audio_chunks
            and raw_audio_samples is not None
            and (raw_audio_samples == 0 or raw_audio_samples == offset_before)
            and new_tts_boundary
        )
        tts_segment_end = bool(tts_is_last_chunk or stage_tts_eos) and new_tts_boundary
        if eos_cursor is not None:
            # An EOS scanned on a step that still had audio ends the turn's next empty step.
            if tts_segment_end:
                eos_cursor.pending_turn = None
            elif scanned_eos and new_tts_boundary:
                eos_cursor.pending_turn = output_turn_id
        if tts_segment_end and terminal_turn_state is not None:
            terminal_turn_state.tts_eos_done = True
            if chunk_key is not None:
                terminal_turn_state.last_tts_chunk = chunk_key
        stage_turn_end_new = bool(stage_turn_end) and (
            terminal_turn_state is None or not terminal_turn_state.turn_eos_done
        )
        if stage_turn_end_new and terminal_turn_state is not None:
            terminal_turn_state.turn_eos_done = True
        unit_end_of_turn = stage_turn_end_new or (finished and not context.auto_responds)

        text_turn_id = output_turn_id if output_turn_id is not None else context.turn_id
        text_turn_state = request_state.turn(text_turn_id) if request_state is not None else None
        if audio_chunks:
            delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id, chunk_key=chunk_key)
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
                delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id, chunk_key=chunk_key)
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

        if tts_segment_end and _bool_metadata(mm_output, (TEXT_WITHOUT_AUDIO_KEY,), default=False) and text:
            # Venus text also carries delegation control. Deliver it even
            # when a mixed speech unit reaches codec EOS without a waveform.
            delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id, chunk_key=chunk_key)
            result = runtime_result(
                stage_role="tts",
                is_listen=False,
                data_plane_request_id=request_id,
                text=delta_text,
                audio_data="",
                audio_duration_ms=0,
                audio_format=context.response_format,
                audio_text_mark=False,
                end_of_turn=unit_end_of_turn,
                abort_data_plane_request=True,
            )
            if output_turn_id is not None:
                result["model_turn_id"] = output_turn_id
            if delta_text and text_turn_state is not None:
                text_turn_state.has_text = True
            yield result
            return

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

    def _project_stage0_text(
        self,
        *,
        request_id: str | None,
        request_state: _RequestState,
        completion: object,
        mm_output: dict[str, object],
        finished: bool,
        text_input: bool,
        context: DuplexDataPlaneContext,
        runtime_result: Callable[..., dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        if request_state.terminal:
            return
        metrics_attached = False
        has_text = False

        def make_result(**values: object) -> dict[str, object]:
            nonlocal metrics_attached
            if text_input and "error_code" not in values:
                # Text input releases the reply past partial PCM, as on the Talker path.
                values["continue_with_partial_input"] = True
            result = runtime_result(stage_role="llm", data_plane_request_id=request_id, **values)
            if metrics_attached or (has_text and not values.get("text")):
                # stage_metrics counts once per snapshot: on the first text event when
                # the snapshot has text, or on the first decision when it has none.
                result.pop("stage_metrics", None)
            else:
                metrics_attached = True
            return result

        start = request_state.stage0_consumed_tokens
        _, new_token_ids = _stage0_snapshot_ids(completion, start)
        if not new_token_ids:
            return
        if self.tokenizer is None:
            yield make_result(
                error_code="runtime_data_plane_missing_tokenizer",
                error="MiniCPM-o Stage0 text projection requires a fast tokenizer.",
            )
            return
        if request_state.stage0_decoder is None:
            request_state.stage0_decoder = DecodeStream(skip_special_tokens=True)
        decoded_text: list[str] = []
        for index, token_id in enumerate(new_token_ids):
            delta = ""
            # Partial units are scanned again; the decoder only accepts new tokens.
            if start + index >= request_state.stage0_decoded_tokens:
                delta = request_state.stage0_decoder.step(self.tokenizer, token_id) or ""
                request_state.stage0_decoded_tokens = start + index + 1
            decoded_text.append(delta)
        turn_id = context.active_response_turn_id if context.active_response_turn_id is not None else context.turn_id
        if text_input and context.model_turn_id is not None:
            turn_id = context.model_turn_id
        special_token_ids = _special_token_ids(mm_output)
        listen_id = special_token_ids.get("listen_token_id")
        turn_eos_id = special_token_ids.get("turn_eos_token_id")
        terminators = {
            token_id
            for token_id in (
                listen_id,
                special_token_ids.get("chunk_eos_token_id"),
                special_token_ids.get("chunk_tts_eos_token_id"),
            )
            if token_id is not None
        }
        unit_start = 0
        if request_state.stage0_skip_open_unit:
            terminator = next((i for i, token_id in enumerate(new_token_ids) if token_id in terminators), None)
            if terminator is None:
                request_state.stage0_consumed_tokens = start + len(new_token_ids)
                return
            request_state.stage0_skip_open_unit = False
            unit_start = terminator + 1
        has_text = any(decoded_text[unit_start:])
        text_parts: list[str] = []
        for index, token_id in enumerate(new_token_ids[unit_start:], start=unit_start):
            if decoded_text[index]:
                text_parts.append(decoded_text[index])
            if token_id not in terminators:
                continue
            if text_parts:
                yield make_result(is_listen=False, text="".join(text_parts), end_of_turn=False, model_turn_id=turn_id)
                text_parts.clear()
            # Later text or a SPEAK token reopens a turn after an intermediate EOS.
            ends_turn = index > unit_start and new_token_ids[index - 1] == turn_eos_id
            unit_start = index + 1
            if ends_turn:
                yield make_result(is_listen=False, text="", end_of_turn=True, model_turn_id=turn_id)
                turn_id += 1
            elif token_id == listen_id:
                yield make_result(
                    is_listen=True,
                    model_listen=True,
                    listen_source="model_listen",
                    end_of_turn=False,
                    model_turn_id=turn_id,
                )
            else:
                yield make_result(
                    is_listen=False, text="", end_of_turn=False, segment_complete=True, model_turn_id=turn_id
                )
        if text_parts:
            yield make_result(is_listen=False, text="".join(text_parts), end_of_turn=False, model_turn_id=turn_id)
        # A resumable stop leaves ``finished`` false and sets finish_reason. An
        # unterminated tail (e.g. a length cap) fails, even if it holds turn_eos.
        if (finished or getattr(completion, "finish_reason", None) is not None) and unit_start < len(new_token_ids):
            unit_start = len(new_token_ids)
            yield make_result(
                error_code="runtime_data_plane_unterminated_unit",
                error=(
                    "MiniCPM-o Stage0 stopped without a unit terminator "
                    f"(finish_reason={getattr(completion, 'finish_reason', None)!r})."
                ),
            )
        request_state.stage0_consumed_tokens = start + unit_start

    @staticmethod
    def _unscanned_token_ids(cursor: _EOSCursor | None, completion: object) -> list[int]:
        """Token ids to scan for a TTS EOS; cumulative ids replay earlier turns, so skip scanned ones."""
        if cursor is None:
            return _completion_token_ids(completion)
        scanned = cursor.scanned_tokens
        cumulative_len, unscanned = _token_ids_from(getattr(completion, "cumulative_token_ids", None), scanned)
        step_ids = _coerce_int_list(getattr(completion, "token_ids", None))
        if not cumulative_len or cumulative_len < scanned:
            # Incomplete snapshots must not rewind the cursor and replay old EOS tokens.
            cursor.gap_step_ids.extend(step_ids)
            if len(cursor.gap_step_ids) > _MAX_GAP_STEP_IDS:
                cursor.gap_step_ids.clear()
            return step_ids
        # Skip the step ids already scanned during the gap, as far as the restored list matches them.
        matched = 0
        for seen, token_id in zip(cursor.gap_step_ids, unscanned):
            if seen != token_id:
                break
            matched += 1
        if unscanned:
            # Once the list grows past the cursor, leftovers (e.g. from a replayed gap step) must not hide a later EOS.
            cursor.gap_step_ids.clear()
        cursor.scanned_tokens = cumulative_len
        return step_ids or unscanned[matched:]

    def segment_text_delta(
        self,
        request_id: str | None,
        text: object,
        *,
        turn_id: int | None = None,
        chunk_key: tuple[int, int] | None = None,
    ) -> str:
        if not isinstance(text, str) or not text:
            return ""
        if request_id is None:
            return text
        turn_state = self._requests.setdefault(request_id, _RequestState()).turn(turn_id)
        if chunk_key is not None:
            # Code2Wav carries queued text once per payload, not a cumulative
            # text snapshot. Equal/prefix-related text in different chunks is
            # real content (e.g. "1" + "1"), not a replay.
            if turn_state.last_text_chunk is not None and chunk_key <= turn_state.last_text_chunk:
                return ""
            turn_state.last_text_chunk = chunk_key
            return text
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
            state.audio_restart_candidate = None
            return audio_data
        if num_samples == prev_samples:
            return None
        if num_samples < prev_samples:
            # Below the cursor means one of two things, and they are not
            # distinguishable from this snapshot alone: the producer restarted
            # its buffer, or this is a unit that carries no cumulative audio at
            # all (a continuation placeholder arrives as a single sample).
            # Rewinding the cursor for the second case is expensive and silent:
            # the next real snapshot is then sliced from ~0 and the whole
            # session's audio goes out again as one delta. So defer the
            # decision — hold the candidate and leave the cursor where it is.
            candidate = state.audio_restart_candidate
            if candidate is None or num_samples <= candidate:
                state.audio_restart_candidate = num_samples
                return audio_data
            # A second below-cursor snapshot that grew from the first: the
            # producer really did restart, and this is its delta.
            state.audio_offset = num_samples
            state.audio_restart_candidate = num_samples
            prev_samples = candidate
        else:
            # Above the cursor: whatever the dip was, this stream never
            # restarted, so it is sliced from the cursor as usual.
            state.audio_restart_candidate = None
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
    special_token_ids = _special_token_ids(mm_output)
    listen_id = special_token_ids.get("listen_token_id")
    if listen_id is None:
        return None
    unit_ids = max(token_ids, _coerce_int_list(getattr(completion, "cumulative_token_ids", None)), key=len)
    if MiniCPMO45DuplexPolicy.speech_unit_closed_by_listen(unit_ids, special_token_ids):
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


def _stage0_snapshot_ids(completion: object, start: int) -> tuple[int, list[int]]:
    # A queued snapshot reads its own cumulative ids, even when empty: token_ids may be
    # a live list that has since grown. Only a snapshot without the field uses token_ids.
    snapshot_ids = getattr(completion, "cumulative_token_ids", None)
    if snapshot_ids is None:
        snapshot_ids = getattr(completion, "token_ids", None)
    return _token_ids_from(snapshot_ids, start)


def _token_ids_from(value: object, start: int) -> tuple[int, list[int]]:
    """A token list's length and its ids from ``start``, coercing only that tail.

    Resident requests resend the whole cumulative list each step, so coercing the
    consumed prefix would cost the full history on every update. A list's raw
    length is its token count, since producers emit one int per token.
    """
    if isinstance(value, (list, tuple)):
        return len(value), _coerce_int_list(value[start:])
    ids = _coerce_int_list(value)
    return len(ids), ids[start:]


def _audio_num_samples(audio_data: object) -> int | None:
    try:
        import torch

        if isinstance(audio_data, torch.Tensor):
            return int(audio_data.numel())
        return int(np.asarray(audio_data, dtype=np.float32).size)
    except Exception:
        return None


def is_last_tts_chunk(mm_output: dict[str, object]) -> bool:
    """Whether a Code2Wav output carries the connector's flush of a unit's remaining codes (``tts_is_last_chunk``)."""
    return _bool_metadata(mm_output, ("tts_is_last_chunk",), default=False)


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
