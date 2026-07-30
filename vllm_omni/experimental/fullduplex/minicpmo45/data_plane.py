from __future__ import annotations

from collections.abc import Iterator, Mapping

from vllm_omni.experimental.fullduplex.base.data_plane import (
    BaseDataPlaneSession,
    DataPlaneContext,
    RequestState,
    audio_num_samples,
    audio_text_marks,
    bool_metadata,
    coerce_int,
    completion_token_ids,
    fallback_audio_text_marks,
    llm_output_text,
    output_epoch_from_metadata,
    output_stage_metrics,
    output_turn_id_from_metadata,
    runtime_result,
    sample_rate_hz,
    special_token_ids,
)
from vllm_omni.experimental.fullduplex.output import get_duplex_output_decision

MiniCPMO45DataPlaneContext = DataPlaneContext


class MiniCPMO45DataPlaneSession(BaseDataPlaneSession):

    def begin_request(self, request_id: str) -> None:
        state = self._requests.setdefault(request_id, RequestState())
        state.terminal = False
        for turn_state in state.turns.values():
            turn_state.tts_eos_done = False

    def project_output(
        self,
        output: object,
        *,
        context: DataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        context = context or DataPlaneContext()
        stage_metrics = output_stage_metrics(output)

        def _runtime_result(**values: object) -> dict[str, object]:
            result = runtime_result(**values)
            if stage_metrics is not None:
                result["stage_metrics"] = stage_metrics
            return result

        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = None
        request_state = self._requests.setdefault(request_id, RequestState()) if request_id is not None else None

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

        out_turn_id = output_turn_id_from_metadata(mm_output)
        output_epoch = output_epoch_from_metadata(mm_output)
        expected_turn_id = context.active_response_turn_id
        stale_turn = expected_turn_id is not None and out_turn_id is not None and out_turn_id < expected_turn_id
        if expected_turn_id is None and out_turn_id is not None:
            stale_turn = out_turn_id < context.turn_id
        stale_epoch = output_epoch is not None and output_epoch != context.epoch
        if context.auto_responds and (stale_turn or stale_epoch):
            return

        mm_text = llm_output_text(mm_output)
        if mm_text:
            text = mm_text
            if request_state is not None:
                request_state.uses_segment_text_metadata = True
        elif request_state is not None and request_state.uses_segment_text_metadata:
            text = ""

        finished = bool(getattr(output, "finished", False))
        tts_is_last_chunk = bool_metadata(mm_output, ("tts_is_last_chunk",), default=False)
        token_ids = completion_token_ids(completion)
        native_decision = _native_decision(completion, mm_output, token_ids=token_ids, finished=finished)
        if native_decision == "listen":
            listen_result = _runtime_result(
                stage_role="llm",
                is_listen=True,
                model_listen=True,
                listen_source="model_listen",
                data_plane_request_id=request_id,
                end_of_turn=False,
            )
            if out_turn_id is not None:
                listen_result["model_turn_id"] = out_turn_id
            yield listen_result
            return

        raw_audio = next((mm_output[key] for key in ("audio", "model_outputs", "latent") if key in mm_output), None)
        raw_audio_samples = audio_num_samples(raw_audio)
        offset_before = self.audio_offset(request_id)
        audio_chunks = list(
            self.encode_audio_chunks_with_duration(
                mm_output,
                request_id=request_id,
                response_format=context.response_format,
                speed=context.speed,
            )
        )
        stage_turn_end = bool_metadata(mm_output, ("turn_end", "end_of_turn"), default=False)
        terminal_turn_state = request_state.turn(out_turn_id) if request_state is not None else None
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

        text_turn_id = out_turn_id if out_turn_id is not None else context.turn_id
        text_turn_state = request_state.turn(text_turn_id) if request_state is not None else None
        if audio_chunks:
            delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id)
            last_idx = len(audio_chunks) - 1
            sr = sample_rate_hz(mm_output)
            marks = audio_text_marks(mm_output)
            fb_marks = fallback_audio_text_marks(audio_chunks, delta_text)
            audio_results: list[dict[str, object]] = []
            for idx, (audio, duration_ms) in enumerate(audio_chunks):
                native_result = _runtime_result(
                    stage_role="tts",
                    is_listen=False,
                    data_plane_request_id=request_id,
                    text=delta_text if idx == 0 else "",
                    audio_data=audio,
                    audio_format=context.response_format,
                    audio_duration_ms=duration_ms,
                    audio_text_mark=idx == last_idx,
                    sample_rate_hz=sr,
                    end_of_turn=unit_end_of_turn and idx == last_idx,
                    abort_data_plane_request=tts_segment_end and idx == last_idx,
                )
                if out_turn_id is not None:
                    native_result["model_turn_id"] = out_turn_id
                if marks and idx == last_idx:
                    native_result["audio_text_marks"] = marks
                    native_result["audio_text_marks_are_cumulative"] = True
                elif idx < len(fb_marks) and fb_marks[idx]:
                    native_result["audio_text_marks"] = fb_marks[idx]
                    native_result["audio_text_marks_are_cumulative"] = True
                audio_results.append(native_result)

            if context.auto_responds:
                if delta_text and text_turn_state is not None:
                    text_turn_state.has_text = True
                future_model_turn = (
                    context.active_response_turn_id is not None
                    and out_turn_id is not None
                    and out_turn_id > context.active_response_turn_id
                )
                response_turn_bound = context.active_response_id is not None and (
                    out_turn_id is None
                    or context.active_response_turn_id is None
                    or context.active_response_turn_id == out_turn_id
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
            terminal_result = _runtime_result(
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
            if out_turn_id is not None:
                terminal_result["model_turn_id"] = out_turn_id
            yield terminal_result
            return

        if context.active_response_id is not None and unit_end_of_turn:
            terminal_result = _runtime_result(
                stage_role="tts",
                is_listen=False,
                data_plane_request_id=request_id,
                text="",
                audio_data="",
                audio_format=context.response_format,
                audio_text_mark=False,
                end_of_turn=True,
            )
            if out_turn_id is not None:
                terminal_result["model_turn_id"] = out_turn_id
            yield terminal_result
            return

        if request_id is not None and context.auto_responds and unit_end_of_turn and context.active_response_id is None:
            if request_state is not None and out_turn_id is None:
                request_state.pending_audio_without_text.clear()
            terminal_result = _runtime_result(
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
            if out_turn_id is not None:
                terminal_result["model_turn_id"] = out_turn_id
            yield terminal_result
            return

        if (
            finished
            and context.auto_responds
            and context.active_response_id is not None
            and request_id is not None
            and not unit_end_of_turn
        ):
            listen_result = _runtime_result(
                stage_role="llm",
                is_listen=True,
                model_listen=False,
                listen_source="auto_response_segment_complete",
                reason="auto_response_segment_complete",
                data_plane_request_id=request_id,
                end_of_turn=False,
            )
            if out_turn_id is not None:
                listen_result["model_turn_id"] = out_turn_id
            yield listen_result
            return

        if not text:
            if context.auto_responds:
                return
            if finished:
                listen_result = _runtime_result(
                    stage_role="llm",
                    is_listen=True,
                    model_listen=False,
                    listen_source="data_plane_finished_without_output",
                    reason="data_plane_finished_without_output",
                    data_plane_request_id=request_id,
                    end_of_turn=False,
                )
                if out_turn_id is not None:
                    listen_result["model_turn_id"] = out_turn_id
                yield listen_result
            return
        if context.auto_responds:
            return
        if "audio" in context.modalities:
            yield _runtime_result(
                stage_role="tts",
                error_code="runtime_data_plane_text_without_audio",
                error="MiniCPM-o native duplex data-plane produced text without audio.",
                data_plane_request_id=request_id,
            )
            return
        yield _runtime_result(
            stage_role="llm",
            is_listen=False,
            data_plane_request_id=request_id,
            text=text if isinstance(text, str) else "",
            audio_data="",
            end_of_turn=unit_end_of_turn,
        )


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
    listen_id = special_token_ids(mm_output).get("listen_token_id")
    if listen_id is None:
        return None
    stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
    if coerce_int(stop_reason) == listen_id:
        return "listen"
    return "listen" if token_ids and token_ids[-1] == listen_id else None
