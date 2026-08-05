# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterator, Mapping

from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.base.data_plane import (
    BaseDataPlaneSession,
    DataPlaneContext,
    RequestState,
    audio_text_marks,
    bool_metadata,
    completion_token_ids,
    fallback_audio_text_marks,
    llm_output_text,
    output_epoch_from_metadata,
    output_stage_metrics,
    output_turn_id_from_metadata,
    runtime_result,
    sample_rate_hz,
)
from vllm_omni.experimental.fullduplex.output import get_duplex_output_decision
from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

logger = init_logger(__name__)

_IM_END_TOKEN_ID = Qwen3OmniDuplexPolicy.IM_END_TOKEN_ID

Qwen3OmniDataPlaneContext = DataPlaneContext


class Qwen3OmniDataPlaneSession(BaseDataPlaneSession):
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
        token_ids = completion_token_ids(completion)
        turn_end = bool_metadata(mm_output, ("turn_end", "end_of_turn"), default=False)
        tts_is_last_chunk = bool_metadata(mm_output, ("tts_is_last_chunk",), default=False)
        eos_emitted = bool_metadata(mm_output, ("eos_emitted",), default=False)

        im_end_detected = _IM_END_TOKEN_ID in token_ids
        unit_end_of_turn = turn_end or (finished and not context.auto_responds)

        text_turn_id = out_turn_id if out_turn_id is not None else context.turn_id
        audio_chunks = list(
            self.encode_audio_chunks_with_duration(
                mm_output,
                request_id=request_id,
                response_format=context.response_format,
                speed=context.speed,
            )
        )

        tts_segment_end = tts_is_last_chunk or eos_emitted or (im_end_detected and not audio_chunks and finished)
        terminal_turn_state = request_state.turn(out_turn_id) if request_state is not None else None
        if tts_segment_end and terminal_turn_state is not None:
            if terminal_turn_state.tts_eos_done:
                tts_segment_end = False
            else:
                terminal_turn_state.tts_eos_done = True

        if audio_chunks:
            delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id)
            last_idx = len(audio_chunks) - 1
            sr = sample_rate_hz(mm_output)
            marks = audio_text_marks(mm_output)
            fb_marks = fallback_audio_text_marks(audio_chunks, delta_text)
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
                    end_of_turn=(unit_end_of_turn or tts_segment_end) and idx == last_idx,
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
                yield native_result
            return

        if tts_segment_end or (unit_end_of_turn and context.active_response_id is not None):
            terminal_result = _runtime_result(
                stage_role="tts",
                is_listen=False,
                data_plane_request_id=request_id,
                text="",
                audio_data="",
                audio_format=context.response_format,
                audio_text_mark=False,
                end_of_turn=unit_end_of_turn or tts_segment_end,
                abort_data_plane_request=tts_segment_end,
            )
            if out_turn_id is not None:
                terminal_result["model_turn_id"] = out_turn_id
            yield terminal_result
            return

        if finished and not text:
            if context.auto_responds:
                return
            yield _runtime_result(
                stage_role="llm",
                is_listen=True,
                model_listen=False,
                listen_source="data_plane_finished_without_output",
                reason="data_plane_finished_without_output",
                data_plane_request_id=request_id,
                end_of_turn=False,
            )
            return

        if not text or context.auto_responds:
            return

        if "audio" in context.modalities:
            yield _runtime_result(
                stage_role="tts",
                error_code="runtime_data_plane_text_without_audio",
                error="Qwen3-Omni duplex data-plane produced text without audio.",
                data_plane_request_id=request_id,
            )
            return

        delta_text = self.segment_text_delta(request_id, text, turn_id=text_turn_id)
        if not delta_text and not unit_end_of_turn:
            return

        yield _runtime_result(
            stage_role="llm",
            is_listen=False,
            data_plane_request_id=request_id,
            text=delta_text,
            audio_data="",
            end_of_turn=unit_end_of_turn,
        )
