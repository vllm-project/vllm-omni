# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn-commit JoyAI observations with optional Qwen3-ASR speech input."""

from __future__ import annotations

import asyncio
import io
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np
import pybase64
from PIL import Image
from vllm.config import ModelConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputAction, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import (
    DuplexAuxiliaryPolicy,
    DuplexAuxiliaryRequest,
    DuplexInputCapacityError,
    DuplexModelPlugin,
    DuplexModelSessionState,
    DuplexRuntimeConfigError,
    duplex_history_digest,
    duplex_text_history,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import (
    LONG_TERM_COMPRESS_PROMPT,
    MID_TERM_SUMMARY_PROMPT,
    QA_QUERY_LABEL,
    QA_RESPONSE_LABEL,
    SILENT_SYSTEM_PROMPT,
    USER_QUERY_HEADER,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import (
    JoyAIDataPlane,
    JoyAISessionState,
    QueryReference,
    WorkingObservation,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.memory import (
    RECENT_MESSAGES,
    SUMMARY_CHUNK_MESSAGES,
    SUMMARY_MAX_TOKENS,
    EncodedVisualImage,
    VisualClosureReason,
    VisualFrame,
    VisualMemoryCapacityError,
    VisualMemorySource,
    VisualSummary,
)
from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import (
    _clean_qwen3_asr_transcript,
    _extract_completed_action_text,
    render_joyai_memory_prefix,
)

if TYPE_CHECKING:
    from transformers import ProcessorMixin


class JoyAIDuplexPlugin(DuplexModelPlugin):
    plugin_id = "joyai-vl-interaction"
    supports_auxiliary_requests = True
    joyai_stage_id = 0
    asr_stage_id: int | None = None
    audio_stage_id = 2

    def __init__(self, encode_audio: Callable[..., str]) -> None:
        self.data_plane = JoyAIDataPlane(encode_audio)
        self.processor = None
        self.max_model_len = 0
        self.max_images = 0  # Loaded with the processor from the stage's multimodal config.
        self._mid_prompt_tokens = 0
        self._long_prompt_tokens = 0
        self._timestamp_tokens = 0
        self._processor_lock = asyncio.Lock()
        self._memory_budget_error: str | None = None

    def configure_pipeline(self, pipeline_config: PipelineConfig) -> None:
        stages = {stage.model_stage: stage.stage_id for stage in pipeline_config.stages}
        self.joyai_stage_id = stages["joyvl"]
        self.asr_stage_id = stages.get("asr")
        self.audio_stage_id = stages["code2wav"]
        self.data_plane.joyai_stage_id = self.joyai_stage_id
        self.data_plane.audio_stage_id = self.audio_stage_id

    def input_stage_id(self, payload: object) -> int:
        if isinstance(payload, Mapping) and payload.get("audio"):
            if self.asr_stage_id is None:
                raise DuplexRuntimeConfigError("JoyAI speech input requires the ASR pipeline")
            return self.asr_stage_id
        return self.joyai_stage_id

    def split_transcription_input(self, payload: object) -> tuple[object, dict[str, object] | None]:
        if not isinstance(payload, Mapping):
            return payload, None
        segments = payload.get("committed_audio_segments")
        if not isinstance(segments, list) or len(segments) < 2:
            return payload, None
        raw = pybase64.b64decode(payload["audio"], validate=True)
        if sum(segment["byte_count"] for segment in segments) != len(raw):
            raise DuplexRuntimeConfigError("Committed audio boundaries do not match the retained PCM")
        boundary = segments[0]["byte_count"]
        first = {**payload, "audio": pybase64.b64encode(raw[:boundary]).decode("ascii")}
        first["realtime_item_id"] = segments[0]["item_id"]
        first.pop("committed_audio_segments", None)
        first.pop("video_frames", None)
        remaining = {**payload, "audio": pybase64.b64encode(raw[boundary:]).decode("ascii")}
        remaining["committed_audio_segments"] = segments[1:]
        return first, remaining

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities(
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=True,
            supports_turn_commit_only=True,
            supports_core_resumable_request=False,
            required_input_modalities=frozenset(),
            optional_input_modalities=frozenset({"video", "audio"} if self.asr_stage_id is not None else {"video"}),
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=max_sessions > 1,
            supports_multi_session_same_replica=max_sessions > 1,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_image_input=True,
            supports_text_only_turn=True,
            adapter_patterns=["turn_commit"],
            signal_sources=["client_event", "server_policy"],
            stage_handoff_transport="stage_connector",
            target_barge_in_latency_ms=None,
        )

    def create_session_state(self) -> JoyAISessionState:
        return JoyAISessionState(audio_buffer=JoyAIFrameBuffer(accepts_audio=self.asr_stage_id is not None))

    def validate_client_extra_body(self, extra_body: object) -> None:
        super().validate_client_extra_body(extra_body)
        if isinstance(extra_body, Mapping) and (extra_body.get("auto_response") or extra_body.get("realtime_tools")):
            raise DuplexRuntimeConfigError(
                "JoyAI video observations use explicit commits; tools use the follow-up adapter"
            )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        runtime_config = self.runtime_config_for_update(config, {})
        async with self._processor_lock:
            if self.processor is None:
                if model_config is None:
                    raise DuplexRuntimeConfigError("JoyAI requires its comprehension-stage model configuration")
                max_model_len = model_config.max_model_len
                max_images = model_config.multimodal_config.get_limit_per_prompt("image")
                if max_images < 1:
                    raise DuplexRuntimeConfigError("JoyAI visual memory requires limit_mm_per_prompt.image >= 1.")
                if self._memory_budget_error is not None:
                    raise DuplexRuntimeConfigError(self._memory_budget_error)
                from vllm.transformers_utils.processor import cached_processor_from_config

                processor = await asyncio.to_thread(
                    cached_processor_from_config,
                    model_config,
                )
                try:
                    budget = await asyncio.to_thread(
                        self._prepare_memory_budget, processor, max_model_len=max_model_len
                    )
                except DuplexRuntimeConfigError as exc:
                    # The plugin belongs to one static stage configuration.
                    # Cache only deterministic validation failures, not load errors or cancellation.
                    self._memory_budget_error = str(exc)
                    raise
                # Publish only validated state. A cancelled preparation worker
                # may still finish, but cannot mutate this shared plugin.
                self.max_model_len = max_model_len
                self.max_images = max_images
                self._mid_prompt_tokens, self._long_prompt_tokens, self._timestamp_tokens = budget
                self.processor = processor
        return runtime_config

    def runtime_config_for_update(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("JoyAI's current TTS profile uses CustomVoice speakers")
        extra = config.extra_body or {}
        return {
            "instructions": config.instructions,
            "initial_user_text": config.initial_user_text,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **{
                key: extra.get(key, default)
                for key, default in (("tts_speaker", "Vivian"), ("tts_language", "Auto"), ("tts_instruct", ""))
            },
        }

    def configure_sampling_params(
        self, *, runtime_config: dict[str, object], defaults: tuple[object, ...]
    ) -> tuple[object, ...]:
        configured = []
        for index, params in enumerate(defaults):
            if isinstance(params, SamplingParams):
                params = params.clone()
                params.output_kind = RequestOutputKind.FINAL_ONLY
                if index == self.joyai_stage_id:
                    params.detokenize = True
                    params.skip_special_tokens = False
                    for key in ("temperature", "max_tokens"):
                        if runtime_config.get(key) is not None:
                            setattr(params, key, runtime_config[key])
            configured.append(params)
        return tuple(configured)

    @staticmethod
    def _sync_visual_history(state: JoyAISessionState, history: Sequence[dict[str, object]]) -> tuple[str, ...]:
        images = tuple(
            part["image_url"]["url"]
            for message in history
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        )
        previous = state.conversation_images
        if images[: len(previous)] != previous:
            # Conservatively revoke derived blocks as well as raw images. An
            # image edit can arrive after its raw source has been compressed.
            state.visual_memory.clear()
            state.observations.clear()
            state.queries.clear()
            state.closed_frame_id = -1
            state.visual_source = None
            state.frames = []
            state.conversation_images = ()
        return images

    def _visual_source(self, state: JoyAISessionState) -> VisualMemorySource | None:
        memory = state.visual_memory
        self._close_working_chunk(state)
        if state.visual_source is not None and memory.matches(state.visual_source):
            return state.visual_source
        state.visual_source = self._choose_visual_source(state)
        return state.visual_source

    def _choose_visual_source(self, state: JoyAISessionState) -> VisualMemorySource | None:
        memory = state.visual_memory
        mid_tokens = sum(entry.token_count + self._timestamp_tokens for entry in memory.mid_term)
        long_budget = self.max_model_len - memory.long_max_tokens - self._long_prompt_tokens
        if len(memory.mid_term) >= memory.mid_group or (
            len(memory.mid_term) >= 2 and mid_tokens + memory.mid_max_tokens > long_budget
        ):
            return memory.long_source(closure_reason="count" if len(memory.mid_term) >= memory.mid_group else "context")
        closed = [frame for frame in memory.pending if frame.id <= state.closed_frame_id]
        if not closed:
            return None
        # A preparation budget may split a closed chunk. Its remaining prefix
        # is still eligible even below the ordinary chunk threshold.
        return memory.frame_source(
            min(len(closed), memory.chunk_frames),
            closure_reason=self._frame_closure_reason(state, closed) or "context",
        )

    def _frame_closure_reason(self, state: JoyAISessionState, frames: list[VisualFrame]) -> VisualClosureReason | None:
        memory = state.visual_memory
        if len(frames) >= memory.chunk_frames:
            return "count"
        if not frames:
            return None
        if sum(len(frame.encoded_image.data) for frame in frames) >= memory.max_encoded_bytes // 2:
            return "capacity"
        if len(frames) >= self.max_images:
            return "image_limit"
        frame_tokens = sum(frame.token_count + self._timestamp_tokens for frame in frames)
        if frame_tokens + self._mid_prompt_tokens + memory.mid_max_tokens >= self.max_model_len:
            return "context"
        return None

    def _close_working_chunk(self, state: JoyAISessionState) -> None:
        """Close the interactive view while retaining its source for consolidation."""
        working = [frame for frame in state.visual_memory.pending if frame.id > state.closed_frame_id]
        if self._frame_closure_reason(state, working) is not None:
            state.closed_frame_id = working[-1].id

    def auxiliary_policy(
        self,
        *,
        state: DuplexModelSessionState,
        history_revision: int,
        history: tuple[dict[str, object], ...],
    ) -> DuplexAuxiliaryPolicy:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        self._sync_visual_history(state, history)
        source = self._visual_source(state)
        return DuplexAuxiliaryPolicy(source_key=source or history_revision, allow_during_response=source is not None)

    def auxiliary_source_matches(
        self,
        *,
        request: DuplexAuxiliaryRequest,
        state: DuplexModelSessionState,
        history: tuple[dict[str, object], ...],
    ) -> bool:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        self._sync_visual_history(state, history)
        if isinstance(request.model_source, VisualMemorySource):
            return state.visual_memory.matches(request.model_source)
        return super().auxiliary_source_matches(request=request, state=state, history=history)

    def auxiliary_preempted_by_input(
        self,
        *,
        payload: Mapping[str, object],
        state: DuplexModelSessionState,
        has_new_user_input: bool,
    ) -> bool:
        return bool(has_new_user_input or payload.get("audio") or not payload.get("video_frames"))

    def prepare_prompt_config(
        self, config: dict[str, object], *, state: DuplexModelSessionState, payload: dict[str, object]
    ) -> dict[str, object]:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        state.memory.validate(duplex_text_history(config.get("conversation", [])))
        images = self._sync_visual_history(state, config.get("conversation", []))
        self._close_working_chunk(state)
        previous = state.conversation_images
        new_images = [*images[len(previous) :], *payload.get("video_frames", [])]
        # Append each conversation image once, alongside newly committed camera
        # frames, so retained camera history cannot displace a newer user image.
        frames = [*state.frames, *new_images][-4:]
        return {
            **config,
            "joyai_state": state,
            "joyai_frames": frames,
            "joyai_conversation_images": images,
            "joyai_new_frames": new_images,
            "joyai_pending_frames": tuple(
                frame for frame in state.visual_memory.pending if frame.id > state.closed_frame_id
            ),
            "joyai_observations": tuple(state.observations),
            "joyai_queries": tuple(state.queries),
            "joyai_closed_frame_id": state.closed_frame_id,
            "joyai_visual_summaries": tuple([*state.visual_memory.long_term, *state.visual_memory.mid_term]),
            "joyai_summary": state.memory.text,
            "joyai_summary_messages": state.memory.covered_messages,
        }

    async def prepare_auxiliary_request(
        self,
        *,
        history: tuple[tuple[str, str], ...],
        runtime_config: Mapping[str, object],
        state: DuplexModelSessionState,
    ) -> DuplexAuxiliaryRequest | None:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        visual = state.visual_memory
        source = self._visual_source(state)
        if source is not None:
            return await asyncio.to_thread(
                self._prepare_visual_summary,
                source,
                tuple(visual.pending if source.kind == "frames" else visual.mid_term),
                visual.mid_max_tokens if source.kind == "frames" else visual.long_max_tokens,
            )
        # Snapshot model-owned memory before yielding; preparation never mutates
        # it. Prompt invalidation and result application stay on the mailbox.
        current = state.memory.matches(history)
        covered = state.memory.covered_messages if current else 0
        previous_summary = state.memory.text if current else ""
        # Camera-only commits keep their ledger positions for fingerprints but
        # must not displace text from the verbatim recent-history window.
        cutoff = len(history)
        recent = 0
        while cutoff and recent < RECENT_MESSAGES:
            cutoff -= 1
            role, text = history[cutoff]
            if role in {"user", "assistant"} and text:
                recent += 1
        # Textless observations retain their ledger indices but consume no
        # summary budget, so a silent camera span cannot stall consolidation.
        end = covered
        pending_text = 0
        while end < cutoff and pending_text < SUMMARY_CHUNK_MESSAGES:
            role, text = history[end]
            if role in {"user", "assistant"} and text:
                pending_text += 1
            end += 1
        end = self._summary_boundary(history, covered, end)
        if end <= covered:
            return None

        def prepare() -> DuplexAuxiliaryRequest | None:
            target = end
            while target > covered:
                transcript = "\n".join(
                    f"[{index + 1}] {role}: {text}"
                    for index, (role, text) in enumerate(history[covered:target], start=covered)
                    if role in {"user", "assistant"} and text
                )
                content = LONG_TERM_COMPRESS_PROMPT.format(
                    merged_range=f"conversation turns 1-{target}",
                    summaries_text=(
                        f"Previous committed summary:\n{previous_summary}\n\nCommitted conversation:\n{transcript}"
                    ),
                    length_instruction=f" (at most {SUMMARY_MAX_TOKENS} tokens) ",
                )
                prompt = self.processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
                )
                if (
                    len(self.processor.tokenizer(text=[prompt])["input_ids"][0]) + SUMMARY_MAX_TOKENS
                    <= self.max_model_len
                ):
                    return DuplexAuxiliaryRequest(
                        stage_id=self.joyai_stage_id,
                        prompt={"prompt": prompt},
                        sampling_params=SamplingParams(
                            temperature=0.0,
                            max_tokens=SUMMARY_MAX_TOKENS,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                            detokenize=True,
                        ),
                        history_length=target,
                        history_digest=duplex_history_digest(history[:target]),
                    )
                target -= 1
                target = self._summary_boundary(history, covered, target)
            return None

        return await asyncio.to_thread(prepare)

    @staticmethod
    def _summary_boundary(history: tuple[tuple[str, str], ...], start: int, end: int) -> int:
        """Prefer complete Q&A, while allowing older text-bearing silent turns.

        The recent-text window already protects the current query. Older user
        messages can be committed without an assistant entry when JoyAI stays
        silent; requiring such an entry would stall the prefix indefinitely.
        """
        for index in range(end - 1, start - 1, -1):
            if history[index][0] == "assistant":
                return index + 1
        # Silent turns can advance on their own, but a question whose answer
        # lies just outside the chunk must remain with that answer. Empty
        # camera observations between them do not break the pair.
        next_role = next((role for role, text in history[end:] if role in {"user", "assistant"} and text), None)
        if next_role == "assistant":
            for index in range(end - 1, start - 1, -1):
                if history[index][0] == "user" and history[index][1]:
                    end = index
                    break
        return end if any(role == "user" and text for role, text in history[start:end]) else start

    def _prepare_visual_summary(
        self,
        source: VisualMemorySource,
        entries: tuple[VisualFrame, ...] | tuple[VisualSummary, ...],
        output_limit: int,
    ) -> DuplexAuxiliaryRequest | None:
        # The caller snapshots immutable encoded frames before offloading.
        # Decode only this prompt's images in the preparation worker.
        entries = entries[: len(source.covered_ids)]
        closure_reason = source.closure_reason
        if source.kind == "frames" and len(entries) > self.max_images:
            entries = entries[: self.max_images]
            closure_reason = "image_limit"
        while entries:
            images = []
            if source.kind == "frames":
                first, last = entries[0].time_s, entries[-1].time_s
                content = [
                    {
                        "type": "text",
                        "text": MID_TERM_SUMMARY_PROMPT.format(
                            chunk_index=source.chunk_index,
                            frame_range=f"{first:.1f} seconds-{last:.1f} seconds",
                            length_instruction="",
                            preferred_time_span="10 seconds",
                        ),
                    }
                ]
                for entry in entries:
                    content.extend([{"type": "text", "text": f"<{entry.time_s:.1f} seconds>"}, {"type": "image"}])
                visual_tokens = sum(entry.token_count for entry in entries) - len(entries)
            else:
                if len(entries) < 2:
                    return None
                content = LONG_TERM_COMPRESS_PROMPT.format(
                    merged_range=f"{entries[0].start_s:.1f} seconds-{entries[-1].end_s:.1f} seconds",
                    summaries_text="\n\n".join(
                        f"<{entry.start_s:.1f} seconds-{entry.end_s:.1f} seconds>\n{entry.text}" for entry in entries
                    ),
                    length_instruction="",
                )
                visual_tokens = 0
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            if (
                len(self.processor.tokenizer(text=[prompt])["input_ids"][0]) + visual_tokens + output_limit
                <= self.max_model_len
            ):
                covered = source.covered_ids[: len(entries)]
                images = [entry.encoded_image.decode() for entry in entries] if source.kind == "frames" else []
                return DuplexAuxiliaryRequest(
                    stage_id=self.joyai_stage_id,
                    prompt={"prompt": prompt, "multi_modal_data": {"image": images} if images else {}},
                    sampling_params=SamplingParams(
                        temperature=0.8 if images else 0.3,
                        top_p=0.9 if images else 0.7,
                        top_k=40 if images else 30,
                        repetition_penalty=1.1,
                        presence_penalty=1.0 if images else 0.5,
                        max_tokens=output_limit,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                        detokenize=True,
                    ),
                    history_length=0,
                    history_digest="",
                    model_source=VisualMemorySource(
                        source.kind,
                        source.generation,
                        covered,
                        source.chunk_index,
                        closure_reason,
                    ),
                    timeout_s=300.0,
                    retry_delays_s=(1.0, 5.0, 30.0),
                )
            entries = entries[:-1]
            closure_reason = "context"
        return None

    def apply_auxiliary_result(
        self, *, request: DuplexAuxiliaryRequest, output: object, state: DuplexModelSessionState
    ) -> bool:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        completion = next(iter(getattr(output, "outputs", ()) or ()), None)
        text = (getattr(completion, "text", "") or "").strip()
        # A truncated or action-only completion is not a usable memory replacement.
        if not text or getattr(completion, "finish_reason", None) == "length":
            return False
        if any(marker in text for marker in ("</silence>", "</delegation>")):
            return False
        text = text.removeprefix("</response>").strip()
        tokens = len(self.processor.tokenizer(text=[text])["input_ids"][0])
        if isinstance(request.model_source, VisualMemorySource):
            applied = state.visual_memory.apply(request.model_source, text=text, token_count=tokens)
            if applied and request.model_source.kind == "frames":
                state.closed_frame_id = max(state.closed_frame_id, request.model_source.covered_ids[-1])
                remaining = {frame.id for frame in state.visual_memory.pending}
                state.observations = [
                    replace(item, frame_ids=frame_ids)
                    for item in state.observations
                    if (frame_ids := tuple(frame_id for frame_id in item.frame_ids if frame_id in remaining))
                ]
            return applied
        if not text or tokens > SUMMARY_MAX_TOKENS:
            return False
        state.memory.text = text
        state.memory.covered_messages = request.history_length
        state.memory.covered_digest = request.history_digest
        return True

    async def prepare_append_plan(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        # Bind on the session loop before yielding. Close/cancel may remove this
        # request while image decoding runs; the worker never mutates live state.
        config = dict(session_config)
        state = config.pop("joyai_state")
        config["conversation"] = deepcopy(config.get("conversation", []))
        self.data_plane.begin_request(request_id)
        request = self.data_plane.requests[request_id]
        request.state = state
        request.frames = config["joyai_frames"]
        request.conversation_images = config.pop("joyai_conversation_images")
        request.modalities = tuple(config["modalities"])
        request.previous_response = state.last_response_text
        request.visual_generation = state.visual_memory.generation
        request.frame_time_s = monotonic() - state.visual_memory.start_time
        awaiting_transcript = isinstance(payload, Mapping) and bool(payload.get("audio"))
        if not awaiting_transcript:
            request.query, request.query_index = self._current_query(config, runtime_config)
        item_ids = config.get("conversation_item_ids", ())
        if 0 <= request.query_index < len(item_ids):
            request.query_item_id = item_ids[request.query_index]
        if request.query_item_id and any(
            request.query_item_id in item.history_anchor_item_ids for item in config.get("joyai_observations", ())
        ):
            request.query = ""
        request.history_anchor_item_ids = tuple(
            item_id for item_id in item_ids[-(RECENT_MESSAGES + SUMMARY_CHUNK_MESSAGES) :] if item_id
        )
        latest_query, latest_index = self._latest_query(config, runtime_config)
        latest_item_id = item_ids[latest_index] if 0 <= latest_index < len(item_ids) else None
        # Speech owns a new query whose text arrives after this plan. Its
        # committed history anchor retains the eventual transcript; the previous
        # text query must not acquire this request's answer as its own memory.
        if not awaiting_transcript and latest_query and (latest_item_id is not None or latest_index == -1):
            query_digest = duplex_history_digest((("user", latest_query),))
            request.query_reference = next(
                (
                    query
                    for query in state.queries
                    if (query.item_id, query.text_digest) == (latest_item_id, query_digest)
                ),
                QueryReference(latest_item_id, query_digest, request.frame_time_s, state.visual_memory.next_id),
            )
        config["joyai_new_frame_time"] = request.frame_time_s

        def prepare():
            new_images = self._decode_images(config.get("joyai_new_frames", []))
            counts = self._image_token_counts(new_images)
            config["joyai_new_images"] = new_images
            config["joyai_new_image_tokens"] = counts
            plan = self.plan_append(
                request_id=request_id,
                fence=fence,
                session_config=config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload=payload,
                final=final,
                sampling_params=sampling_params,
            )
            return plan, [(EncodedVisualImage.from_image(image), tokens) for image, tokens in zip(new_images, counts)]

        try:
            plan, new_frames = await asyncio.to_thread(prepare)
            try:
                state.visual_memory.check_capacity([image for image, _ in new_frames])
            except VisualMemoryCapacityError as exc:
                raise DuplexInputCapacityError(str(exc), code="visual_memory_capacity") from exc
            request.new_frames = new_frames
            return plan
        except BaseException:
            self.data_plane.close_stream(request_id)
            raise

    def commit_append_plan(self, request_id: str) -> None:
        request = self.data_plane.requests[request_id]
        request.state.frames = request.frames
        request.state.conversation_images = request.conversation_images
        if request.visual_generation == request.state.visual_memory.generation:
            first_frame_id = request.state.visual_memory.next_id
            frames = []
            for image, tokens in request.new_frames:
                frames.append(
                    request.state.visual_memory.append(
                        source_id=request_id, image=image, token_count=tokens, time_s=request.frame_time_s
                    )
                )
            if frames:
                request.state.observations.append(
                    WorkingObservation(
                        request_id,
                        tuple(frame.id for frame in frames),
                        request.query,
                        request.query_index,
                        query_item_id=request.query_item_id,
                        history_anchor_item_ids=request.history_anchor_item_ids,
                    )
                )
            request.response_frame_id = frames[-1].id if frames else first_frame_id
            request.response_time_s = (
                request.frame_time_s
                if frames
                else max(
                    [frame.time_s for frame in request.state.visual_memory.pending]
                    + [
                        entry.end_s
                        for entry in (*request.state.visual_memory.long_term, *request.state.visual_memory.mid_term)
                    ],
                    default=0.0,
                )
            )
            if request.query_reference is not None and not any(
                (query.item_id, query.text_digest)
                == (request.query_reference.item_id, request.query_reference.text_digest)
                for query in request.state.queries
            ):
                request.state.queries.append(request.query_reference)
                del request.state.queries[: -(RECENT_MESSAGES + SUMMARY_CHUNK_MESSAGES)]

    @staticmethod
    def _decode_images(frames: list[str]) -> list[Image.Image]:
        images = []
        for encoded in frames:
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            with Image.open(io.BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
                if image.width * image.height > 262144:
                    ratio = math.sqrt(262144 / (image.width * image.height))
                    image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
                images.append(image.convert("RGB").copy())
        return images

    def _image_token_counts(self, images: list[Image.Image]) -> list[int]:
        return [self._image_token_count(self.processor, height=image.height, width=image.width) for image in images]

    @staticmethod
    def _image_token_count(processor: ProcessorMixin, *, height: int, width: int) -> int:
        image_processor = processor.image_processor
        return image_processor.get_number_of_image_patches(height, width, {}) // image_processor.merge_size**2

    def _prepare_memory_budget(self, processor: ProcessorMixin, *, max_model_len: int) -> tuple[int, int, int]:
        memory = self.create_session_state().visual_memory

        def overhead(content: str) -> int:
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            return len(processor.tokenizer(text=[prompt])["input_ids"][0])

        mid_prompt_tokens = overhead(
            MID_TERM_SUMMARY_PROMPT.format(
                chunk_index=1,
                frame_range="0.0 seconds-999999999.9 seconds",
                length_instruction="",
                preferred_time_span="10 seconds",
            )
        )
        long_prompt_tokens = overhead(
            LONG_TERM_COMPRESS_PROMPT.format(
                merged_range="0.0 seconds-999999999.9 seconds",
                summaries_text="",
                length_instruction="",
            )
        )
        # Conservative scheduling hint for each timestamp and template boundary.
        # The actual full prompt is tokenized again before every submission.
        timestamp_tokens = overhead("<999999999.9 seconds-999999999.9 seconds>")
        image_tokens = self._image_token_count(processor, height=512, width=512)
        if (
            mid_prompt_tokens + timestamp_tokens + image_tokens + memory.mid_max_tokens > max_model_len
            or long_prompt_tokens + 2 * (memory.mid_max_tokens + timestamp_tokens) + memory.long_max_tokens
            > max_model_len
        ):
            raise DuplexRuntimeConfigError(
                "JoyAI visual memory requires a context that fits one image summary and two mid-term summaries; "
                "increase the JoyAI stage max_model_len."
            )
        return mid_prompt_tokens, long_prompt_tokens, timestamp_tokens

    @staticmethod
    def _latest_query(session_config: Mapping[str, object], runtime_config: Mapping[str, object]) -> tuple[str, int]:
        query = runtime_config.get("initial_user_text") or ""
        query_index = -1
        for index, (role, text) in enumerate(duplex_text_history(session_config.get("conversation", []))):
            if role == "user" and text:
                query, query_index = text, index
        return query, query_index

    @staticmethod
    def _current_query(session_config: Mapping[str, object], runtime_config: Mapping[str, object]) -> tuple[str, int]:
        query = runtime_config.get("initial_user_text") or ""
        query_index = -1
        for index, (role, text) in enumerate(duplex_text_history(session_config.get("conversation", []))):
            if role == "user" and text:
                query, query_index = text, index
            elif role == "assistant" and text:
                query, query_index = "", -1
        return query, query_index

    def _working_messages(
        self, session_config: Mapping[str, object], runtime_config: Mapping[str, object]
    ) -> tuple[list[dict[str, object]], list[Image.Image], list[int]]:
        """Render retained visual observations in order, without owning played Q&A."""
        pending = {frame.id: frame for frame in session_config.get("joyai_pending_frames", ())}
        groups = []
        represented: set[int] = set()
        for observation in session_config.get("joyai_observations", ()):
            frames = [pending[frame_id] for frame_id in observation.frame_ids if frame_id in pending]
            if frames:
                groups.append((frames[0].id, frames, observation))
                represented.update(frame.id for frame in frames)
        # Sources restored independently of the observation ledger still retain
        # their original frame order and timestamps.
        for frame in pending.values():
            if frame.id not in represented:
                groups.append((frame.id, [frame], WorkingObservation("", (frame.id,), "")))
        groups.sort(key=lambda item: item[0])
        messages: list[dict[str, object]] = []
        images: list[Image.Image] = []
        counts: list[int] = []
        last_query = ""
        last_query_key = None
        item_ids = session_config.get("conversation_item_ids", ())
        user_queries = {
            item_ids[index] if index < len(item_ids) and item_ids[index] else index: text
            for index, (role, text) in enumerate(duplex_text_history(session_config.get("conversation", [])))
            if role == "user" and text
        }
        query, query_index = self._current_query(session_config, runtime_config)
        current_query_key = (
            (item_ids[query_index] if 0 <= query_index < len(item_ids) and item_ids[query_index] else query_index)
            if query
            else None
        )
        retained_queries = {-1: runtime_config.get("initial_user_text") or "", **user_queries}
        for _, frames, observation in groups:
            content = []
            query_key = observation.query_item_id or observation.query_index
            included_query_key = None
            # Text edits/deletions remain authoritative in Session.history.
            if (
                observation.query
                and (observation.query != last_query or query_key != last_query_key)
                and retained_queries.get(query_key) == observation.query
            ):
                content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{observation.query}"})
                last_query = observation.query
                last_query_key = included_query_key = query_key
            for frame in frames:
                content.extend([{"type": "text", "text": f"<{frame.time_s:.1f} seconds>"}, {"type": "image"}])
                images.append(frame.encoded_image.decode())
                counts.append(frame.token_count)
            messages.append(
                {
                    "role": "user",
                    "content": content,
                    "_query_key": included_query_key,
                    "_history_anchors": observation.history_anchor_item_ids,
                    "_response_item_id": observation.response_item_id,
                }
            )
            if observation.silent:
                messages.append({"role": "assistant", "content": "</silence>"})
        new_images = session_config.get("joyai_new_images", [])
        new_counts = session_config.get("joyai_new_image_tokens", [])
        if not images and not new_images:
            # Closed sources remain stored for retries, not as the next
            # interactive observation's legacy last-four-frame fallback.
            frames_key = "joyai_new_frames" if session_config.get("joyai_closed_frame_id", -1) >= 0 else "joyai_frames"
            new_images = self._decode_images(session_config.get(frames_key, []))
            new_counts = self._image_token_counts(new_images)
        content = []
        if query and (query != last_query or current_query_key != last_query_key or not new_images):
            content.append({"type": "text", "text": f"{USER_QUERY_HEADER}\n{query}"})
        time_s = session_config.get("joyai_new_frame_time", 0.0)
        for image, tokens in zip(new_images, new_counts):
            content.extend([{"type": "text", "text": f"<{time_s:.1f} seconds>"}, {"type": "image"}])
            images.append(image)
            counts.append(tokens)
        if not content:
            content.append(
                {"type": "text", "text": "Observe the current scene and choose whether to speak or stay silent."}
            )
        messages.append({"role": "user", "content": content, "_query_key": current_query_key, "_current": True})
        return messages, images, counts

    @staticmethod
    def _archived_qa_prefix(session_config, runtime_config, history):
        """Render bounded reference metadata using only current committed text."""
        item_ids = session_config.get("conversation_item_ids", ())
        canonical = {
            item_ids[index]: (role, text)
            for index, (role, text) in enumerate(duplex_text_history(session_config.get("conversation", [])))
            if index < len(item_ids) and item_ids[index] is not None and text
        }
        seed = runtime_config.get("initial_user_text") or ""
        if seed:
            canonical[-1] = ("user", seed)
        selected = {message["_item_key"] for message in history}
        latest_query, latest_index = JoyAIDuplexPlugin._latest_query(session_config, runtime_config)
        latest_key = item_ids[latest_index] if 0 <= latest_index < len(item_ids) else -1
        closed = session_config.get("joyai_closed_frame_id", -1)
        lines = []
        prefix_keys = set()
        active_query = ""
        for query in session_config.get("joyai_queries", ()):
            key = query.item_id if query.item_id is not None else -1
            role, text = canonical.get(key, ("", ""))
            if role != "user" or duplex_history_digest((("user", text),)) != query.text_digest:
                continue
            if query.first_frame_id > closed:
                continue
            if key == latest_key and latest_query:
                active_query = latest_query
                prefix_keys.add(key)
            responses = [
                response
                for response in query.responses
                if response.source_frame_id <= closed
                and response.item_id in selected
                and canonical.get(response.item_id, ("", ""))[0] == "assistant"
            ]
            if responses:
                parts = [f"#{len(lines) + 1} [{QA_QUERY_LABEL}@{query.time_s:.1f} seconds] {text}"]
                parts.extend(
                    f"[{QA_RESPONSE_LABEL}@{response.time_s:.1f} seconds] {canonical[response.item_id][1]}"
                    for response in responses
                )
                lines.append("\n".join(parts))
                prefix_keys.add(key)
                prefix_keys.update(response.item_id for response in responses)
        return "\n".join(lines), active_query, prefix_keys

    @staticmethod
    def _interleave_history(
        history: list[dict[str, object]], working_messages: list[dict[str, object]], prefix_keys: set
    ) -> list[dict[str, object]]:
        """Place committed Q&A at its visual source, resolving item references anew."""
        by_key = {message["_item_key"]: message for message in history}
        positions = {message["_item_key"]: index for index, message in enumerate(history)}
        current_query_key = working_messages[-1].get("_query_key")
        query_has_position = current_query_key in by_key and any(
            current_query_key in message.get("_history_anchors", ()) for message in working_messages[:-1]
        )
        claimed = {
            key
            for message in working_messages
            for key in (message.get("_query_key"), message.get("_response_item_id"))
            if key is not None and not (message.get("_current") and query_has_position and key == current_query_key)
        }
        messages = []
        cursor = 0

        def append_history(end: int) -> None:
            nonlocal cursor
            while cursor < end:
                message = history[cursor]
                if message["_item_key"] not in claimed:
                    content = message["content"]
                    if message["role"] == "user":
                        content = [{"type": "text", "text": f"{USER_QUERY_HEADER}\n{content}"}]
                    messages.append({"role": message["role"], "content": content})
                cursor += 1

        for source in working_messages:
            if source.get("_current"):
                append_history(len(history))
            else:
                anchor = next(
                    (positions[key] for key in reversed(source.get("_history_anchors", ())) if key in positions), -1
                )
                append_history(anchor + 1)
            content = source["content"]
            query_key = source.get("_query_key")
            if isinstance(content, list) and query_key is not None:
                # Budget-trimmed or deleted historical queries stay removed.
                # The current instruction remains available with its images.
                if (
                    query_key in prefix_keys
                    or (source.get("_current") and query_has_position)
                    or (query_key not in by_key and query_key != current_query_key)
                ):
                    content = [part for part in content if not part.get("text", "").startswith(USER_QUERY_HEADER)]
            messages.append({"role": source["role"], "content": content})
            response = by_key.get(source.get("_response_item_id"))
            if response is not None and response["role"] == "assistant":
                messages.append({"role": "assistant", "content": response["content"]})
        return messages

    @staticmethod
    def _drop_oldest_visual_frame(messages: list[dict[str, object]]) -> None:
        """Trim one frame and its timestamp at an explicit model-context boundary."""
        for index, message in enumerate(messages):
            content = message["content"]
            if message["role"] != "user" or not isinstance(content, list):
                continue
            for part_index, part in enumerate(content):
                if part.get("type") == "image":
                    del content[part_index - 1 : part_index + 1]
                    if not content:
                        del messages[index]
                        if index < len(messages) and messages[index]["role"] == "assistant":
                            del messages[index]
                    return

    @staticmethod
    def _bounded_text_history(
        session_config: dict[str, object], runtime_config: dict[str, object]
    ) -> tuple[list[dict[str, object]], dict[str, object] | None, str]:
        """Select unsummarized recent turns and retain the latest instruction."""
        history = []
        item_ids = session_config.get("conversation_item_ids", ())
        query = runtime_config.get("initial_user_text") or ""
        latest_instruction = None
        if query:
            latest_instruction = {"role": "user", "content": query, "_item_key": -1}
            history.append(latest_instruction)
        for index, (role, text) in enumerate(duplex_text_history(session_config.get("conversation", []))):
            if role not in {"user", "assistant"}:
                continue
            if text:
                message = {
                    "role": role,
                    "content": ("</response> " if role == "assistant" else "") + text,
                    "_item_key": item_ids[index] if index < len(item_ids) and item_ids[index] else index,
                }
                if role == "user":
                    query = text
                    latest_instruction = message
                if index >= session_config.get("joyai_summary_messages", 0):
                    history.append(message)
        # Keep interactive prefill bounded even while summary generation fails
        # or falls behind. The complete unsummarized ledger stays available to
        # prepare_auxiliary_request for eventual consolidation.
        history = history[-(RECENT_MESSAGES + SUMMARY_CHUNK_MESSAGES) :]
        # A reply does not retire a standing instruction. Resolve it from the
        # current conversation even when summaries or the history bound cover it.
        if latest_instruction is not None and latest_instruction not in history:
            history = [latest_instruction, *history[-(RECENT_MESSAGES + SUMMARY_CHUNK_MESSAGES - 1) :]]
        response_refs = {item.response_item_id for item in session_config.get("joyai_observations", ())}
        while history and history[0]["role"] == "assistant" and history[0]["_item_key"] not in response_refs:
            history.pop(0)
        return history, latest_instruction, query

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        if not final or not isinstance(payload, Mapping):
            raise DuplexRuntimeConfigError("JoyAI requires committed input")
        self.input_stage_id(payload)
        stage_sampling_params = session_config.get("stage_sampling_params")
        if stage_sampling_params is not None:
            sampling_params = stage_sampling_params[self.joyai_stage_id]
        if not isinstance(sampling_params, SamplingParams):
            raise DuplexRuntimeConfigError("JoyAI requires autoregressive sampling parameters")
        if payload.get("audio"):
            return self._plan_speech_append(session_config, runtime_config, payload, sampling_params)
        history, latest_instruction, query = self._bounded_text_history(session_config, runtime_config)
        # Rendering attaches fresh queries to their visual observation and
        # leaves already-observed items at their committed history position.
        unsummarized_history = history.copy()
        working_messages, images, image_token_counts = self._working_messages(session_config, runtime_config)
        while len(images) > self.max_images:
            self._drop_oldest_visual_frame(working_messages)
            images.pop(0)
            image_token_counts.pop(0)
        system = SILENT_SYSTEM_PROMPT
        if runtime_config.get("instructions"):
            system += "\n\n" + runtime_config["instructions"]
        summary = session_config.get("joyai_summary")
        visual_summaries = [
            f"<{entry.start_s:.1f} seconds-{entry.end_s:.1f} seconds>\n{entry.text}"
            for entry in session_config.get("joyai_visual_summaries", ())
            if query or entry.chunk_index is not None
        ]
        budget = self.max_model_len - (sampling_params.max_tokens or 256)
        rebudget_history = False
        while True:
            archived_qa, active_query, prefix_keys = self._archived_qa_prefix(session_config, runtime_config, history)
            prefix = render_joyai_memory_prefix(
                summary if query else "", visual_summaries, archived_qa=archived_qa, active_query=active_query
            )
            messages = [
                {"role": "system", "content": system},
                *self._interleave_history(
                    [message for message in history if message["_item_key"] not in prefix_keys],
                    working_messages,
                    prefix_keys,
                ),
            ]
            if prefix:
                first_user = next(message for message in messages if message["role"] == "user")
                content = first_user["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                first_user["content"] = [{"type": "text", "text": prefix}, *content]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_tokens = self.processor.tokenizer(text=[prompt])["input_ids"][0]
            if len(text_tokens) + sum(image_token_counts) - len(images) <= budget:
                if rebudget_history:
                    # Omit summaries against minimal history first, then
                    # restore recent turns once instead of once per summary.
                    history = unsummarized_history.copy()
                    rebudget_history = False
                    continue
                break
            if len(history) > (1 if latest_instruction is not None else 0):
                history.pop(1 if history[0] is latest_instruction else 0)
                while history and history[0]["role"] == "assistant":
                    history.pop(0)
            elif summary:
                # Derived memory must not displace standing instructions or
                # make a previously admissible current observation fail.
                summary = ""
                # Rebudget the recent turns now that the summary no longer
                # consumes their space. Covered history remains summarized.
                rebudget_history = True
            elif visual_summaries:
                # Current observations/query take priority, as in the original
                # working chunk. Omitted blocks remain in bounded model memory.
                visual_summaries.pop(0)
                rebudget_history = True
            elif len(images) > 1:
                self._drop_oldest_visual_frame(working_messages)
                images.pop(0)
                image_token_counts.pop(0)
            else:
                raise DuplexRuntimeConfigError("JoyAI instruction and current observation exceed the context budget")
        return DuplexAppendPlan(
            prompt={
                "prompt": prompt,
                "multi_modal_data": {"image": images} if images else {},
                "additional_information": {
                    key: runtime_config[key] for key in ("tts_speaker", "tts_language", "tts_instruct")
                },
            }
        )

    def _plan_speech_append(self, session_config, runtime_config, payload, sampling_params) -> DuplexAppendPlan:
        """Keep ASR handoff budgeting deferred until the transcript is available."""
        history, _, _ = self._bounded_text_history(session_config, runtime_config)
        # The new spoken query is not known yet. Keep the initial standing
        # instruction separate from evictable history until ASR supplies it.
        instruction = runtime_config.get("initial_user_text") or ""
        history = [{"role": item["role"], "content": item["content"]} for item in history if item["_item_key"] != -1]
        images = []
        for encoded in session_config.get("joyai_frames", [])[-4:]:
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            with Image.open(io.BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
                image.thumbnail((448, 448))
                images.append(image.convert("RGB").copy())
        system = SILENT_SYSTEM_PROMPT
        if runtime_config.get("instructions"):
            system += "\n\n" + runtime_config["instructions"]
        budget = self.max_model_len - (sampling_params.max_tokens or 256)
        image_token_counts = self._image_token_counts(images)
        prompt_data = {
            "additional_information": {
                key: runtime_config[key] for key in ("tts_speaker", "tts_language", "tts_instruct")
            },
        }
        if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != 16000:
            raise DuplexRuntimeConfigError("JoyAI speech input requires normalized 16 kHz pcm_f32le")
        audio = np.frombuffer(pybase64.b64decode(payload["audio"], validate=True), dtype="<f4")
        if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
            raise DuplexRuntimeConfigError("JoyAI speech commit requires non-silent finite PCM")
        prompt_data["additional_information"].update(
            joyai_prompt_context={
                "system": system,
                "history": history,
                "instruction": instruction,
                "summary": session_config.get("joyai_summary") or "",
                "visual_summaries": [
                    f"<{entry.start_s:.1f} seconds-{entry.end_s:.1f} seconds>\n{entry.text}"
                    for entry in session_config.get("joyai_visual_summaries", ())
                ],
            },
            joyai_prompt_budget=budget,
            joyai_image_token_counts=image_token_counts,
            deferred_multi_modal_data={"image": images} if images else {},
        )
        if payload.get("realtime_item_id"):
            prompt_data["additional_information"]["realtime_item_id"] = payload["realtime_item_id"]
        prompt_data["prompt"] = (
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
            "Please transcribe this speech.<|im_end|>\n<|im_start|>assistant\n"
        )
        prompt_data["multi_modal_data"] = {"audio": (audio, 16000)}
        return DuplexAppendPlan(prompt=prompt_data)

    def user_transcript(self, *, stage_id: int, output: object, prompt: object, finished: bool) -> str | None:
        if stage_id != self.asr_stage_id or not finished or not getattr(output, "outputs", None):
            return None
        return _clean_qwen3_asr_transcript(_extract_completed_action_text(output)) or None

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        if stage_id != self.joyai_stage_id or not segment_finished:
            return None
        observation = self.data_plane.requests.get(output.request_id)
        if observation is None:
            return DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE)
        action = observation.parse(output)
        if (
            "audio" not in observation.modalities
            or action.action is not Action.RESPONSE
            or not action.text
            or observation.is_repeat(action)
        ):
            return DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE)
        return None

    def project_intermediate_output(self, *, stage_id: int, output: object, context: object) -> bool:
        return stage_id == self.joyai_stage_id and context.segment_finished
