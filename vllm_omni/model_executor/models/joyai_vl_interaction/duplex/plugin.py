# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn-commit JoyAI observations with optional Qwen3-ASR speech input."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable, Mapping
from copy import deepcopy

import numpy as np
import pybase64
from vllm.config import ModelConfig
from vllm.multimodal.image import open_image
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputAction, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, DuplexModelSessionState, DuplexRuntimeConfigError
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import SILENT_SYSTEM_PROMPT
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import (
    JoyAIDataPlane,
    JoyAISessionState,
    Observation,
    advance_chunk,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.delegation import (
    DELEGATION_TOOL_NAME,
    DelegationResult,
    accept_result,
    delegation_system_section,
    expire_unanswered,
    expired_results,
    reconcile_delegation_runtime,
    render_result_blocks,
    result_records,
    settle_response,
    unanswered_results,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer
from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import (
    _clean_qwen3_asr_transcript,
    _extract_completed_action_text,
    render_joyai_session_prompt,
)


class JoyAIDuplexPlugin(DuplexModelPlugin):
    plugin_id = "joyai-vl-interaction"
    joyai_stage_id = 0
    asr_stage_id: int | None = None
    audio_stage_id = 2
    #: Accepted delegation results are session state: a client ``session.update``
    #: must not be able to inject or drop them through ``extra_body``.
    private_runtime_config_keys = frozenset(
        {
            "joyai_delegation_results",
            "joyai_chunk_id",
            "joyai_chunk_spoken",
            "_joyai_delegation_displayed",
            "_joyai_chunk_id",
        }
    )

    def __init__(self, encode_audio: Callable[..., str]) -> None:
        self.data_plane = JoyAIDataPlane(encode_audio)
        self.processor = None
        self.max_model_len = 0
        self._processor_lock = asyncio.Lock()

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
        if not isinstance(extra_body, Mapping):
            return
        if extra_body.get("auto_response"):
            raise DuplexRuntimeConfigError("JoyAI observations use explicit camera commits, not auto responses")
        tools = extra_body.get("realtime_tools")
        if tools is not None and not self.declares_delegation(tools):
            raise DuplexRuntimeConfigError(
                "JoyAI sessions only declare the delegate tool; any other tool is unsupported"
            )

    @staticmethod
    def declares_delegation(tools: object) -> bool:
        """Whether a ``tools`` declaration asks for exactly the JoyAI delegation function."""
        if not isinstance(tools, list) or not tools:
            return False
        for tool in tools:
            if not isinstance(tool, Mapping):
                return False
            if tool.get("type", "function") != "function" or tool.get("name") != DELEGATION_TOOL_NAME:
                return False
        return True

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        async with self._processor_lock:
            if self.processor is None:
                if model_config is None:
                    raise DuplexRuntimeConfigError("JoyAI requires its comprehension-stage model configuration")
                from vllm.transformers_utils.processor import cached_processor_from_config

                self.processor = await asyncio.to_thread(
                    cached_processor_from_config,
                    model_config,
                )
                self.max_model_len = model_config.max_model_len
        return self.runtime_config_for_update(config, {})

    def runtime_config_for_update(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("JoyAI's current TTS profile uses CustomVoice speakers")
        extra = config.extra_body or {}
        updated: dict[str, object] = {
            "instructions": config.instructions,
            "initial_user_text": config.initial_user_text,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **{
                key: extra.get(key, default)
                for key, default in (("tts_speaker", "Vivian"), ("tts_language", "Auto"), ("tts_instruct", ""))
            },
        }
        # Delegation and chunk state are per-session bookkeeping, not session
        # configuration: an update must carry them forward or accepted results
        # would vanish and the chunk identity would restart.
        for key in ("joyai_delegation_results", "joyai_chunk_id", "joyai_chunk_spoken"):
            if key in current:
                updated[key] = current[key]
        # The declared tools are session configuration and only the delegate
        # function is accepted, so the validated declaration passes through.
        tools = extra.get("realtime_tools")
        if tools is not None:
            updated["realtime_tools"] = tools
        return updated

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

    def prepare_prompt_config(
        self, config: dict[str, object], *, state: DuplexModelSessionState, payload: dict[str, object]
    ) -> dict[str, object]:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        images = tuple(
            part["image_url"]["url"]
            for message in config.get("conversation", [])
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        )
        previous = state.conversation_images
        if images[: len(previous)] == previous:
            frames = state.frames
            new_images = images[len(previous) :]
        else:
            # Rebuild the visual window after conversation images are edited or deleted.
            frames = []
            new_images = images
        # Append each conversation image once, alongside newly committed camera
        # frames, so retained camera history cannot displace a newer user image.
        frames = [*frames, *new_images, *payload.get("video_frames", [])][-4:]
        return {**config, "joyai_state": state, "joyai_frames": frames, "joyai_conversation_images": images}

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
        request.delegation_tools = self.delegation_enabled(runtime_config)
        # The chunk identity is session state decided here, not inside the
        # worker, so the follow-up prompt and the published call agree on it.
        planned = self._advance_chunk(runtime_config, request, has_camera=bool(payload.get("video_frames")))
        # The planner runs off-loop: hand it the same frozen view of the session
        # that the rest of the plan sees, never the live plugin state.
        config["joyai_chunk_id"] = request.chunk_id
        try:
            plan = await asyncio.to_thread(
                self.plan_append,
                request_id=request_id,
                fence=fence,
                session_config=config,
                runtime_config=planned,
                seq=seq,
                turn_seq=turn_seq,
                payload=payload,
                final=final,
                sampling_params=sampling_params,
            )
        except BaseException:
            self.data_plane.close_stream(request_id)
            raise
        plan.prompt["_joyai_chunk_id"] = request.chunk_id
        return plan

    def commit_append_plan(self, request_id: str) -> None:
        request = self.data_plane.requests[request_id]
        request.state.frames = request.frames
        request.state.conversation_images = request.conversation_images

    def _advance_chunk(
        self, runtime_config: Mapping[str, object], request: Observation, *, has_camera: bool
    ) -> dict[str, object]:
        """Give this observation its chunk identity.

        JoyAI's interaction is chunk based: a spoken reply answers the goal of
        the current chunk and stays the goal until the chunk closes. Both the
        chunk id and the "a reply of it stored text" flag live in the session
        runtime config, so two sessions of one deployment cannot advance each
        other's chunk.
        """
        current = runtime_config.get("joyai_chunk_id")
        chunk, spoken = advance_chunk(
            current=current,
            spoken=runtime_config.get("joyai_chunk_spoken") is True,
            has_camera=has_camera,
        )
        request.chunk_id = chunk
        if chunk == current and spoken == (runtime_config.get("joyai_chunk_spoken") is True):
            return dict(runtime_config)
        return {**runtime_config, "joyai_chunk_id": chunk, "joyai_chunk_spoken": spoken}

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
        has_audio = bool(payload.get("audio"))
        self.input_stage_id(payload)
        delegation = self.delegation_enabled(runtime_config)
        history = []
        query = runtime_config.get("initial_user_text") or ""
        if has_audio and query:
            history.append({"role": "user", "content": query})
        for message in session_config.get("conversation", []):
            role, content = message.get("role"), message.get("content")
            if role not in {"user", "assistant"}:
                continue
            if isinstance(content, list):
                text = "\n".join(
                    part.get("text", "") if part.get("type") == "text" else part.get("transcript", "")
                    for part in content
                    if part.get("type") in {"text", "audio_url"}
                )
            else:
                text = content if isinstance(content, str) else ""
            if text:
                if role == "user":
                    query = text
                history.append({"role": role, "content": ("</response> " if role == "assistant" else "") + text})
        history = history[-8:]
        while history and history[0]["role"] == "assistant":
            history.pop(0)
        # The active instruction is attached to the latest observation below.
        if not has_audio and history and history[-1]["role"] == "user":
            history.pop()
        images = []
        for encoded in session_config.get("joyai_frames", [])[-4:]:
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            with open_image(io.BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
                image.thumbnail((448, 448))
                images.append(image.convert("RGB").copy())
        system = SILENT_SYSTEM_PROMPT
        if delegation:
            system += "\n\n" + delegation_system_section()
        if runtime_config.get("instructions"):
            system += "\n\n" + runtime_config["instructions"]
        stage_sampling_params = session_config.get("stage_sampling_params")
        if stage_sampling_params is not None:
            sampling_params = stage_sampling_params[self.joyai_stage_id]
        if not isinstance(sampling_params, SamplingParams):
            raise DuplexRuntimeConfigError("JoyAI requires autoregressive sampling parameters")
        # The plan runs on a worker thread and must not read live session state,
        # so the chunk identity is the one the session loop bound to this request.
        chunk = session_config.get("joyai_chunk_id")
        results: list[DelegationResult] = []
        rendered_call_ids: list[str] = []
        results_block = ""
        if delegation:
            results = unanswered_results(runtime_config, chunk=chunk)
            results_block, rendered_call_ids = render_result_blocks(results)
        # The rendered block is part of the prompt, so the prompt renderer below
        # counts its tokens against the budget; subtracting its characters here
        # too would reserve the same context twice.
        budget = self.max_model_len - (sampling_params.max_tokens or 256)
        # Qwen3-VL expands each image placeholder to this many visual tokens.
        # Infer the grid from image sizes once; budget trimming needs no pixels.
        image_processor = self.processor.image_processor
        image_token_counts = [
            image_processor.get_number_of_image_patches(image.height, image.width, {}) // image_processor.merge_size**2
            for image in images
        ]
        prompt_data = {
            "additional_information": {
                key: runtime_config[key] for key in ("tts_speaker", "tts_language", "tts_instruct")
            },
        }
        if has_audio:
            if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != 16000:
                raise DuplexRuntimeConfigError("JoyAI speech input requires normalized 16 kHz pcm_f32le")
            audio = np.frombuffer(pybase64.b64decode(payload["audio"], validate=True), dtype="<f4")
            if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
                raise DuplexRuntimeConfigError("JoyAI speech commit requires non-silent finite PCM")
            prompt_data["additional_information"].update(
                joyai_prompt_context={"system": system, "history": history},
                joyai_prompt_budget=budget,
                joyai_image_token_counts=image_token_counts,
                deferred_multi_modal_data={"image": images} if images else {},
            )
            if results_block:
                prompt_data["additional_information"]["joyai_delegation_context"] = results_block
            if payload.get("realtime_item_id"):
                prompt_data["additional_information"]["realtime_item_id"] = payload["realtime_item_id"]
            prompt_data["prompt"] = (
                "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                "Please transcribe this speech.<|im_end|>\n<|im_start|>assistant\n"
            )
            prompt_data["multi_modal_data"] = {"audio": (audio, 16000)}
        else:
            try:
                prompt_data["prompt"], images = render_joyai_session_prompt(
                    self.processor,
                    system=system,
                    history=history,
                    query=query,
                    images=images,
                    image_token_counts=image_token_counts,
                    budget=budget,
                    context_blocks=[results_block] if results_block else (),
                )
            except OmniClientError as exc:
                raise DuplexRuntimeConfigError(str(exc)) from exc
            prompt_data["multi_modal_data"] = {"image": images} if images else {}
        if rendered_call_ids:
            # Private plan-time marks: the session books them only once the
            # stage accepts this append, and reconciles them with the live
            # runtime, which may have accepted a newer result meanwhile. Only
            # the results this prompt actually rendered are booked; a result
            # trimmed out of the block stays available for a later prompt.
            prompt_data["_joyai_delegation_displayed"] = rendered_call_ids
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
            or action.action not in {Action.RESPONSE, Action.DELEGATE}
            or not action.text
            or (action.action is Action.DELEGATE and not observation.delegation_tools)
            or observation.is_repeat(action)
        ):
            return DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE)
        return None

    def project_intermediate_output(self, *, stage_id: int, output: object, context: object) -> bool:
        return stage_id == self.joyai_stage_id and context.segment_finished

    # ---- delegation: the shared tool-call lifecycle, model side ---------- #

    @staticmethod
    def delegation_enabled(runtime_config: Mapping[str, object]) -> bool:
        tools = runtime_config.get("realtime_tools")
        return JoyAIDuplexPlugin.declares_delegation(tools)

    def parse_function_call(self, model_result: Mapping[str, object]) -> dict[str, str] | None:
        """Publish this observation's delegation as one shared function call.

        A call is terminal for its observation even when it also spoke a note:
        the reply that will matter is the one answering its result. The chunk
        flag is not set here -- it lives in the session runtime, and a call
        alone must not close the chunk (there is no spoken reply to repeat).
        """
        call = model_result.get("joyai_function_call")
        if not isinstance(call, Mapping):
            return None
        call_id, name, arguments = call.get("call_id"), call.get("name"), call.get("arguments")
        if not isinstance(call_id, str) or not call_id or name != DELEGATION_TOOL_NAME:
            return None
        if not isinstance(arguments, str) or not arguments:
            return None
        return {"call_id": call_id, "name": name, "arguments": arguments}

    @staticmethod
    def has_pending_tool_followup(runtime_config: Mapping[str, object]) -> bool:
        """Whether an accepted result is still waiting for an observation.

        JoyAI observations are driven by camera commits, so the next commit
        carries the result; no ``response.create`` is needed.
        """
        return bool(unanswered_results(runtime_config))

    @staticmethod
    def extra_prompt_config(runtime_config: Mapping[str, object]) -> Mapping[str, object]:
        records = result_records(runtime_config)
        if not records:
            return {}
        return {"joyai_delegation_results": records}

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
        *,
        history_len: int | None = None,
    ) -> dict[str, object] | None:
        """Prepare one accepted result without consuming anything.

        ``name`` and ``arguments`` are the session's immutable call origin, so a
        client cannot rebind the result to another question. Refusals raise
        ``DuplexRuntimeConfigError`` and leave the call pending for a retry.
        """
        del config
        output = item.get("output")
        if not isinstance(output, str):
            raise DuplexRuntimeConfigError(
                "function_call_output requires a string output",
                code="invalid_function_call_output",
            )
        if item.get("name") != DELEGATION_TOOL_NAME:
            raise DuplexRuntimeConfigError(
                "function_call_output is not for the JoyAI delegate tool",
                code="invalid_function_call_output",
            )
        try:
            return accept_result(current, item, accepted_at=history_len if isinstance(history_len, int) else 0)
        except ValueError as exc:
            raise DuplexRuntimeConfigError(str(exc), code="invalid_function_call_output") from exc

    def runtime_patch_for_submitted_prompt(self, prompt: dict[str, object]) -> Mapping[str, object] | None:
        """Book the results this accepted prompt displays."""
        displayed = prompt.pop("_joyai_delegation_displayed", None)
        chunk = prompt.pop("_joyai_chunk_id", None)
        patch: dict[str, object] = {}
        if isinstance(displayed, list) and displayed:
            patch["_joyai_delegation_displayed"] = [call_id for call_id in displayed if isinstance(call_id, str)]
        if isinstance(chunk, str):
            patch["joyai_chunk_id"] = chunk
        return patch or None

    def reconcile_submitted_runtime(
        self,
        current: Mapping[str, object],
        updated: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        """Book this prompt's display marks without dropping newer results."""
        displayed = updated.get("_joyai_delegation_displayed")
        if not isinstance(displayed, list) or not displayed:
            return None
        call_ids = [call_id for call_id in displayed if isinstance(call_id, str)]
        chunk = updated.get("joyai_chunk_id")
        live = reconcile_delegation_runtime(current, displayed=call_ids, chunk=chunk)
        if live is None:
            return None
        # The reconciler's return value is installed as the whole runtime, so
        # keep every key this prompt was planned against.
        reconciled = dict(current)
        reconciled["joyai_delegation_results"] = live
        return reconciled

    def runtime_config_after_response(
        self,
        runtime_config: Mapping[str, object],
        *,
        committed: bool,
        awaiting_playback: bool = False,
        response_id: str | None = None,
        release_owner: bool = False,
    ) -> dict[str, object] | None:
        """Apply one finished response to the results its prompt displayed.

        A stored reply answers them. A reply that stored nothing (a repeat, a
        silent decision, a failure) leaves them unanswered, and the next frame
        carries them again. ``awaiting_playback`` keeps the decision open: the
        reply may still store its text when the playback acknowledgement
        arrives, and that path calls this hook again.
        """
        del response_id, release_owner
        if awaiting_playback:
            return None
        chunk = runtime_config.get("joyai_chunk_id")
        expired = expired_results(runtime_config, chunk=chunk)
        updated = settle_response(runtime_config, committed=committed, chunk=chunk, expired=expired)
        if not committed:
            return updated
        # A stored reply closes the chunk at its next camera frame. The flag is
        # session state, so it travels with the runtime rather than a plugin slot.
        spoken = dict(updated if updated is not None else runtime_config)
        spoken["joyai_chunk_spoken"] = True
        return spoken

    def retire_pending_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object] | None:
        """The conversation a pending result belonged to was cancelled.

        JoyAI writes each prompt from the session's own context, so there is no
        queued frame to retire. What does become unreachable is a result the
        cancelled turn never answered: its question is gone with the turn, and
        it must not surface in a later observation as if it had been asked.
        """
        del config
        return expire_unanswered(current)
