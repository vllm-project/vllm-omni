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
from PIL import Image
from vllm.config import ModelConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputAction, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, DuplexModelSessionState, DuplexRuntimeConfigError
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import SILENT_SYSTEM_PROMPT
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import JoyAIDataPlane, JoyAISessionState
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
        if isinstance(extra_body, Mapping) and (extra_body.get("auto_response") or extra_body.get("realtime_tools")):
            raise DuplexRuntimeConfigError(
                "JoyAI video observations use explicit commits; tools use the follow-up adapter"
            )

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
        try:
            return await asyncio.to_thread(
                self.plan_append,
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
        except BaseException:
            self.data_plane.close_stream(request_id)
            raise

    def commit_append_plan(self, request_id: str) -> None:
        request = self.data_plane.requests[request_id]
        request.state.frames = request.frames
        request.state.conversation_images = request.conversation_images

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
            with Image.open(io.BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
                image.thumbnail((448, 448))
                images.append(image.convert("RGB").copy())
        system = SILENT_SYSTEM_PROMPT
        if runtime_config.get("instructions"):
            system += "\n\n" + runtime_config["instructions"]
        stage_sampling_params = session_config.get("stage_sampling_params")
        if stage_sampling_params is not None:
            sampling_params = stage_sampling_params[self.joyai_stage_id]
        if not isinstance(sampling_params, SamplingParams):
            raise DuplexRuntimeConfigError("JoyAI requires autoregressive sampling parameters")
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
                )
            except OmniClientError as exc:
                raise DuplexRuntimeConfigError(str(exc)) from exc
            prompt_data["multi_modal_data"] = {"image": images} if images else {}
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
