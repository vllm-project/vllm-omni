# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn-commit JoyAI observations on the existing three-stage pipeline."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Callable, Mapping
from copy import deepcopy

import pybase64
from PIL import Image
from vllm.config import ModelConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputAction, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, DuplexModelSessionState, DuplexRuntimeConfigError
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import SILENT_SYSTEM_PROMPT, USER_QUERY_HEADER
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import JoyAIDataPlane, JoyAISessionState


class JoyAIDuplexPlugin(DuplexModelPlugin):
    plugin_id = "joyai-vl-interaction"

    def __init__(self, encode_audio: Callable[..., str]) -> None:
        self.data_plane = JoyAIDataPlane(encode_audio)
        self.processor = None
        self.max_model_len = 0
        self._processor_lock = asyncio.Lock()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities(
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=True,
            supports_turn_commit_only=True,
            supports_core_resumable_request=False,
            required_input_modalities=frozenset(),
            optional_input_modalities=frozenset({"video"}),
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
        return JoyAISessionState()

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
                    raise DuplexRuntimeConfigError("JoyAI requires the stage-0 model configuration")
                from transformers import AutoProcessor

                self.processor = await asyncio.to_thread(
                    AutoProcessor.from_pretrained,
                    model_config.model,
                    trust_remote_code=model_config.trust_remote_code,
                    revision=getattr(model_config, "revision", None),
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
                if index == 0:
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
        if not final or not isinstance(payload, Mapping) or payload.get("audio"):
            raise DuplexRuntimeConfigError("JoyAI video profile requires committed text/image/video input")
        history = []
        query = runtime_config.get("initial_user_text") or ""
        for message in session_config.get("conversation", []):
            role, content = message.get("role"), message.get("content")
            if role not in {"user", "assistant"}:
                continue
            if isinstance(content, list):
                text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
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
        if history and history[-1]["role"] == "user":
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
        while True:
            content = [{"type": "image"} for _ in images]
            content.append(
                {
                    "type": "text",
                    "text": f"{USER_QUERY_HEADER}\n{query}"
                    if query
                    else "Observe the current scene and choose whether to speak or stay silent.",
                }
            )
            messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_tokens = self.processor.tokenizer(text=[prompt])["input_ids"][0]
            if len(text_tokens) + sum(image_token_counts) - len(images) <= budget:
                break
            if history:
                history.pop(0)
                while history and history[0]["role"] == "assistant":
                    history.pop(0)
            elif len(images) > 1:
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
        if stage_id != 0 or not segment_finished:
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
        return stage_id == 0 and context.segment_finished
