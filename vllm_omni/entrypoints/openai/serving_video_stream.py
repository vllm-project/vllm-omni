# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming video WebSocket handlers for multi-model pipelines.

Shared protocol (see :mod:`video_stream_base`):
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "..."}  # base64 JPEG/PNG frame
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav"}
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}

Model-specific handlers:
    :class:`QwenOmniStreamingVideoHandler` — Qwen3-Omni (thinker -> talker -> code2wav);
        turns start on ``video.query`` only.
    :class:`AuraStreamingVideoHandler` — AURA Omni (ASR -> AURA -> TTS -> code2wav);
        auto-trigger on buffered frames; ``video.query`` is ignored.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vllm_omni.model_executor.stage_input_processors.aura_cross_turn_penalty import (
    CrossTurnPenalty,
    merge_penalty_sampling_params,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    build_aura_streaming_turn_additional_information,
    frames_to_video_tuple,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    AuraSessionState,
    DEFAULT_AURA_SYSTEM_PROMPT,
    create_streaming_session,
    unregister_session,
)
from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    _DEFAULT_CONFIG_TIMEOUT,
    _DEFAULT_IDLE_TIMEOUT,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
    _decode_frame_bytes,
)
from vllm_omni.entrypoints.openai.video_stream_base import (
    OmniStreamingVideoHandler as OmniStreamingVideoHandlerBase,
)

__all__ = [
    "AuraSessionState",
    "AuraStreamingVideoHandler",
    "AuraStreamingVideoSessionConfig",
    "QwenOmniStreamingVideoHandler",
    "StreamingVideoSessionConfig",
    "create_streaming_video_handler",
]

_AURA_PIPELINE_NAMES = frozenset({"aura_omni"})
_AURA_ADDITIONAL_INFO_KEY = "_aura_additional_information"


def _resolve_deploy_pipeline(engine_client: Any) -> str | None:
    """Read ``pipeline:`` from the engine deploy YAML (entrypoints-only; no engine field)."""
    config_path = getattr(engine_client, "config_path", None)
    if config_path is None:
        return None
    from pathlib import Path

    from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config

    path = Path(config_path)
    if not path.exists():
        if path.parent != Path("."):
            return None
        bare_name = path.name if path.name.endswith(".yaml") else f"{path.name}.yaml"
        candidate = _DEPLOY_DIR / bare_name
        if not candidate.exists():
            return None
        path = candidate
    pipeline = load_deploy_config(path).pipeline
    return str(pipeline) if pipeline else None


class QwenOmniStreamingVideoHandler(OmniStreamingVideoHandlerBase):
    """Qwen-Omni pipeline: manual ``video.query`` trigger and image_pil prompts."""

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        return False

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        n_buf = len(frame_buffer)
        if n_buf <= config.num_frames:
            frames = list(frame_buffer)
        else:
            stride = max(1, n_buf // config.num_frames)
            idx = [i * stride for i in range(config.num_frames - 1)] + [n_buf - 1]
            frames = [frame_buffer[i] for i in idx]

        prewarmed = prewarmed_frames or {}
        user_content: list[dict] = []
        for frame_b64 in frames:
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME:
                continue
            if cached is not None:
                pil, pil_uuid = cached
                user_content.append(
                    {
                        "type": "image_pil",
                        "image_pil": pil,
                        "uuid": pil_uuid,
                    }
                )
            else:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    }
                )

        if len(audio_buffer) > 0:
            wav_b64 = self._pcm_to_wav_b64(bytes(audio_buffer))
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_b64,
                        "format": "wav",
                    },
                }
            )

        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}

        messages: list[dict[str, Any]] = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        recent_history = message_history[-2:] if len(message_history) > 2 else message_history
        for hist_msg in recent_history:
            messages.append(self._text_only_message(hist_msg))

        messages.append(user_message)

        return messages, user_message

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
        request_id: str | None = None,
    ) -> None:
        del request_id
        message_history.append(user_message)
        message_history.append({"role": "assistant", "content": response_text})

    _build_messages = build_engine_prompt


class AuraStreamingVideoSessionConfig(StreamingVideoSessionConfig):
    """Session config for AURA streaming video."""

    auto_trigger: bool = Field(default=True, description="Auto-start a turn after enough frames.")
    auto_trigger_min_frames: int = Field(default=2, ge=1, description="Minimum buffered frames to auto-trigger.")
    max_frames_per_round: int = Field(default=16, ge=2, description="Max frames per video_tuple.")
    pruning_enabled: bool = Field(default=True, description="Enable SessionHistory pruning.")
    max_rounds: int = Field(default=45, ge=1, description="Sliding-window round limit before pruning.")
    num_rounds_keep: int = Field(default=30, ge=1, description="Rounds to keep in sliding window after pruning.")
    max_context_qas: int = Field(default=10, ge=1, description="Max QAs in compressed context history.")
    max_1qna_rounds: int = Field(default=4, ge=1, description="Max rounds per 1QNA context-history QA.")
    aura_system_prompt: str | None = Field(default=None, description="Override AURA system prompt.")
    video_fps: float = Field(default=2.0, gt=0.0, description="FPS metadata for video_tuple.")
    cross_turn_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description="Cross-turn repetition penalty strength (0=disabled, 2.0–3.0 recommended).",
    )
    cross_turn_lookback: int = Field(
        default=10,
        ge=1,
        description="Number of recent assistant responses for cross-turn penalty window.",
    )
    cross_turn_ngram_sizes: list[int] = Field(
        default_factory=lambda: [3, 4, 5],
        description="N-gram sizes for bad_words hard blocking in cross-turn penalty.",
    )
    stream_text_deltas: bool = Field(
        default=False,
        description=(
            "When false (default), accumulate assistant text server-side and only "
            "emit response.text.done to the client. Audio streaming is unaffected."
        ),
    )
    tts_task_type: str | None = Field(default=None, description="Qwen3-TTS task type override.")
    tts_language: str | None = Field(default=None, description="Qwen3-TTS language override.")
    tts_speaker: str | None = Field(default=None, description="CustomVoice speaker name.")
    tts_ref_audio: str | None = Field(default=None, description="Base TTS reference audio path.")
    tts_ref_text: str | None = Field(default=None, description="Base TTS reference transcript.")
    tts_instruct: str | None = Field(default=None, description="VoiceDesign / style instruct text.")
    tts_pass_token_ids: bool | None = Field(
        default=None,
        description="Pass AURA assistant token ids directly to Qwen3-TTS.",
    )


class AuraStreamingVideoHandler(OmniStreamingVideoHandlerBase):
    """AURA pipeline: frame-only auto trigger (no ``video.query`` / interrupt)."""

    def supports_manual_query_turn(self) -> bool:
        return False

    def supports_query_interrupt(self) -> bool:
        return False

    def releases_turn_after_text_done(self) -> bool:
        return True

    def create_message_history(self, config: StreamingVideoSessionConfig) -> AuraSessionState:
        aura_config = self._as_aura_config(config)
        return create_streaming_session(
            max_rounds=aura_config.max_rounds,
            num_rounds_keep=aura_config.num_rounds_keep,
            pruning_enabled=aura_config.pruning_enabled,
            max_context_qas=aura_config.max_context_qas,
            max_1qna_rounds=aura_config.max_1qna_rounds,
            system_prompt=aura_config.aura_system_prompt or DEFAULT_AURA_SYSTEM_PROMPT,
        )

    def on_session_end(self, message_history: Any) -> None:
        if isinstance(message_history, AuraSessionState) and message_history.session_id:
            unregister_session(message_history.session_id)

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        config = self._as_aura_config(trigger.config)
        if not config.auto_trigger:
            return False
        return trigger.frame_count >= config.auto_trigger_min_frames and not trigger.is_turn_locked

    def auto_trigger_frame_count(
        self,
        frame_buffer: list[str],
        message_history: Any,
    ) -> int:
        del frame_buffer
        if isinstance(message_history, AuraSessionState):
            return len(message_history.turn_frame_arrays)
        return 0

    def on_frame_buffered(
        self,
        raw_bytes: bytes,
        frame_b64: str,
        message_history: Any,
        config: StreamingVideoSessionConfig,
    ) -> None:
        del frame_b64, config
        if not isinstance(message_history, AuraSessionState):
            return
        frame = _decode_frame_bytes(raw_bytes)
        message_history.append_turn_frame(frame)

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: Any,
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del frame_buffer, prewarmed_frames
        aura_config = self._as_aura_config(config)
        if not isinstance(message_history, AuraSessionState):
            raise TypeError("AURA streaming requires AuraSessionState message history")

        frames = list(message_history.turn_frame_arrays)
        video_array, metadata = frames_to_video_tuple(
            frames,
            fps=aura_config.video_fps,
            max_frames=aura_config.max_frames_per_round,
        )

        user_content: list[dict[str, Any]] = []
        if len(audio_buffer) > 0:
            wav_b64 = self._pcm_to_wav_b64(bytes(audio_buffer))
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_b64,
                        "format": "wav",
                    },
                }
            )
        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}
        messages = [user_message]

        system_prompt = aura_config.aura_system_prompt or DEFAULT_AURA_SYSTEM_PROMPT
        additional_information = build_aura_streaming_turn_additional_information(
            session_id=message_history.session_id,
            video_array=video_array,
            video_metadata=metadata,
            system_prompt=system_prompt,
            skip_asr=len(audio_buffer) == 0,
            include_tts="audio" in aura_config.modalities,
            tts_task_type=aura_config.tts_task_type,
            tts_language=aura_config.tts_language,
            tts_speaker=aura_config.tts_speaker,
            tts_ref_audio=aura_config.tts_ref_audio,
            tts_ref_text=aura_config.tts_ref_text,
            tts_instruct=aura_config.tts_instruct,
            tts_pass_token_ids=aura_config.tts_pass_token_ids,
        )
        user_message[_AURA_ADDITIONAL_INFO_KEY] = additional_information

        return messages, user_message

    def on_turn_complete(
        self,
        message_history: Any,
        user_message: dict[str, Any],
        response_text: str,
        request_id: str | None = None,
    ) -> None:
        del user_message
        if not isinstance(message_history, AuraSessionState):
            return
        message_history.commit_turn(
            response_text=response_text,
            request_id=request_id,
        )

    async def _ensure_cross_turn_penalty(
        self,
        config: AuraStreamingVideoSessionConfig,
        message_history: AuraSessionState,
    ) -> CrossTurnPenalty | None:
        if message_history.cross_turn_penalty is not None:
            return message_history.cross_turn_penalty
        if config.cross_turn_penalty <= 0 or self._engine_client is None:
            return None
        try:
            tokenizer = await self._engine_client.get_tokenizer()
        except Exception:
            return None
        message_history.cross_turn_penalty = CrossTurnPenalty(
            tokenizer,
            window=config.cross_turn_lookback,
            logit_penalty=config.cross_turn_penalty,
            ngram_sizes=config.cross_turn_ngram_sizes,
        )
        return message_history.cross_turn_penalty

    async def _receive_config(self, websocket) -> StreamingVideoSessionConfig | None:
        import asyncio
        import json

        from pydantic import ValidationError

        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._config_timeout)
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        config_data = {k: v for k, v in msg.items() if k != "type"}
        alias_map = {
            "num_sample_frames": "num_frames",
            "evs_enabled": "enable_frame_filter",
            "evs_threshold": "frame_filter_threshold",
        }
        for old_key, new_key in alias_map.items():
            if old_key in config_data and new_key not in config_data:
                config_data[new_key] = config_data[old_key]

        try:
            return AuraStreamingVideoSessionConfig(**config_data)
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

    async def _process_query_engine(
        self,
        websocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event,
        prewarmed_frames: dict[str, tuple[Any, str]],
        release_turn_lock=None,
    ) -> None:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        messages, user_message = self.build_engine_prompt(
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
        )
        additional_information = user_message.pop(_AURA_ADDITIONAL_INFO_KEY, None)

        if isinstance(message_history, AuraSessionState) and isinstance(additional_information, dict):
            deferred = additional_information.get("deferred_multi_modal_data")
            message_history.pending_turn_video = deferred if isinstance(deferred, dict) else None

        aura_config = self._as_aura_config(config)
        penalty_kwargs: dict[str, Any] = {}
        if isinstance(message_history, AuraSessionState):
            penalty = await self._ensure_cross_turn_penalty(aura_config, message_history)
            if penalty is not None:
                penalty_kwargs = penalty.build_sampling_kwargs()

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if config.sampling_params_list or penalty_kwargs:
            request_kwargs["sampling_params_list"] = merge_penalty_sampling_params(
                config.sampling_params_list,
                penalty_kwargs,
            )

        try:
            chat_request = ChatCompletionRequest(**request_kwargs)
        except Exception as e:
            await self._send_error(websocket, f"Failed to build request: {e}")
            return

        if isinstance(additional_information, dict):
            chat_request.additional_information = additional_information  # type: ignore[attr-defined]

        try:
            engine_prompt = await self._preprocess_to_engine_prompt(chat_request)
        except Exception as e:
            await self._send_error(websocket, f"Preprocess failed: {e}")
            return

        await self._run_engine_generation(
            websocket,
            config,
            message_history,
            user_message,
            request_id,
            interrupt_event,
            engine_prompt,
            release_turn_lock=release_turn_lock,
        )

    async def _run_engine_generation(
        self,
        websocket,
        config: StreamingVideoSessionConfig,
        message_history: Any,
        user_message: dict[str, Any],
        request_id: str,
        interrupt_event,
        engine_prompt: Any,
        release_turn_lock=None,
    ) -> None:
        """Stream engine outputs; release turn lock after assistant text when configured."""
        from vllm_omni.entrypoints.openai import video_stream_envs
        from vllm_omni.outputs import OmniRequestOutput

        await websocket.send_json({"type": "response.start"})
        text_parts: list[str] = []
        text_done_sent = False
        turn_lock_released = False
        audio_chunk_count = 0
        audio_chunks_drained = 0
        previous_text = ""
        interrupted = False

        async_chunk_mode = video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK
        streaming = async_chunk_mode == "on"
        aura_config = self._as_aura_config(config)
        stream_text_deltas = aura_config.stream_text_deltas
        audio_tail_tensors: list[Any] = []

        async def _try_release_turn_lock(full_text: str) -> None:
            nonlocal turn_lock_released
            if release_turn_lock is None or turn_lock_released:
                return
            turn_lock_released = True
            await release_turn_lock(
                message_history=message_history,
                user_message=user_message,
                response_text=full_text,
                request_id=request_id,
            )

        try:
            result_gen = self._engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
                output_modalities=config.modalities,
            )

            async for output in result_gen:
                if interrupt_event.is_set():
                    if not interrupted:
                        interrupted = True
                    continue

                if not isinstance(output, OmniRequestOutput):
                    continue

                out_type = getattr(output, "final_output_type", "text")

                if out_type == "audio":
                    if streaming and not text_done_sent:
                        full_text = "".join(text_parts)
                        await websocket.send_json({"type": "response.text.done", "text": full_text})
                        text_done_sent = True
                        await _try_release_turn_lock(full_text)

                    audio_chunk_count += 1
                    if streaming:
                        b64, audio_chunks_drained = self._extract_audio_delta_b64(
                            output,
                            audio_chunks_drained,
                        )
                        if b64:
                            await websocket.send_json(
                                {
                                    "type": "response.audio.delta",
                                    "data": b64,
                                    "format": "wav",
                                }
                            )
                    else:
                        audio_data = self._get_audio_data(output)
                        if audio_data is not None:
                            audio_tail_tensors = list(audio_data) if isinstance(audio_data, list) else [audio_data]
                else:
                    delta_text, previous_text = self._extract_text_delta(output, previous_text)
                    if delta_text:
                        text_parts.append(delta_text)
                        if streaming and stream_text_deltas:
                            await websocket.send_json({"type": "response.text.delta", "delta": delta_text})

            if not text_done_sent:
                full_text = "".join(text_parts)
                await websocket.send_json({"type": "response.text.done", "text": full_text})
                text_done_sent = True
                await _try_release_turn_lock(full_text)

            if not streaming and audio_tail_tensors:
                import torch

                try:
                    coalesced = (
                        audio_tail_tensors[0] if len(audio_tail_tensors) == 1 else torch.cat(audio_tail_tensors, dim=-1)
                    )
                    tail_np = self._tensor_to_1d_np(coalesced)
                    b64, _ = self._encode_tail(tail_np, 0, new_drained=len(audio_tail_tensors), is_first=True)
                    if b64:
                        await websocket.send_json(
                            {
                                "type": "response.audio.delta",
                                "data": b64,
                                "format": "wav",
                            }
                        )
                except Exception:
                    pass

            if audio_chunk_count > 0:
                await websocket.send_json({"type": "response.audio.done"})

            if release_turn_lock is None and not turn_lock_released:
                response_text = "".join(text_parts)
                self.on_turn_complete(message_history, user_message, response_text, request_id)

        except Exception:
            await self._send_error(websocket, "Query processing failed")

        if not text_done_sent:
            full_text = "".join(text_parts)
            await websocket.send_json({"type": "response.text.done", "text": full_text})

    @staticmethod
    def _as_aura_config(config: StreamingVideoSessionConfig) -> AuraStreamingVideoSessionConfig:
        if isinstance(config, AuraStreamingVideoSessionConfig):
            return config
        return AuraStreamingVideoSessionConfig(**config.model_dump())


def create_streaming_video_handler(
    chat_service: Any,
    idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    engine_client: Any | None = None,
) -> OmniStreamingVideoHandlerBase:
    """Create the handler for ``/v1/video/chat/stream``.

    Routes to :class:`AuraStreamingVideoHandler` when the deploy YAML
    ``pipeline`` is ``aura_omni``.
    """
    pipeline = _resolve_deploy_pipeline(engine_client) if engine_client is not None else None
    if pipeline in _AURA_PIPELINE_NAMES:
        return AuraStreamingVideoHandler(
            chat_service=chat_service,
            idle_timeout=idle_timeout,
            config_timeout=config_timeout,
            engine_client=engine_client,
        )

    return QwenOmniStreamingVideoHandler(
        chat_service=chat_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
        engine_client=engine_client,
    )
