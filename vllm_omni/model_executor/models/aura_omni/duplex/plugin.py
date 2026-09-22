# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA full-duplex model plugin: turn-commit four-stage NewRequest."""

from __future__ import annotations

import binascii
from collections.abc import Mapping
from typing import Any

import numpy as np
import pybase64 as base64
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, EncodeAudio
from vllm_omni.model_executor.models.aura_omni.duplex.capabilities import aura_duplex_capabilities
from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import (
    AuraDataPlaneContext,
    AuraDataPlaneSession,
)
from vllm_omni.model_executor.models.aura_omni.duplex.session import AuraServingSessionState
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    DEFAULT_AURA_SYSTEM_PROMPT,
    SILENT_TEXT,
    is_effectively_silent,
)

# AURA v1 (Qwen3-VL) silent / ChatML turn-end ids.
AURA_SILENT_TOKEN_ID = 151669
AURA_IM_END_TOKEN_ID = 151645
AURA_SILENT_TOKEN_IDS = frozenset({AURA_SILENT_TOKEN_ID})

_PRIVATE_KEYS = frozenset(
    {
        "aura_system_prompt",
        "aura_history_revision",
    }
)

_TTS_EXTRA_KEYS = (
    "tts_task_type",
    "tts_language",
    "tts_speaker",
    "tts_instruct",
    "tts_ref_audio",
    "tts_ref_text",
    "tts_x_vector_only_mode",
)


def _decode_pcm_f32le(payload: Mapping[str, object]) -> tuple[np.ndarray, int]:
    audio = payload.get("audio")
    sample_rate_hz = payload.get("sample_rate_hz", 16000)
    if not isinstance(audio, str) or not isinstance(sample_rate_hz, int):
        raise ValueError("AURA duplex commit payload requires pcm_f32le audio and sample_rate_hz")
    try:
        raw = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("AURA duplex audio is not valid base64") from exc
    if len(raw) % 4:
        raise ValueError("AURA duplex pcm_f32le payload has a partial sample")
    values = np.frombuffer(raw, dtype="<f4").copy()
    return values, sample_rate_hz


def _video_frames_to_mm(frames: object) -> dict[str, object]:
    if not isinstance(frames, list) or not frames:
        return {}
    from io import BytesIO

    from PIL import Image

    images: list[object] = []
    for frame in frames:
        if not isinstance(frame, str) or not frame:
            continue
        try:
            raw = base64.b64decode(frame, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            images.append(np.asarray(Image.open(BytesIO(raw)).convert("RGB")))
        except Exception:
            continue
    if not images:
        return {}
    # One <|video_pad|> for the whole clip. Two separate images with one
    # <|image_pad|> crash Stage1 (mm_items['image'][1]). Qwen3-VL's temporal
    # patch wants at least two frames; a single sticky frame is duplicated.
    if len(images) == 1:
        images = [images[0], images[0]]
    video = np.stack(images, axis=0)
    n_frames = int(video.shape[0])
    metadata = {
        "fps": 2.0,
        "duration": n_frames / 2.0,
        "total_num_frames": n_frames,
        "frames_indices": list(range(n_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return {"video": [(video, metadata)]}


def _completion_token_ids(completion: object | None) -> list[int]:
    if completion is None:
        return []
    for attr in ("token_ids", "cumulative_token_ids"):
        value = getattr(completion, attr, None)
        if isinstance(value, list | tuple):
            out: list[int] = []
            for item in value:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
            if out:
                return out
    return []


def _first_completion(output: object) -> object | None:
    outputs = getattr(output, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return outputs[0]
    return None


class AuraDuplexPlugin(DuplexModelPlugin):
    """Turn-commit AURA: one ephemeral four-stage request per utterance."""

    plugin_id = "aura"
    private_runtime_config_keys = _PRIVATE_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = AuraDataPlaneSession(encode_audio)

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        configured = list(defaults)
        if len(configured) > 1 and isinstance(configured[1], SamplingParams):
            stage1 = configured[1].clone()
            stop_ids = list(stage1.stop_token_ids or [])
            for stop_id in (AURA_SILENT_TOKEN_ID, AURA_IM_END_TOKEN_ID):
                if stop_id not in stop_ids:
                    stop_ids.append(stop_id)
            stage1.stop_token_ids = stop_ids
            # 151669 is a stop id: keep it in the completion so decide_output /
            # aura2tts can see <|silent|> instead of an empty detokenized string.
            stage1.include_stop_str_in_output = True
            stage1.skip_special_tokens = False
            configured[1] = stage1
        # Codec EOS (2150) and the 240-token cap are defaults for an unset
        # Talker config. A value already set by yaml or the caller is kept,
        # even when max_tokens is longer than 240.
        if len(configured) > 2 and isinstance(configured[2], SamplingParams):
            stage2 = configured[2].clone()
            if not stage2.stop_token_ids:
                stage2.stop_token_ids = [2150]
            if stage2.max_tokens is None:
                stage2.max_tokens = 240
            configured[2] = stage2
        return tuple(configured)

    def draining_stage_ids(self, *, stage_count: int) -> frozenset[int]:
        # Talker is stage 2; Code2Wav is the last stage. Stage 0/1 are the
        # input gate and are idle once a turn is released.
        if stage_count <= 2:
            return frozenset()
        return frozenset(range(2, stage_count))

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
        del request_id, session_config, seq, turn_seq, sampling_params

        def _skip() -> DuplexAppendPlan:
            return DuplexAppendPlan(
                prompt={
                    "prompt_token_ids": [0],
                    "additional_information": {
                        "aura_skip_submit": True,
                        "session_id": fence.session_id,
                        "epoch": fence.epoch,
                        "turn_id": fence.turn_id,
                    },
                }
            )

        if not isinstance(payload, Mapping):
            if not final:
                return _skip()
            raise ValueError("AURA duplex plan_append expects a mapping payload")
        if not final and not payload.get("aura_turn_commit"):
            return _skip()

        mm = _video_frames_to_mm(payload.get("video_frames"))
        audio_b64 = payload.get("audio")
        is_speech = bool(payload.get("is_speech", True))
        has_audio = isinstance(audio_b64, str) and bool(audio_b64)
        if has_audio:
            wav, sample_rate_hz = _decode_pcm_f32le(payload)
        else:
            wav, sample_rate_hz = np.zeros(0, dtype=np.float32), int(payload.get("sample_rate_hz") or 16000)
        # Vision-follow commits often carry empty or near-silent PCM with
        # is_speech=False. All-zero audio crashes Qwen3ASRProcessor, and an
        # empty Stage0 prompt is rejected ("decoder prompt cannot be empty").
        # Also guard is_speech=True + empty-audio + frames (prompt was "").
        sample_rate_hz = int(sample_rate_hz or payload.get("sample_rate_hz") or 16000)
        wav_rms = float(np.sqrt(np.mean(np.square(wav.astype(np.float32, copy=False))))) if wav.size else 0.0
        near_silent = wav.size == 0 or not np.any(wav) or wav_rms < 1e-3
        if mm and (not has_audio or (not is_speech and near_silent)):
            n = max(1600, int(sample_rate_hz) // 10)
            wav = np.linspace(1e-4, -1e-4, n, dtype=np.float32)
            has_audio = True
        if not has_audio and not mm:
            raise ValueError("AURA duplex commit requires audio and/or video_frames")
        system_prompt = runtime_config.get("aura_system_prompt", DEFAULT_AURA_SYSTEM_PROMPT)

        additional_information: dict[str, object] = {
            "aura_system_prompt": system_prompt,
            "aura_session_id": fence.session_id,
            "aura_duplex": True,
            "session_id": fence.session_id,
            "epoch": fence.epoch,
            "turn_id": fence.turn_id,
            # asr2aura ignores Stage0 text when False (placeholder zeros).
            "is_speech": is_speech,
            "tts_task_type": runtime_config.get("tts_task_type", "CustomVoice"),
            "tts_language": runtime_config.get("tts_language", "Chinese"),
            "tts_speaker": runtime_config.get("tts_speaker", "Vivian"),
        }
        item_id = payload.get("realtime_item_id")
        if isinstance(item_id, str) and item_id:
            additional_information["realtime_item_id"] = item_id
        for key in _TTS_EXTRA_KEYS:
            if key in additional_information:
                continue
            value = runtime_config.get(key)
            if value is not None:
                additional_information[key] = value
        if mm:
            additional_information["deferred_multi_modal_data"] = mm

        asr_prompt = (
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|>"
            "Please transcribe this speech.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        multi_modal_data: dict[str, object] = {}
        if has_audio:
            multi_modal_data["audio"] = (wav, sample_rate_hz)
        # Never submit an empty decoder prompt: Stage0 always needs the ASR pad
        # when audio (or a vision-driven pad) is present.
        prompt: dict[str, Any] = {
            "prompt": asr_prompt if has_audio else "",
            "multi_modal_data": multi_modal_data,
            "additional_information": additional_information,
        }
        return DuplexAppendPlan(prompt=prompt)

    def project_intermediate_output(
        self,
        *,
        stage_id: int,
        output: object,
        context: object,
    ) -> bool:
        """Project Stage1 thinker text to the client without short-circuiting TTS."""
        del output, context
        return stage_id == 1

    def user_transcript(
        self,
        *,
        stage_id: int,
        output: object,
        prompt: object,
        finished: bool,
    ) -> str | None:
        """Stage0 ASR text for a spoken turn, so the demo can show what was said.

        Vision-follow sets ``is_speech`` false; that audio is a silent pad and
        must not open a user bubble. The pipeline still forwards Stage0.
        """
        if stage_id != 0 or not finished:
            return None
        info = prompt.get("additional_information") if isinstance(prompt, dict) else None
        if isinstance(info, dict) and info.get("is_speech") is False:
            return None
        from vllm_omni.model_executor.stage_input_processors.aura_omni import (
            _extract_text,
            _normalize_asr_transcript,
            is_effectively_silent,
        )

        text = _normalize_asr_transcript(_extract_text(output))
        if not text or is_effectively_silent(text):
            return None
        return text

    def plan_partial_stage_output(
        self,
        orchestrator: Any,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: Any,
    ):
        from vllm_omni.model_executor.models.aura_omni.duplex.sentence_tts import (
            plan_partial_stage_output,
        )

        return plan_partial_stage_output(orchestrator, stage_id, replica_id, output, req_state)

    def partial_stage_followup(self, plan: Any, req_state: Any) -> Any:
        """Close the Talker stream after a resumable final sentence.

        The sentence text was already forwarded with ``queue_close_after``.
        Flipping the prompt flag here makes the second submit a close-only
        sentinel instead of another copy of that sentence.
        """
        if plan is None or not getattr(plan, "queue_close_after", False):
            return None
        prompt = getattr(req_state, "prompt", None)
        info = prompt.get("additional_information") if isinstance(prompt, dict) else None
        if isinstance(info, dict):
            info["aura_tts_partial"] = True
            info["aura_tts_close_only"] = True
        from vllm_omni.engine.duplex.plugin import PartialStageForward
        from vllm_omni.model_executor.models.aura_omni.duplex.sentence_tts import (
            SentenceTtsOutput,
        )

        request_id = str(getattr(plan.output, "request_id", ""))
        return PartialStageForward(
            output=SentenceTtsOutput(request_id, ""),
            is_final_update=True,
            close_only=True,
        )

    def commit_model_context(self, *, session_id: str | None, assistant_text: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            return
        from vllm_omni.model_executor.models.aura_omni.duplex.history import get_or_create_session_history

        get_or_create_session_history(session_id).commit_turn(assistant_text or SILENT_TEXT)

    def release_concurrent_turn_requests(
        self,
        *,
        stage_id: int,
        segment_finished: bool,
        output: object,
        context: object,
    ) -> bool:
        """After Stage1 text/silent final, next commit may start while TTS drains."""
        del context
        if stage_id != 1:
            return False
        if segment_finished:
            return True
        return bool(getattr(output, "finished", False))

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
        del final_stage_id, segment_output_metadata
        # Stage1 silent → short-circuit TTS/Code2Wav.
        if stage_id != 1 or not segment_finished:
            return None
        completion = _first_completion(output)
        token_ids = _completion_token_ids(completion) or list(segment_token_ids)
        text = getattr(completion, "text", None) if completion is not None else None
        cumulative = getattr(completion, "cumulative_text", None) if completion is not None else None
        text_blob = " ".join(part for part in (text, cumulative) if isinstance(part, str))
        # Empty finished Stage1 is silent: stop_token 151669 is often omitted from
        # token_ids/text unless include_stop_str_in_output is set.
        is_silent = (
            any(sid in token_ids for sid in AURA_SILENT_TOKEN_IDS)
            or (isinstance(text, str) and is_effectively_silent(text))
            or (isinstance(cumulative, str) and is_effectively_silent(cumulative))
            or SILENT_TEXT in text_blob
        )
        if not is_silent and not token_ids and not text_blob.strip():
            is_silent = True
        if not is_silent and token_ids:
            is_silent = token_ids[0] in AURA_SILENT_TOKEN_IDS or (
                len(token_ids) > 3 and any(sid in token_ids[:6] for sid in AURA_SILENT_TOKEN_IDS)
            )
        if not is_silent:
            return None
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata={
                "duplex_direct_response": True,
                "model_listen": True,
                "listen_source": "aura_silent",
                "silent_text": SILENT_TEXT,
            },
        )

    def create_session_state(self) -> AuraServingSessionState:
        return AuraServingSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return aura_duplex_capabilities(max_sessions=max_sessions)

    def validate_client_extra_body(self, extra_body: object) -> None:
        if extra_body is None:
            return
        if not isinstance(extra_body, dict):
            raise ValueError("AURA duplex extra_body must be an object")

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: object | None
    ) -> dict[str, object]:
        del model_config
        extra = config.extra_body if isinstance(config.extra_body, dict) else {}
        system_prompt = extra.get("aura_system_prompt") or config.instructions or DEFAULT_AURA_SYSTEM_PROMPT
        runtime: dict[str, object] = {
            "aura_system_prompt": str(system_prompt),
            "instructions": str(system_prompt),
            "tts_task_type": str(extra.get("tts_task_type") or "CustomVoice"),
            "tts_language": str(extra.get("tts_language") or "Chinese"),
            "tts_speaker": str(extra.get("tts_speaker") or "Vivian"),
        }
        for key in _TTS_EXTRA_KEYS:
            if key in runtime:
                continue
            if key in extra and extra[key] is not None:
                runtime[key] = extra[key]
        return runtime

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        updated = dict(current)
        extra = config.extra_body if isinstance(config.extra_body, dict) else {}
        # Same precedence as prepare_runtime_config: explicit extra, then
        # instructions, then the prompt already installed (or the default).
        # setdefault would keep the creation-time prompt and ignore a later
        # session.update.instructions.
        explicit = extra.get("aura_system_prompt") if "aura_system_prompt" in extra else None
        if isinstance(explicit, str) and explicit:
            prompt = explicit
        elif config.instructions:
            prompt = str(config.instructions)
        else:
            current_prompt = current.get("aura_system_prompt")
            prompt = (
                str(current_prompt)
                if isinstance(current_prompt, str) and current_prompt
                else DEFAULT_AURA_SYSTEM_PROMPT
            )
        updated["aura_system_prompt"] = prompt
        updated["instructions"] = str(config.instructions) if config.instructions else prompt
        for key in _TTS_EXTRA_KEYS:
            if key in extra and extra[key] is not None:
                updated[key] = extra[key]
        return updated

    def data_plane_context(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> AuraDataPlaneContext:
        del active_response_turn_id, active_response_id
        return AuraDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )


__all__ = ["AURA_SILENT_TOKEN_ID", "AuraDuplexPlugin"]
