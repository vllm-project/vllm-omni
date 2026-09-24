# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Engine-owned VAD/commit turns with concurrent input and interruptible output.

Qwen3-Omni is a turn model: this plugin does not advertise native streaming KV
or model-driven turn decisions. Each committed utterance is a fresh request.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pybase64 as base64
from PIL import Image
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, DuplexModelPlugin, DuplexRuntimeConfigError
from vllm_omni.model_executor.models.qwen3_omni.duplex.session import QwenDuplexSessionState

logger = init_logger(__name__)

#: Mirrors ``limit_mm_per_prompt.image`` in ``deploy/qwen3_omni_duplex.yaml``.
#: The session already refuses a ninth stored image, but that budget lives in
#: the engine and this one in the deploy config; the prompt is the only place
#: that can keep them from drifting apart into a rejected submission.
MAX_PROMPT_IMAGES = 8


class QwenDataPlane(DuplexDataPlane):
    def __init__(self, encode_audio):
        self.encode_audio = encode_audio
        self.terminal: set[str] = set()

    def begin_request(self, request_id):
        self.terminal.discard(request_id)

    def is_terminal(self, request_id):
        return request_id in self.terminal

    def mark_terminal(self, request_id):
        self.terminal.add(request_id)

    def close_stream(self, request_id):
        pass

    def close_session(self, session_id, *, active_request_id=None):
        self.terminal = {
            rid for rid in self.terminal if not duplex_resource_request_belongs_to_session(rid, session_id)
        }

    def project(self, result, *, context=None):
        if not isinstance(result, dict):
            return
        context = context or {}
        for output in result.get("data_plane_outputs", []):
            rid = output.request_id
            if self.is_terminal(rid):
                continue
            completion = next(iter(getattr(output, "outputs", ()) or ()), None)
            stage_id = getattr(output, "stage_id", 0)
            text = getattr(completion, "text", "") if stage_id == 0 else ""
            mm = getattr(output, "multimodal_output", None)
            if not isinstance(mm, Mapping) or "audio" not in mm:
                mm = getattr(completion, "multimodal_output", {}) or {}
            audio = mm.get("audio")
            encoded = None
            duration_ms = 0
            rate = mm.get("sr", mm.get("sample_rate", mm.get("sample_rate_hz", 24000)))
            if isinstance(rate, (list, tuple)):
                rate = next((value for value in rate if value is not None), 24000)
            rate = int(rate)
            if rate <= 0:
                raise ValueError("Invalid output audio sample rate")
            if audio is not None and "audio" in context.get("modalities", ()):
                chunks = audio if isinstance(audio, (list, tuple)) else [audio]
                arrays = [
                    np.asarray(
                        chunk.detach().float().cpu().numpy() if hasattr(chunk, "detach") else chunk, dtype=np.float32
                    ).reshape(-1)
                    for chunk in chunks
                ]
                audio = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
                duration_ms = audio.size * 1000 / rate
                if audio.size:
                    encoded = self.encode_audio(
                        audio, rate, context.get("response_format", "pcm16"), context.get("speed")
                    )
            yield {
                "supported": True,
                "data_plane_request_id": rid,
                "text": text or "",
                "audio_data": encoded or "",
                "audio_format": context.get("response_format", "pcm16"),
                "audio_duration_ms": duration_ms,
                # Thinker text and codec chunks are independent streams. Only
                # the final audio boundary can conservatively attest all text.
                "audio_text_mark": False,
                "text_requires_complete_audio": "audio" in context.get("modalities", ()),
                "audio_complete": bool(output.finished) and stage_id == 2,
                "sample_rate_hz": rate,
                "end_of_turn": bool(output.finished),
                "model_turn_id": context.get("active_response_turn_id")
                if context.get("active_response_turn_id") is not None
                else context.get("turn_id", 0),
                "runtime_impl": "qwen3_commit",
                "uses_model_runner_scheduler": True,
                "runner_kv_backed": True,
            }


class Qwen3OmniDuplexPlugin(DuplexModelPlugin):
    plugin_id = "qwen3-omni"
    projects_intermediate_outputs = True

    def __init__(self, encode_audio):
        self.data_plane = QwenDataPlane(encode_audio)
        self.processor = None
        self._processor_lock = asyncio.Lock()

    def capabilities(self, *, max_sessions):
        return DuplexCapabilities(
            supports_model_native_turn_policy=False,
            supports_input_append=True,
            supports_barge_in=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=True,
            supports_turn_commit_only=True,
            supports_core_resumable_request=False,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=max_sessions > 1,
            supports_multi_session_same_replica=max_sessions > 1,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_audio_truncate=True,
            supports_chat_completions=True,
            supports_image_input=True,
            supports_text_only_turn=True,
            adapter_patterns=["turn_commit"],
            signal_sources=["client_event", "server_policy"],
            stage_handoff_transport="scheduler_data_plane",
            target_barge_in_latency_ms=None,
        )

    def create_session_state(self):
        return QwenDuplexSessionState()

    def validate_client_extra_body(self, extra_body):
        if not isinstance(extra_body, Mapping):
            return
        if extra_body.get("auto_response"):
            raise DuplexRuntimeConfigError(
                "Qwen requires server VAD or explicit commits; native auto_response is unsupported"
            )
        if extra_body.get("realtime_tools"):
            raise DuplexRuntimeConfigError("Qwen duplex tool calls are not implemented")

    async def prepare_runtime_config(self, config, *, model_config):
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("Qwen duplex does not support reference voice audio")
        if model_config is None and self.processor is None:
            raise DuplexRuntimeConfigError("Qwen duplex requires the stage-0 model configuration")
        async with self._processor_lock:
            if self.processor is None:
                from transformers import AutoProcessor

                self.processor = await asyncio.to_thread(
                    AutoProcessor.from_pretrained,
                    model_config.model,
                    trust_remote_code=model_config.trust_remote_code,
                    revision=getattr(model_config, "revision", None),
                )
        return self.runtime_config_for_update(config, {})

    def runtime_config_for_update(self, config, current):
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("Qwen duplex does not support reference voice audio")
        return {
            "instructions": config.instructions,
            "initial_user_text": config.initial_user_text,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

    def configure_sampling_params(self, *, runtime_config, defaults):
        params = copy.deepcopy(defaults)
        for value in params:
            if value is not None:
                value.output_kind = RequestOutputKind.DELTA
        if params and params[0] is not None:
            if runtime_config.get("temperature") is not None:
                params[0].temperature = float(runtime_config["temperature"])
            if isinstance(runtime_config.get("max_tokens"), int):
                params[0].max_tokens = runtime_config["max_tokens"]
        return params

    def prepare_prompt_config(self, config, *, state, payload):
        history = config.get("conversation", [])
        # The history accessor copies message dicts but preserves content identity.
        # Keep payloads privately, never put PCM into public conversation events.
        has_audio = bool(payload.get("audio"))
        # A committed audio item need not be last: camera/text items can arrive
        # before response.create. Search only the unanswered user turn so a new
        # payload can never overwrite an earlier, already answered utterance.
        current_contents = []
        if has_audio:
            for message in reversed(history):
                if message.get("role") != "user":
                    break
                content = message.get("content")
                if isinstance(content, list) and any(part.get("type") == "audio_url" for part in content):
                    current_contents.append(content)
        current_is_in_history = bool(current_contents)
        if current_is_in_history:
            # The runner concatenates multiple commits made before a response.
            # Associate that single payload with all of its source audio items.
            state.audio_history = [
                (keys, value)
                for keys, value in state.audio_history
                if not any(key is content for key in keys for content in current_contents)
            ]
            state.audio_history.append((tuple(current_contents), payload))
        # Reserve one of the four audio slots for the actual submitted input,
        # including when deferred input has no trailing user history entry.
        history_limit = 4 if current_is_in_history or not has_audio else 3
        retained = state.audio_history[-history_limit:]
        while len(retained) > 1 and sum(len(value.get("audio", "")) for _, value in retained) > 8 * 1024 * 1024:
            retained.pop(0)
        state.audio_history = retained
        # Match the official demo's unit of history: consecutive user items
        # followed by their assistant reply(s). Prune whole turns, before
        # formatting, so a camera item cannot rescue an orphaned answer.
        turns = []
        for index, message in enumerate(history):
            role = message.get("role")
            if not turns or role == "system" or (role == "user" and turns[-1][1][-1].get("role") != "user"):
                turns.append((index, []))
            turns[-1][1].append(message)

        messages = []
        for index, turn in turns:
            # Keep the existing recent-message window, rounded outwards to a
            # complete turn. Never retain an answer whose input was cut off.
            if index + len(turn) <= len(history) - 16:
                continue
            if turn[0].get("role") == "assistant":
                continue
            resolved = []
            included_audio = set()
            for message in turn:
                content = message.get("content")
                if message.get("role") == "user" and isinstance(content, list):
                    if any(part.get("type") == "audio_url" for part in content):
                        audio = next((value for keys, value in retained if any(key is content for key in keys)), None)
                        if audio is None:
                            break
                        if id(audio) not in included_audio:
                            resolved.append({"role": "user", "audio_payload": audio})
                            included_audio.add(id(audio))
                        # Preserve text/images in mixed source items as well.
                        other_parts = [part for part in content if part.get("type") != "audio_url"]
                        if other_parts:
                            resolved.append({"role": "user", "content": other_parts})
                        continue
                resolved.append(message)
            else:
                messages.extend(resolved)
        if has_audio and not current_is_in_history:
            # History is context, not a replacement for the current request.
            # Re-encoding only history here makes Qwen answer an old question.
            messages.append({"role": "user", "audio_payload": payload})
        return {**config, "qwen_messages": messages}

    @staticmethod
    def format_history(messages):
        """Merge user items as Qwen's official web_demo.format_history does.

        This is a model-input view, not a mutation of addressable session items.
        Empty assistant messages remain boundaries for interrupted/unheard turns.
        """
        formatted = []
        pending = []
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                pending.extend([{"type": "text", "text": content}] if isinstance(content, str) else content or [])
                continue
            if pending:
                formatted.append({"role": "user", "content": pending})
                pending = []
            formatted.append(message)
        if pending:
            # The official demo puts media before the final turn's instruction.
            media = [part for part in pending if part.get("type") != "text"]
            text = [part for part in pending if part.get("type") == "text"]
            formatted.append({"role": "user", "content": media + text})
        return formatted

    @staticmethod
    def _trim_prompt_images(messages, images):
        """Drop the oldest images, with their placeholders, down to the prompt limit.

        The processor pairs ``{"type": "image"}`` parts with ``mm["image"]``
        positionally, and the placeholders read across *messages* in the same
        order as *images*, so a dropped image has to take its placeholder with
        it. Oldest first keeps what the client shared most recently.
        """
        excess = len(images) - MAX_PROMPT_IMAGES
        if excess <= 0:
            return messages, images
        logger.warning(
            "Qwen duplex prompt carries %d images; dropping the %d oldest to stay within %d",
            len(images),
            excess,
            MAX_PROMPT_IMAGES,
        )
        remaining = excess
        for message in messages:
            if remaining <= 0:
                break
            content = message.get("content")
            if not isinstance(content, list):
                continue
            kept = []
            for part in content:
                if remaining > 0 and isinstance(part, dict) and part.get("type") == "image":
                    remaining -= 1
                    continue
                kept.append(part)
            message["content"] = kept
        return [message for message in messages if message.get("content") != []], images[excess:]

    @staticmethod
    def _audio_content(payload):
        raw = base64.b64decode(payload["audio"], validate=True)
        audio = np.frombuffer(raw, dtype="<f4").copy()
        if not audio.size or not np.isfinite(audio).all():
            raise DuplexRuntimeConfigError("Audio must contain finite PCM samples")
        return [{"type": "audio"}], audio

    async def prepare_append_plan(self, **kwargs):
        # History mutation has already happened on the session loop. The worker
        # owns this snapshot and never reads or mutates live session state.
        snapshot = copy.deepcopy(kwargs)
        return await asyncio.to_thread(self.plan_append, **snapshot)

    def plan_append(
        self, *, request_id, fence, session_config, runtime_config, seq, turn_seq, payload, final, sampling_params
    ):
        if not final or not isinstance(payload, dict):
            raise DuplexRuntimeConfigError("Qwen generation requires a committed audio turn")
        messages = []
        if runtime_config.get("instructions"):
            messages.append({"role": "system", "content": runtime_config["instructions"]})
        history = session_config.get("qwen_messages")
        if history is None and payload.get("audio"):
            # Nothing prepared the conversation, so the committed payload is
            # the whole turn. An empty prepared list is not that: it means the
            # preparation ran and found nothing to say.
            history = [{"role": "user", "audio_payload": payload}]
        if not history:
            raise DuplexRuntimeConfigError("Qwen generation requires committed audio or conversation history")
        audios = []
        images = []
        for message in history:
            if "audio_payload" not in message:
                content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "image_url":
                            encoded = part["image_url"]["url"].split(",", 1)[1]
                            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                                image.thumbnail((448, 448))
                                images.append(image.convert("RGB").copy())
                            parts.append({"type": "image"})
                        elif part.get("type") == "text":
                            parts.append(part)
                    messages.append({"role": message["role"], "content": parts})
                else:
                    messages.append(message)
                continue
            content, audio = self._audio_content(message["audio_payload"])
            audios.append((audio, 16000))
            messages.append({"role": "user", "content": content})
        if payload.get("audio") and runtime_config.get("initial_user_text"):
            messages.append({"role": "user", "content": runtime_config["initial_user_text"]})
        messages, images = self._trim_prompt_images(messages, images)
        messages = self.format_history(messages)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        mm = {"audio": audios} if audios else {}
        if images:
            mm["image"] = images
        logger.info(
            "Qwen duplex visual input request=%s images=%d sizes=%s audios=%d",
            request_id,
            len(images),
            [image.size for image in images],
            len(audios),
        )
        # Opt-in capture of the exact decoded images submitted to the processor.
        # Request IDs are hashed because they need not be safe path components.
        debug_dir = os.environ.get("VLLM_OMNI_QWEN_VISUAL_DEBUG_DIR")
        if debug_dir:
            try:
                target = Path(debug_dir) / hashlib.sha256(request_id.encode()).hexdigest()[:24]
                target.mkdir(parents=True, exist_ok=True)
                filenames = []
                for index, image in enumerate(images):
                    filename = f"image_{index:02d}.png"
                    image.save(target / filename)
                    filenames.append(filename)
                (target / "input.json").write_text(
                    json.dumps(
                        {"request_id": request_id, "images": filenames, "audio_count": len(audios), "prompt": prompt},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.info("Qwen duplex visual input capture request=%s path=%s", request_id, target)
            except (OSError, ValueError):
                logger.exception("Failed to capture Qwen visual input request=%s", request_id)
        return DuplexAppendPlan(prompt={"prompt": prompt, "multi_modal_data": mm})

    def decide_output(self, **kwargs):
        return None

    def data_plane_context(self, **kwargs):
        return kwargs
