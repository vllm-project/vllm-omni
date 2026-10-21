# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Manual-turn Realtime sessions for Ming3's Chat -> Talker pipeline.

Ming3 consumes complete audio turns, not the upstream transcription model's
buffer_realtime_audio generator. Reuse Chat preprocessing and streaming so
voice controls and the async thinker/talker handoff have one implementation.
"""

import asyncio
import copy
import hashlib
import io
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pybase64 as base64
import soundfile as sf
from starlette.websockets import WebSocket
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection

from vllm_omni.entrypoints.openai.ming3_voices import resolve_stored_voice, voice_id
from vllm_omni.model_executor.models.ming_flash_omni.omni_voice import (
    DEFAULT_IP_VOICE,
    ip_voice_name,
    validate_voice_controls,
)
from vllm_omni.utils.audio_resample import StreamingAudioResampler

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat


class Ming3RealtimeConnection(RealtimeConnection):
    generation_task: asyncio.Task[None] | None

    def __init__(self, websocket: WebSocket, serving: "OmniOpenAIServingChat") -> None:
        super().__init__(websocket, serving)
        self._session = {
            "id": self.connection_id,
            "type": "realtime",
            "model": websocket.query_params.get("model"),
            "output_modalities": ["audio"],
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None},
                "output": {"format": {"type": "audio/pcm", "rate": 24000}},
            },
        }
        self._messages: list[dict[str, Any]] = []
        self._input_audio = bytearray()
        self._audio_started = False
        self._spoken_voice = None
        self._response_started = asyncio.Event()

    async def send(self, event):
        if getattr(event, "type", None) == "session.created":
            await self.send_json({"type": "session.created", "session": self._session})
        else:
            await super().send(event)

    async def send_json(self, payload: dict[str, Any]) -> None:
        try:
            await self.websocket.send_text(json.dumps(payload))
        except Exception:
            self._is_connected = False
            raise

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    async def send_error(self, message, code=None):
        await self.send_json(
            {"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message}}
        )

    async def handle_event(self, event):
        try:
            if not isinstance(event, dict):
                raise ValueError("Realtime event must be an object")
            kind = event.get("type")
            if kind == "session.update":
                await self._update_session(event)
            elif kind == "input_audio_buffer.append":
                audio = base64.b64decode(event["audio"], validate=True)
                if len(audio) % 2:
                    raise ValueError("PCM16 audio must contain an even number of bytes")
                if len(self._input_audio) + len(audio) > self._max_audio_filesize_mb * 1024**2:
                    raise ValueError("Input audio buffer exceeds maximum size")
                self._input_audio.extend(audio)
            elif kind == "input_audio_buffer.clear":
                self._input_audio.clear()
                await self.send_json({"type": "input_audio_buffer.cleared"})
            elif kind == "input_audio_buffer.commit":
                if not self._input_audio:
                    raise ValueError("Cannot commit an empty audio buffer")
                item = {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "audio": base64.b64encode(self._input_audio).decode(),
                            "format": "pcm16",
                        }
                    ],
                }
                await self._add_item(item)
                self._input_audio.clear()
                await self.send_json({"type": "input_audio_buffer.committed", "item_id": self._last_item_id})
            elif kind == "conversation.item.create":
                await self._add_item(event["item"])
            elif kind == "response.create":
                if self.generation_task is not None and not self.generation_task.done():
                    raise ValueError("A response is already in progress")
                if not self._messages:
                    raise ValueError("Create a conversation item or commit input audio first")
                options = event.get("response") or {}
                if not isinstance(options, dict):
                    raise ValueError("response must be an object")
                session = copy.deepcopy(self._session)
                for key in ("instructions", "output_modalities", "max_output_tokens"):
                    if key in options:
                        session[key] = options[key]
                if "audio" in options:
                    audio = options["audio"]
                    if not isinstance(audio, dict) or not isinstance(audio.get("output", {}), dict):
                        raise ValueError("response.audio.output must be an object")
                    session["audio"]["output"].update(audio.get("output", {}))
                await self._validate_voice(session)
                self._response_started.clear()
                self.generation_task = asyncio.create_task(self._respond(session, options))
                # Enter the response's try/finally before accepting a cancel event.
                await self._response_started.wait()
            elif kind == "response.cancel":
                if self.generation_task is not None and not self.generation_task.done():
                    self.generation_task.cancel()
            else:
                raise ValueError(f"Unsupported Realtime event: {kind}")
        except (ValueError, KeyError, TypeError) as exc:
            await self.send_error(str(exc), "invalid_event")

    async def cleanup(self):
        await super().cleanup()
        if self.generation_task is not None:
            with suppress(asyncio.CancelledError):
                await self.generation_task

    async def _update_session(self, event):
        update = event.get("session", {})
        if not isinstance(update, dict):
            raise ValueError("session must be an object")
        session = copy.deepcopy(self._session)
        for key in ("model", "instructions", "additional_information", "output_modalities", "max_output_tokens"):
            if key in update:
                session[key] = update[key]
        # Retain compatibility with the upstream top-level model field.
        session["model"] = event.get("model") or session["model"]
        if not session["model"]:
            raise ValueError("Specify model in session.update or the WebSocket query")
        if self._check_model(session["model"]) is not None:
            raise ValueError(f"Unknown model: {session['model']}")
        audio = update.get("audio", {})
        if not isinstance(audio, dict):
            raise ValueError("session.audio must be an object")
        for side in ("input", "output"):
            if side in audio:
                if not isinstance(audio[side], dict):
                    raise ValueError(f"session.audio.{side} must be an object")
                session["audio"][side].update(audio[side])
        if "voice" in update:  # Beta Realtime session shape.
            session["audio"]["output"]["voice"] = update["voice"]
        for side in ("input", "output"):
            fmt = session["audio"][side]["format"]
            session["audio"][side]["format"] = self._pcm_format(fmt)
        if session["audio"]["input"].get("turn_detection") is not None:
            raise ValueError("Ming3 Realtime uses manual turns; set audio.input.turn_detection to null")
        controls = session.get("additional_information")
        if controls is not None and not isinstance(controls, dict):
            raise ValueError("additional_information must be an object")
        await self._validate_voice(session)
        self._session = session
        self._is_model_validated = True
        await self.send_json({"type": "session.updated", "session": session})

    @staticmethod
    def _pcm_format(fmt):
        if not isinstance(fmt, dict) or fmt.get("type", "audio/pcm") != "audio/pcm" or fmt.get("rate", 24000) != 24000:
            raise ValueError("Ming3 Realtime supports PCM16 mono at 24000 Hz")
        return {"type": "audio/pcm", "rate": 24000}

    @staticmethod
    def _voice_key(session):
        controls = session.get("additional_information") or {}
        selected = session["audio"]["output"].get("voice")
        if selected is None:
            selected = controls.get("voice_name", controls.get("voice"))
        references = {
            key: controls[key]
            for key in (
                "ref_audio",
                "reference_audio",
                "prompt_wav",
                "spk_emb",
                "prompt_text",
                "prompt_wav_lat",
                "prompt_wav_emb",
                "prompt_audio_mode",
            )
            if controls.get(key) is not None
        }
        if references:
            fingerprint = hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest()
            return ("reference", voice_id(selected) if selected is not None else None, fingerprint)
        if selected is None and controls.get("voice_mode") != "reference":
            selected = DEFAULT_IP_VOICE
        return ("reference", voice_id(selected) if selected is not None else None)

    async def _validate_voice(self, session):
        controls = session.get("additional_information") or {}
        selected = session["audio"]["output"].get("voice")
        if selected is None:
            selected = controls.get("voice_name", controls.get("voice"))
        validate_voice_controls(controls)
        key = self._voice_key(session)
        if self._audio_started and key != self._spoken_voice:
            raise ValueError("Voice cannot be changed after the session has emitted audio")
        if (
            self.generation_task is not None
            and not self.generation_task.done()
            and key != self._voice_key(self._session)
        ):
            raise ValueError("Voice cannot be changed during a response")
        session["audio"]["output"]["format"] = self._pcm_format(session["audio"]["output"]["format"])
        if session.get("output_modalities") not in (["audio"], ["text"]):
            raise ValueError("output_modalities must be ['audio'] or ['text']")
        if selected is not None:
            if any(controls.get(k) is not None for k in ("ref_audio", "reference_audio", "prompt_wav", "spk_emb")):
                raise ValueError("Choose either a stored voice or reference audio/speaker embedding")
            selected = voice_id(selected)
            # Validate before acknowledging session.updated, including missing samples.
            if ip_voice_name(selected) is None:
                await resolve_stored_voice(self.serving._ming3_speech_service, selected)
            session["audio"]["output"]["voice"] = selected

    async def _add_item(self, item):
        if not isinstance(item, dict):
            raise ValueError("item must be an object")
        if item.get("type") != "message" or item.get("role") not in {"user", "assistant", "system"}:
            raise ValueError("Only conversation message items are supported")
        parts = item.get("content", [])
        if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
            raise ValueError("item.content must be an array of objects")
        content = []
        for part in parts:
            if part.get("type") in {"input_text", "text"}:
                if not isinstance(part.get("text"), str):
                    raise ValueError("Conversation text must be a string")
                content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "input_audio":
                fmt = part.get("format", "pcm16")
                if fmt not in {"pcm16", "wav"}:
                    raise ValueError("Conversation audio format must be pcm16 or wav")
                audio = base64.b64decode(part["audio"], validate=True)
                if not audio:
                    raise ValueError("Conversation audio cannot be empty")
                if len(audio) > self._max_audio_filesize_mb * 1024**2:
                    raise ValueError("Conversation audio exceeds maximum size")
                if fmt == "wav":
                    try:
                        info = sf.info(io.BytesIO(audio))
                    except sf.LibsndfileError as exc:
                        raise ValueError("Invalid WAV audio") from exc
                    if info.format not in {"WAV", "WAVEX", "RF64"}:
                        raise ValueError("Conversation audio must contain WAV data")
                    if not info.frames:
                        raise ValueError("Conversation audio cannot be empty")
                else:
                    if len(audio) % 2:
                        raise ValueError("Invalid PCM16 audio")
                    buffer = io.BytesIO()
                    sf.write(buffer, np.frombuffer(audio, dtype="<i2"), 24000, format="WAV", subtype="PCM_16")
                    audio = buffer.getvalue()
                # Native Chat media loading limits the encoded file, including
                # the WAV header added to raw PCM or a committed input buffer.
                if len(audio) > self._max_audio_filesize_mb * 1024**2:
                    raise ValueError("Conversation audio exceeds maximum size")
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(audio).decode(),
                            "format": "wav",
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported conversation content: {part.get('type')}")
        if not content:
            raise ValueError("Message content cannot be empty")
        self._messages.append({"role": item["role"], "content": content})
        self._last_item_id = item.get("id") or f"item_{uuid4().hex}"
        await self.send_json({"type": "conversation.item.created", "item": {**item, "id": self._last_item_id}})

    async def _respond(self, session, options):
        response_id, item_id = f"resp_{uuid4().hex}", f"item_{uuid4().hex}"
        result = None
        transcript = ""
        status = "completed"
        limit_reached = False
        sent_audio = False
        resampler = None
        audio_output = session.get("output_modalities") != ["text"]
        common = {"response_id": response_id, "item_id": item_id, "output_index": 0, "content_index": 0}
        messages = copy.deepcopy(self._messages)
        history_index = len(self._messages)
        try:
            self._response_started.set()
            await self.send_json({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}})
            if not self._is_model_validated:
                raise ValueError("Send session.update before response.create")
            controls = dict(session.get("additional_information") or {})
            voice = session["audio"]["output"].get("voice")
            if voice is not None:
                controls["voice_name"] = voice_id(voice)
            if session.get("instructions"):
                messages.insert(0, {"role": "system", "content": session["instructions"]})
            request = ChatCompletionRequest(
                model=session["model"],
                messages=messages,
                stream=True,
                modalities=["text", "audio"] if audio_output else ["text"],
                audio={"format": "wav"},
                additional_information=controls,
                max_completion_tokens=options.get("max_output_tokens", session.get("max_output_tokens"))
                if options.get("max_output_tokens", session.get("max_output_tokens")) != "inf"
                else None,
            )
            result = await self.serving.create_chat_completion(request)
            if isinstance(result, ErrorResponse):
                raise ValueError(result.error.message)
            await self.send_json(
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }
            )
            await self.send_json(
                {
                    "type": "response.content_part.added",
                    **common,
                    "part": {"type": "audio", "transcript": ""} if audio_output else {"type": "text", "text": ""},
                }
            )
            async for event in result:
                for line in event.splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise ValueError(str(data["error"]))
                    for choice in data.get("choices", []):
                        # Terminal choices may have no content; later audio
                        # choices must not overwrite the Thinker's limit.
                        limit_reached |= choice.get("finish_reason") == "length"
                        content = choice.get("delta", {}).get("content")
                        if not content:
                            continue
                        if data.get("modality") == "audio":
                            samples, rate = sf.read(io.BytesIO(base64.b64decode(content)), dtype="float32")
                            if samples.ndim > 1:
                                samples = samples.mean(axis=-1)
                            if resampler is None:
                                resampler = StreamingAudioResampler(rate, 24000)
                            elif resampler.source_rate != rate:
                                raise ValueError("Audio sample rate changed within a response")
                            sent_audio |= await self._emit_audio(resampler.process(samples), common, session)
                        else:
                            transcript += content
                            kind = (
                                "response.output_audio_transcript.delta"
                                if audio_output
                                else "response.output_text.delta"
                            )
                            await self.send_json({"type": kind, **common, "delta": content})
            if resampler is not None:
                tail = resampler.process(np.empty(0, dtype=np.float32), final=True)
                sent_audio |= await self._emit_audio(tail, common, session)
            if audio_output:
                if not sent_audio:
                    raise ValueError("Audio generation completed without producing audio")
                await self.send_json({"type": "response.output_audio.done", **common})
            await self.send_json(
                {
                    "type": "response.output_audio_transcript.done" if audio_output else "response.output_text.done",
                    **common,
                    "transcript" if audio_output else "text": transcript,
                }
            )
            if limit_reached:
                status = "incomplete"
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as exc:
            status = "failed"
            await self.send_error(str(exc), "processing_error")
        finally:
            # Keep the text already delivered to the client when a response is cancelled.
            if transcript:
                self._messages.insert(history_index, {"role": "assistant", "content": transcript})
            if result is not None and hasattr(result, "aclose"):
                await result.aclose()
            if self._is_connected:
                part = (
                    {"type": "audio", "transcript": transcript}
                    if audio_output
                    else {"type": "text", "text": transcript}
                )
                item = {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed" if status == "completed" else "incomplete",
                    "content": [part],
                }
                await self.send_json({"type": "response.content_part.done", **common, "part": part})
                await self.send_json(
                    {"type": "response.output_item.done", "response_id": response_id, "output_index": 0, "item": item}
                )
                response: dict[str, object] = {"id": response_id, "status": status, "output": [item]}
                if status == "incomplete":
                    response["status_details"] = {"type": "incomplete", "reason": "max_output_tokens"}
                await self.send_json({"type": "response.done", "response": response})

    async def _emit_audio(self, samples, common, session):
        if not samples.size:
            return False
        if not self._audio_started:
            # The response uses a validated session snapshot; its voice is
            # immutable. Reference audio may be large, so fingerprint it once.
            self._spoken_voice = self._voice_key(session)
            self._audio_started = True
        await self.send_json({"type": "response.output_audio.delta", **common, "delta": self._pcm16_b64(samples)})
        return True
