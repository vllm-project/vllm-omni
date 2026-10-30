# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-owned text intermediate on the resident DRVEXO Thinker.

Audio rendering uses the existing media pipeline. Recognition sees only the
current recording and a static transcription instruction, never chat history.
The resulting text then follows the ordinary decision/chat path.
"""

from __future__ import annotations

import copy
import io
import json
import time
import uuid

import anyio
import numpy as np
import pybase64 as base64
import soundfile as sf
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.multimodal.media.connector import MediaConnector
from vllm.sampling_params import RequestOutputKind, SamplingParams

TRANSCRIBE = "请准确转写这段音频，只输出听到的原文，不要解释。"
AUDIO_TYPES = frozenset(("input_audio", "audio_url"))


def split_current(messages):
    """Return current audio and accompanying written instructions, if any."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("A current user message is required")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("Messages must be objects")
    for message in messages[:-1]:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in AUDIO_TYPES for part in content
        ):
            raise ValueError("Committed audio turns must use their returned transcription text")
    current = messages[-1]
    content = current.get("content")
    if not isinstance(content, list):
        return None
    audio = [part for part in content if isinstance(part, dict) and part.get("type") in AUDIO_TYPES]
    if not audio:
        return None
    if current.get("role") != "user" or len(audio) != 1:
        raise ValueError("Text intermediate requires exactly one current user audio input")
    if any(not isinstance(part, dict) or part.get("type") not in AUDIO_TYPES | {"text"} for part in content):
        raise ValueError("Only text and audio are supported in this request")
    written = [part.get("text") for part in content if part.get("type") == "text"]
    if not all(isinstance(value, str) for value in written):
        raise ValueError("Accompanying instructions must be text")
    return copy.deepcopy(audio[0]), "\n".join(written).strip()


def exact_silence(audio):
    """Detect digital zero only; do not pretend this is a noise/VAD model."""
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[1], (int, float)):
        audio = audio[0]
    if isinstance(audio, list):
        if not audio:
            raise ValueError("Empty audio input")
        if all(isinstance(value, (int, float)) for value in audio):
            return exact_silence(np.asarray(audio))
        return all(exact_silence(item) for item in audio)
    if isinstance(audio, np.ndarray):
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("Empty or nonfinite audio input")
        return not np.any(audio != 0)
    return False


async def current_audio_to_text(serving, request):
    parts = split_current(request.messages)
    if parts is None:
        return request, None
    audio, written = parts
    # render_chat_async returns processed features, not the source waveform.
    # Decode once for activity validation, then render those exact samples via
    # a local data URI (also avoids a second fetch of mutable remote media).
    if audio["type"] == "input_audio":
        value = audio.get("input_audio") or {}
        uri = f"data:audio/{value.get('format', 'wav')};base64,{value.get('data', '')}"
    else:
        value = audio.get("audio_url")
        uri = value.get("url") if isinstance(value, dict) else value
    if not isinstance(uri, str) or not uri:
        raise ValueError("An audio source is required")
    config = getattr(serving, "model_config", None)
    connector = MediaConnector(
        media_io_kwargs=getattr(request, "media_io_kwargs", None),
        allowed_local_media_path=getattr(config, "allowed_local_media_path", "") or "",
        allowed_media_domains=getattr(config, "allowed_media_domains", None),
    )
    waveform, sample_rate = await connector.fetch_audio_async(uri)
    if exact_silence(waveform):
        raise ValueError("Audio contains digital silence; no user transcript was committed")
    encoded = io.BytesIO()
    sf.write(encoded, waveform, int(sample_rate), format="WAV", subtype="FLOAT")
    audio = {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(encoded.getvalue()).decode(), "format": "wav"},
    }
    # A separate owned request prevents chat or caller generation settings from
    # influencing recognition. Media decoding and model length limits are reused.
    messages = [{"role": "user", "content": [{"type": "text", "text": TRANSCRIBE}, audio]}]
    internal = request.model_copy(
        update={
            "messages": messages,
            "modalities": ["text"],
            "stream": False,
            "n": 1,
            "decision_head_enabled": False,
            "tools": None,
            "tool_choice": None,
            "max_tokens": 192,
            "max_completion_tokens": 192,
            "chat_template": None,
            "chat_template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
            "continue_final_message": False,
            "additional_information": None,
            "model_intermediate_buffer": None,
            "voice": None,
            "speaker": None,
            "instructions": None,
            "language": None,
            "truncate_prompt_tokens": None,
            "cache_salt": uuid.uuid4().hex,
        }
    )
    _, prompts = await serving._preprocess_chat(
        internal,
        messages,
        default_template=serving.chat_template,
        default_template_content_format=serving.chat_template_content_format,
        default_template_kwargs={"enable_thinking": False},
        tool_dicts=None,
        tool_parser=None,
        renderer=serving.renderer,
        add_generation_prompt=True,
        continue_final_message=False,
        documents=None,
        add_special_tokens=False,
    )
    if len(prompts) != 1:
        raise ValueError("Expected one prepared audio transcription request")
    prompt = prompts[0]
    if exact_silence((prompt.get("multi_modal_data") or {}).get("audio")):
        raise ValueError("Audio contains digital silence; no user transcript was committed")
    engine = serving.engine_client
    params = [value.clone() for value in engine.default_sampling_params_list]
    params[0] = SamplingParams(temperature=0, max_tokens=192, seed=20261021, output_kind=RequestOutputKind.FINAL_ONLY)
    identifier = "drvexo-transcription-" + uuid.uuid4().hex
    started = time.perf_counter()
    generator = engine.generate(
        prompt=prompt, request_id=identifier, sampling_params_list=params, output_modalities=["text"]
    )
    final = None
    completed = False
    try:
        async for output in generator:
            if output.error:
                raise RuntimeError("Transcription engine failed")
            final = output
        completed = True
    finally:
        with anyio.CancelScope(shield=True):
            await generator.aclose()
            if not completed:
                await engine.abort(identifier)
    if final is None or len(final.outputs) != 1:
        raise RuntimeError("Transcription produced no unambiguous result")
    completion = final.outputs[0]
    if completion.finish_reason != "stop":
        raise RuntimeError("Transcription is incomplete; no decision was issued")
    text = completion.text.strip()
    if not text:
        raise ValueError("No speech was transcribed; no user transcript was committed")
    current_text = "\n".join(value for value in (written, text) if value)
    converted = copy.deepcopy(request.messages)
    converted[-1]["content"] = current_text
    metadata = {
        "text": text,
        "current_text": current_text,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": len(prompt.get("prompt_token_ids", [])),
        "completion_tokens": len(completion.token_ids),
        "source": "shared_backbone",
        "request_id": identifier,
    }
    return request.model_copy(update={"messages": converted}), metadata


def attach_transcription(response, metadata):
    if isinstance(response, ChatCompletionResponse):
        return response.model_copy(update={"input_transcription": metadata})
    if not hasattr(response, "__aiter__"):
        return response

    async def stream():
        attached = False
        try:
            async for chunk in response:
                if not attached and chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                    event = json.loads(chunk[6:])
                    if "error" not in event:
                        event["input_transcription"] = metadata
                        chunk = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                        attached = True
                yield chunk
        finally:
            # A disconnect cancels the surrounding Starlette task group. Let
            # every inner response adapter reach its owned engine cleanup.
            with anyio.CancelScope(shield=True):
                await response.aclose()

    return stream()
