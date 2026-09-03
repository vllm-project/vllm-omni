# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone stage serving: /v1/stage/run endpoint handlers.

Entry handler runs the speech pipeline and returns raw multimodal_output.
Downstream handler accepts upstream stage_output, runs the engine, and
returns audio via AudioMixin.create_audio.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from fastapi.responses import JSONResponse, Response
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

if TYPE_CHECKING:
    from fastapi import Request

logger = init_logger(__name__)

MAX_CODEC_ELEMENTS = 2 * 1024 * 1024

_audio_mixin = AudioMixin()


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-serializable types."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "items") and callable(obj.items):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def _extract_multimodal_output(final_output: Any) -> Any | None:
    """Extract the first non-None multimodal_output from engine output."""
    for co in getattr(final_output, "outputs", []):
        mm = getattr(co, "multimodal_output", None)
        if mm:
            return mm
    return None


def _clean_codec_frames(mm_output: Any) -> Any:
    """Filter invalid codec frames from multimodal output before serialization.

    Drops frames with any negative value (padding) or all-zero values
    (prefill padding, EOS). Operates on the tensor before JSON conversion.
    """
    codes = mm_output.get("codes") if hasattr(mm_output, "get") else None
    if codes is None:
        return mm_output
    audio = codes.get("audio") if hasattr(codes, "get") else None
    if not isinstance(audio, torch.Tensor) or audio.ndim != 2 or audio.numel() == 0:
        return mm_output
    valid = (audio >= 0).all(dim=1) & audio.any(dim=1)
    filtered = audio[valid]
    if filtered.shape[0] < audio.shape[0]:
        logger.debug(
            "[stage_run] filtered %d/%d invalid codec frames",
            audio.shape[0] - filtered.shape[0],
            audio.shape[0],
        )
    if hasattr(codes, "__setitem__"):
        codes["audio"] = filtered
    else:
        codes.audio = filtered
    return mm_output


def _extract_sample_rate(mm: Any) -> int:
    """Read sample rate from multimodal output, defaulting to 24kHz."""
    sr_raw = mm.get("sr") if hasattr(mm, "get") else None
    if sr_raw is None:
        return 24000
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    return sr_val.item() if hasattr(sr_val, "item") else int(sr_val)


async def run_entry_speech(
    raw_request: Request,
    handler: Any,
    body: dict,
    request_id: str,
) -> JSONResponse:
    """Speech entry stage: run speech generation, return serialized multimodal_output."""
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    speech_request = OpenAICreateSpeechRequest.model_validate(body)

    if speech_request.ref_audio is not None:
        raise ValueError(
            "ref_audio is not supported in standalone mode. Standalone stages "
            "bypass the talker2code2wav payload contract (frame filtering, ref "
            "code prepending, ICL conditioning). Use co-located mode for "
            "voice cloning / Base task requests."
        )

    engine_client = raw_request.app.state.engine_client
    _, generator, _ = await handler._prepare_speech_generation(
        speech_request,
        request_id=request_id,
    )

    final_output = None
    try:
        async for output in generator:
            final_output = output
    except asyncio.CancelledError:
        await engine_client.abort(request_id)
        raise

    if final_output is None:
        raise ValueError("No output generated")

    mm_output = _extract_multimodal_output(final_output)
    if mm_output is None and len(final_output.outputs) > 1:
        logger.warning(
            "[stage_run] request %s produced %d outputs but none had multimodal_output",
            request_id,
            len(final_output.outputs),
        )

    if mm_output is not None:
        mm_output = _clean_codec_frames(mm_output)
    stage_output = _to_json_safe(mm_output) if mm_output else None
    return JSONResponse(
        {
            "request_id": request_id,
            "stage_output": stage_output,
            "finished": final_output.finished,
        }
    )


async def run_downstream_audio(
    raw_request: Request,
    body: dict,
    request_id: str,
) -> Response | JSONResponse:
    """Final audio stage: accept codec tokens, return WAV via AudioMixin."""
    from vllm import SamplingParams

    engine_client = raw_request.app.state.engine_client
    stage_output = body["stage_output"]

    prompt_token_ids = _parse_codec_tokens(stage_output, request_id)

    # max_tokens caps generation length. The stage's deploy YAML sets the real
    # limit (e.g., 65536 for Qwen3-TTS code2wav). The caller can override
    # via the request body. vLLM's default (16) is too low for codec decoding.
    max_tokens = body.get("max_tokens", 65536)

    generator = engine_client.generate(
        prompt={"prompt_token_ids": prompt_token_ids},
        request_id=request_id,
        output_modalities=["audio"],
        sampling_params=SamplingParams(max_tokens=max_tokens, detokenize=False),
    )

    final_output = None
    try:
        async for output in generator:
            final_output = output
    except asyncio.CancelledError:
        await engine_client.abort(request_id)
        raise

    if final_output is None:
        raise ValueError("No output generated")

    mm_output = _extract_multimodal_output(final_output)
    if mm_output is None:
        raise ValueError("No audio in engine output")

    audio_data = mm_output.get("audio") if hasattr(mm_output, "get") else None
    if audio_data is None:
        raise ValueError("No audio key in multimodal output")

    if isinstance(audio_data, torch.Tensor):
        audio_np = audio_data.cpu().float().numpy()
    elif isinstance(audio_data, np.ndarray):
        audio_np = audio_data.astype(np.float32)
    else:
        audio_np = np.array(audio_data, dtype=np.float32)

    sample_rate = _extract_sample_rate(mm_output)
    audio_response = _audio_mixin.create_audio(
        CreateAudio(
            audio_tensor=audio_np,
            sample_rate=sample_rate,
            response_format=body.get("response_format") or "wav",
            speed=float(body["speed"]) if body.get("speed") is not None else 1.0,
            base64_encode=False,
        )
    )

    return Response(
        content=audio_response.audio_data,
        media_type=audio_response.media_type,
        headers={"X-Request-Id": request_id},
    )


def _parse_codec_tokens(stage_output: dict, request_id: str) -> list[int]:
    """Extract, validate, and flatten codec tokens from stage_output.

    Raises ValueError on invalid input instead of returning error responses.
    """
    codes = stage_output.get("codes", {})
    codec_data = codes.get("audio") if isinstance(codes, dict) else None
    if codec_data is None:
        codec_data = stage_output.get("codes.audio")
    if not codec_data:
        raise ValueError("No codec data in stage_output")

    flat_len = sum(len(row) for row in codec_data) if isinstance(codec_data[0], list) else len(codec_data)
    if flat_len > MAX_CODEC_ELEMENTS:
        raise ValueError(f"Codec data too large ({flat_len} elements, max {MAX_CODEC_ELEMENTS})")

    if isinstance(codec_data[0], list):
        num_quantizers = len(codec_data[0])
        return [codec_data[frame][q] for q in range(num_quantizers) for frame in range(len(codec_data))]
    return list(codec_data)
