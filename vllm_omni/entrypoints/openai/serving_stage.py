# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone stage serving: downstream handler utilities.

Shared utilities and downstream-only handlers for /v1/stage/run.
Entry mode logic remains in api_server.py (per-handler dispatch).
"""

from __future__ import annotations

import asyncio
import io
import wave
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from fastapi.responses import JSONResponse, Response
from vllm.logger import init_logger

if TYPE_CHECKING:
    from fastapi import Request

logger = init_logger(__name__)

MAX_CODEC_ELEMENTS = 2 * 1024 * 1024


def extract_multimodal_output(final_output: Any) -> Any | None:
    for co in getattr(final_output, "outputs", []):
        mm = getattr(co, "multimodal_output", None)
        if mm:
            return mm
    return None


def serialize(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if hasattr(obj, "items") and callable(obj.items):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(x) for x in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def stage_output_response(mm_output: Any, final_output: Any, request_id: str) -> JSONResponse:
    return JSONResponse(
        {
            "request_id": request_id,
            "stage_output": serialize(mm_output) if mm_output else None,
            "finished": getattr(final_output, "finished", True) if final_output else True,
        }
    )


def extract_sample_rate(mm: Any) -> int:
    sr_raw = mm.get("sr") if hasattr(mm, "get") else None
    if sr_raw is None:
        return 24000
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    return sr_val.item() if hasattr(sr_val, "item") else int(sr_val)


def build_wav_response(audio_data: Any, sample_rate: int, request_id: str) -> Response:
    if isinstance(audio_data, torch.Tensor):
        audio_np = audio_data.cpu().float().numpy()
    elif isinstance(audio_data, np.ndarray):
        audio_np = audio_data.astype(np.float32)
    else:
        audio_np = np.array(audio_data, dtype=np.float32)

    if audio_np.size > 0 and np.all(np.isfinite(audio_np)):
        abs_max = np.abs(audio_np).max()
        if abs_max <= 1.0:
            audio_np = audio_np * 32767
        elif abs_max > 32768:
            audio_np = audio_np * (32767 / abs_max)
    audio_np = np.clip(audio_np, -32768, 32767)
    audio_int16 = audio_np.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"X-Request-Id": request_id},
    )


def _parse_codec_tokens(stage_output: dict, request_id: str) -> list[int] | JSONResponse:
    """Extract and validate codec tokens from stage_output. Returns token list or error response."""
    codes = stage_output.get("codes", {})
    codec_data = codes.get("audio") if isinstance(codes, dict) else None
    if codec_data is None:
        codec_data = stage_output.get("codes.audio")
    if not codec_data:
        return JSONResponse(
            {"error": "No codec data in stage_output", "request_id": request_id},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    flat_len = sum(len(row) for row in codec_data) if isinstance(codec_data[0], list) else len(codec_data)
    if flat_len > MAX_CODEC_ELEMENTS:
        return JSONResponse(
            {
                "error": f"Codec data too large ({flat_len} elements, max {MAX_CODEC_ELEMENTS})",
                "request_id": request_id,
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    codec_tensor = torch.tensor(codec_data, dtype=torch.long)
    if codec_tensor.ndim == 2:
        return codec_tensor.transpose(0, 1).reshape(-1).tolist()
    return codec_tensor.tolist()


async def _generate_and_collect(
    engine_client: Any,
    prompt_token_ids: list[int],
    request_id: str,
    output_modalities: list[str] | None = None,
    kv_transfer_params: dict | None = None,
) -> Any:
    """Run engine.generate() and collect final output with cancellation support."""
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=65536, detokenize=False)
    if kv_transfer_params:
        sp.extra_args = {"kv_transfer_params": kv_transfer_params}

    kwargs: dict[str, Any] = {
        "prompt": {"prompt_token_ids": prompt_token_ids},
        "request_id": request_id,
        "sampling_params": sp,
    }
    if output_modalities:
        kwargs["output_modalities"] = output_modalities

    generator = engine_client.generate(**kwargs)
    final_output = None
    try:
        async for output in generator:
            final_output = output
    except asyncio.CancelledError:
        await engine_client.abort(request_id)
        raise
    return final_output


async def run_downstream_audio(
    raw_request: Request,
    body: dict,
    request_id: str,
) -> Response | JSONResponse:
    """Final audio stage: accept codec tokens, return WAV."""
    engine_client = raw_request.app.state.engine_client
    stage_output = body["stage_output"]

    tokens_or_error = _parse_codec_tokens(stage_output, request_id)
    if isinstance(tokens_or_error, JSONResponse):
        return tokens_or_error

    kvtp = body.get("kv_transfer_params")
    final_output = await _generate_and_collect(
        engine_client,
        tokens_or_error,
        request_id,
        output_modalities=["audio"],
        kv_transfer_params=kvtp,
    )
    if final_output is None:
        return JSONResponse(
            {"error": "No output generated", "request_id": request_id},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    mm_output = extract_multimodal_output(final_output)
    if mm_output is None:
        return JSONResponse(
            {"error": "No audio output", "request_id": request_id},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    audio_data = mm_output.get("audio") if hasattr(mm_output, "get") else None
    if audio_data is None:
        return JSONResponse(
            {"error": "No audio in multimodal output", "request_id": request_id},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    return build_wav_response(audio_data, extract_sample_rate(mm_output), request_id)


async def run_downstream_intermediate(
    raw_request: Request,
    body: dict,
    request_id: str,
) -> JSONResponse:
    """Intermediate stage: accept stage_output, return stage_output for next stage."""
    engine_client = raw_request.app.state.engine_client
    stage_output = body["stage_output"]

    tokens_or_error = _parse_codec_tokens(stage_output, request_id)
    if isinstance(tokens_or_error, JSONResponse):
        return tokens_or_error

    kvtp = body.get("kv_transfer_params")
    final_output = await _generate_and_collect(
        engine_client,
        tokens_or_error,
        request_id,
        kv_transfer_params=kvtp,
    )
    if final_output is None:
        return JSONResponse(
            {"error": "No output generated", "request_id": request_id},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )

    return stage_output_response(extract_multimodal_output(final_output), final_output, request_id)
