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

# Mirrors OpenAICreateSpeechRequest: entry mode validates these strictly, so
# downstream mode must enforce the same contract instead of coercing silently.
_VALID_RESPONSE_FORMATS = ("wav", "pcm", "flac", "mp3", "opus")
_DOWNSTREAM_MAX_TOKENS_DEFAULT = 65536

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

    Model-agnostic baseline: drops negative-padded and all-zero prefill/EOS
    rows. Model-specific transfer details (codebook-range checks, ref-code
    prepending, trim metadata) stay in each model's stage input processor;
    see the standalone disaggregation guide for the resulting limitations.
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
            "bypass the talker2code2wav ref-code prepending and ICL conditioning "
            "contract. Use co-located mode for voice cloning / Base task requests."
        )
    engine_client = raw_request.app.state.engine_client
    _, generator, tts_params = await handler._prepare_speech_generation(
        speech_request,
        request_id=request_id,
    )

    if tts_params.get("ref_audio"):
        raise ValueError(
            "Voice cloning is not supported in standalone mode. The resolved "
            "voice acquires reference conditioning that requires the co-located "
            "payload contract. Use co-located mode for this voice."
        )
    x_vec_only = (tts_params.get("x_vector_only_mode") or [True])[0]
    if not x_vec_only:
        raise ValueError(
            "ICL conditioning is not supported in standalone mode. The resolved "
            "voice uses in-context learning that requires the co-located payload "
            "contract. Use co-located mode for this voice."
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


def _validate_downstream_controls(body: dict) -> tuple[int, str, float]:
    """Validate downstream-only request controls. Raises ValueError."""
    max_tokens = body.get("max_tokens", _DOWNSTREAM_MAX_TOKENS_DEFAULT)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError(f"'max_tokens' must be an integer, got {max_tokens!r}")
    if max_tokens < 1:
        raise ValueError(f"'max_tokens' must be >= 1, got {max_tokens}")
    response_format = body.get("response_format") or "wav"
    if response_format not in _VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"Unsupported 'response_format' {response_format!r}. Supported: {list(_VALID_RESPONSE_FORMATS)}."
        )
    speed_raw = body.get("speed", 1.0)
    try:
        speed = float(speed_raw)
    except (TypeError, ValueError):
        raise ValueError(f"'speed' must be a number in [0.25, 4.0], got {speed_raw!r}") from None
    if not 0.25 <= speed <= 4.0:
        raise ValueError(f"'speed' must be in [0.25, 4.0], got {speed}")
    return max_tokens, response_format, speed


async def run_downstream_audio(
    raw_request: Request,
    body: dict,
    request_id: str,
) -> Response | JSONResponse:
    """Final audio stage: accept codec tokens, return WAV via AudioMixin."""
    from vllm import SamplingParams

    engine_client = raw_request.app.state.engine_client
    if not isinstance(body.get("stage_output"), dict):
        raise ValueError("'stage_output' must be a JSON object carrying upstream codec tokens")
    stage_output = body["stage_output"]

    prompt_token_ids = _parse_codec_tokens(stage_output)

    # max_tokens caps generation length. The stage's deploy YAML sets the real
    # limit (e.g., 65536 for Qwen3-TTS code2wav). The caller can override
    # via the request body. vLLM's default (16) is too low for codec decoding.
    max_tokens, response_format, speed = _validate_downstream_controls(body)

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
            response_format=response_format,
            speed=speed,
            base64_encode=False,
        )
    )

    return Response(
        content=audio_response.audio_data,
        media_type=audio_response.media_type,
        headers={"X-Request-Id": request_id},
    )


def _check_codec_int(value: Any, frame: int | None = None) -> int:
    """Validate a single codec token id. Raises ValueError."""
    if isinstance(value, bool) or not isinstance(value, int):
        where = f" at frame {frame}" if frame is not None else ""
        raise ValueError(f"Codec token{where} must be an integer, got {value!r}")
    if value < 0:
        where = f" at frame {frame}" if frame is not None else ""
        raise ValueError(
            f"Negative codec token{where} ({value}): upstream output looks unfiltered. "
            "Entry stages filter padding before serialization."
        )
    return value


def _parse_codec_tokens(stage_output: dict) -> list[int]:
    """Extract, validate, and flatten codec tokens from stage_output.

    Expects the ``{"codes": {"audio": [[frame...], ...]}}`` envelope produced
    by the entry stage. Rows are per-frame quantizer ids; the code2wav stage
    consumes them codebook-major flat, matching the canonical
    ``talker2code2wav_full_payload`` wire format.

    Raises ValueError on invalid input instead of returning error responses.
    """
    if not isinstance(stage_output, dict):
        raise ValueError(f"'stage_output' must be a JSON object, got {type(stage_output).__name__}")
    codes = stage_output.get("codes", {})
    codec_data = codes.get("audio") if isinstance(codes, dict) else None
    if codec_data is None:
        codec_data = stage_output.get("codes.audio")
    if not codec_data:
        raise ValueError("No codec data in stage_output")
    if not isinstance(codec_data, list):
        raise ValueError(f"'codes.audio' must be a list of frames, got {type(codec_data).__name__}")

    if isinstance(codec_data[0], list):
        num_quantizers = len(codec_data[0])
        if num_quantizers == 0:
            raise ValueError("Codec frames have zero quantizers")
        for i, row in enumerate(codec_data):
            if not isinstance(row, list) or len(row) != num_quantizers:
                got = len(row) if isinstance(row, list) else type(row).__name__
                raise ValueError(f"Ragged codec data at frame {i}: expected {num_quantizers} quantizers, got {got}")
            for v in row:
                _check_codec_int(v, frame=i)
        flat_len = len(codec_data) * num_quantizers
        if flat_len > MAX_CODEC_ELEMENTS:
            raise ValueError(f"Codec data too large ({flat_len} elements, max {MAX_CODEC_ELEMENTS})")
        return [codec_data[frame][q] for q in range(num_quantizers) for frame in range(len(codec_data))]
    for v in codec_data:
        _check_codec_int(v)
    if len(codec_data) > MAX_CODEC_ELEMENTS:
        raise ValueError(f"Codec data too large ({len(codec_data)} elements, max {MAX_CODEC_ELEMENTS})")
    return list(codec_data)
