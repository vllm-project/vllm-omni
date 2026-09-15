# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``/v1/videos*`` endpoint helpers peeled from ``api_server.py``.

PUT HERE:
  - Multipart form parsing, upload limits, job run/cleanup, response factories,
    app-state runtime context, and other route-adjacent ``/v1/videos*`` logic
    that used to live in ``api_server.py``.
  - May import ``video_api_utils`` for decode/encode (helpers → utils OK).

DO NOT PUT HERE:
  - Shared media encode/decode backends — keep those in ``video_api_utils``.

LONGEVITY:
  - This package ``helpers.py`` is the **longer home** for video generation
    endpoint logic through the helper (P0.2) + router (P0.3) stages.
  - The later P1.3 video modality PR (#5227) may further split this file (e.g. form /
    jobs) and move ``video_api_utils`` into ``media.py`` here; until then keep
    growing here, not in root utils.

See ``video/generation/README.md`` (utils vs helpers, no overlap).
"""

import asyncio
import io
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from http import HTTPStatus
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import File, Form, HTTPException, Request, UploadFile
from PIL import Image
from vllm.entrypoints.launchers.launcher import terminate_if_errored
from vllm.entrypoints.serve import create_error_response
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.diffusion.models.interface import ReferenceVideoDecodeSpec
from vllm_omni.entrypoints.openai.app_state import Omnivideo
from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.protocol.videos import (
    SecondStr,
    SizeStr,
    VideoAction,
    VideoError,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoResponse,
)
from vllm_omni.entrypoints.openai.serving_video import (
    OmniOpenAIServingVideo,
    ReferenceAudio,
    ReferenceImage,
    ReferenceVideo,
)
from vllm_omni.entrypoints.openai.storage import STORAGE_MANAGER
from vllm_omni.entrypoints.openai.stores import VIDEO_STORE
from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.entrypoints.openai.video_api_utils import (
    VideoFrames,
    _decode_image_bytes,
    _validate_image_pixel_limit,
    decode_audio_url,
    decode_input_reference,
)
from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)


MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES = 30 * 1024 * 1024

MINIMAX_H3_MAX_REFERENCE_VIDEO_BYTES = 50 * 1024 * 1024

MINIMAX_H3_MAX_REFERENCE_AUDIO_BYTES = 15 * 1024 * 1024

MINIMAX_H3_MAX_REFERENCE_COUNT = 12

MINIMAX_H3_REFERENCE_IMAGE_FORMATS = frozenset({"jpeg", "png", "webp", "heic", "heif"})

MINIMAX_H3_REFERENCE_VIDEO_SUFFIXES = frozenset({".mp4", ".mov"})

MINIMAX_H3_REFERENCE_AUDIO_SUFFIXES = frozenset({".wav", ".mp3"})

CONTROL_REFERENCE_IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
CONTROL_REFERENCE_VIDEO_SUFFIXES = frozenset({".mkv", ".mov", ".mp4", ".webm"})
CONTROL_REFERENCE_MAX_BYTES = 512 * 1024 * 1024

VIDEO_SYNC_TIMEOUT_S = float(os.environ.get("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", 600.0))


def _resolve_video_runtime_context(raw_request: Request) -> tuple[str | None, list[Any] | None]:
    app_model_name = None
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    if serving_models and getattr(serving_models, "base_model_paths", None):
        base_paths = serving_models.base_model_paths
        if base_paths:
            app_model_name = base_paths[0].name

    app_stage_configs = getattr(raw_request.app.state, "stage_configs", None)
    return app_model_name, app_stage_configs


def _parse_form_json(value: str | None, expected_type: type | None = None) -> Any:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid JSON in form field.",
        ) from exc
    if expected_type is not None and not isinstance(parsed, expected_type):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid JSON in form field: expected {expected_type.__name__}, got {type(parsed).__name__}.",
        )
    return parsed


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def _stage_engine_args(stage_cfg: Any) -> Any:
    return _config_get(stage_cfg, "engine_args", {}) or {}


def _diffusion_model_classes(stage_configs: list[Any] | None) -> list[type]:
    if not stage_configs:
        return []

    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    model_classes: list[type] = []
    for stage_cfg in stage_configs:
        if get_stage_type(stage_cfg) != "diffusion":
            continue
        model_class_name = _config_get(_stage_engine_args(stage_cfg), "model_class_name")
        if not model_class_name:
            continue
        model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
        if model_cls is not None:
            model_classes.append(model_cls)
    return model_classes


def _normalize_reference_video_decode_spec(spec: ReferenceVideoDecodeSpec) -> ReferenceVideoDecodeSpec:
    max_frames = spec.max_frames
    if max_frames is not None:
        try:
            max_frames = int(max_frames)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid reference video decode spec: max_frames must be an integer.",
            ) from exc
        if max_frames <= 0:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid reference video decode spec: max_frames must be positive.",
            )

    keep = str(spec.keep or "first").strip().lower()
    if keep not in {"first", "last"}:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid reference video decode spec: keep must be either 'first' or 'last'.",
        )
    return ReferenceVideoDecodeSpec(max_frames=max_frames, keep=cast(Literal["first", "last"], keep))


def _reference_video_decode_spec(
    req: VideoGenerationRequest,
    stage_configs: list[Any] | None,
) -> ReferenceVideoDecodeSpec:
    video_params = req.resolve_video_params()
    extra_params = req.extra_params if isinstance(req.extra_params, dict) else {}
    for model_cls in _diffusion_model_classes(stage_configs):
        resolver = getattr(model_cls, "reference_video_decode_spec", None)
        if resolver is None:
            continue
        try:
            spec = resolver(num_frames=video_params.num_frames, extra_args=extra_params)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc
        if spec is not None:
            return _normalize_reference_video_decode_spec(spec)
    return ReferenceVideoDecodeSpec(max_frames=video_params.num_frames, keep="first")


def video_response_from_request(model_name: str, req: VideoGenerationRequest) -> VideoResponse:
    video_params = req.resolve_video_params()
    duration_s = None
    if req.seconds is not None:
        duration_s = float(req.seconds)
    elif video_params.num_frames is not None and video_params.fps is not None:
        duration_s = video_params.num_frames / video_params.fps

    resp = VideoResponse(
        model=model_name,
        status=VideoGenerationStatus.QUEUED,
        size=req.size,
        prompt=req.prompt,
        quality=req.quality or "default",
        fps=video_params.fps,
        num_frames=video_params.num_frames,
        duration_s=duration_s,
    )
    resp.seconds = str(req.seconds or resp.seconds)
    return resp


def _status_code_for_video_failure(error: VideoError | None) -> int:
    if error is None:
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if isinstance(error.code, int):
        if 400 <= error.code < 600:
            return error.code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "HTTPException":
        status_text, _, _ = error.message.partition(":")
        try:
            status_code = int(status_text)
        except ValueError:
            return HTTPStatus.INTERNAL_SERVER_ERROR.value
        if 400 <= status_code < 600:
            return status_code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "EngineDeadError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value
    if error.code == "EngineGenerateError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    return HTTPStatus.INTERNAL_SERVER_ERROR.value


def _video_error_from_exception(exc: Exception) -> VideoError:
    if isinstance(exc, HTTPException):
        message = str(exc.detail) if exc.detail else str(exc)
        return VideoError(code=exc.status_code, message=message)

    if isinstance(exc, OmniClientError):
        return VideoError(code=exc.status_code, message=exc.message)

    if isinstance(exc, (EngineGenerateError, EngineDeadError)):
        err = create_error_response(exc)
        return VideoError(code=err.error.code, message=err.error.message)

    return VideoError(
        code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        message=str(exc),
    )


async def _cleanup_video(storage_key: str | None) -> None:
    if storage_key is None:
        return
    try:
        await STORAGE_MANAGER.delete(storage_key)
    except Exception:
        logger.warning("Failed to cleanup partial video file '%s'", storage_key, exc_info=True)


def _cleanup_video_references(
    reference_video: ReferenceVideo | None,
    reference_audio: ReferenceAudio | None,
    control_path: str | None = None,
) -> None:
    if reference_video is not None:
        for path in reference_video.cleanup_paths:
            if os.path.exists(path):
                os.unlink(path)
    if reference_audio is not None:
        cleanup_paths = reference_audio.cleanup_paths or tuple(_reference_list(reference_audio.path))
        for path in cleanup_paths:
            if os.path.exists(path):
                os.unlink(path)
    if control_path is not None and os.path.exists(control_path):
        os.unlink(control_path)


def _unpack_video_generation_result(
    result: Sequence[object],
) -> tuple[bytes, dict[str, float], float, VideoAction | None, dict[str, object]]:
    video_metadata: dict[str, object] = {}
    if len(result) == 5:
        video_bytes, stage_durations, peak_memory_mb, action, raw_metadata = result
        if isinstance(raw_metadata, dict):
            video_metadata = {str(key): value for key, value in raw_metadata.items()}
    else:
        video_bytes, stage_durations, peak_memory_mb, action = result
    return (
        cast(bytes, video_bytes),
        cast(dict[str, float], stage_durations),
        float(cast(float, peak_memory_mb)),
        cast(VideoAction | None, action),
        video_metadata,
    )


async def _run_video_generation_job(
    handler: OmniOpenAIServingVideo,
    request: VideoGenerationRequest,
    video_id: str,
    reference_image: ReferenceImage | None = None,
    reference_video: ReferenceVideo | None = None,
    reference_audio: ReferenceAudio | None = None,
    control_path: str | None = None,
    app_state: Any | None = None,
) -> None:
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        logger.warning("Video job %s missing before generation task started; skipping", video_id)
        _cleanup_video_references(reference_video, reference_audio, control_path)
        return

    await VIDEO_STORE.update_fields(video_id, {"status": VideoGenerationStatus.IN_PROGRESS})
    started_at = time.perf_counter()
    storage_key: str | None = None
    try:
        settings = handler._resolve_video_output_settings(request)
        storage_key = f"{video_id}.{settings.output_format}"
        video_bytes, stage_durations, peak_memory_mb, action, video_metadata = _unpack_video_generation_result(
            await handler.generate_video_bytes(
                request,
                video_id,
                reference_image=reference_image,
                reference_video=reference_video,
                reference_audio=reference_audio,
            )
        )

        save_context = await STORAGE_MANAGER.save(video_bytes, storage_key)
        logger.info("Video request %s persisted %s output file.", video_id, save_context.key)

        updated_fields = {
            "status": VideoGenerationStatus.COMPLETED,
            "progress": 100,
            "file_name": storage_key,
            "media_type": settings.media_type,
            "completed_at": save_context.created_at,
            "inference_time_s": time.perf_counter() - started_at,
            "stage_durations": stage_durations,
            "peak_memory_mb": peak_memory_mb,
            "action": action,
        }
        updated_fields.update({key: value for key, value in video_metadata.items() if value is not None})
        if save_context.expires_at is not None:
            updated_fields["expires_at"] = save_context.expires_at

        await VIDEO_STORE.update_fields(video_id, updated_fields)
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Video generation failed (engine error) for id=%s", video_id)

        await _cleanup_video(storage_key)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
        # Background tasks can't propagate exceptions to FastAPI handlers.
        # Actively signal shutdown when the engine is dead.
        if app_state is not None and isinstance(exc, EngineDeadError):
            terminate_if_errored(
                server=app_state.server,
                engine=app_state.engine_client,
            )
    except Exception as exc:
        logger.exception("Video generation failed for id=%s", video_id)

        await _cleanup_video(storage_key)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
    except asyncio.CancelledError:
        await _cleanup_video(storage_key)
        await VIDEO_STORE.pop(video_id)
        raise
    finally:
        _cleanup_video_references(reference_video, reference_audio, control_path)


async def _persist_uploaded_video_references(uploads: list[UploadFile]) -> list[str]:
    paths: list[str] = []
    try:
        for upload in uploads:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in {".mkv", ".mov", ".mp4", ".webm"}:
                suffix = ".mp4"
            fd, path = tempfile.mkstemp(prefix="vllm_omni_video_reference_", suffix=suffix)
            paths.append(path)
            with os.fdopen(fd, "wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    output.write(chunk)
    except Exception:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
        raise
    return paths


async def _persist_uploaded_control_reference(
    upload: UploadFile,
    *,
    max_bytes: int = CONTROL_REFERENCE_MAX_BYTES,
) -> str:
    """Stream one model control upload to request-scoped local storage."""
    kind = _uploaded_media_kind(upload)
    suffix = Path(upload.filename or "").suffix.lower()
    supported_suffixes = CONTROL_REFERENCE_IMAGE_SUFFIXES | CONTROL_REFERENCE_VIDEO_SUFFIXES
    if kind == "audio" or (suffix and suffix not in supported_suffixes):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="control_reference must be an image or video file.",
        )
    if not suffix:
        suffix = ".png" if kind == "image" else ".mp4"

    declared_size = getattr(upload, "size", None)
    if isinstance(declared_size, Integral) and int(declared_size) > max_bytes:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"control_reference exceeds the {max_bytes // (1024 * 1024)} MiB size limit.",
        )

    fd, path = tempfile.mkstemp(prefix="vllm_omni_control_reference_", suffix=suffix)
    size = 0
    persisted = False
    try:
        with os.fdopen(fd, "wb") as output:
            while chunk := await upload.read(min(1024 * 1024, max_bytes - size + 1)):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST.value,
                        detail=f"control_reference exceeds the {max_bytes // (1024 * 1024)} MiB size limit.",
                    )
                output.write(chunk)
        if size == 0:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="control_reference must not be empty.",
            )
        persisted = True
        return path
    finally:
        if not persisted:
            with suppress(OSError):
                os.unlink(path)


def _validate_control_upload(
    handler: OmniOpenAIServingVideo,
    request: VideoGenerationRequest,
    control_reference: UploadFile | None,
    control_type: str | None,
) -> str | None:
    """Validate a generic uploaded control against model capabilities."""
    if control_reference is None:
        if control_type is not None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="control_type requires a control_reference upload.",
            )
        return None
    if control_type is None or not control_type.strip():
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="control_reference requires control_type.",
        )

    normalized_type = control_type.strip().lower()
    supported_types = frozenset(getattr(handler, "supported_control_upload_types", ()))
    if normalized_type not in supported_types:
        supported = ", ".join(sorted(supported_types)) or "none"
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                f"Control upload type '{normalized_type}' is not supported by this model. "
                f"Supported control upload types: {supported}."
            ),
        )

    existing = (request.extra_params or {}).get(normalized_type)
    if existing is not None:
        if not isinstance(existing, Mapping) and existing is not True:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"extra_params.{normalized_type} must be an object when control_reference is uploaded.",
            )
        if isinstance(existing, Mapping) and any(existing.get(key) is not None for key in ("control", "control_path")):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    "Provide either control_reference or "
                    f"extra_params.{normalized_type}.control/control_path, not both."
                ),
            )
    return normalized_type


def _attach_control_upload(
    request: VideoGenerationRequest,
    control_type: str,
    control_path: str | None = None,
) -> None:
    """Declare an uploaded control and attach its persisted path when available."""
    extra_params = dict(request.extra_params or {})
    existing = extra_params.get(control_type)
    control_params = dict(existing) if isinstance(existing, Mapping) else {}
    if control_path is not None:
        control_params["control_path"] = control_path
    extra_params[control_type] = control_params
    request.extra_params = extra_params


def _reference_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _uploaded_media_kind(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("audio/"):
        return "audio"
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}:
        return "image"
    if suffix in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
        return "audio"
    return "video"


def _minimax_h3_upload_limit(upload: UploadFile) -> int:
    kind = _uploaded_media_kind(upload)
    if kind == "image":
        return MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES
    if kind == "audio":
        return MINIMAX_H3_MAX_REFERENCE_AUDIO_BYTES
    return MINIMAX_H3_MAX_REFERENCE_VIDEO_BYTES


async def _read_upload_limited(upload: UploadFile, *, max_bytes: int | None = None) -> bytes:
    """Read an upload with an optional hard byte limit."""
    if max_bytes is None:
        return await upload.read()

    declared_size = getattr(upload, "size", None)
    if isinstance(declared_size, Integral) and int(declared_size) > max_bytes:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Uploaded reference exceeds the {max_bytes // (1024 * 1024)} MiB size limit.",
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, max_bytes - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Uploaded reference exceeds the {max_bytes // (1024 * 1024)} MiB size limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_minimax_h3_image_payload(
    payload: bytes,
    *,
    filename: str | None,
    allow_non_image: bool = False,
) -> None:
    """Validate a H3 image before converting it to a format-less PIL image."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image_format = str(image.format or "").lower()
            if image_format in MINIMAX_H3_REFERENCE_IMAGE_FORMATS:
                _validate_image_pixel_limit(image)
    except InvalidInputReferenceError as exc:
        raise HTTPException(HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc
    except Image.DecompressionBombError as exc:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid uploaded image reference: {filename}; image exceeds the decoder pixel limit.",
        ) from exc
    except (OSError, ValueError) as exc:
        if allow_non_image:
            return
        raise HTTPException(400, detail=f"Invalid uploaded image reference: {filename}") from exc

    if image_format not in MINIMAX_H3_REFERENCE_IMAGE_FORMATS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "MiniMax H3 reference images must use JPG, JPEG, PNG, WEBP, HEIC, or HEIF; "
                f"got {image_format or 'unknown'}."
            ),
        )


async def _persist_uploaded_media_references(
    uploads: list[UploadFile],
) -> tuple[list[Image.Image], list[str], list[str]]:
    """Persist a mixed MiniMax H3 multipart reference list.

    Images are decoded in memory; videos and audio remain files because H3's
    reference encoders need the original container streams (including video
    soundtracks).
    """
    images: list[Image.Image] = []
    videos: list[str] = []
    audios: list[str] = []
    paths: list[str] = []
    if len(uploads) > MINIMAX_H3_MAX_REFERENCE_COUNT:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"MiniMax H3 accepts at most {MINIMAX_H3_MAX_REFERENCE_COUNT} total references.",
        )
    try:
        for upload in uploads:
            kind = _uploaded_media_kind(upload)
            payload = await _read_upload_limited(upload, max_bytes=_minimax_h3_upload_limit(upload))
            if kind == "image":
                try:
                    _validate_minimax_h3_image_payload(payload, filename=upload.filename)
                    images.append(
                        _decode_image_bytes(
                            payload,
                            source=f"uploaded image reference: {upload.filename}",
                        )
                    )
                except InvalidInputReferenceError as exc:
                    raise HTTPException(HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc
                continue
            suffix = Path(upload.filename or "").suffix.lower()
            if kind == "video" and suffix and suffix not in MINIMAX_H3_REFERENCE_VIDEO_SUFFIXES:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="MiniMax H3 reference videos must use an MP4 or MOV file.",
                )
            if kind == "audio" and suffix and suffix not in MINIMAX_H3_REFERENCE_AUDIO_SUFFIXES:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="MiniMax H3 reference audio must use a WAV or MP3 file.",
                )
            if not suffix or len(suffix) > 8:
                suffix = ".mp3" if kind == "audio" else ".mp4"
            fd, path = tempfile.mkstemp(prefix="vllm_omni_reference_", suffix=suffix)
            paths.append(path)
            with os.fdopen(fd, "wb") as output:
                output.write(payload)
            if kind == "audio":
                audios.append(path)
            else:
                videos.append(path)
    except Exception:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)
        raise
    return images, videos, audios


async def _parse_video_form(
    raw_request: Request,
    prompt: str = Form(...),
    input_reference: UploadFile | None = File(default=None),
    input_references: list[UploadFile] | None = File(default=None),
    control_reference: UploadFile | None = File(default=None),
    control_type: str | None = Form(default=None),
    image_reference: str | None = Form(default=None),
    video_reference: str | None = Form(default=None),
    audio_reference: str | None = Form(default=None),
    model: str | None = Form(default=None),
    seconds: SecondStr | None = Form(default=None),
    size: SizeStr | None = Form(default=None),
    user: str | None = Form(default=None),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    num_frames: int | None = Form(default=None),
    fps: float | None = Form(default=None, ge=1, allow_inf_nan=False),
    aspect_ratio: str | None = Form(default=None),
    short_edge: int | None = Form(default=None, ge=1),
    num_outputs_per_prompt: int = Form(default=1, ge=1, le=10),
    start_time_seconds: float | None = Form(default=None, ge=0.0),
    quality: str | None = Form(default=None),
    num_inference_steps: int | None = Form(default=None),
    guidance_scale: float | None = Form(default=None),
    guidance_scale_2: float | None = Form(default=None),
    boundary_ratio: float | None = Form(default=None),
    flow_shift: float | None = Form(default=None),
    true_cfg_scale: float | None = Form(default=None),
    seed: int | None = Form(default=None),
    generate_sound: bool | None = Form(default=None),
    sound_duration: float | None = Form(default=None, gt=0.0),
    negative_prompt: str | None = Form(default=None),
    enable_frame_interpolation: bool | None = Form(default=None),
    frame_interpolation_exp: int | None = Form(default=None, ge=1),
    frame_interpolation_scale: float | None = Form(default=None, gt=0.0),
    frame_interpolation_model_path: str | None = Form(default=None),
    lora: str | None = Form(default=None),
    extra_params: str | None = Form(default=None),
    return_stage_metrics: bool | None = Form(default=None),
) -> tuple[
    VideoGenerationRequest,
    "OmniOpenAIServingVideo",
    str,
    ReferenceImage | None,
    ReferenceVideo | None,
    ReferenceAudio | None,
    str | None,
]:
    """FastAPI dependency that parses video form data, validates inputs,
    resolves the handler, and decodes any reference image.

    Used by both ``POST /v1/videos`` (async) and ``POST /v1/videos/sync``.
    """
    input_references = input_references or []
    input_reference_bytes: bytes | None = None
    parsed_image_reference = _parse_form_json(image_reference)
    parsed_video_reference = _parse_form_json(video_reference)
    parsed_audio_reference = _parse_form_json(audio_reference)

    if input_references and any(
        item is not None for item in (parsed_image_reference, parsed_video_reference, input_reference)
    ):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Provide input_references alone, without input_reference, image_reference, or video_reference.",
        )
    if input_reference is not None and (parsed_image_reference is not None or parsed_video_reference is not None):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=(
                "Provide only one of input_reference, image_reference, or video_reference when using "
                "input_reference; image_reference and video_reference may be combined."
            ),
        )

    request_data: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "seconds": seconds,
        "size": size,
        "image_reference": parsed_image_reference,
        "video_reference": parsed_video_reference,
        "audio_reference": parsed_audio_reference,
        "user": user,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "aspect_ratio": aspect_ratio,
        "short_edge": short_edge,
        "num_outputs_per_prompt": num_outputs_per_prompt,
        "start_time_seconds": start_time_seconds,
        "quality": quality,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "guidance_scale_2": guidance_scale_2,
        "boundary_ratio": boundary_ratio,
        "flow_shift": flow_shift,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed,
        "generate_sound": generate_sound,
        "sound_duration": sound_duration,
        "negative_prompt": negative_prompt,
        "enable_frame_interpolation": enable_frame_interpolation,
        "frame_interpolation_exp": frame_interpolation_exp,
        "frame_interpolation_scale": frame_interpolation_scale,
        "frame_interpolation_model_path": frame_interpolation_model_path,
        "lora": _parse_form_json(lora, expected_type=dict),
        "extra_params": _parse_form_json(extra_params, expected_type=dict),
        "return_stage_metrics": return_stage_metrics,
    }
    request_data = {k: v for k, v in request_data.items() if v is not None}
    request = VideoGenerationRequest(**request_data)

    handler = Omnivideo(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Video generation handler not initialized.",
        )
    logger.info("Video generation handler: %s", type(handler).__name__)
    try:
        app_model_name, app_stage_configs = _resolve_video_runtime_context(raw_request)
        effective_model_name = handler.model_name or app_model_name or request.model or "unknown"
        if request.model is not None and effective_model_name is not None and request.model != effective_model_name:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    f"Model mismatch: request specifies '{request.model}' but server is running "
                    f"'{effective_model_name}'."
                ),
            )
        handler.set_stage_configs_if_missing(app_stage_configs)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video generation setup failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation setup failed: {str(e)}",
        )

    normalized_control_type = _validate_control_upload(handler, request, control_reference, control_type)
    if normalized_control_type is not None:
        # Make the selected transfer mode visible while choosing the model's
        # reference-video decode policy. The upload itself stays unpersisted
        # until reference parsing succeeds, preserving failure cleanup.
        _attach_control_upload(request, normalized_control_type)

    supports_mixed_reference_inputs = bool(getattr(handler, "supports_mixed_reference_inputs", False))
    if input_reference is not None:
        input_reference_bytes = await _read_upload_limited(
            input_reference,
            max_bytes=_minimax_h3_upload_limit(input_reference) if supports_mixed_reference_inputs else None,
        )
        if supports_mixed_reference_inputs:
            input_reference_kind = _uploaded_media_kind(input_reference)
            _validate_minimax_h3_image_payload(
                input_reference_bytes,
                filename=input_reference.filename,
                allow_non_image=input_reference_kind != "image",
            )

    decode_spec = ReferenceVideoDecodeSpec()
    if not input_references and (parsed_video_reference is not None or input_reference_bytes is not None):
        stage_configs = (
            handler.stage_configs
            or app_stage_configs
            or getattr(getattr(handler, "_engine_client", None), "stage_configs", None)
        )
        decode_spec = _reference_video_decode_spec(request, stage_configs)
    reference_image = None
    reference_video = None
    reference_audio: ReferenceAudio | None = None
    if input_references:
        if not supports_mixed_reference_inputs:
            video_paths = await _persist_uploaded_video_references(input_references)
            reference_video = ReferenceVideo(data=video_paths, cleanup_paths=tuple(video_paths))
            images, audio_paths = [], []
        else:
            images, video_paths, audio_paths = await _persist_uploaded_media_references(input_references)
        if images:
            reference_image = ReferenceImage(data=images if len(images) > 1 else images[0])
        if video_paths:
            reference_video = ReferenceVideo(data=video_paths, cleanup_paths=tuple(video_paths))
        if audio_paths:
            reference_audio = ReferenceAudio(path=audio_paths, cleanup_paths=tuple(audio_paths))
    else:
        video_paths: list[str] = []
        try:
            image_items = _reference_list(request.image_reference)
            video_items = _reference_list(request.video_reference)
            image_data = []
            for item in image_items:
                media_data = await decode_input_reference(item, None, None)
                if not isinstance(media_data, Image.Image):
                    raise InvalidInputReferenceError("image_reference did not decode to an image")
                image_data.append(media_data)

            video_frames: list[Image.Image] | None = None
            for item in video_items:
                media_data = await decode_input_reference(
                    None,
                    item,
                    None,
                    max_video_frames=decode_spec.max_frames,
                    video_keep=decode_spec.keep,
                )
                if not isinstance(media_data, VideoFrames):
                    raise InvalidInputReferenceError("video_reference did not decode to a video")
                if media_data.source_path is not None:
                    video_paths.append(media_data.source_path)
                else:
                    if len(video_items) != 1:
                        raise InvalidInputReferenceError(
                            "multiple video URL references must be downloadable source videos"
                        )
                    video_frames = list(media_data)

            if input_reference_bytes is not None:
                media_data = await decode_input_reference(
                    None,
                    None,
                    input_reference_bytes,
                    max_video_frames=decode_spec.max_frames,
                    video_keep=decode_spec.keep,
                )
                if isinstance(media_data, Image.Image):
                    image_data.append(media_data)
                elif isinstance(media_data, VideoFrames):
                    if media_data.source_path is not None:
                        video_paths.append(media_data.source_path)
                    else:
                        video_frames = list(media_data)

            if image_data:
                reference_image = ReferenceImage(data=image_data if len(image_data) > 1 else image_data[0])
            if video_paths:
                reference_video = ReferenceVideo(data=video_paths, cleanup_paths=tuple(video_paths))
            elif video_frames is not None:
                reference_video = ReferenceVideo(data=video_frames)
        except InvalidInputReferenceError as exc:
            for path in video_paths:
                if os.path.exists(path):
                    os.unlink(path)
            raise HTTPException(400, detail=str(exc) or "Invalid input reference.") from exc

    audio_paths = [] if reference_audio is None else list(_reference_list(reference_audio.path))
    if request.audio_reference is not None:
        try:
            for audio_reference in _reference_list(request.audio_reference):
                audio_paths.append(await decode_audio_url(audio_reference.audio_url))
        except InvalidInputReferenceError as exc:
            _cleanup_video_references(reference_video, reference_audio)
            cleanup_paths = set(() if reference_audio is None else reference_audio.cleanup_paths)
            for path in audio_paths:
                if path not in cleanup_paths and os.path.exists(path):
                    os.unlink(path)
            raise HTTPException(400, detail=str(exc)) from exc
    if audio_paths:
        cleanup_paths = (
            tuple(audio_paths)
            if reference_audio is None
            else reference_audio.cleanup_paths
            + tuple(path for path in audio_paths if path not in reference_audio.cleanup_paths)
        )
        reference_audio = ReferenceAudio(
            path=audio_paths if len(audio_paths) > 1 else audio_paths[0],
            cleanup_paths=cleanup_paths,
        )

    control_path: str | None = None
    if control_reference is not None and normalized_control_type is not None:
        try:
            control_path = await _persist_uploaded_control_reference(
                control_reference,
                max_bytes=CONTROL_REFERENCE_MAX_BYTES,
            )
            _attach_control_upload(request, normalized_control_type, control_path)
        except (asyncio.CancelledError, HTTPException, OSError, TypeError, ValueError):
            _cleanup_video_references(reference_video, reference_audio, control_path)
            raise

    return (
        request,
        handler,
        effective_model_name,
        reference_image,
        reference_video,
        reference_audio,
        control_path,
    )
