# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech / TTS endpoint router.

Owns all ``/v1/audio/`` routes: speech synthesis, voice management, and the
streaming speech websocket.  Imported and registered by
:mod:`vllm_omni.entrypoints.openai.api_server` via ``include_router``.
"""

from http import HTTPStatus

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.entrypoints.openai.app_state import OmniAudioGenerate, Omnispeech
from vllm_omni.entrypoints.openai.errors import (
    InvalidPresetVoiceReferenceError,
    InvalidVoiceReferenceError,
    _create_speech_error_json_response,
    _error_response_to_json_response,
)
from vllm_omni.entrypoints.openai.protocol.audio import (
    BatchSpeechRequest,
    OpenAICreateAudioGenerateRequest,
    OpenAICreateSpeechRequest,
)
from vllm_omni.entrypoints.serve.utils.errors import (
    _create_engine_error_json_response,
)

logger = init_logger(__name__)
speech_router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers (speech-family local)
# ---------------------------------------------------------------------------


def _unsupported_api_response(raw_request: Request, api_name: str) -> JSONResponse:
    """404 for a route whose serving handler is not configured on this server."""
    message = f"The model does not support {api_name}"
    base_server = getattr(raw_request.app.state, "serving_tokenization", None)
    if base_server is None:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail=message)
    err = base_server.create_error_response(
        message=message,
        err_type="NotFoundError",
        status_code=HTTPStatus.NOT_FOUND,
    )
    return _error_response_to_json_response(err, status_code=HTTPStatus.NOT_FOUND)


# ---------------------------------------------------------------------------
# Speech synthesis
# ---------------------------------------------------------------------------


@speech_router.post(
    "/v1/audio/speech",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}, "text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech(request: OpenAICreateSpeechRequest, raw_request: Request):
    """Generate speech audio from text using the loaded TTS model.

    Args:
        request: Speech synthesis request in OpenAI-compatible format.
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        The generated audio response, or an OpenAI-style error payload when
        the request cannot be fulfilled.

    Raises:
        HTTPException: If the server does not support speech generation or the
        synthesis request fails unexpectedly.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return _unsupported_api_response(raw_request, "Speech API")
    try:
        result = await handler.create_speech(request, raw_request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        return result
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@speech_router.post(
    "/v1/audio/speech/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_speech_batch(request: BatchSpeechRequest, raw_request: Request):
    handler = Omnispeech(raw_request)
    if handler is None:
        return _unsupported_api_response(raw_request, "Speech API")
    try:
        result = await handler.create_speech_batch(request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        # exclude_none so optional per-item fields are omitted rather than
        # serialized as null: errored items drop `usage`/`audio_data`/`media_type`,
        # successful items drop `error`. Matches the documented batch response shape.
        return JSONResponse(content=result.model_dump(exclude_none=True))
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


@speech_router.post(
    "/v1/audio/generate",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"audio/*": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_audio_generate(request: OpenAICreateAudioGenerateRequest, raw_request: Request):
    handler = OmniAudioGenerate(raw_request)
    if handler is None:
        return _unsupported_api_response(raw_request, "Audio Generate API")
    try:
        result = await handler.create_audio_generate(request, raw_request)
        if isinstance(result, ErrorResponse):
            return _error_response_to_json_response(result)
        return result
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Voice management
# ---------------------------------------------------------------------------


@speech_router.get(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def list_voices(raw_request: Request):
    """List available TTS voices exposed by the loaded speech model.

    Args:
        raw_request: Raw FastAPI request for accessing app state.

    Returns:
        A JSON payload containing the sorted set of supported speaker names, or
        an OpenAI-style error response when the current server configuration
        does not support the Speech API.
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return _create_speech_error_json_response(
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    speakers = sorted(handler._get_available_voices())

    # Get uploaded speakers details
    uploaded_speakers = []
    if hasattr(handler, "uploaded_speakers"):
        for voice_name, info in handler.uploaded_speakers.items():
            voice_entry = {
                "name": info.get("name", voice_name),
                "consent": info.get("consent", ""),
                "created_at": info.get("created_at", 0),
                "file_size": info.get("file_size", 0),
                "mime_type": info.get("mime_type", ""),
                "embedding_source": info.get("embedding_source", "audio"),
                "embedding_dim": info.get("embedding_dim"),
            }
            if info.get("ref_text"):
                voice_entry["ref_text"] = info["ref_text"]
            if info.get("speaker_description"):
                voice_entry["speaker_description"] = info["speaker_description"]
            uploaded_speakers.append(voice_entry)

    return JSONResponse(content={"voices": speakers, "uploaded_voices": uploaded_speakers})


@speech_router.post(
    "/v1/audio/voices",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def upload_voice(
    raw_request: Request,
    audio_sample: UploadFile | None = File(None),
    speaker_embedding: str | None = Form(None),
    consent: str = Form(...),
    name: str = Form(...),
    ref_text: str | None = Form(None),
    speaker_description: str | None = Form(None),
):
    """Upload a new voice for voice cloning.

    Accepts either an audio file or a pre-computed speaker embedding vector.
    These are mutually exclusive: provide one or the other.

    When using ``audio_sample``, the server stores the audio and extracts the
    speaker embedding on first use (Base task models only).

    When using ``speaker_embedding``, pass a JSON-encoded list of floats
    (1024-dim for 0.6B, 2048-dim for 1.7B). The voice is stored as a
    safetensors file and is immediately ready for use.

    Args:
        audio_sample: Audio file (max 10MB). Mutually exclusive with speaker_embedding.
        speaker_embedding: JSON-encoded float list. Mutually exclusive with audio_sample.
        consent: Consent recording ID
        name: Name for the new voice
        ref_text: Optional transcript of the audio for ICL (in-context
            learning) mode. When provided, voice clone requests using this
            voice will produce higher quality results.
        speaker_description: Optional free-form description of the voice
            (e.g. "warm speaker", "energetic narrator").
        raw_request: Raw FastAPI request

    Returns:
        JSON response with voice information
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return _create_speech_error_json_response(
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    try:
        if speaker_embedding is not None and audio_sample is not None:
            return _create_speech_error_json_response("'audio_sample' and 'speaker_embedding' are mutually exclusive")
        if speaker_embedding is not None:
            result = await handler.upload_voice_embedding(speaker_embedding, consent, name)
        elif audio_sample is not None:
            result = await handler.upload_voice(
                audio_sample,
                consent,
                name,
                ref_text=ref_text,
                speaker_description=speaker_description,
            )
        else:
            return _create_speech_error_json_response("Either 'audio_sample' or 'speaker_embedding' must be provided")

        return JSONResponse(content={"success": True, "voice": result})

    except ValueError as e:
        return _create_speech_error_json_response(str(e))
    except Exception as e:
        logger.exception(f"Failed to upload voice: {e}")
        return _create_speech_error_json_response(
            f"Failed to upload voice: {str(e)}",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@speech_router.delete(
    "/v1/audio/voices/{name}",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def delete_voice(name: str, raw_request: Request):
    """Delete an uploaded voice.

    Deletes the voice sample and associated metadata. Also removes any
    cached voice clone prompts for this voice.

    Args:
        name: Name of the voice to delete
        raw_request: Raw FastAPI request

    Returns:
        JSON response indicating success or failure
    """
    handler = Omnispeech(raw_request)
    if handler is None:
        return _create_speech_error_json_response(
            "The model does not support Speech API",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    try:
        # Delete the voice
        await handler.delete_voice(name)
        return JSONResponse(content={"success": True, "message": f"Voice '{name}' deleted successfully"})
    except InvalidPresetVoiceReferenceError as e:
        return _create_speech_error_json_response(
            str(e),
            err_type="ForbiddenError",
            status_code=HTTPStatus.FORBIDDEN,
        )
    except InvalidVoiceReferenceError as e:
        return _create_speech_error_json_response(
            str(e),
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
    except ValueError as e:
        return _create_speech_error_json_response(str(e))
    except Exception as e:
        logger.exception(f"Failed to delete voice '{name}': {e}")
        return _create_speech_error_json_response(
            f"Failed to delete voice: {str(e)}",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


# ---------------------------------------------------------------------------
# Streaming speech WebSocket
# ---------------------------------------------------------------------------


@speech_router.websocket("/v1/audio/speech/stream")
async def streaming_speech(websocket: WebSocket):
    """WebSocket endpoint for streaming text input TTS.

    Accepts text incrementally and returns audio for the buffered text on
    input.done, which flushes without closing the connection. See
    serving_speech_stream.py for protocol.
    """
    handler = getattr(websocket.app.state, "openai_streaming_speech", None)
    if handler is None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "error",
                "message": "Streaming speech is not available",
            }
        )
        await websocket.close()
        return
    await handler.handle_session(websocket)
