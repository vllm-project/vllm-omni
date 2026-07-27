# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WebSocket handler for streaming generated video chunks.

Protocol:
    Client -> Server:
        {"type": "session.start", "model": "...", "prompt": "...", "format": "m4s", ...}
        {"type": "session.prompt_update", "prompt": "..."}  # currently unsupported
        {"type": "session.stop"}
        {"type": "session.ping"}  # optional; refreshes stall clock

    Server -> Client:
        {"type": "video.start", "request_id": "...", "config": {...}, "format": "m4s"}
        <binary frame: fragmented MP4 bytes>
        {"type": "session.done", ...}
        {"type": "session.pong"}  # reply to session.ping
        {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import time
import uuid
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest, VideoParams
from vllm_omni.entrypoints.openai.stage_params import (
    build_stage_sampling_params_list,
    get_default_sampling_params_list,
)
from vllm_omni.entrypoints.openai.utils import get_stage_type, parse_lora_request
from vllm_omni.entrypoints.openai.video_api_utils import StreamingVideoFormat, create_streaming_video_encoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

logger = init_logger(__name__)

_DEFAULT_STALL_TIMEOUT = 60.0
_DEFAULT_START_TIMEOUT = 10.0
_CONTROL_POLL_INTERVAL = 1.0
_MAX_START_MESSAGE_SIZE = 4 * 1024 * 1024
_MAX_CONTROL_MESSAGE_SIZE = 128 * 1024


class _SessionProgress:
    """Tracks last server-side progress for stall-timeout enforcement."""

    def __init__(self) -> None:
        self._last_at = time.monotonic()

    def touch(self) -> None:
        self._last_at = time.monotonic()

    def stalled_for(self, stall_timeout: float) -> bool:
        return time.monotonic() - self._last_at > stall_timeout


class OmniStreamingVideoOutputHandler:
    """Handles `/v1/realtime/video` sessions for generated video output."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_name: str | None = None,
        stage_configs: list[Any] | None = None,
        stall_timeout: float = _DEFAULT_STALL_TIMEOUT,
        start_timeout: float = _DEFAULT_START_TIMEOUT,
    ) -> None:
        self._engine_client = engine_client
        self._model_name = model_name
        self._stage_configs = stage_configs
        self._stall_timeout = stall_timeout
        self._start_timeout = start_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        await websocket.accept()

        request_id: str | None = None
        stop_event = asyncio.Event()
        send_lock = asyncio.Lock()
        control_task: asyncio.Task[None] | None = None
        stopped = False
        chunks_sent = 0

        try:
            start = await self._receive_start(websocket)
            if start is None:
                return
            request, output_format = start

            request_id = f"video_stream-{uuid.uuid4().hex}"
            progress = _SessionProgress()
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "video.start",
                        "request_id": request_id,
                        "format": output_format,
                        "config": request.model_dump(exclude_none=True),
                    }
                )
            progress.touch()

            control_task = asyncio.create_task(
                self._control_loop(websocket, request_id, stop_event, send_lock, progress)
            )
            async for chunk in self._stream_video_bytes(request, request_id, output_format, progress):
                if stop_event.is_set():
                    stopped = True
                    break
                async with send_lock:
                    await websocket.send_bytes(chunk)
                chunks_sent += 1
                progress.touch()

            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "session.done",
                        "request_id": request_id,
                        "chunks": chunks_sent,
                        "stopped": stopped or stop_event.is_set(),
                    }
                )
        except WebSocketDisconnect:
            logger.info("Streaming video output: client disconnected")
            if request_id is not None:
                await self._abort_request(request_id)
        except Exception as e:
            if stop_event.is_set():
                try:
                    async with send_lock:
                        await websocket.send_json(
                            {
                                "type": "session.done",
                                "request_id": request_id,
                                "chunks": chunks_sent,
                                "stopped": True,
                            }
                        )
                except Exception:
                    logger.debug("Failed to send stopped session.done", exc_info=True)
                return
            logger.exception("Streaming video output session error: %s", e)
            await self._send_error(websocket, str(e))
        finally:
            if control_task is not None:
                control_task.cancel()
                await asyncio.gather(control_task, return_exceptions=True)

    async def _receive_start(self, websocket: WebSocket) -> tuple[VideoGenerationRequest, StreamingVideoFormat] | None:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._start_timeout)
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.start")
            return None

        if len(raw) > _MAX_START_MESSAGE_SIZE:
            await self._send_error(websocket, "session.start message too large")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.start")
            return None

        if not isinstance(msg, dict):
            await self._send_error(websocket, "session.start must be a JSON object")
            return None

        if msg.get("type") != "session.start":
            await self._send_error(websocket, f"Expected session.start, got: {msg.get('type')}")
            return None

        try:
            output_format = self._coerce_output_format(msg.pop("format", "m4s"))
        except ValueError as e:
            await self._send_error(websocket, str(e))
            return None

        request_data = self._coerce_request_data(msg)
        try:
            request = VideoGenerationRequest(**request_data)
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session.start: {e}")
            return None

        configured_model = self._model_name
        if request.model is not None and configured_model is not None and request.model != configured_model:
            await self._send_error(
                websocket,
                f"Model mismatch: request specifies '{request.model}' but server is running '{configured_model}'.",
            )
            return None

        return request, output_format

    @staticmethod
    def _coerce_request_data(msg: dict[str, Any]) -> dict[str, Any]:
        request_data = {k: v for k, v in msg.items() if k not in {"type", "format", "sampling_params"}}
        sampling_params = msg.get("sampling_params")
        if isinstance(sampling_params, dict):
            extra_params = dict(request_data.get("extra_params") or {})
            for key, value in sampling_params.items():
                if key in VideoGenerationRequest.model_fields and key not in request_data:
                    request_data[key] = value
                else:
                    extra_params[key] = value
            if extra_params:
                request_data["extra_params"] = extra_params
        return request_data

    @staticmethod
    def _coerce_output_format(raw_format: Any) -> StreamingVideoFormat:
        if raw_format == "m4s":
            return cast(StreamingVideoFormat, raw_format)
        raise ValueError("Invalid session.start format: expected 'm4s'.")

    async def _control_loop(
        self,
        websocket: WebSocket,
        request_id: str,
        stop_event: asyncio.Event,
        send_lock: asyncio.Lock,
        progress: _SessionProgress,
    ) -> None:
        while not stop_event.is_set():
            if progress.stalled_for(self._stall_timeout):
                await self._send_error(
                    websocket,
                    "Stall timeout: no generation progress",
                    send_lock=send_lock,
                )
                await self._abort_request(request_id)
                stop_event.set()
                return

            poll_timeout = min(_CONTROL_POLL_INTERVAL, self._stall_timeout)
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=poll_timeout,
                )
            except asyncio.TimeoutError:
                continue

            if len(raw) > _MAX_CONTROL_MESSAGE_SIZE:
                await self._send_error(websocket, "control message too large", send_lock=send_lock)
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON message", send_lock=send_lock)
                continue

            if not isinstance(msg, dict):
                await self._send_error(websocket, "WebSocket messages must be JSON objects", send_lock=send_lock)
                continue

            msg_type = msg.get("type")
            if msg_type == "session.stop":
                await self._abort_request(request_id)
                stop_event.set()
                return
            if msg_type == "session.ping":
                progress.touch()
                try:
                    async with send_lock:
                        await websocket.send_json({"type": "session.pong"})
                except Exception:
                    pass
                continue
            if msg_type == "session.prompt_update":
                await self._send_error(
                    websocket,
                    "session.prompt_update is not supported by this backend yet",
                    send_lock=send_lock,
                )
                continue
            await self._send_error(websocket, f"Unknown message type: {msg_type}", send_lock=send_lock)

    async def _abort_request(self, request_id: str) -> None:
        try:
            abort = getattr(self._engine_client, "abort", None)
            if callable(abort):
                result = abort(request_id)
                if inspect.isawaitable(result):
                    await result
        except Exception:
            logger.debug("Failed to abort streaming video request %s", request_id, exc_info=True)

    async def _stream_video_bytes(
        self,
        request: VideoGenerationRequest,
        request_id: str,
        output_format: StreamingVideoFormat,
        progress: _SessionProgress,
    ) -> AsyncGenerator[bytes, None]:
        """Yield encoded video bytes from diffusion streaming outputs."""
        prompt, gen_params, vp = self._build_prompt_and_sampling_params(request)
        video_codec_options = {"preset": "ultrafast", "threads": "0", "tune": "zerolatency"}
        if isinstance(request.extra_params, dict) and "video_codec_options" in request.extra_params:
            video_codec_options = request.extra_params["video_codec_options"]

        output_fps = vp.fps or gen_params.resolved_frame_rate or 16
        encoder = create_streaming_video_encoder(
            output_format=output_format,
            fps=output_fps,
            video_codec_options=video_codec_options,
        )
        completed = False
        try:
            async for result in self._iter_generation_outputs(prompt, gen_params, request_id, progress):
                if error := getattr(result, "error", None):
                    raise RuntimeError(str(error))
                videos = self._extract_video_outputs(result)
                for video in videos:
                    chunk = encoder.encode(video)
                    if chunk:
                        yield chunk

            completed = True
            final_chunk = encoder.close()
            if final_chunk:
                yield final_chunk
        finally:
            if not completed:
                encoder.close()

    def _build_prompt_and_sampling_params(
        self,
        request: VideoGenerationRequest,
    ) -> tuple[OmniTextPrompt, OmniDiffusionSamplingParams, VideoParams]:
        """Build text-only diffusion inputs for the streaming endpoint."""
        prompt: OmniTextPrompt = OmniTextPrompt(prompt=request.prompt)
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt

        gen_params = self._resolve_default_sampling_params()
        vp = request.resolve_video_params()
        if vp.width is not None and vp.height is not None:
            gen_params.width = vp.width
            gen_params.height = vp.height
        if vp.num_frames is not None:
            gen_params.num_frames = vp.num_frames
        if vp.fps is not None:
            gen_params.fps = vp.fps
            gen_params.frame_rate = float(vp.fps)

        provided_fields = request.model_fields_set
        if "num_inference_steps" in provided_fields and request.num_inference_steps is not None:
            gen_params.num_inference_steps = request.num_inference_steps
        if "guidance_scale" in provided_fields and request.guidance_scale is not None:
            gen_params.guidance_scale = request.guidance_scale
        if "guidance_scale_2" in provided_fields and request.guidance_scale_2 is not None:
            gen_params.guidance_scale_2 = request.guidance_scale_2
        if "true_cfg_scale" in provided_fields and request.true_cfg_scale is not None:
            gen_params.true_cfg_scale = request.true_cfg_scale
        if "seed" in provided_fields and request.seed is not None:
            gen_params.seed = request.seed
        if "boundary_ratio" in provided_fields and request.boundary_ratio is not None:
            gen_params.boundary_ratio = request.boundary_ratio
        if "flow_shift" in provided_fields and request.flow_shift is not None:
            gen_params.extra_args["flow_shift"] = request.flow_shift

        if request.extra_params is not None:
            if not isinstance(request.extra_params, dict):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="extra_params must be a JSON object/dict.",
                )
            gen_params.extra_args.update(request.extra_params)

        self._apply_lora(request.lora, gen_params)
        return prompt, gen_params, vp

    def _resolve_default_sampling_params(self) -> OmniDiffusionSamplingParams:
        default_sampling_params_list = getattr(self._engine_client, "default_sampling_params_list", None)
        if default_sampling_params_list:
            for params in default_sampling_params_list:
                if isinstance(params, OmniDiffusionSamplingParams):
                    return copy.deepcopy(params)
        return OmniDiffusionSamplingParams()

    @staticmethod
    def _apply_lora(lora_body: Any, gen_params: OmniDiffusionSamplingParams) -> None:
        try:
            lora_request, lora_scale = parse_lora_request(lora_body)
        except ValueError as e:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=str(e),
            ) from e

        if lora_request is None:
            return

        gen_params.lora_request = lora_request
        if lora_scale is not None:
            gen_params.lora_scale = lora_scale

    async def _iter_generation_outputs(
        self,
        prompt: OmniTextPrompt,
        gen_params: OmniDiffusionSamplingParams,
        request_id: str,
        progress: _SessionProgress,
    ) -> AsyncGenerator[Any, None]:
        stage_configs = self._stage_configs or getattr(self._engine_client, "stage_configs", None)
        if not stage_configs:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                detail="Stage configs not found. Start server with an omni diffusion model.",
            )

        for stage in stage_configs:
            stage_type = get_stage_type(stage)
            if stage_type != "diffusion":
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail=f"Video generation only supports diffusion stages, found '{stage_type}' stage.",
                )

        engine_client = cast(Any, self._engine_client)
        sampling_params_list = build_stage_sampling_params_list(
            list(stage_configs),
            get_default_sampling_params_list(engine_client),
            diffusion_params=gen_params,
            replace_diffusion_params=True,
        )

        async for output in engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            progress.touch()
            yield output

    @staticmethod
    def _normalize_video_outputs(videos: Any) -> list[Any]:
        if videos is None:
            return []
        if hasattr(videos, "ndim") and videos.ndim == 5:
            return [videos[i] for i in range(videos.shape[0])]
        if isinstance(videos, list):
            if not videos:
                return []
            first = videos[0]
            if hasattr(first, "ndim") and first.ndim == 5:
                flattened: list[Any] = []
                for item in videos:
                    if hasattr(item, "ndim") and item.ndim == 5:
                        flattened.extend([item[i] for i in range(item.shape[0])])
                    else:
                        flattened.append(item)
                return flattened
            if isinstance(first, list):
                return videos
            if hasattr(first, "ndim") and first.ndim == 3:
                return [videos]
            return videos
        return [videos]

    def _extract_video_outputs(self, result: Any) -> list[Any]:
        videos = None
        if hasattr(result, "images") and result.images:
            videos = result.images
        elif hasattr(result, "request_output"):
            request_output = result.request_output
            if isinstance(request_output, dict) and request_output.get("images"):
                videos = request_output["images"]
            elif not isinstance(request_output, dict):
                if hasattr(request_output, "images") and request_output.images:
                    videos = request_output.images
                elif hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                    videos = request_output.multimodal_output.get("video")
        if videos is None and hasattr(result, "multimodal_output") and result.multimodal_output:
            videos = result.multimodal_output.get("video")

        normalized = self._normalize_video_outputs(videos)
        if not normalized:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No video outputs found in generation result.",
            )
        return normalized

    @staticmethod
    async def _send_error(
        websocket: WebSocket,
        message: str,
        code: int | None = None,
        *,
        send_lock: asyncio.Lock | None = None,
    ) -> None:
        try:
            payload: dict[str, Any] = {"type": "error", "message": message}
            if code is not None:
                payload["code"] = code
            if send_lock is None:
                await websocket.send_json(payload)
            else:
                async with send_lock:
                    await websocket.send_json(payload)
        except Exception:
            pass
