# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
import uuid
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException, Request
from PIL import Image
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.image_api_utils import parse_size
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoData,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from vllm_omni.entrypoints.openai.video_api_utils import decode_input_reference, encode_video_base64
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams, OmniTextPrompt
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

logger = init_logger(__name__)


class OmniOpenAIServingVideo:
    """OpenAI-style video generation handler for omni diffusion models."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_name: str | None = None,
        stage_configs: list[Any] | None = None,
    ) -> None:
        self._engine_client = engine_client
        self._model_name = model_name
        self._stage_configs = stage_configs

    @classmethod
    def for_diffusion(
        cls,
        diffusion_engine: EngineClient,
        model_name: str,
        stage_configs: list[Any] | None = None,
    ) -> OmniOpenAIServingVideo:
        return cls(
            diffusion_engine,
            model_name=model_name,
            stage_configs=stage_configs,
        )

    async def generate_videos(
        self,
        request: VideoGenerationRequest,
        raw_request: Request | None = None,
        *,
        input_reference_bytes: bytes | None = None,
    ) -> VideoGenerationResponse:
        if request.stream:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Streaming video generation is not supported yet.",
            )

        request_id = f"video_gen_{uuid.uuid4().hex}"
        model_name = self._resolve_model_name(raw_request)

        if request.model is not None and model_name is not None and request.model != model_name:
            logger.warning(
                "Model mismatch: request specifies '%s' but server is running '%s'. Using server model.",
                request.model,
                model_name,
            )

        prompt: OmniTextPrompt = {"prompt": request.prompt}
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt

        input_image = None
        try:
            input_image = decode_input_reference(request.input_reference, input_reference_bytes)
        except ValueError as exc:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=str(exc),
            ) from exc

        gen_params = OmniDiffusionSamplingParams(num_outputs_per_prompt=request.n)

        width, height, num_frames, fps = self._resolve_video_params(request)
        if input_image is not None and width is not None and height is not None:
            target_size = (width, height)
            if input_image.size != target_size:
                input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
        if input_image is not None:
            prompt["multi_modal_data"] = {"image": input_image}

        if width is not None and height is not None:
            gen_params.width = width
            gen_params.height = height
        if num_frames is not None:
            gen_params.num_frames = num_frames
        if fps is not None:
            gen_params.fps = fps
            gen_params.frame_rate = float(fps)

        if request.num_inference_steps is not None:
            gen_params.num_inference_steps = request.num_inference_steps
        if request.guidance_scale is not None:
            gen_params.guidance_scale = request.guidance_scale
        if request.guidance_scale_2 is not None:
            gen_params.guidance_scale_2 = request.guidance_scale_2
        if request.true_cfg_scale is not None:
            gen_params.true_cfg_scale = request.true_cfg_scale
        if request.seed is not None:
            gen_params.seed = request.seed
        if request.boundary_ratio is not None:
            gen_params.boundary_ratio = request.boundary_ratio

        logger.info(
            "Boundary ratio parse: request=%s gen_params=%s",
            request.boundary_ratio,
            gen_params.boundary_ratio,
        )
        if request.flow_shift is not None:
            gen_params.extra_args["flow_shift"] = request.flow_shift

        self._apply_lora(request.lora, gen_params)

        logger.info(
            "Video sampling params: steps=%s guidance=%s guidance_2=%s seed=%s",
            gen_params.num_inference_steps,
            gen_params.guidance_scale,
            gen_params.guidance_scale_2,
            gen_params.seed,
        )

        result = await self._run_generation(prompt, gen_params, request_id, raw_request)
        _t_encode_start = time.perf_counter()
        videos = self._extract_video_outputs(result)
        audios = self._extract_audio_outputs(result, expected_count=len(videos))
        output_fps = fps or 24
        audio_sample_rate = self._resolve_audio_sample_rate(result)

        video_data = [
            VideoData(
                b64_json=(
                    encode_video_base64(video, fps=output_fps)
                    if audios[idx] is None
                    else encode_video_base64(
                        video,
                        fps=output_fps,
                        audio=audios[idx],
                        audio_sample_rate=audio_sample_rate,
                    )
                )
            )
            for idx, video in enumerate(videos)
        ]
        _t_encode_ms = (time.perf_counter() - _t_encode_start) * 1000
        logger.info("Video response encoding (MP4+base64): %.2f ms", _t_encode_ms)
        return VideoGenerationResponse(created=int(time.time()), data=video_data)

    def _resolve_model_name(self, raw_request: Request | None) -> str | None:
        if self._model_name:
            return self._model_name
        if raw_request is None:
            return None
        serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
        if serving_models and getattr(serving_models, "base_model_paths", None):
            base_paths = serving_models.base_model_paths
            if base_paths:
                return base_paths[0].name
        return None

    @staticmethod
    def _resolve_video_params(request: VideoGenerationRequest) -> tuple[int | None, int | None, int | None, int | None]:
        width = request.width or (request.video_params.width if request.video_params else None)
        height = request.height or (request.video_params.height if request.video_params else None)
        num_frames = request.num_frames or (request.video_params.num_frames if request.video_params else None)
        fps = request.fps or (request.video_params.fps if request.video_params else None)
        seconds = request.seconds

        if request.size:
            width, height = parse_size(request.size)

        if fps is None:
            fps = 24  # Default FPS if not specified

        if num_frames is None and seconds is not None:
            num_frames = int(seconds) * int(fps)

        return width, height, num_frames, fps

    @staticmethod
    def _apply_lora(lora_body: Any, gen_params: OmniDiffusionSamplingParams) -> None:
        if lora_body is None:
            return
        if not isinstance(lora_body, dict):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora field: expected an object.",
            )
        lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
        lora_path = (
            lora_body.get("local_path")
            or lora_body.get("path")
            or lora_body.get("lora_path")
            or lora_body.get("lora_local_path")
        )
        lora_scale = lora_body.get("scale")
        if lora_scale is None:
            lora_scale = lora_body.get("lora_scale")
        lora_int_id = lora_body.get("int_id")
        if lora_int_id is None:
            lora_int_id = lora_body.get("lora_int_id")
        if lora_int_id is None and lora_path:
            lora_int_id = stable_lora_int_id(str(lora_path))

        if not lora_name or not lora_path:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid lora object: both name and path are required.",
            )

        gen_params.lora_request = LoRARequest(str(lora_name), int(lora_int_id), str(lora_path))
        if lora_scale is not None:
            gen_params.lora_scale = float(lora_scale)

    async def _run_generation(
        self,
        prompt: OmniTextPrompt,
        gen_params: OmniDiffusionSamplingParams,
        request_id: str,
        raw_request: Request | None,
    ) -> Any:
        has_stage_list = hasattr(self._engine_client, "stage_list")
        logger.info(
            "Video generation routing: stage_configs=%s, has_stage_list=%s, engine_type=%s",
            "present"
            if (self._stage_configs or (getattr(raw_request.app.state, "stage_configs", None) if raw_request else None))
            else "missing",
            has_stage_list,
            type(self._engine_client).__name__,
        )
        stage_configs = (
            self._stage_configs
            or (getattr(raw_request.app.state, "stage_configs", None) if raw_request else None)
            or getattr(self._engine_client, "stage_configs", None)
        )

        if not stage_configs:
            if not hasattr(self._engine_client, "stage_list"):
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail="Stage configs not found. Start server with an omni diffusion model.",
                )

        # Video generation endpoint only supports diffusion stages.
        if stage_configs:
            for stage in stage_configs:
                # Extract stage_type: dicts and OmegaConf objects use .get(), others use getattr
                if hasattr(stage, "get"):
                    stage_type = stage.get("stage_type", "llm")
                else:
                    stage_type = getattr(stage, "stage_type", "llm")

                if stage_type != "diffusion":
                    raise HTTPException(
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                        detail=f"Video generation only supports diffusion stages, found '{stage_type}' stage.",
                    )

        # Common generation logic for both paths
        engine_client = cast(AsyncOmni, self._engine_client)
        stage_list = getattr(engine_client, "stage_list", None)
        if isinstance(stage_list, list):
            sampling_params_list: list[OmniSamplingParams] = [gen_params for _ in stage_list]
        else:
            sampling_params_list = [gen_params]

        result = None
        async for output in engine_client.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            result = output

        if result is None:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No output generated from video generation pipeline.",
            )
        return result

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
            if isinstance(first, Image.Image):
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
            elif hasattr(request_output, "images") and request_output.images:
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
    def _extract_audio_outputs(result: Any, expected_count: int) -> list[Any | None]:
        audio = None
        if hasattr(result, "multimodal_output") and result.multimodal_output:
            audio = result.multimodal_output.get("audio")
        elif hasattr(result, "request_output"):
            request_output = result.request_output
            if isinstance(request_output, dict) and request_output.get("multimodal_output"):
                mm_output = request_output.get("multimodal_output") or {}
                audio = mm_output.get("audio")
            elif hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                audio = request_output.multimodal_output.get("audio")

        if audio is None:
            return [None] * expected_count

        if isinstance(audio, (list, tuple)):
            if len(audio) == expected_count and any(hasattr(item, "shape") or hasattr(item, "ndim") for item in audio):
                return list(audio)
            if expected_count == 1:
                return [audio]

        if hasattr(audio, "ndim") and getattr(audio, "ndim", None) is not None and audio.ndim > 1:
            first_dim = getattr(audio, "shape", [0])[0]
            if first_dim == expected_count:
                return [audio[i] for i in range(expected_count)]

        if expected_count == 1:
            return [audio]

        return [audio] + [None] * max(expected_count - 1, 0)

    def _resolve_audio_sample_rate(self, result: Any) -> int:
        result_sample_rate = self._extract_audio_sample_rate_from_result(result)
        if result_sample_rate is not None:
            return result_sample_rate

        model_config = getattr(self._engine_client, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        config_sample_rate = self._extract_audio_sample_rate_from_config(hf_config)
        if config_sample_rate is not None:
            return config_sample_rate

        return 24000

    @classmethod
    def _extract_audio_sample_rate_from_result(cls, result: Any) -> int | None:
        multimodal_output = getattr(result, "multimodal_output", None)
        if isinstance(multimodal_output, dict):
            sample_rate = cls._coerce_audio_sample_rate(
                multimodal_output.get("audio_sample_rate")
                or multimodal_output.get("sample_rate")
                or multimodal_output.get("sampling_rate")
                or multimodal_output.get("sr")
            )
            if sample_rate is not None:
                return sample_rate

        request_output = getattr(result, "request_output", None)
        if isinstance(request_output, dict):
            multimodal_output = request_output.get("multimodal_output") or {}
            if isinstance(multimodal_output, dict):
                return cls._coerce_audio_sample_rate(
                    multimodal_output.get("audio_sample_rate")
                    or multimodal_output.get("sample_rate")
                    or multimodal_output.get("sampling_rate")
                    or multimodal_output.get("sr")
                )
        elif hasattr(request_output, "multimodal_output"):
            multimodal_output = getattr(request_output, "multimodal_output", None)
            if isinstance(multimodal_output, dict):
                return cls._coerce_audio_sample_rate(
                    multimodal_output.get("audio_sample_rate")
                    or multimodal_output.get("sample_rate")
                    or multimodal_output.get("sampling_rate")
                    or multimodal_output.get("sr")
                )

        return None

    @classmethod
    def _extract_audio_sample_rate_from_config(cls, config: Any) -> int | None:
        if config is None:
            return None

        for attr_name in ("output_sampling_rate", "audio_sample_rate", "sample_rate", "sampling_rate"):
            raw_value = config.get(attr_name) if isinstance(config, dict) else getattr(config, attr_name, None)
            sample_rate = cls._coerce_audio_sample_rate(raw_value)
            if sample_rate is not None:
                return sample_rate

        for component_name in ("vocoder", "audio_vae"):
            component = (
                config.get(component_name) if isinstance(config, dict) else getattr(config, component_name, None)
            )
            if component is None:
                continue

            sample_rate = cls._extract_audio_sample_rate_from_config(component)
            if sample_rate is not None:
                return sample_rate

            component_config = (
                component.get("config") if isinstance(component, dict) else getattr(component, "config", None)
            )
            sample_rate = cls._extract_audio_sample_rate_from_config(component_config)
            if sample_rate is not None:
                return sample_rate

        return None

    @staticmethod
    def _coerce_audio_sample_rate(value: Any) -> int | None:
        if value is None:
            return None

        try:
            sample_rate = value.item() if hasattr(value, "item") else value
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            return None

        return sample_rate if sample_rate > 0 else None
