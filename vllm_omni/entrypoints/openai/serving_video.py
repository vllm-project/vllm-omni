# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, cast

from fastapi import HTTPException
from PIL import Image
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoData,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from vllm_omni.entrypoints.openai.utils import get_stage_type, parse_lora_request
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes, encode_video_base64
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams, OmniTextPrompt

logger = init_logger(__name__)


@dataclass
class ReferenceImage:
    """Reference class for tracking additional metadata if needed"""

    data: Image.Image


@dataclass
class VideoGenerationArtifacts:
    """Normalized outputs and profiler metadata extracted from one request."""

    videos: list[Any]
    audios: list[Any | None]
    audio_sample_rate: int
    output_fps: int
    stage_durations: dict[str, float]
    peak_memory_mb: float


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

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def stage_configs(self) -> list[Any] | None:
        return self._stage_configs

    def set_stage_configs_if_missing(self, stage_configs: list[Any] | None) -> None:
        if self._stage_configs is None and stage_configs is not None:
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

    async def _run_and_extract(
        self,
        request: VideoGenerationRequest,
        reference_id: str,
        *,
        reference_image: ReferenceImage | None = None,
    ) -> VideoGenerationArtifacts:
        """Run the generation pipeline and extract video/audio/profiler outputs."""
        prompt: OmniTextPrompt = OmniTextPrompt(prompt=request.prompt)
        if request.negative_prompt is not None:
            prompt["negative_prompt"] = request.negative_prompt

        gen_params = OmniDiffusionSamplingParams()

        input_image = None if reference_image is None else reference_image.data
        vp = request.resolve_video_params()
        if input_image is not None and vp.width is not None and vp.height is not None:
            target_size = (vp.width, vp.height)
            if input_image.size != target_size:
                input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
        if input_image is not None:
            prompt["multi_modal_data"] = {"image": input_image}
        if vp.width is not None and vp.height is not None:
            gen_params.width = vp.width
            gen_params.height = vp.height
        if vp.num_frames is not None:
            gen_params.num_frames = vp.num_frames
        if vp.fps is not None:
            gen_params.fps = vp.fps
            gen_params.frame_rate = float(vp.fps)

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

        # Apply model-specific extra parameters
        if request.extra_params is not None:
            if not isinstance(request.extra_params, dict):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="extra_params must be a JSON object/dict.",
                )
            # Merge extra_params into extra_args
            gen_params.extra_args.update(request.extra_params)
            logger.info("Applied extra_params: %s", request.extra_params)

        self._apply_lora(request.lora, gen_params)

        logger.info(
            "Video sampling params: steps=%s guidance=%s guidance_2=%s seed=%s",
            gen_params.num_inference_steps,
            gen_params.guidance_scale,
            gen_params.guidance_scale_2,
            gen_params.seed,
        )

        result = await self._run_generation(prompt, gen_params, reference_id)
        videos = self._extract_video_outputs(result)
        audios = self._extract_audio_outputs(result, expected_count=len(videos))
        audio_sample_rate = self._resolve_audio_sample_rate(result)
        output_fps = vp.fps or self._resolve_fps(result) or 24
        return VideoGenerationArtifacts(
            videos=videos,
            audios=audios,
            audio_sample_rate=audio_sample_rate,
            output_fps=output_fps,
            stage_durations=self._extract_stage_durations(result),
            peak_memory_mb=self._extract_peak_memory_mb(result),
        )

    async def generate_videos(
        self,
        request: VideoGenerationRequest,
        reference_id: str,
        *,
        reference_image: ReferenceImage | None = None,
    ) -> VideoGenerationResponse:
        artifacts = await self._run_and_extract(request, reference_id, reference_image=reference_image)
        _t_encode_start = time.perf_counter()
        video_data = [
            VideoData(
                b64_json=(
                    encode_video_base64(video, fps=artifacts.output_fps)
                    if artifacts.audios[idx] is None
                    else encode_video_base64(
                        video,
                        fps=artifacts.output_fps,
                        audio=artifacts.audios[idx],
                        audio_sample_rate=artifacts.audio_sample_rate,
                    )
                )
            )
            for idx, video in enumerate(artifacts.videos)
        ]
        _t_encode_ms = (time.perf_counter() - _t_encode_start) * 1000
        logger.info("Video response encoding (MP4+base64): %.2f ms", _t_encode_ms)
        return VideoGenerationResponse(
            created=int(time.time()),
            data=video_data,
            stage_durations=artifacts.stage_durations,
            peak_memory_mb=artifacts.peak_memory_mb,
        )

    async def generate_video_bytes(
        self,
        request: VideoGenerationRequest,
        reference_id: str,
        *,
        reference_image: ReferenceImage | None = None,
    ) -> tuple[bytes, dict[str, float], float]:
        """Generate a video and return raw MP4 bytes, bypassing base64 encoding."""
        artifacts = await self._run_and_extract(request, reference_id, reference_image=reference_image)
        if len(artifacts.videos) > 1:
            logger.warning(
                "Video request %s generated %d outputs; returning only the first.",
                reference_id,
                len(artifacts.videos),
            )
        audio = artifacts.audios[0]
        _t_encode_start = time.perf_counter()
        video_bytes = _encode_video_bytes(
            artifacts.videos[0],
            fps=artifacts.output_fps,
            **({"audio": audio, "audio_sample_rate": artifacts.audio_sample_rate} if audio is not None else {}),
        )
        _t_encode_ms = (time.perf_counter() - _t_encode_start) * 1000
        logger.info("Video response encoding (MP4 bytes): %.2f ms", _t_encode_ms)
        return video_bytes, artifacts.stage_durations, artifacts.peak_memory_mb

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

    async def _run_generation(
        self,
        prompt: OmniTextPrompt,
        gen_params: OmniDiffusionSamplingParams,
        request_id: str,
    ) -> Any:
        stage_configs = self._stage_configs or getattr(self._engine_client, "stage_configs", None)

        if not stage_configs:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                detail="Stage configs not found. Start server with an omni diffusion model.",
            )

        # Video generation endpoint only supports diffusion stages.
        for stage in stage_configs:
            stage_type = get_stage_type(stage)
            if stage_type != "diffusion":
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    detail=f"Video generation only supports diffusion stages, found '{stage_type}' stage.",
                )

        # Common generation logic for both paths
        engine_client = cast(AsyncOmni, self._engine_client)
        sampling_params_list: list[OmniSamplingParams] = [gen_params for _ in stage_configs]

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

    @staticmethod
    def _resolve_fps(result: Any) -> int | None:
        """Extract fps from multimodal_output if the model reported it."""
        multimodal_output = getattr(result, "multimodal_output", None)
        if isinstance(multimodal_output, dict):
            fps = multimodal_output.get("fps")
            if fps is not None:
                try:
                    fps_val = fps.item() if hasattr(fps, "item") else int(fps)
                    if fps_val > 0:
                        return fps_val
                except (TypeError, ValueError):
                    pass

        request_output = getattr(result, "request_output", None)
        if isinstance(request_output, dict):
            mm = request_output.get("multimodal_output") or {}
            if isinstance(mm, dict):
                fps = mm.get("fps")
                if fps is not None:
                    try:
                        fps_val = fps.item() if hasattr(fps, "item") else int(fps)
                        if fps_val > 0:
                            return fps_val
                    except (TypeError, ValueError):
                        pass
        elif hasattr(request_output, "multimodal_output"):
            mm = getattr(request_output, "multimodal_output", None)
            if isinstance(mm, dict):
                fps = mm.get("fps")
                if fps is not None:
                    try:
                        fps_val = fps.item() if hasattr(fps, "item") else int(fps)
                        if fps_val > 0:
                            return fps_val
                    except (TypeError, ValueError):
                        pass

        return None

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

    @staticmethod
    def _extract_stage_durations(result: Any) -> dict[str, float]:
        stage_durations = getattr(result, "stage_durations", None)
        return stage_durations if isinstance(stage_durations, dict) else {}

    @staticmethod
    def _extract_peak_memory_mb(result: Any) -> float:
        peak_memory_mb = getattr(result, "peak_memory_mb", 0.0)
        try:
            return float(peak_memory_mb or 0.0)
        except (TypeError, ValueError):
            return 0.0
