# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
An high-level API client adapter that forwards ComfyUI inputs to vLLM-Omni's REST API,
and transforms the API responses back to ComfyUI formats.

The image generation part is derived from dougbtv/comfyui-vllm-omni by Doug (@dougbtv).
Original source at https://github.com/dougbtv/comfyui-vllm-omni, distributed under the MIT License.
"""

import asyncio
import json
from typing import Any

import aiohttp
import av.error
import torch
from comfy_api.input import AudioInput, VideoInput

from .format import (
    audio_to_base64,
    audio_to_bytes,
    base64_to_audio,
    base64_to_image_tensor,
    bytes_to_audio,
    bytes_to_video,
    image_tensor_to_base64,
    image_tensor_to_png_bytes,
    mask_tensor_to_png_bytes,
    video_to_base64,
    video_to_bytes,
)
from .logger import get_logger, pretty_printer
from .models import lookup_model_spec, lookup_params_builder
from .types import (
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MAX_TOTAL_REFERENCES,
    MINIMAX_H3_CONTROL_TYPES,
    AudioFormat,
    MiniMaxH3Control,
)
from .validators import validate_minimax_h3_control

logger = get_logger(__name__)


# crf=0 is lossless, not a default: 1px structure hints must survive the encode.
MINIMAX_H3_CONTROL_VIDEO_ENCODING = {"format": "mp4", "codec": "h264", "crf": 0}


def _add_video_upload(
    form: aiohttp.FormData,
    field_name: str,
    video: VideoInput,
    filename: str,
    *,
    format: str | None = None,
    codec: str | None = None,
    crf: float | None = None,
) -> None:
    form.add_field(
        field_name,
        video_to_bytes(video, filename, format=format, codec=codec, crf=crf),
        filename=filename,
        content_type="video/mp4",
    )


async def url_json(session: aiohttp.ClientSession, url: str, verb: str = "get", **kwargs) -> dict[str, Any]:
    try:
        async with getattr(session, verb)(url, **kwargs) as response:
            if not response.ok:
                error_text = await response.text()
                raise (ValueError if response.status < 500 else RuntimeError)(
                    f"vLLM-Omni API returned status {response.status}: {error_text}"
                )
            try:
                return await response.json()
            except aiohttp.ContentTypeError as e:
                raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")
    except aiohttp.ClientError as e:
        raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")


async def url_bytes(session: aiohttp.ClientSession, url: str, verb: str = "get", **kwargs) -> bytes:
    try:
        async with getattr(session, verb)(url, **kwargs) as response:
            if not response.ok:
                error_text = await response.text()
                raise (ValueError if response.status < 500 else RuntimeError)(
                    f"vLLM-Omni API returned status {response.status}: {error_text}"
                )
            return await response.read()
    except aiohttp.ClientError as e:
        raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")


class VLLMOmniClient:
    def __init__(
        self,
        base_url: str,
        timeout: float | None = None,
        poll_interval: float = 5.0,
        max_poll_duration: float = 60 * 30,
    ):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.poll_interval = poll_interval
        self.max_poll_duration = max_poll_duration

    async def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        sampling_params: dict | None = None,
        lora: dict | None = None,
    ) -> torch.Tensor:
        """Run text-to-image generation via DALLE API"""
        await self._check_model_exist(model)

        size = f"{width}x{height}"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if sampling_params is not None:
            payload.update(sampling_params)
        if lora is not None:
            payload["lora"] = lora
        logger.debug("img gen payload: %s", payload)

        url = self.base_url + "/images/generations"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")
                    if "data" not in data:
                        raise RuntimeError("API response missing 'data' field - expected OpenAI DALL-E format")
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(f"API returned image #{idx} without 'b64_json' field")
                        base64_str = img["b64_json"]
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)
                        logger.debug("Image #%d has shape %s", idx, tensor.shape)

                    batch_tensor = torch.stack(image_tensors, dim=0)
                    logger.debug("batch_tensor output has shape: %s", batch_tensor.shape)
                    return batch_tensor

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def edit_image(
        self,
        *,
        model: str,
        prompt: str,
        image: torch.Tensor,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        mask: torch.Tensor | None = None,
        sampling_params: dict | None = None,
        lora: dict | None = None,
    ) -> torch.Tensor:
        """Run image editing via DALLE API"""
        await self._check_model_exist(model)

        size = f"{width}x{height}"
        image_filename = "image.png"  # Required for multipart form
        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field(
            "image",
            image_tensor_to_png_bytes(image, image_filename),
            filename=image_filename,
            content_type="image/png",
        )
        form.add_field("prompt", prompt)
        form.add_field("size", size)
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)
        if sampling_params is not None:
            for k, v in sampling_params.items():
                form.add_field(k, str(v))
        if lora is not None:
            form.add_field("lora", json.dumps(lora, ensure_ascii=False))
        if mask is not None:
            mask_filename = "mask.png"
            form.add_field(
                "mask",
                image_tensor_to_png_bytes(mask, mask_filename),
                filename=mask_filename,
                content_type="image/png",
            )

        url = self.base_url + "/images/edits"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(url, data=form) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    if "data" not in data:
                        raise RuntimeError("API response missing 'data' field - expected OpenAI DALL-E format")
                    if not data["data"]:
                        raise RuntimeError("API returned empty data array")

                    image_tensors = []
                    for idx, img in enumerate(data["data"]):
                        if "b64_json" not in img:
                            raise RuntimeError(f"API returned image #{idx} without 'b64_json' field")
                        base64_str = img["b64_json"]
                        tensor = base64_to_image_tensor(base64_str)
                        image_tensors.append(tensor)

                    return torch.stack(image_tensors, dim=0)

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def generate_image_chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: str | None = None,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        lora: dict | None = None,
    ) -> torch.Tensor:
        payload = VLLMOmniClient._prepare_chat_completion_messages(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            audio=audio,
            video=video,
            sampling_params=sampling_params,
            modalities=["image"],
            # === below are additional `extra_body` fields, handled by **kwargs ===
            lora=lora,
        )
        choices = await self._generate_base_chat_completion(model, payload)

        image_tensors = []
        for idx, img_content in enumerate(choices[0]["message"]["content"]):
            base64_str = img_content.get("image_url", {}).get("url", "")
            if not base64_str:
                raise RuntimeError(f"API returned image #{idx} without image url")
            tensor = base64_to_image_tensor(base64_str)
            image_tensors.append(tensor)

        return torch.stack(image_tensors, dim=0)

    async def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        negative_prompt: str | None = None,
        frame: torch.Tensor | None = None,
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
        references: dict | None = None,
        control: MiniMaxH3Control | None = None,
        sampling_params: dict | None = None,
        model_params: dict | None = None,
        lora: dict | None = None,
        spec_model: str | None = None,
        **extra_params,
    ) -> VideoInput:
        """Post a video job and return the decoded result.

        ``spec_model`` names the model whose payload spec builds the request, for
        deployments that serve a known model under a different ``model`` alias. It
        never reaches the wire; defaults to ``model``.
        """
        if frame is not None and references is not None:
            raise ValueError("Provide only one of frame or references, not both.")
        if frame is not None and (first_frame is not None or last_frame is not None):
            raise ValueError("Provide either frame or first_frame/last_frame, not both.")
        if references is not None and (first_frame is not None or last_frame is not None):
            raise ValueError("Provide either first_frame/last_frame or references, not both.")

        if control is not None and any(value is not None for value in (frame, first_frame, last_frame, references)):
            raise ValueError("MiniMax-H3 control cannot be combined with frame, keyframes, or references.")
        if control is not None:
            validate_minimax_h3_control(control)
            conflicting_namespaces = set(extra_params) & set(MINIMAX_H3_CONTROL_TYPES)
            if conflicting_namespaces:
                conflicts = ", ".join(sorted(conflicting_namespaces))
                raise ValueError(f"Conflicting MiniMax-H3 control namespaces: {conflicts}.")

        spec, matched_pattern = lookup_model_spec(spec_model or model)
        if (first_frame is not None or last_frame is not None) and (
            matched_pattern is None or "MiniMax-H3" not in matched_pattern
        ):
            raise ValueError("first_frame and last_frame are supported only for MiniMax-H3; use frame for this model.")

        # === regular payload fields ===
        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("prompt", prompt)
        form.add_field("width", str(width))
        form.add_field("height", str(height))
        form.add_field("num_frames", str(num_frames))
        form.add_field("fps", str(fps))
        if negative_prompt:
            form.add_field("negative_prompt", negative_prompt)
        if sampling_params is not None:
            for k, v in sampling_params.items():
                form.add_field(k, str(v))
        if lora is not None:
            form.add_field("lora", json.dumps(lora, ensure_ascii=False))

        # === multimodal inputs (first-last-frames, references, etc.) ===
        input_reference_image: torch.Tensor | None = None
        keyframe_images: list[tuple[str, torch.Tensor]] = []
        video_task: str | None = None

        if frame is not None:
            input_reference_image = frame
            video_task = "fl2va"
        elif first_frame is not None or last_frame is not None:
            frame_indices: list[int] = []
            if first_frame is not None:
                keyframe_images.append(("first_frame.png", first_frame))
                frame_indices.append(0)
            if last_frame is not None:
                keyframe_images.append(("last_frame.png", last_frame))
                frame_indices.append(-1)
            if len(keyframe_images) == 1:
                input_reference_image = keyframe_images[0][1]
            extra_params["frame_indices"] = frame_indices
            video_task = "fl2va"
        elif references is not None:
            reference_formats = (
                ("image", MAX_REFERENCE_IMAGES, "png", "image/png", image_tensor_to_png_bytes),
                ("video", MAX_REFERENCE_VIDEOS, "mp4", "video/mp4", video_to_bytes),
                ("audio", MAX_REFERENCE_AUDIOS, "mp3", "audio/mpeg", audio_to_bytes),
            )
            supported_inputs = {f"{kind}_{i}" for kind, limit, *_ in reference_formats for i in range(1, limit + 1)}
            connected = {name: value for name, value in references.items() if value is not None}
            unsupported = connected.keys() - supported_inputs
            if unsupported:
                raise ValueError(f"Unsupported reference input(s): {', '.join(sorted(unsupported))}.")
            if not any(name.startswith(("image_", "video_")) for name in connected):
                raise ValueError(
                    "references requires at least one image or video; audio-only inputs are not supported."
                )
            if len(connected) > MAX_TOTAL_REFERENCES:
                raise ValueError(
                    f"references supports at most {MAX_TOTAL_REFERENCES} inputs in total "
                    f"(up to {MAX_REFERENCE_IMAGES} images, {MAX_REFERENCE_VIDEOS} videos, "
                    f"and {MAX_REFERENCE_AUDIOS} audios)."
                )
            for kind, limit, extension, content_type, encode in reference_formats:
                for index in range(1, limit + 1):
                    name = f"{kind}_{index}"
                    if name in connected:
                        filename = f"{name}.{extension}"
                        form.add_field(
                            "input_references",
                            encode(connected[name], filename),
                            filename=filename,
                            content_type=content_type,
                        )
            video_task = "ref2va"
        else:
            video_task = "t2va"

        if input_reference_image is not None:
            image_filename = keyframe_images[0][0] if keyframe_images else "image.png"
            form.add_field(
                "input_reference",
                image_tensor_to_png_bytes(input_reference_image, image_filename),
                filename=image_filename,
                content_type="image/png",
            )

        if len(keyframe_images) == 2:
            for image_filename, image in keyframe_images:
                form.add_field(
                    "input_references",
                    image_tensor_to_png_bytes(image, image_filename),
                    filename=image_filename,
                    content_type="image/png",
                )
        if control is not None:
            control_type = control["control_type"]
            form.add_field("control_type", control_type)
            control_video = control.get("control_video")
            if control_video is not None:
                control_filename = "control.mp4"
                _add_video_upload(
                    form,
                    "control_reference",
                    control_video,
                    control_filename,
                    **MINIMAX_H3_CONTROL_VIDEO_ENCODING,
                )
            source_video = control.get("source_video")
            if source_video is not None:
                source_filename = "source.mp4"
                _add_video_upload(
                    form,
                    "source_reference",
                    source_video,
                    source_filename,
                    **MINIMAX_H3_CONTROL_VIDEO_ENCODING,
                )
            mask = control.get("mask")
            if mask is not None:
                mask_filename = "mask.png"
                form.add_field(
                    "mask_reference",
                    mask_tensor_to_png_bytes(mask, mask_filename),
                    filename=mask_filename,
                    content_type="image/png",
                )
            mask_video = control.get("mask_video")
            if mask_video is not None:
                mask_filename = "mask.mp4"
                _add_video_upload(
                    form,
                    "mask_reference",
                    mask_video,
                    mask_filename,
                    **MINIMAX_H3_CONTROL_VIDEO_ENCODING,
                )
            extra_params = {
                **extra_params,
                control_type: {"control_context_scale": control["control_context_scale"]},
            }

        # === model specific params. Either use a specialized builder, or add flattened fields as-is ===
        model_params_type = None
        if model_params is not None:
            model_params = dict(model_params)
            model_params_type = model_params.pop("type", None)

        params_builder = lookup_params_builder(spec_model or model, model_params_type)
        if params_builder is not None:
            form_fields = params_builder(
                model_params or {},
                extra_params={**extra_params, "task": video_task},
                width=width,
                height=height,
            )
            for k, v in form_fields.items():
                form.add_field(k, v if isinstance(v, str) else str(v))
        else:
            if model_params is not None:
                for k, v in model_params.items():
                    form.add_field(k, str(v))
            if extra_params:
                form.add_field("extra_params", json.dumps(extra_params, ensure_ascii=False))

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Start the video generation job
            url = f"{self.base_url}/videos"
            data = await url_json(session, url, "post", data=form)
            if (job_id := data.get("id", None)) is None:
                raise RuntimeError("API response missing job 'id' field - expected OpenAI compliant format")
            if (job_status := data.get("status", None)) is None:
                raise RuntimeError("API response missing job 'status' field - expected OpenAI compliant format")

            # Poll for video generation job completion
            deadline = asyncio.get_running_loop().time() + self.max_poll_duration
            url = f"{self.base_url}/videos/{job_id}"
            while job_status not in {"completed", "failed"}:
                await asyncio.sleep(self.poll_interval)

                data = await url_json(session, url)
                if (job_status := data.get("status", None)) is None:
                    raise RuntimeError("API response missing job 'status' field - expected OpenAI compliant format")
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(f"Timed out waiting for video job {job_id} to complete")

            if job_status == "failed":
                raise RuntimeError(f"Video job failed: {data}")

            # Retrieve completed content
            video_bytes = await url_bytes(session, f"{url}/content")

            # Decode video and make a best effort at cleaning up server resources
            try:
                return bytes_to_video(video_bytes)
            finally:
                try:
                    await url_json(session, url, "delete")
                except Exception as exc:
                    logger.warning("Failed to clean up video job %s: %s", job_id, exc)

    async def generate_understanding_chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        modalities: list[str] = ["text", "audio"],
        **extra_body,
    ) -> tuple[str | None, AudioInput | None]:
        # Response may contain two choices: one with text, one with audio
        payload = VLLMOmniClient._prepare_chat_completion_messages(
            model=model,
            prompt=prompt,
            negative_prompt=None,
            image=image,
            audio=audio,
            video=video,
            sampling_params=sampling_params,
            modalities=modalities,
            **extra_body,
        )

        choices = await self._generate_base_chat_completion(model, payload)
        text_response = None
        audio_base64 = None
        for choice in choices:
            try:
                text_response = choice["message"]["content"]
            except (KeyError, TypeError):
                # Either this case (text response) or the audio response case will be hit. Checking None's later.
                pass
            try:
                audio_base64 = choice["message"]["audio"]["data"]
            except (KeyError, TypeError):
                # Either this case (text response) or the audio response case will be hit. Checking None's later.
                pass
        if audio_base64 is None and text_response is None:
            raise RuntimeError(
                "API response missing both '.message.audio' and 'message.content' fields."
                f"The choices object is {choices}"
            )
        if audio_base64 is not None:
            audio = base64_to_audio(audio_base64)
            logger.debug(
                "audio sample rate %d, audio shape %s, duration in second %f",
                audio["sample_rate"],
                audio["waveform"].shape,
                audio["waveform"].shape[2] / audio["sample_rate"],
            )
        else:
            audio = None
        return text_response, audio

    async def generate_speech(
        self,
        *,
        model: str,
        input: str,
        voice: str,
        response_format: AudioFormat,
        speed: float,
        **extra_params,
    ) -> AudioInput:
        await self._check_model_exist(model)

        ref_audio: AudioInput | None = extra_params.pop("ref_audio", None)

        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            **extra_params,
        }

        if ref_audio is not None:
            audio_base64 = audio_to_base64(ref_audio)
            payload["ref_audio"] = audio_base64

        logger.debug("Omni TTS payload: %s", pretty_printer.pformat(payload))

        url = self.base_url + "/audio/speech"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        audio_bytes = await response.read()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    try:
                        audio = bytes_to_audio(audio_bytes)
                    except av.error.InvalidDataError as e:
                        raise ValueError(
                            f"Invalid audio data received from vLLM-Omni: {e}"
                            "Check if you have input unsupported arguments (such as 'voice')"
                        )
                    return audio

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {url}: {e}")

    async def _generate_base_chat_completion(self, model: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        logger.debug("Omni payload: %s", pretty_printer.pformat(payload))
        await self._check_model_exist(model)

        url = self.base_url + "/chat/completions"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status}: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response from vLLM-Omni: {e}")

                    logger.debug(
                        "chat completion response: %s",
                        pretty_printer.pformat(data),
                    )

                    try:
                        return data["choices"]
                    except (KeyError, TypeError):
                        raise RuntimeError("Invalid JSON response from vLLM-Omni: missing 'choices' field")

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {self.base_url}: {e}")

    async def _check_model_exist(self, model: str):
        url = self.base_url + "/models"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(
                    url,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise (ValueError if response.status < 500 else RuntimeError)(
                            f"vLLM-Omni API returned status {response.status} "
                            f"when getting hosted model list: {error_text}"
                        )

                    try:
                        data = await response.json()
                    except aiohttp.ContentTypeError as e:
                        raise RuntimeError(f"Invalid JSON response when getting hosted model list from vLLM-Omni: {e}")

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network error connecting to vLLM-Omni at {self.base_url}: {e}")
        try:
            model_list = data["data"]
            model_found = next((True for m in model_list if m["id"] == model), False)
        except (KeyError, TypeError):
            raise RuntimeError(f"Invalid JSON response of the hosted model list: {data}")

        if not model_found:
            raise ValueError(f"Model {model} not served at {self.base_url}.")

    @staticmethod
    def _prepare_chat_completion_messages(
        *,
        model: str,
        prompt: str,
        negative_prompt: str | None,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        modalities: list[str] | None = None,  # diffusion don't have this field
        **extra_body,
    ):
        message_content: list[dict] = [{"type": "text", "text": prompt}]
        if image is not None:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_tensor_to_base64(image)},
                }
            )
        if audio is not None:
            message_content.append({"type": "audio_url", "audio_url": {"url": audio_to_base64(audio)}})
        if video is not None:
            message_content.append({"type": "video_url", "video_url": {"url": video_to_base64(video)}})
        messages = [{"role": "user", "content": message_content}]

        payload: dict[str, Any] = {"messages": messages, "model": model}
        if modalities:
            payload["modalities"] = modalities

        combined_extra_body: dict[str, Any] = {}
        if sampling_params is not None:
            spec, _ = lookup_model_spec(model)
            is_single_sampling_param = isinstance(sampling_params, dict) or len(sampling_params) == 1

            if (spec is None and is_single_sampling_param) or (spec is not None and spec["stages"] == ["diffusion"]):
                # Diffusion format: extra_body directly contains sampling params.
                # Validation has already taken care of matching sampling params' types and length. Safe to take [0].
                # * Use this mode if the model is a simple one-stage diffusion model.
                # * Fallback to this mode if model is not registered and a single sampling param is provided.
                sampling_params = sampling_params if isinstance(sampling_params, dict) else sampling_params[0]
                combined_extra_body: dict[str, Any] = sampling_params.copy()
                if "n" in combined_extra_body:
                    combined_extra_body["num_outputs_per_prompt"] = combined_extra_body.pop("n")
            else:
                # AR format: the payload has a sampling_params_list field, containing a list.
                sampling_params_list = sampling_params if isinstance(sampling_params, list) else [sampling_params]
                payload["sampling_params_list"] = sampling_params_list

        if negative_prompt:
            combined_extra_body["negative_prompt"] = negative_prompt

        if extra_body:
            combined_extra_body.update(extra_body)

        # Add extra_body only if it has any content.
        if combined_extra_body:
            payload["extra_body"] = combined_extra_body

        # Place to inject any model-specific payload adjustment
        spec, _ = lookup_model_spec(model)
        if spec:
            preprocessor = spec.get("payload_preprocessor", None)
            if preprocessor is not None:
                payload = preprocessor(payload)

        return payload
