# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
from typing import Literal

import torch
from comfy_api.input import AudioInput, VideoInput

from .utils.api_client import VLLMOmniClient
from .utils.latent_mask import _align_frame_count, _video_latent_t
from .utils.logger import get_logger
from .utils.models import lookup_model_spec
from .utils.types import (
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_CONTROL_TYPES,
    AudioFormat,
    AutoregressionSamplingParams,
    DiffusionSamplingParams,
    FastH3Deployment,
    LatentMaskEditing,
    MiniMaxH3Control,
    MiniMaxH3ControlType,
    MiniMaxH3ModelSpecificParams,
    QwenTTSModelSpecificParams,
    VideoReferences,
    WanModelSpecificParams,
)
from .utils.validators import (
    add_sampling_parameters_to_stage,
    validate_minimax_h3_control,
    validate_model_and_sampling_params_types,
)

logger = get_logger(__name__)

FASTH3_INFERENCE_STEPS = 4
FASTH3_FPS = 24
# A FastH3 deployment may expose H3 under any --served-model-name, so the payload
# spec has to be named outright rather than recovered from that alias.
FASTH3_SPEC_MODEL = "MiniMax-H3"


def _resolve_fast_h3_deployment(deployment: dict) -> tuple[str, str]:
    if not isinstance(deployment, dict):
        raise ValueError("FastH3 deployment must be provided by a FastH3 Deployment node.")

    url = str(deployment.get("url") or "").strip().rstrip("/")
    model = str(deployment.get("model") or "").strip()
    if not url or not model:
        raise ValueError("FastH3 deployment requires both URL and model.")
    return url, model


class _VLLMOmniGenerateBase:
    """Base class for vLLM-Omni generation nodes with shared functionality."""

    CATEGORY = "vLLM-Omni"

    @classmethod
    def VALIDATE_INPUTS(cls, url, model) -> str | Literal[True]:
        """
        Can only validate this model's own input. Cannot check inputs from other nodes.
        See: https://docs.comfy.org/custom-nodes/backend/server_overview#validate_inputs
        """
        if not url:
            return "URL must not be empty"
        if not model:
            return "Model must not be empty"
        return True


class VLLMOmniGenerateImage(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                # "video": ("VIDEO",),
                # "audio": ("AUDIO",),
                "sampling_params": ("SAMPLING_PARAMS",),
                "lora": ("REMOTE_LORA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    async def generate(
        self,
        url: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str | None = None,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        audio: AudioInput | None = None,  # Hidden & unused
        video: VideoInput | None = None,  # Hidden & unused
        sampling_params: dict | list[dict] | None = None,
        lora: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            logger.info("Uncaught kwargs: %s", kwargs)
        logger.debug("Got sampling params: %s", sampling_params)
        validate_model_and_sampling_params_types(model, sampling_params)
        if image is None and mask is not None:
            raise ValueError("Mask input provided without an image input.")

        client = VLLMOmniClient(url)

        spec, pattern = lookup_model_spec(model)
        is_bagel = pattern is not None and "bagel" in pattern.lower()

        # Prefer DALL-E compatible API for simple (one-stage) diffusion models
        if (spec is None or spec["stages"] == ["diffusion"]) and not is_bagel:
            # The number of sampling parameter groups should have been validated.
            # Now, simply convert single-item list to dict.
            if isinstance(sampling_params, list):
                sampling_params = sampling_params[0]
            if audio is None and image is None and video is None:
                # No multimodal input --- use DALL-E image generation
                logger.info("Using DALL-E image generation endpoint")
                output = await client.generate_image(
                    model=model,
                    prompt=prompt,
                    width=width,
                    height=height,
                    negative_prompt=negative_prompt,
                    sampling_params=sampling_params,
                    lora=lora,
                )
                return (output,)
            elif image is not None and audio is None and video is None:
                # Image and text input --- use DALL-E image edit
                logger.info("Using DALL-E image edit endpoint")
                output = await client.edit_image(
                    model=model,
                    prompt=prompt,
                    image=image,
                    width=width,
                    height=height,
                    negative_prompt=negative_prompt,
                    mask=mask,
                    sampling_params=sampling_params,
                    lora=lora,
                )
                return (output,)

        logger.info("Using chat completion endpoint")
        sampling_params = add_sampling_parameters_to_stage(
            model, sampling_params, "diffusion", width=width, height=height
        )
        logger.debug("Edited sampling params: %s", sampling_params)

        output = await client.generate_image_chat_completion(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            audio=audio,
            video=video,
            sampling_params=sampling_params,
            lora=lora,
        )

        return (output,)


class VLLMOmniGenerateVideo(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"}),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 832, "min": 1}),
                "height": ("INT", {"default": 480, "min": 1}),
                "fps": ("INT", {"default": 16, "min": 1}),
                "duration": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.1,
                        "step": 0.1,
                        "round": 0.001,
                        "tooltip": (
                            "Clip length in seconds, converted to frames with the fps above. "
                            "Models that only accept certain frame counts (e.g. MiniMax-H3) round to "
                            "their own lattice, so the served clip can be slightly longer than requested."
                        ),
                    },
                ),
            },
            "optional": {
                "frame": ("IMAGE",),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "references": ("VIDEO_REFERENCES",),
                "control": ("MINIMAX_H3_CONTROL",),
                "sampling_params": ("SAMPLING_PARAMS",),
                "lora": ("REMOTE_LORA",),
                "model_params": ("VIDEO_PARAMS",),
                "fast_h3": ("FASTH3_DEPLOYMENT",),
                "latent_edit": ("LATENT_MASK_EDITING",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        url,
        model,
        frame=None,
        first_frame=None,
        last_frame=None,
        references=None,
        control=None,
        fast_h3=None,
        **_kwargs,
    ) -> str | Literal[True]:
        base = super().VALIDATE_INPUTS(url, model)
        if base is not True:
            return base
        if frame is not None and references is not None:
            return "Provide only one of frame or references, not both."
        if frame is not None and (first_frame is not None or last_frame is not None):
            return "Provide either frame or first_frame/last_frame, not both."
        if references is not None and (first_frame is not None or last_frame is not None):
            return "Provide either first_frame/last_frame or references, not both."
        if fast_h3 is not None and any(value is not None for value in (frame, first_frame, last_frame, references)):
            return (
                "FastH3 Preview supports T2VA only; disconnect frame, first_frame, last_frame, and references inputs."
            )
        if control is not None and any(value is not None for value in (frame, first_frame, last_frame, references)):
            return "MiniMax-H3 control cannot be combined with frame, keyframes, or references."
        if control is not None:
            try:
                validate_minimax_h3_control(control)
            except ValueError as exc:
                return str(exc)
        return True

    async def generate(
        self,
        url: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        negative_prompt: str | None = None,
        frame: torch.Tensor | None = None,
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
        references: dict | None = None,
        control: MiniMaxH3Control | None = None,
        sampling_params: dict | list[dict] | None = None,
        model_params: dict | None = None,
        lora: dict | None = None,
        fast_h3: dict | None = None,
        latent_edit: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            logger.info("Uncaught kwargs: %s", kwargs)
        logger.debug("Got sampling params: %s", sampling_params)
        logger.debug("Got model params: %s", model_params)

        # Which spec builds the payload. Only a FastH3 deployment separates this
        # from the served name; every other path keeps them equal.
        spec_model = model

        if fast_h3 is not None:
            if any(value is not None for value in (frame, first_frame, last_frame, references)):
                raise ValueError(
                    "FastH3 Preview supports T2VA only; disconnect frame, first_frame, "
                    "last_frame, and references inputs."
                )
            if lora is not None:
                raise ValueError(
                    "FastH3 is already fused into the selected server; disconnect the request-level LoRA input."
                )

            url, model = _resolve_fast_h3_deployment(fast_h3)
            logger.info("Using FastH3 deployment at %s", url)
            fps = FASTH3_FPS
            # The served name is whatever the operator passed to --served-model-name.
            # Left alone, lookup_model_spec would miss H3 for an alias such as
            # "fasth3", drop the params builder, and send a t2va request carrying no
            # aspect_ratio -- which the server refuses.
            spec_model = FASTH3_SPEC_MODEL

            if sampling_params is None:
                sampling_params = DiffusionSamplingParams()
            elif isinstance(sampling_params, list):
                if len(sampling_params) != 1:
                    raise ValueError("FastH3 expects a single diffusion sampling params group.")
                sampling_params = sampling_params[0].__class__(sampling_params[0])
            else:
                sampling_params = sampling_params.__class__(sampling_params)
            sampling_params["num_inference_steps"] = FASTH3_INFERENCE_STEPS

            # FastH3 owns both modality shifts. Sending the ordinary H3 values
            # from a connected H3 Params node would turn a deployment choice
            # into a request-level override, which the server intentionally
            # rejects when it differs from the fused adapter contract.
            if model_params is not None:
                model_params = model_params.__class__(model_params)
                model_params.pop("flow_shift", None)
                model_params.pop("audio_flow_shift", None)

        # Frames stay the wire unit. Convert after the FastH3 branch above, so the
        # duration is measured against the fps the server will actually apply.
        num_frames = max(1, round(duration * fps))

        validate_model_and_sampling_params_types(spec_model, sampling_params)

        # Currently, all video generation models are single-stage diffusion models
        if isinstance(sampling_params, list):
            if len(sampling_params) != 1:
                raise ValueError(
                    "Video generation expects a single sampling params group. "
                    "Please provide one Diffusion sampling node."
                )
            sampling_params = sampling_params[0]

        if sampling_params is not None:
            sampling_params.pop("type", None)  # internal fields
        # model_params["type"] is also an internal field, but is used in api_client
        # if model_params is not None:
        #     model_params.pop("type", None)  # internal fields

        client = VLLMOmniClient(url)
        output = await client.generate_video(
            model=model,
            spec_model=spec_model,
            prompt=prompt,
            frame=frame,  # frame present => fl2va / Wan I2V
            first_frame=first_frame,
            last_frame=last_frame,
            references=references,
            control=control,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            negative_prompt=negative_prompt,
            sampling_params=sampling_params,
            lora=lora,
            model_params=model_params,
            latent_edit=latent_edit,
        )
        return (output,)


class VLLMOmniUnderstanding(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "Qwen/Qwen2.5-Omni-7B"}),
                "prompt": ("STRING", {"multiline": True}),
                "output_text": ("BOOLEAN", {"default": True}),
                "output_audio": ("BOOLEAN", {"default": True}),
                "use_audio_in_video": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "sampling_params": ("SAMPLING_PARAMS",),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("text_response", "audio_response")
    FUNCTION = "generate"

    @classmethod
    def VALIDATE_INPUTS(cls, url, model, output_text, output_audio) -> str | Literal[True]:  # type: ignore[reportIncompatibleMethodOverride]
        super_validation = super().VALIDATE_INPUTS(url, model)
        if isinstance(super_validation, str):
            return super_validation
        if not output_text and not output_audio:
            return "At least one of output_text or output_audio must be True."
        return True

    async def generate(
        self,
        url: str,
        model: str,
        prompt: str,
        image: torch.Tensor | None = None,
        audio: AudioInput | None = None,
        video: VideoInput | None = None,
        sampling_params: dict | list[dict] | None = None,
        output_text: bool = True,
        output_audio: bool = True,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> tuple[str, AudioInput]:
        if kwargs:
            logger.info("Uncaught kwargs: %s", kwargs)
        logger.debug("Got sampling params: %s", sampling_params)
        validate_model_and_sampling_params_types(model, sampling_params)

        client = VLLMOmniClient(url)
        spec, pattern = lookup_model_spec(model)
        is_bagel = pattern is not None and "bagel" in pattern.lower()

        if is_bagel:
            # A lot of special handlings here...
            if output_audio:
                raise ValueError("BAGEL models do not support audio output.")
            if audio is not None or video is not None:
                raise ValueError("BAGEL models do not support audio or video input.")
            (
                text_response,
                _,
            ) = await client.generate_understanding_chat_completion(
                model=model,
                prompt=prompt,
                image=image,
                audio=None,
                video=None,
                sampling_params=sampling_params,
                modalities=["text"],
            )
        else:
            modalities = []
            if output_text:
                modalities.append("text")
            if output_audio:
                modalities.append("audio")

            if use_audio_in_video and video is not None:
                use_audio_in_video = True
            else:
                use_audio_in_video = False

            (
                text_response,
                audio,
            ) = await client.generate_understanding_chat_completion(
                model=model,
                prompt=prompt,
                image=image,
                audio=audio,
                video=video,
                sampling_params=sampling_params,
                modalities=modalities,
                # == extra kwargs ==
                mm_processor_kwargs={"use_audio_in_video": use_audio_in_video},
            )

        if text_response is None:
            text_response = ""
        if audio is None:
            channels = 1
            duration = 1
            sample_rate = 44100
            num_samples = int(round(duration * sample_rate))
            waveform = torch.zeros((1, channels, num_samples), dtype=torch.float32)
            audio = {"waveform": waveform, "sample_rate": sample_rate}

        return (text_response, audio)


class VLLMOmniTTS(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": (
                    "STRING",
                    {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"},
                ),
                "input": ("STRING", {"multiline": True}),
                "voice": ("STRING", {"default": "Vivian"}),
                "response_format": (["mp3", "opus", "aac", "flac", "wav", "pcm"],),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01},
                ),
            },
            "optional": {
                "model_specific_params": ("TTS_PARAMS",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    async def generate(
        self,
        url: str,
        model: str,
        input: str,
        voice: str,
        response_format: AudioFormat,
        speed: float,
        model_specific_params: dict | None,
        **kwargs,
    ) -> tuple[AudioInput]:
        logger.info("Got extra kwargs in TTS: %s", kwargs)

        is_qwen_tts = "qwen3-tts" in model.lower()
        if not is_qwen_tts and isinstance(model_specific_params, QwenTTSModelSpecificParams):
            raise ValueError(
                "You have provided Qwen-specific TTS params."
                "However, the model appears to not be a Qwen TTS model (no 'Qwen3-TTS' in model name)."
            )

        combined_params = {**kwargs, **(model_specific_params or {})}

        client = VLLMOmniClient(url)

        audio = await client.generate_speech(
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            **combined_params,
        )
        return (audio,)


class VLLMOmniGenerateMusic(_VLLMOmniGenerateBase):
    """Generate a song from lyrics and a musical description with MiniMax Music 3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "MiniMaxAI/MiniMax-Music3"}),
                "instructions": (
                    "STRING",
                    {
                        "multiline": True,
                        "display_name": "caption",
                        "tooltip": "Music caption: describe genre, instruments, tempo and mood.",
                    },
                ),
                "lyrics": ("STRING", {"multiline": True}),
                "max_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 300.0,
                        "min": 1,
                        "max": 360,
                        "step": 0.01,
                        "tooltip": "Upper limit; rounded down to whole 25 Hz audio frames. May end earlier.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**53 - 1, "control_after_generate": True}),
                "response_format": (["wav", "mp3", "flac", "opus"],),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    async def generate(
        self,
        url: str,
        model: str,
        lyrics: str,
        instructions: str,
        response_format: AudioFormat,
        max_duration_seconds: float,
        seed: int = 0,
    ) -> tuple[AudioInput]:
        audio = await VLLMOmniClient(url.rstrip("/")).generate_speech(
            model=model,
            input=lyrics,
            instructions=instructions,
            voice="default",
            speed=1.0,
            response_format=response_format,
            max_new_tokens=int(max_duration_seconds * 25),
            seed=seed,
        )
        return (audio,)


class VLLMOmniVoiceClone(_VLLMOmniGenerateBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "http://localhost:8000/v1"}),
                "model": ("STRING", {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}),
                "input": ("STRING", {"multiline": True}),
                "voice": ("STRING", {"default": "Vivian"}),
                "response_format": (["mp3", "opus", "aac", "flac", "wav", "pcm"],),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01},
                ),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True}),
                "x_vector_only_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model_specific_params": ("TTS_PARAMS",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    async def generate(
        self,
        url: str,
        model: str,
        input: str,
        voice: str,
        response_format: AudioFormat,
        speed: float,
        ref_audio: AudioInput,
        ref_text: str,
        x_vector_only_mode: bool,
        model_specific_params: dict | None,
        **kwargs,
    ):
        is_qwen_tts = "qwen3-tts" in model.lower()
        if not is_qwen_tts and isinstance(model_specific_params, QwenTTSModelSpecificParams):
            raise ValueError(
                "You have provided Qwen-specific TTS params."
                "However, the model appears to not be a Qwen TTS model (no 'Qwen3-TTS' in model name)."
            )

        combined_params = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode,
            **kwargs,
            **(model_specific_params or {}),
        }

        client = VLLMOmniClient(url)

        audio = await client.generate_speech(
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            **combined_params,
        )
        return (audio,)


class VLLMOmniARSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                # === Put seed at last. ===
                # Whenever a field named "seed" is present, ComfyUI adds another field called "control after generate"
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "step": 1,
                        "tooltip": "-1 means to not provide a seed.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAMPLING_PARAMS",)
    RETURN_NAMES = ("AR sampling params",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def get_params(self, seed, **kwargs):
        params = AutoregressionSamplingParams(kwargs)
        if seed >= 0:
            params["seed"] = seed
        return (params,)


class VLLMOmniDiffusionSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of images to generate",
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Number of denoising steps (higher = better quality, slower).",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 7.5,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale (higher = more prompt adherence).",
                    },
                ),
                "true_cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "True CFG scale for advanced control (model-specific).",
                    },
                ),
                "vae_use_slicing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable VAE slicing for reduced memory usage (slight quality trade-off)",
                    },
                ),
                "vae_use_tiling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable VAE tiling for reduced memory usage (slight quality trade-off)",
                    },
                ),
                # === Put seed at last. ===
                # Whenever a field named "seed" is present, ComfyUI adds another field called "control after generate"
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "step": 1,
                        "tooltip": "-1 means to not provide a seed.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAMPLING_PARAMS",)
    RETURN_NAMES = ("diffusion sampling params",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def get_params(self, seed, **kwargs):
        params = DiffusionSamplingParams(kwargs)
        if seed >= 0:
            params["seed"] = seed
        return (params,)


class VLLMOmniSamplingParamsList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param1": ("SAMPLING_PARAMS",),
            },
            "optional": {
                "param2": ("SAMPLING_PARAMS",),
                "param3": ("SAMPLING_PARAMS",),
            },
        }

    RETURN_TYPES = ("SAMPLING_PARAMS",)
    RETURN_NAMES = ("param list",)
    FUNCTION = "aggregate"
    CATEGORY = "vLLM-Omni/Sampling Params"

    def aggregate(self, param1: dict, param2: dict | None = None, param3: dict | None = None):
        for i, p in enumerate((param1, param2, param3)):
            if isinstance(p, list):
                raise ValueError(
                    f"Input {i} is a Multi-Stage Sampling Params List. "
                    f"Expected a single sampling parameters node (either AR or Diffusion)."
                )

        params = [param1]
        if param2 is not None:
            params.append(param2)
        if param3 is not None:
            params.append(param3)
        return (params,)


class VLLMOmniRemoteLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_path": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": ""}),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "int_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "step": 1,
                        "tooltip": "0 means it is not set and the server can derive it.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("REMOTE_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "get_lora"
    CATEGORY = "vLLM-Omni"

    @classmethod
    def VALIDATE_INPUTS(cls, local_path, name) -> str | Literal[True]:
        if not local_path.strip() or not name.strip():
            return "Both local_path and name must be provided."
        return True

    def get_lora(self, local_path: str, name: str, scale: float, int_id: int):
        local_path = local_path.strip()
        name = name.strip()
        lora = {
            "local_path": local_path or None,
            "name": name or None,
            "scale": float(scale),
            "int_id": int(int_id) if int_id > 0 else None,
        }
        return (lora,)


class VLLMOmniFastH3Deployment:
    """Select a vLLM-Omni service that fused FastH3 at startup."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "http://localhost:8000/v1",
                        "tooltip": "URL of a vLLM-Omni server started with a FastH3 --lora-path.",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "MiniMaxAI/MiniMax-H3",
                        "tooltip": "Model name exposed by the FastH3 deployment.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("FASTH3_DEPLOYMENT",)
    RETURN_NAMES = ("deployment",)
    FUNCTION = "get_deployment"
    CATEGORY = "vLLM-Omni"
    DESCRIPTION = (
        "Selects a server with FastH3 fused at startup. The connected Generate Video node uses T2VA, "
        "four inference steps, and 24 FPS without sending a request-level LoRA."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, url, model) -> str | Literal[True]:
        try:
            _resolve_fast_h3_deployment({"url": url, "model": model})
        except ValueError as exc:
            return str(exc)
        return True

    def get_deployment(self, url: str, model: str):
        url, model = _resolve_fast_h3_deployment({"url": url, "model": model})
        return (FastH3Deployment({"url": url, "model": model}),)


class VLLMOmniQwenTTSParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_type": (
                    ["CustomVoice", "VoiceDesign", "Base"],
                    {"default": "CustomVoice"},
                ),
                "language": (
                    ["Auto", "Chinese", "English", "Japanese", "Korean"],
                    {"default": "Auto"},
                ),
                "instructions": ("STRING", {"multiline": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1}),
            }
        }

    RETURN_TYPES = ("TTS_PARAMS",)
    RETURN_NAMES = ("Qwen TTS params",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/TTS Params"

    def get_params(self, **kwargs):
        return (QwenTTSModelSpecificParams(kwargs),)


class VLLMOmniWanParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guidance_scale_2": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "boundary_ratio": (
                    "FLOAT",
                    {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "flow_shift": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("VIDEO_PARAMS",)
    RETURN_NAMES = ("Wan video params",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Video Params"

    def get_params(self, **kwargs):
        return (WanModelSpecificParams(kwargs),)


class VLLMOmniMiniMaxH3Params:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_flow_shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "flow_shift": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("VIDEO_PARAMS",)
    RETURN_NAMES = ("MiniMax-H3 video params",)
    FUNCTION = "get_params"
    CATEGORY = "vLLM-Omni/Video Params"

    def get_params(self, **kwargs):
        params = MiniMaxH3ModelSpecificParams(kwargs)
        params["type"] = "minimax_h3"
        return (params,)


class VLLMOmniMiniMaxH3Control:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_type": (list(MINIMAX_H3_CONTROL_TYPES),),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Controls the selected mode's conditioning strength.",
                    },
                ),
            },
            "optional": {
                "control_video": (
                    "VIDEO",
                    {"tooltip": "Required for structure modes; optional for inpaint."},
                ),
                "source_video": (
                    "VIDEO",
                    {"tooltip": "Optional source video; requires mask or mask_video."},
                ),
                "mask": (
                    "MASK",
                    {"tooltip": "Static mask; white/1 marks areas to regenerate. Do not combine with mask_video."},
                ),
                "mask_video": (
                    "VIDEO",
                    {"tooltip": "Temporal mask video; do not combine with mask."},
                ),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_CONTROL",)
    RETURN_NAMES = ("control",)
    FUNCTION = "get_control"
    CATEGORY = "vLLM-Omni/Video Params"

    def get_control(
        self,
        control_type: MiniMaxH3ControlType,
        strength: float,
        control_video: VideoInput | None = None,
        source_video: VideoInput | None = None,
        mask: torch.Tensor | None = None,
        mask_video: VideoInput | None = None,
    ):
        control = MiniMaxH3Control(
            control_type=control_type,
            control_context_scale=strength,
        )
        if control_video is not None:
            control["control_video"] = control_video
        if source_video is not None:
            control["source_video"] = source_video
        if mask is not None:
            control["mask"] = mask
        if mask_video is not None:
            control["mask_video"] = mask_video
        validate_minimax_h3_control(control)
        return (control,)


class VLLMOmniVideoReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "video_1": ("VIDEO",),
                "video_2": ("VIDEO",),
                # Append ports to preserve connections in saved workflows.
                **{f"image_{i}": ("IMAGE",) for i in range(3, MAX_REFERENCE_IMAGES + 1)},
                **{f"audio_{i}": ("AUDIO",) for i in range(3, MAX_REFERENCE_AUDIOS + 1)},
                **{f"video_{i}": ("VIDEO",) for i in range(3, MAX_REFERENCE_VIDEOS + 1)},
            },
        }

    RETURN_TYPES = ("VIDEO_REFERENCES",)
    RETURN_NAMES = ("references",)
    FUNCTION = "get_references"
    CATEGORY = "vLLM-Omni"

    def get_references(
        self,
        image_1: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
        audio_1: AudioInput | None = None,
        audio_2: AudioInput | None = None,
        video_1: VideoInput | None = None,
        video_2: VideoInput | None = None,
        image_3: torch.Tensor | None = None,
        image_4: torch.Tensor | None = None,
        image_5: torch.Tensor | None = None,
        image_6: torch.Tensor | None = None,
        image_7: torch.Tensor | None = None,
        image_8: torch.Tensor | None = None,
        image_9: torch.Tensor | None = None,
        audio_3: AudioInput | None = None,
        video_3: VideoInput | None = None,
        **kwargs,
    ):
        if kwargs:
            logger.info("Uncaught kwargs: %s", kwargs)
        refs = VideoReferences()
        for kind, values in (
            ("image", (image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9)),
            ("video", (video_1, video_2, video_3)),
            ("audio", (audio_1, audio_2, audio_3)),
        ):
            for index, value in enumerate(values, start=1):
                if value is not None:
                    refs[f"{kind}_{index}"] = value
        return (refs,)


class VLLMOmniLatentMaskEditing:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "source_video": ("VIDEO",),
                "source_audio": ("AUDIO",),
                "video_mask": ("MASK",),
                "audio_mask": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT_MASK_EDITING",)
    RETURN_NAMES = ("latent_edit",)
    FUNCTION = "get_latent_edit"
    CATEGORY = "vLLM-Omni"

    def get_latent_edit(
        self,
        source_video: VideoInput | None = None,
        source_audio: AudioInput | None = None,
        video_mask: torch.Tensor | None = None,
        audio_mask: float = -1.0,
        **kwargs,
    ):
        if kwargs:
            logger.info("Uncaught kwargs: %s", kwargs)
        edit = LatentMaskEditing()
        if source_video is not None:
            edit["source_video"] = source_video
        if source_audio is not None:
            edit["source_audio"] = source_audio
        if video_mask is not None:
            edit["video_mask"] = video_mask
        if audio_mask >= 0.0:
            edit["audio_mask"] = audio_mask
        return (edit,)


class VLLMOmniMiniMaxH3TemporalMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.01}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 15.0}),
                "mode": (["continuation", "extension"],),
                "preserve_fraction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("MASK", "FLOAT", "IMAGE", "MASK")
    RETURN_NAMES = ("mask", "preview_fps", "preview_images", "preview_mask")
    FUNCTION = "build"
    CATEGORY = "vLLM-Omni"

    def build(self, images, source_fps, duration, mode, preserve_fraction=0.5):
        if images.shape[0] <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Source must contain frames and have a positive finite FPS.")
        if not math.isfinite(duration) or duration <= 0 or not 0 <= preserve_fraction <= 1:
            raise ValueError("Invalid duration or preserve_fraction.")
        frames = _align_frame_count(max(1, round(duration * 24)))
        source_seconds = images.shape[0] / source_fps
        if mode == "extension":
            if frames / 24 <= source_seconds:
                raise ValueError("Extension output must be longer than the source; increase duration.")
            boundary = source_seconds
        elif mode == "continuation":
            boundary = min(source_seconds, frames / 24) * preserve_fraction
        else:
            raise ValueError(f"Unknown temporal mask mode: {mode}")
        available = min(frames, math.floor(boundary * 24 + 1e-8))
        prefix = 0 if available < 5 else 5 + 17 * ((available - 5) // 17)
        preserved = _video_latent_t(prefix) if prefix else 0
        total = _video_latent_t(frames)
        mask = torch.ones(total, 1, 1)
        mask[:preserved] = 0
        indices = torch.arange(frames, device=images.device)
        source_indices = (indices * (source_fps / 24)).floor().long().clamp(max=images.shape[0] - 1)
        preview_images = images.index_select(0, source_indices)
        preview_mask = mask.index_select(0, torch.arange(frames) * total // frames)
        return mask, 24.0, preview_images, preview_mask
