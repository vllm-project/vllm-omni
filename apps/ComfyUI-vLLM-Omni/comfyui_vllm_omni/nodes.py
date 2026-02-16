from typing import Literal, cast

import torch
from comfy_api.input import AudioInput, VideoInput

from .utils.api_client import VLLMOmniClient
from .utils.logger import get_logger
from .utils.models import lookup_model_spec
from .utils.types import AudioFormat
from .utils.validators import (
    add_sampling_parameters_to_stage,
    validate_model_and_sampling_params_types,
)

logger = get_logger(__name__)


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
        **kwargs,
    ):
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
            sampling_params = cast(dict | None, sampling_params)
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
        )

        return (output,)


class VLLMOmniComprehension(_VLLMOmniGenerateBase):
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
            ) = await client.generate_comprehension_chat_completion(
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
            ) = await client.generate_comprehension_chat_completion(
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
        extra_params_type = None if model_specific_params is None else model_specific_params["type"]
        if not is_qwen_tts and extra_params_type == "qwen-tts":
            raise ValueError(
                "You have provided Qwen-specific TTS params."
                "However, the model appears to not be a Qwen TTS model (no 'Qwen3-TTS' in model name)."
            )

        combined_params = {**kwargs, **(model_specific_params or {})}
        combined_params.pop("type", None)  # Internal fields in model_specific_params

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
        extra_params_type = None if model_specific_params is None else model_specific_params["type"]
        if not is_qwen_tts and extra_params_type == "qwen-tts":
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
        combined_params.pop("type", None)  # Internal fields in model_specific_params

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
        params = {
            "type": "autoregression",  # for internal use, removed before sending the request
            **kwargs,
        }
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
        params = {
            "type": "diffusion",  # for internal use, removed before sending the request
            **kwargs,
        }
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
        return ({"type": "qwen-tts", **kwargs},)
