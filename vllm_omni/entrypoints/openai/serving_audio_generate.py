import asyncio

import torch
from fastapi import Request
from fastapi.responses import Response
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import (
    AudioResponse,
    CreateAudio,
    OpenAICreateAudioGenerateRequest,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class OmniOpenAIServingAudioGenerate(OpenAIServing, AudioMixin):
    """Serving class for audio generation via diffusion models (e.g. Stable Audio)."""

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.pop("model_name", None)
        super().__init__(*args, **kwargs)
        self.diffusion_mode = False

    @classmethod
    def for_diffusion(cls, *args, **kwargs) -> "OmniOpenAIServingAudioGenerate":
        """Create an instance configured to run in diffusion mode."""
        instance = cls(*args, **kwargs)
        instance.diffusion_mode = True
        return instance

    def _is_stable_audio_model(self) -> bool:
        return self.engine_client.model_type == "StableAudioPipeline"

    async def create_audio_generate(
        self,
        request: OpenAICreateAudioGenerateRequest,
        raw_request: Request | None = None,
    ):
        """
        Generate audio using diffusion-based models (e.g. Stable Audio).

        This endpoint is designed for audio generation models as
        opposed to TTS models that specifically generate speech.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"audiogen-{random_uuid()}"

        try:
            default_sr = 44100  # Default sample rate for Stable Audio

            # Build prompt for diffusion audio generation
            prompt = {
                "prompt": request.input,
            }
            if request.negative_prompt:
                prompt["negative_prompt"] = request.negative_prompt

            # Build sampling params for diffusion
            sampling_params_list = [OmniDiffusionSamplingParams(num_outputs_per_prompt=1)]

            # Create generator if seed provided
            if request.seed is not None:
                from vllm_omni.platforms import current_omni_platform

                rng = torch.Generator(device=current_omni_platform.device_type).manual_seed(request.seed)
                sampling_params_list[0].generator = rng

            if request.guidance_scale is not None:
                sampling_params_list[0].guidance_scale = request.guidance_scale

            if request.num_inference_steps is not None:
                sampling_params_list[0].num_inference_steps = request.num_inference_steps

            # Set up audio duration parameters
            if request.audio_length is not None:
                audio_length = request.audio_length
                audio_start = request.audio_start if request.audio_start is not None else 0.0
                audio_end_in_s = audio_start + audio_length
                sampling_params_list[0].extra_args = {
                    "audio_start_in_s": audio_start,
                    "audio_end_in_s": audio_end_in_s,
                }

            logger.info(
                "Audio generation request %s: prompt=%r",
                request_id,
                request.input[:50] + "..." if len(request.input) > 50 else request.input,
            )

            generator = self.engine_client.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["audio"],
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            # Extract audio from output
            audio_output = None
            if hasattr(final_output, "multimodal_output") and final_output.multimodal_output:
                audio_output = final_output.multimodal_output
            if not audio_output and hasattr(final_output, "request_output"):
                if final_output.request_output and hasattr(final_output.request_output, "multimodal_output"):
                    audio_output = final_output.request_output.multimodal_output

            # Check for audio data using either "audio" or "model_outputs" key
            audio_key = None
            if audio_output:
                if "audio" in audio_output:
                    audio_key = "audio"
                elif "model_outputs" in audio_output:
                    audio_key = "model_outputs"

            if not audio_output or audio_key is None:
                return self.create_error_response("Audio generation model did not produce audio output.")

            audio_tensor = audio_output[audio_key]
            sample_rate = audio_output.get("sr", default_sr)
            if hasattr(sample_rate, "item"):
                sample_rate = sample_rate.item()

            # Convert tensor to numpy
            if hasattr(audio_tensor, "float"):
                audio_tensor = audio_tensor.float().detach().cpu().numpy()

            # Squeeze batch dimension if present, but preserve channel dimension for stereo
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.squeeze()

            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=int(sample_rate),
                response_format=request.response_format or "wav",
                speed=request.speed or 1.0,
                stream_format=request.stream_format,
                base64_encode=False,
            )

            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)
        except Exception as e:
            logger.exception("Audio generation failed: %s", e)
            return self.create_error_response(f"Audio generation failed: {e}")
