import asyncio

from fastapi import Request
from fastapi.responses import Response
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import AudioResponse, CreateAudio, OpenAICreateSpeechRequest
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


class OmniOpenAIServingSpeech(OpenAIServing, AudioMixin):
    async def create_speech(
        self,
        request: OpenAICreateSpeechRequest,
        raw_request: Request | None = None,
    ):
        """
        Create Speech API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createSpeech
        for the API specification. This API mimics the OpenAI
        Create Speech API.

        NOTE: This implementation does not currently support streaming audio
        generation. The full audio is generated before the response is sent.
        """

        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = f"speech-{random_uuid()}"

        try:
            prompt = {"prompt": request.input}
            sampling_params_list = self.engine_client.default_sampling_params_list

            generator = self.engine_client.generate(
                prompt=prompt, request_id=request_id, sampling_params_list=sampling_params_list
            )

            final_output: OmniRequestOutput | None = None
            async for res in generator:
                final_output = res

            if final_output is None:
                return self.create_error_response("No output generated from the model.")

            if final_output.final_output_type != "audio":
                return self.create_error_response(f"Unexpected final output type: {final_output.final_output_type}")

            audio_tensor = final_output.request_output.multimodal_output["audio"].float().detach().cpu().numpy()
            audio_obj = CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=24000,
                response_format=request.response_format,
                speed=request.speed,
                stream_format=request.stream_format,
                base64_encode=False,
            )

            audio_response: AudioResponse = self.create_audio(audio_obj)
            return Response(content=audio_response.audio_data, media_type=audio_response.media_type)

        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))
        except Exception as e:
            return self.create_error_response(f"{e} {e.__cause__}")
