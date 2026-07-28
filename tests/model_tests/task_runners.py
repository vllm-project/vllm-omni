"""
Definitions running and validating individual tasks, e.g., text to image,
image to image, and so on. These are called by the core test runner.
"""

import base64
import io
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from PIL import Image

from tests.helpers.runtime import (
    DiffusionResponse,
    OmniResponse,
    OmniServer,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from tests.model_tests.config_types import ModelTasks
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

OMNI_TEXT_PROMPT = "Hello, briefly describe yourself."
OMNI_SPEECH_PROMPT = "Say: hello world"

_OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

OnlineTaskRunner: TypeAlias = Callable[[OmniServer, OpenAIClientHandler], None]
OfflineTaskRunner: TypeAlias = Callable[[Omni], None]


def _format_omni_chat_prompt(user_text: str) -> str:
    return (
        f"<|im_start|>system\n{_OMNI_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


PROMPT = "A black cat sitting on a bed."
IMAGE_DIMS = (512, 512)
HEIGHT, WIDTH = IMAGE_DIMS
INPUT_IMAGE = Image.new("RGB", IMAGE_DIMS)

# Offline sampling params
IMAGE_GEN_SAMPLING_PARAMS = OmniDiffusionSamplingParams(
    num_inference_steps=4,
    height=HEIGHT,
    width=WIDTH,
    seed=42,
)
TEXT_GEN_SAMPLING_PARAMS = OmniDiffusionSamplingParams(
    num_inference_steps=4,
    seed=42,
)
VIDEO_NUM_FRAMES = 9
VIDEO_GEN_SAMPLING_PARAMS = OmniDiffusionSamplingParams(
    num_inference_steps=4,
    height=HEIGHT,
    width=WIDTH,
    num_frames=VIDEO_NUM_FRAMES,
    seed=42,
)

# Online extra_body for diffusion requests
IMAGE_GEN_EXTRA_BODY = {
    "height": HEIGHT,
    "width": WIDTH,
    "num_inference_steps": 4,
    "seed": 42,
}

# Online form_data for video generation requests (multipart /v1/videos API)
VIDEO_GEN_FORM_DATA = {
    "height": HEIGHT,
    "width": WIDTH,
    "num_inference_steps": 4,
    "num_frames": VIDEO_NUM_FRAMES,
    "seed": 42,
}


### Shared validation
def _validate_images(images: list[Image.Image], expected_n: int = 1):
    """Given a set of images, ensure we got the expected count of images,
    and that all provided images match the expected dimensions."""
    assert len(images) == expected_n
    for img in images:
        assert isinstance(img, Image.Image)
        assert img.size == IMAGE_DIMS
    return images


def _validate_image_gen_determinism(images_a: list[Image.Image], images_b: list[Image.Image]):
    """Ensure that two image sets are valid and that the results match."""
    _validate_images(images_a)
    _validate_images(images_b)
    assert np.array_equal(np.array(images_a[0]), np.array(images_b[0]))


def _get_online_omni_response(responses: list[OmniResponse]) -> OmniResponse:
    assert len(responses) == 1
    return responses[0]


def _validate_text(response: OmniResponse) -> None:
    assert response.completion_tokens


def _validate_speech(response: OmniResponse) -> None:
    assert response.audio_bytes


def _validate_offline_speech(outputs: list[OmniRequestOutput]) -> None:
    audio_stage = next((o for o in outputs if getattr(o, "final_output_type", None) == "audio"), None)
    assert audio_stage is not None
    assert audio_stage.request_output is not None
    assert audio_stage.request_output.outputs[0].multimodal_output.get("audio") is not None


### Output extractor utils for offline / online paths respectively
def _get_offline_images(outputs: list[OmniRequestOutput]) -> list[Image.Image]:
    """Extract the images from an Omni .generate() call."""
    assert len(outputs) == 1
    return outputs[0].images


def _get_online_images(responses: list[DiffusionResponse]) -> list[Image.Image]:
    """Extract the images from a server response."""
    assert len(responses) == 1
    images = responses[0].images
    assert images is not None
    return images


def _validate_video(outputs: list[OmniRequestOutput], expected_n: int = 1):
    """Given a set of outputs, ensure we got video frames with the expected shape."""
    assert len(outputs) == expected_n
    for output in outputs:
        # Video models return numpy arrays via output.images
        images = output.images
        assert len(images) > 0
        for frame_data in images:
            assert isinstance(frame_data, np.ndarray)
            # (num_outputs, num_frames, H, W, C) or (num_frames, H, W, C)
            assert frame_data.ndim in (4, 5), f"Expected 4D or 5D video array, got shape {frame_data.shape}"
            assert frame_data.shape[-3] == HEIGHT, f"Expected height {HEIGHT}, got {frame_data.shape[-3]}"
            assert frame_data.shape[-2] == WIDTH, f"Expected width {WIDTH}, got {frame_data.shape[-2]}"
            assert frame_data.shape[-1] == 3, f"Expected 3 channels (RGB), got {frame_data.shape[-1]}"


### Offline helpers
def _run_offline_t2t(omni: Omni):
    return omni.generate({"prompt": _format_omni_chat_prompt(OMNI_TEXT_PROMPT), "modalities": ["text"]})


def _run_offline_t2s(omni: Omni):
    return omni.generate({"prompt": _format_omni_chat_prompt(OMNI_SPEECH_PROMPT), "modalities": ["audio"]})


def _run_offline_t2i(omni: Omni, params: OmniDiffusionSamplingParams = IMAGE_GEN_SAMPLING_PARAMS):
    return omni.generate({"prompt": PROMPT}, params)


def _run_offline_t2v(omni: Omni, params: OmniDiffusionSamplingParams = VIDEO_GEN_SAMPLING_PARAMS):
    return omni.generate({"prompt": PROMPT}, params)


def _run_offline_i2i(omni: Omni):
    return omni.generate(
        {"prompt": PROMPT, "multi_modal_data": {"image": INPUT_IMAGE}},
        IMAGE_GEN_SAMPLING_PARAMS,
    )


### Online helpers
def _build_online_image_data_url() -> str:
    """Get a valid base 64 encoded data URL corresponding to an image."""
    buf = io.BytesIO()
    INPUT_IMAGE.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _run_online_t2i(
    server: OmniServer, client: OpenAIClientHandler, extra_body: dict | None = None
) -> list[DiffusionResponse]:
    """Run a text to image request through the server."""
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": server.model,
        "messages": messages,
        "extra_body": extra_body or IMAGE_GEN_EXTRA_BODY,
    }
    return client.send_diffusion_request(request_config)


def _run_online_i2i(server: OmniServer, client: OpenAIClientHandler) -> list[DiffusionResponse]:
    """Run an image to image request through the server."""
    image_data_url = _build_online_image_data_url()
    messages = dummy_messages_from_mix_data(
        content_text=PROMPT,
        image_data_url=image_data_url,
    )
    request_config = {
        "model": server.model,
        "messages": messages,
        "extra_body": IMAGE_GEN_EXTRA_BODY,
    }
    return client.send_diffusion_request(request_config)


### Offline task runners
def run_and_validate_text_to_text_request(omni: Omni):
    """Run and validate a text to text request."""
    outputs = _run_offline_t2t(omni)
    assert len(outputs) == 1
    assert outputs[0].request_output is not None
    assert outputs[0].request_output.outputs[0].text is not None


def run_and_validate_text_to_speech_request(omni: Omni):
    """Run and validate a text-in, speech-out request."""
    _validate_offline_speech(_run_offline_t2s(omni))


def run_and_validate_text_to_image_request(omni: Omni):
    """Run and validate a text to image request."""
    _validate_images(_get_offline_images(_run_offline_t2i(omni)))


def run_and_validate_image_to_image_request(omni: Omni):
    """Run and validate an image to image request."""
    _validate_images(_get_offline_images(_run_offline_i2i(omni)))


def run_and_validate_text_to_video_request(omni: Omni):
    """Run and validate a text to video request."""
    _validate_video(_run_offline_t2v(omni))


def run_and_validate_text_to_image_determinism(omni: Omni):
    """Checks for determinism; for now we just keep this for TTI."""
    _validate_image_gen_determinism(
        _get_offline_images(_run_offline_t2i(omni)),
        _get_offline_images(_run_offline_t2i(omni)),
    )


def run_and_validate_text_to_image_multi_output(omni: Omni):
    """Checks for multi-output; for now we just keep this for TTI."""
    params = replace(IMAGE_GEN_SAMPLING_PARAMS, num_outputs_per_prompt=2)
    _validate_images(_get_offline_images(_run_offline_t2i(omni, params)), expected_n=2)


def _run_online_t2v(
    server: OmniServer, client: OpenAIClientHandler, form_data: dict | None = None
) -> list[DiffusionResponse]:
    """Run a text to video request through the server's /v1/videos API."""
    data = dict(form_data or VIDEO_GEN_FORM_DATA)
    data.setdefault("prompt", PROMPT)
    data.setdefault("model", server.model)
    return client.send_video_diffusion_request({"form_data": data})


def _get_online_videos(responses: list[DiffusionResponse]) -> list:
    """Extract the videos from a server response."""
    assert len(responses) == 1
    videos = responses[0].videos
    assert videos is not None
    assert len(videos) > 0
    return videos


### Online task runners
def run_and_validate_online_text_to_image_request(server: OmniServer, client: OpenAIClientHandler):
    """Run and validate a text to image request through the server."""
    _validate_images(_get_online_images(_run_online_t2i(server, client)))


def run_and_validate_online_image_to_image_request(server: OmniServer, client: OpenAIClientHandler):
    """Run and validate an image to image request through the server."""
    _validate_images(_get_online_images(_run_online_i2i(server, client)))


def run_and_validate_online_text_to_image_determinism(server: OmniServer, client: OpenAIClientHandler):
    """Checks for determinism through the server; for now we just keep this for TTI."""
    _validate_image_gen_determinism(
        _get_online_images(_run_online_t2i(server, client)),
        _get_online_images(_run_online_t2i(server, client)),
    )


def run_and_validate_online_text_to_image_multi_output(server: OmniServer, client: OpenAIClientHandler):
    """Checks for multi-output through the server; for now we just keep this for TTI."""
    extra_body = {**IMAGE_GEN_EXTRA_BODY, "num_outputs_per_prompt": 2}
    _validate_images(
        _get_online_images(_run_online_t2i(server, client, extra_body=extra_body)),
        expected_n=2,
    )


def run_and_validate_online_text_to_video_request(server: OmniServer, client: OpenAIClientHandler):
    """Run and validate a text to video request through the server."""
    _get_online_videos(_run_online_t2v(server, client))


def run_and_validate_online_text_to_text_request(server: OmniServer, client: OpenAIClientHandler) -> None:
    """Run and validate a text-in, text-out request."""
    messages = dummy_messages_from_mix_data(content_text=OMNI_TEXT_PROMPT)
    request_config = {
        "model": server.model,
        "messages": messages,
        "modalities": ["text"],
    }
    _validate_text(_get_online_omni_response(client.send_omni_request(request_config)))


def run_and_validate_online_text_to_speech_request(server: OmniServer, client: OpenAIClientHandler) -> None:
    """Run and validate a text-in, speech-out request."""
    messages = dummy_messages_from_mix_data(content_text=OMNI_SPEECH_PROMPT)
    request_config = {
        "model": server.model,
        "messages": messages,
        "modalities": ["audio"],
    }
    _validate_speech(_get_online_omni_response(client.send_omni_request(request_config)))


# TODO: add offline AR task runners (run_and_validate_text_to_text_request,
# run_and_validate_mm_to_text_request, run_and_validate_text_to_speech_request)
# once the following are in place:
#   - OmniModelTestOpts initialization and case-filtering infra (analogous to
#     build_omni_from_diff_accelerations / get_parametrized_options for diffusion)
#   - Confirmed API for Omni.generate() with AR prompts and modality params


@dataclass
class TaskRunner:
    offline_validator: OfflineTaskRunner
    online_validator: OnlineTaskRunner


TASKS_TO_RUNNER_MAP: dict[ModelTasks, TaskRunner] = {
    ModelTasks.TEXT_TO_TEXT: TaskRunner(
        run_and_validate_text_to_text_request,
        run_and_validate_online_text_to_text_request,
    ),
    ModelTasks.TEXT_TO_IMAGE: TaskRunner(
        run_and_validate_text_to_image_request,
        run_and_validate_online_text_to_image_request,
    ),
    ModelTasks.IMAGE_TO_IMAGE: TaskRunner(
        run_and_validate_image_to_image_request,
        run_and_validate_online_image_to_image_request,
    ),
    ModelTasks.TEXT_TO_VIDEO: TaskRunner(
        run_and_validate_text_to_video_request,
        run_and_validate_online_text_to_video_request,
    ),
    ModelTasks.TEXT_TO_AUDIO: TaskRunner(
        run_and_validate_text_to_speech_request,
        run_and_validate_online_text_to_speech_request,
    ),
}
