"""
Definitions running and validating individual tasks, e.g., text to image,
image to image, and so on. These are called by the core test runner.
"""

import base64
import io
from dataclasses import replace

import numpy as np
from PIL import Image

from tests.helpers.runtime import DiffusionResponse, OmniServer, OpenAIClientHandler, dummy_messages_from_mix_data
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

PROMPT = "Dummy prompt"
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

# Online extra_body for diffusion requests
IMAGE_GEN_EXTRA_BODY = {
    "height": HEIGHT,
    "width": WIDTH,
    "num_inference_steps": 4,
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


### Offline helpers
def _run_offline_t2i(omni: Omni, params: OmniDiffusionSamplingParams = IMAGE_GEN_SAMPLING_PARAMS):
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
def run_and_validate_text_to_image_request(omni: Omni):
    """Run and validate a text to image request."""
    _validate_images(_get_offline_images(_run_offline_t2i(omni)))


def run_and_validate_image_to_image_request(omni: Omni):
    """Run and validate an image to image request."""
    _validate_images(_get_offline_images(_run_offline_i2i(omni)))


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
