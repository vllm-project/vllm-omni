# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Definitions running and validating individual tasks, e.g., text to image,
image to image, and so on. These are called by the core test runner.
"""

import base64
import io
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.helpers.runtime import DiffusionResponse, OmniServer, OnlineOmniClient, dummy_messages_from_mix_data
from tests.model_tests.diffusion.config_types import DiffusionTasks
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
VIDEO_GEN_FORM_DATA: dict[str, Any] = {
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
    assert len(images_a) == len(images_b)
    for image_a, image_b in zip(images_a, images_b, strict=True):
        assert np.array_equal(np.array(image_a), np.array(image_b))


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


def _validate_video_frames(videos: list, expected_n: int = 1, expected_frames: int = VIDEO_NUM_FRAMES):
    """Given a list of videos (one array per generation, each holding all of
    that generation's frames), ensure count and shape are correct."""
    assert len(videos) == expected_n
    for video in videos:
        assert isinstance(video, np.ndarray)
        assert video.ndim == 4, f"Expected 4D video array (frames, H, W, C), got shape {video.shape}"
        assert video.shape[-4] == expected_frames, f"Expected {expected_frames} frames, got {video.shape[-4]}"
        assert video.shape[-3] == HEIGHT, f"Expected height {HEIGHT}, got {video.shape[-3]}"
        assert video.shape[-2] == WIDTH, f"Expected width {WIDTH}, got {video.shape[-2]}"
        assert video.shape[-1] == 3, f"Expected 3 channels (RGB), got {video.shape[-1]}"
    return videos


def _get_offline_videos(outputs: list[OmniRequestOutput]) -> list:
    """Extract the list of videos (one array per generation) from an Omni .generate() call.

    Multi-output video generations come back as a single 5D array
    (num_outputs, num_frames, H, W, C) rather than one 4D array per output;
    split it into a flat per-output list here so callers always see one
    4D array per generation.
    """
    assert len(outputs) == 1
    videos = []
    for video in outputs[0].images:
        if video.ndim == 5:
            videos.extend(video)
        else:
            videos.append(video)
    return videos


def _validate_video(outputs: list[OmniRequestOutput], expected_n: int = 1):
    """Given a set of outputs, ensure we got video frames with the expected shape."""
    _validate_video_frames(_get_offline_videos(outputs), expected_n=expected_n)


def _validate_video_gen_determinism(videos_a: list, videos_b: list):
    """Ensure that two video generations are valid and that the results match."""
    _validate_video_frames(videos_a)
    _validate_video_frames(videos_b)
    assert len(videos_a) == len(videos_b)
    for video_a, video_b in zip(videos_a, videos_b, strict=True):
        assert np.array_equal(video_a, video_b)


### Offline helpers
def _run_offline_t2i(omni: Omni, params: OmniDiffusionSamplingParams = IMAGE_GEN_SAMPLING_PARAMS):
    return omni.generate({"prompt": PROMPT}, params)


def _run_offline_t2v(omni: Omni, params: OmniDiffusionSamplingParams = VIDEO_GEN_SAMPLING_PARAMS):
    return omni.generate({"prompt": PROMPT}, params)


def _run_offline_i2i(omni: Omni, params: OmniDiffusionSamplingParams = IMAGE_GEN_SAMPLING_PARAMS):
    return omni.generate(
        {"prompt": PROMPT, "multi_modal_data": {"image": INPUT_IMAGE}},
        params,
    )


def _run_offline_i2v(omni: Omni, params: OmniDiffusionSamplingParams = VIDEO_GEN_SAMPLING_PARAMS):
    return omni.generate(
        {"prompt": PROMPT, "multi_modal_data": {"image": INPUT_IMAGE}},
        params,
    )


### Online helpers
def _build_online_image_data_url() -> str:
    """Get a valid base 64 encoded data URL corresponding to an image."""
    buf = io.BytesIO()
    INPUT_IMAGE.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _run_online_t2i(
    server: OmniServer, client: OnlineOmniClient, extra_body: dict | None = None
) -> list[DiffusionResponse]:
    """Run a text to image request through the server."""
    messages = dummy_messages_from_mix_data(content_text=PROMPT)
    request_config = {
        "model": server.model,
        "messages": messages,
        "extra_body": extra_body or IMAGE_GEN_EXTRA_BODY,
    }
    return client.send_diffusion_request(request_config)


def _run_online_i2i(
    server: OmniServer, client: OnlineOmniClient, extra_body: dict | None = None
) -> list[DiffusionResponse]:
    """Run an image to image request through the server."""
    image_data_url = _build_online_image_data_url()
    messages = dummy_messages_from_mix_data(
        content_text=PROMPT,
        image_data_url=image_data_url,
    )
    request_config = {
        "model": server.model,
        "messages": messages,
        "extra_body": extra_body or IMAGE_GEN_EXTRA_BODY,
    }
    return client.send_diffusion_request(request_config)


### Offline task runners
def run_and_validate_text_to_image_request(omni: Omni):
    """Run and validate a text to image request."""
    _validate_images(_get_offline_images(_run_offline_t2i(omni)))


def run_and_validate_image_to_image_request(omni: Omni):
    """Run and validate an image to image request."""
    _validate_images(_get_offline_images(_run_offline_i2i(omni)))


def run_and_validate_text_to_video_request(omni: Omni):
    """Run and validate a text to video request."""
    _validate_video(_run_offline_t2v(omni))


def run_and_validate_image_to_video_request(omni: Omni, check_t2v_divergence: bool = True):
    """Run and validate an image to video request.

    LTX2Pipeline uses a unified text/image entry and can fall back to T2V if
    the image never reaches conditioning, so also run T2V with the same seed
    and prompt and assert the outputs differ; a regression that silently
    drops the input image would otherwise still pass a "did we get a valid
    video back" check alone. Pipelines that mandate an image and fail closed
    without one (SANA-Video I2V) pass check_t2v_divergence=False: a dropped
    image raises instead of silently degrading, so the T2V comparison is moot.
    """
    i2v_video = _validate_video_frames(_get_offline_videos(_run_offline_i2v(omni)))[0]
    if not check_t2v_divergence:
        return
    t2v_video = _validate_video_frames(_get_offline_videos(_run_offline_t2v(omni)))[0]
    assert not np.array_equal(i2v_video, t2v_video), (
        "I2V output is identical to T2V output with the same seed and prompt; "
        "the input image may not be reaching conditioning."
    )


def run_and_validate_determinism(omni: Omni, task_type: DiffusionTasks):
    """Checks for determinism, dispatching by task type."""
    if task_type == DiffusionTasks.TEXT_TO_IMAGE:
        _validate_image_gen_determinism(
            _get_offline_images(_run_offline_t2i(omni)),
            _get_offline_images(_run_offline_t2i(omni)),
        )
    elif task_type == DiffusionTasks.IMAGE_TO_IMAGE:
        _validate_image_gen_determinism(
            _get_offline_images(_run_offline_i2i(omni)),
            _get_offline_images(_run_offline_i2i(omni)),
        )
    elif task_type == DiffusionTasks.TEXT_TO_VIDEO:
        _validate_video_gen_determinism(
            _get_offline_videos(_run_offline_t2v(omni)),
            _get_offline_videos(_run_offline_t2v(omni)),
        )
    elif task_type == DiffusionTasks.IMAGE_TO_VIDEO:
        _validate_video_gen_determinism(
            _get_offline_videos(_run_offline_i2v(omni)),
            _get_offline_videos(_run_offline_i2v(omni)),
        )
    else:
        raise ValueError(f"Task type {task_type} is not yet supported for determinism checks")


def run_and_validate_multi_output(omni: Omni, task_type: DiffusionTasks):
    """Checks for multi-output, dispatching by task type."""
    if task_type == DiffusionTasks.TEXT_TO_IMAGE:
        params = replace(IMAGE_GEN_SAMPLING_PARAMS, num_outputs_per_prompt=2)
        _validate_images(_get_offline_images(_run_offline_t2i(omni, params)), expected_n=2)
    elif task_type == DiffusionTasks.IMAGE_TO_IMAGE:
        params = replace(IMAGE_GEN_SAMPLING_PARAMS, num_outputs_per_prompt=2)
        _validate_images(_get_offline_images(_run_offline_i2i(omni, params)), expected_n=2)
    elif task_type == DiffusionTasks.TEXT_TO_VIDEO:
        params = replace(VIDEO_GEN_SAMPLING_PARAMS, num_outputs_per_prompt=2)
        _validate_video_frames(_get_offline_videos(_run_offline_t2v(omni, params)), expected_n=2)
    elif task_type == DiffusionTasks.IMAGE_TO_VIDEO:
        params = replace(VIDEO_GEN_SAMPLING_PARAMS, num_outputs_per_prompt=2)
        _validate_video_frames(_get_offline_videos(_run_offline_i2v(omni, params)), expected_n=2)
    else:
        raise ValueError(f"Task type {task_type} is not yet supported for multi-output checks")


def _run_online_t2v(
    server: OmniServer, client: OnlineOmniClient, form_data: dict | None = None
) -> list[DiffusionResponse]:
    """Run a text to video request through the server's /v1/videos API."""
    data = dict(form_data or VIDEO_GEN_FORM_DATA)
    data.setdefault("prompt", PROMPT)
    data.setdefault("model", server.model)
    return client.send_video_diffusion_request({"form_data": data})


def _run_online_i2v(server: OmniServer, client: OnlineOmniClient) -> list[DiffusionResponse]:
    """Run an image to video request through the server's /v1/videos API."""
    data = dict(VIDEO_GEN_FORM_DATA)
    data.setdefault("prompt", PROMPT)
    data.setdefault("model", server.model)
    image_data_url = _build_online_image_data_url()
    return client.send_video_diffusion_request({"form_data": data, "image_reference": image_data_url})


def _get_online_videos(responses: list[DiffusionResponse]) -> list:
    """Extract the videos from a server response."""
    assert len(responses) == 1
    videos = responses[0].videos
    assert videos is not None
    assert len(videos) > 0
    return videos


### Online task runners
def run_and_validate_online_text_to_image_request(server: OmniServer, client: OnlineOmniClient):
    """Run and validate a text to image request through the server."""
    _validate_images(_get_online_images(_run_online_t2i(server, client)))


def run_and_validate_online_image_to_image_request(server: OmniServer, client: OnlineOmniClient):
    """Run and validate an image to image request through the server."""
    _validate_images(_get_online_images(_run_online_i2i(server, client)))


def run_and_validate_online_determinism(server: OmniServer, client: OnlineOmniClient, task_type: DiffusionTasks):
    """Checks for determinism through the server, dispatching by task type.

    Video is skipped online only: send_video_diffusion_request returns encoded
    video bytes, not a frame-level array, and re-encoding the same content
    isn't guaranteed to be byte-identical even for deterministic generation,
    so a reliable check isn't possible with what's implemented here. The
    offline determinism check (run_and_validate_determinism) does cover
    video, since it has direct access to the raw frame arrays.
    """
    if task_type == DiffusionTasks.TEXT_TO_IMAGE:
        _validate_image_gen_determinism(
            _get_online_images(_run_online_t2i(server, client)),
            _get_online_images(_run_online_t2i(server, client)),
        )
    elif task_type == DiffusionTasks.IMAGE_TO_IMAGE:
        _validate_image_gen_determinism(
            _get_online_images(_run_online_i2i(server, client)),
            _get_online_images(_run_online_i2i(server, client)),
        )
    elif task_type in (DiffusionTasks.TEXT_TO_VIDEO, DiffusionTasks.IMAGE_TO_VIDEO):
        pytest.skip(f"Online determinism isn't supported for {task_type}.")
    else:
        raise ValueError(f"Task type {task_type} is not yet supported for determinism checks")


def run_and_validate_online_multi_output(server: OmniServer, client: OnlineOmniClient, task_type: DiffusionTasks):
    """Checks for multi-output through the server, dispatching by task type.

    Video is skipped online only: send_video_diffusion_request explicitly
    disallows more than one request/output per call. The offline multi-output
    check (run_and_validate_multi_output) does cover video, since
    Omni.generate() supports num_outputs_per_prompt directly.
    """
    if task_type == DiffusionTasks.TEXT_TO_IMAGE:
        extra_body = {**IMAGE_GEN_EXTRA_BODY, "num_outputs_per_prompt": 2}
        _validate_images(
            _get_online_images(_run_online_t2i(server, client, extra_body=extra_body)),
            expected_n=2,
        )
    elif task_type == DiffusionTasks.IMAGE_TO_IMAGE:
        extra_body = {**IMAGE_GEN_EXTRA_BODY, "num_outputs_per_prompt": 2}
        _validate_images(
            _get_online_images(_run_online_i2i(server, client, extra_body=extra_body)),
            expected_n=2,
        )
    elif task_type in (DiffusionTasks.TEXT_TO_VIDEO, DiffusionTasks.IMAGE_TO_VIDEO):
        pytest.skip(f"Online multi-output isn't supported for {task_type}.")
    else:
        raise ValueError(f"Task type {task_type} is not yet supported for multi-output checks")


def run_and_validate_online_text_to_video_request(server: OmniServer, client: OnlineOmniClient):
    """Run and validate a text to video request through the server."""
    _get_online_videos(_run_online_t2v(server, client))


def run_and_validate_online_image_to_video_request(server: OmniServer, client: OnlineOmniClient):
    """Run and validate an image to video request through the server."""
    _get_online_videos(_run_online_i2v(server, client))
