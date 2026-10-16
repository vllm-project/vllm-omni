# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    get_qwen_image_21_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler, StepScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def preprocess(tmp_path):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
    return get_qwen_image_21_pre_process_func(SimpleNamespace(model=str(tmp_path)))


def make_request(request_id, sizes, color="red"):
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt={
            "prompt": "A ceramic teapot",
            "multi_modal_data": {"image": [Image.new("RGB", size, color) for size in sizes]},
        },
        sampling_params=OmniDiffusionSamplingParams(
            height=1024, width=1024, num_inference_steps=2, true_cfg_scale=1.0, seed=42
        ),
    )


@pytest.mark.parametrize("scheduler_class", [RequestScheduler, StepScheduler])
@pytest.mark.parametrize(
    "first_sizes,second_sizes,compatible",
    [
        pytest.param([], [(64, 64)], False, id="text-and-edit"),
        pytest.param([(64, 64)], [], False, id="edit-and-text"),
        pytest.param([(64, 64)], [(64, 64), (64, 64)], False, id="image-count"),
        pytest.param([(64, 64)], [(128, 64)], False, id="image-aspect-ratio"),
        pytest.param([(64, 64), (128, 64)], [(128, 64), (64, 64)], False, id="slot-order"),
        pytest.param([], [], True, id="two-text-requests"),
        pytest.param([(64, 64)], [(64, 64)], True, id="different-image-content"),
        pytest.param([(64, 64)], [(128, 128)], True, id="same-resized-layout"),
        pytest.param([(64, 64), (128, 64)], [(128, 128), (256, 128)], True, id="same-multi-image-layout"),
    ],
)
def test_scheduler_groups_by_condition_layout(preprocess, scheduler_class, first_sizes, second_sizes, compatible):
    first = preprocess(make_request("first", first_sizes))
    second = preprocess(make_request("second", second_sizes, color="blue"))
    scheduler = scheduler_class()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(first)
    scheduler.add_request(second)

    scheduled = scheduler.schedule()
    expected_ids = ["first", "second"] if compatible else ["first"]
    assert scheduled.scheduled_request_ids == expected_ids

    scheduler.finish_requests(expected_ids, DiffusionRequestStatus.FINISHED_COMPLETED)
    if not compatible:
        assert scheduler.schedule().scheduled_request_ids == ["second"]
        scheduler.finish_requests("second", DiffusionRequestStatus.FINISHED_COMPLETED)
    assert not scheduler.has_requests()


@pytest.mark.parametrize(
    "prompt", ["A teapot", {"prompt": "A teapot"}, {"prompt": "A teapot", "multi_modal_data": {"image": None}}]
)
def test_text_input_forms_share_compatibility_key(preprocess, prompt):
    request = make_request("text", [])
    request.prompt = prompt
    processed = preprocess(request)
    empty_list_request = preprocess(make_request("empty-list", []))
    assert processed.batch_compatibility_key is not None
    assert processed.batch_compatibility_key == empty_list_request.batch_compatibility_key


@pytest.mark.parametrize("image_type", ["numpy", "tensor"])
def test_preprocess_accepts_array_images_without_truth_testing(preprocess, image_type):
    import numpy as np
    import torch

    pixels = np.full((32, 32, 4), 255, dtype=np.uint8)
    image = torch.from_numpy(pixels).permute(2, 0, 1) if image_type == "tensor" else pixels
    request = make_request("array", [])
    request.prompt["multi_modal_data"]["image"] = image
    processed = preprocess(request)
    assert processed.batch_compatibility_key is not None
    assert processed.allow_mixed_step_phases is False


def test_late_prefill_waits_until_all_decode_requests_finish(preprocess):
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=3))
    for request_id in ("a", "b"):
        scheduler.add_request(preprocess(make_request(request_id, [])))
    wave = scheduler.schedule()
    assert wave.scheduled_request_ids == ["a", "b"]
    scheduler.update_from_output(
        wave,
        BatchRunnerOutput.from_list(
            [RunnerOutput(request_id=request_id, step_index=1, finished=False) for request_id in ("a", "b")]
        ),
    )
    scheduler.add_request(preprocess(make_request("late", [])))
    assert scheduler.schedule().scheduled_request_ids == ["a", "b"]
    scheduler.finish_requests("a", DiffusionRequestStatus.FINISHED_COMPLETED)
    assert scheduler.schedule().scheduled_request_ids == ["b"]
    scheduler.finish_requests("b", DiffusionRequestStatus.FINISHED_COMPLETED)
    assert scheduler.schedule().scheduled_request_ids == ["late"]


def test_preprocess_normalizes_tuple_images_like_a_list(preprocess):
    image = Image.new("RGB", (64, 64), "red")
    tuple_request = make_request("tuple", [])
    tuple_request.prompt["multi_modal_data"]["image"] = (image,)

    processed = preprocess(tuple_request)
    reference = preprocess(make_request("list", [(64, 64)]))

    assert processed.batch_compatibility_key == reference.batch_compatibility_key
    assert processed.batch_compatibility_key[0] == "qwen_image_21"
    assert len(processed.batch_compatibility_key[1]) == 1
    info = processed.prompt["additional_information"]
    assert len(info["prompt_image"]) == 1
    assert len(info["vae_images"]) == 1
    assert len(info["input_image_sizes"]) == 1


def test_preprocess_rejects_more_images_than_the_model_supports(preprocess):
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
        MAX_QWEN_IMAGE_21_INPUT_IMAGES,
    )

    request = make_request("too-many", [(64, 64)] * (MAX_QWEN_IMAGE_21_INPUT_IMAGES + 1))
    with pytest.raises(ValueError, match="At most"):
        preprocess(request)


def test_preprocess_defaults_true_cfg_scale(preprocess):
    request = make_request("cfg", [])
    request.sampling_params.true_cfg_scale = None
    processed = preprocess(request)
    assert processed.sampling_params.true_cfg_scale == 1.0

    request = make_request("cfg-set", [])
    request.sampling_params.true_cfg_scale = 4.0
    assert preprocess(request).sampling_params.true_cfg_scale == 4.0


@pytest.mark.parametrize("size,expected_wider", [((128, 64), True), ((64, 128), False)])
def test_preprocess_derives_output_size_from_condition_aspect(preprocess, size, expected_wider):
    request = make_request("aspect", [size])
    request.sampling_params.height = None
    request.sampling_params.width = None

    processed = preprocess(request)

    height = processed.sampling_params.height
    width = processed.sampling_params.width
    assert height % 32 == 0 and width % 32 == 0
    assert (width > height) is expected_wider
    info = processed.prompt["additional_information"]
    assert info["calculated_height"] == height
    assert info["calculated_width"] == width


def test_preprocess_stages_rgba_vae_images_and_resized_prompt_images(preprocess):
    processed = preprocess(make_request("staged", [(96, 64)]))
    info = processed.prompt["additional_information"]

    (vae_image,) = info["vae_images"]
    # RGB input gains an opaque alpha channel for the RGBA VAE; the frame dim is
    # inserted for the 3D (video-capable) VAE.
    assert vae_image.shape[1] == 4
    assert vae_image.shape[2] == 1
    (prompt_image,) = info["prompt_image"]
    assert isinstance(prompt_image, Image.Image)
    (width, height) = info["input_image_sizes"][0]
    assert prompt_image.size == (width, height)
    assert processed.batch_compatibility_key == ("qwen_image_21", ((width, height),))
