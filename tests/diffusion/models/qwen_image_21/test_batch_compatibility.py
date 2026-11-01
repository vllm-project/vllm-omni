# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    QWEN_IMAGE_21_T2I_PROMPT_TEMPLATE,
    get_qwen_image_21_pre_process_func,
    qwen_image_21_prompt_text,
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


def _stub_tokenizer(text):
    """Stand-in for the Qwen3-VL processor tokenizer: one id per character."""
    return {"input_ids": list(range(len(text)))}


@pytest.fixture
def sized_preprocess(tmp_path, monkeypatch):
    """Pre-process func whose prompt tokenizer is a deterministic stub.

    The stub makes the prompt token count depend only on the templated string,
    so compatibility keys can be compared without a checkpoint on disk.
    """
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as decode_graph_mod
    from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21 as pipeline_mod

    monkeypatch.setitem(pipeline_mod._PROMPT_TOKENIZER_CACHE, str(tmp_path), _stub_tokenizer)
    # Treat this fixture as a CUDA graph-capable serve even in CPU-only CI.
    monkeypatch.setattr(decode_graph_mod.current_omni_platform, "is_cuda", lambda: True)
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(exist_ok=True)
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
    key = processed.batch_compatibility_key
    assert key[:2] == ("qwen_image_21", ((width, height),))
    # ``tmp_path`` has no ``processor/`` subfolder, so the prompt size is unknown
    # and the key stays length-agnostic.
    assert key[2] is None


def test_prompt_scaffold_tracks_the_image_count():
    assert qwen_image_21_prompt_text("hi", 0) == QWEN_IMAGE_21_T2I_PROMPT_TEMPLATE.format("hi")
    one = qwen_image_21_prompt_text("hi", 1)
    two = qwen_image_21_prompt_text("hi", 2)
    assert one.count("<|image_pad|>") == 1
    assert two.count("<|image_pad|>") == 2
    assert "<image2>" not in one
    assert "<image2>" in two


def test_prompt_length_is_part_of_the_compatibility_key(sized_preprocess):
    short = sized_preprocess(make_request("short", [(64, 64)]))
    short_again = sized_preprocess(make_request("short-again", [(64, 64)], color="blue"))
    long_request = make_request("long", [(64, 64)], color="blue")
    long_request.prompt["prompt"] = "A ceramic teapot on a windowsill " * 8
    long = sized_preprocess(long_request)

    # Same reference layout, different prompt text: the layout part still
    # matches, only the text-length slot separates the two requests.
    assert short.batch_compatibility_key[1] == long.batch_compatibility_key[1]
    assert short.batch_compatibility_key[2] == short_again.batch_compatibility_key[2]
    assert long.batch_compatibility_key[2] > short.batch_compatibility_key[2]


def test_text_only_requests_are_keyed_by_prompt_length(sized_preprocess):
    short = sized_preprocess(make_request("short", []))
    long_request = make_request("long", [])
    long_request.prompt = {"prompt": "A ceramic teapot on a windowsill " * 8}
    long = sized_preprocess(long_request)

    assert short.batch_compatibility_key is not None
    assert short.batch_compatibility_key[2] != long.batch_compatibility_key[2]


def test_unsizable_prompt_keeps_the_length_agnostic_key(preprocess):
    key = preprocess(make_request("unsized", [(64, 64)])).batch_compatibility_key
    assert key[0] == "qwen_image_21"
    assert len(key) == 4
    assert key[2] is None
    assert key[3] is None


@pytest.mark.parametrize("od_config", [{"enforce_eager": True}, {"enable_cuda_graph_decode": False}])
def test_eager_serve_keeps_the_padding_friendly_key(tmp_path, monkeypatch, od_config):
    from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21 as pipeline_mod

    monkeypatch.setitem(pipeline_mod._PROMPT_TOKENIZER_CACHE, str(tmp_path), _stub_tokenizer)
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(exist_ok=True)
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))

    preprocess = get_qwen_image_21_pre_process_func(SimpleNamespace(model=str(tmp_path), **od_config))

    # Padding is free when no decode graph can be captured, so the key must not
    # split requests by prompt length.
    assert preprocess(make_request("eager", [(64, 64)])).batch_compatibility_key[2] is None


@pytest.mark.parametrize(
    "config_overrides",
    [
        pytest.param({"parallel_config": SimpleNamespace(tensor_parallel_size=2)}, id="tensor-parallel"),
        pytest.param({"parallel_config": SimpleNamespace(sequence_parallel_size=2)}, id="sequence-parallel"),
        pytest.param({"parallel_config": SimpleNamespace(use_hsdp=True)}, id="hsdp"),
        pytest.param({"extras": {"prefix_kv_cache_dtype": "fp8"}}, id="quantized-prefix-kv"),
        pytest.param({"enable_layerwise_offload": True}, id="layer-wise-offload"),
    ],
)
def test_ineligible_graph_config_keeps_the_padding_friendly_key(tmp_path, monkeypatch, config_overrides):
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as decode_graph_mod
    from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21 as pipeline_mod

    monkeypatch.setitem(pipeline_mod._PROMPT_TOKENIZER_CACHE, str(tmp_path), _stub_tokenizer)
    monkeypatch.setattr(decode_graph_mod.current_omni_platform, "is_cuda", lambda: True)
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(exist_ok=True)
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
    preprocess = get_qwen_image_21_pre_process_func(
        SimpleNamespace(model=str(tmp_path), enable_cuda_graph_decode=True, **config_overrides)
    )

    assert preprocess(make_request("ineligible", [(64, 64)])).batch_compatibility_key[2] is None


def test_non_cuda_graph_request_keeps_the_padding_friendly_key(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph as decode_graph_mod
    from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21 as pipeline_mod

    monkeypatch.setitem(pipeline_mod._PROMPT_TOKENIZER_CACHE, str(tmp_path), _stub_tokenizer)
    monkeypatch.setattr(decode_graph_mod.current_omni_platform, "is_cuda", lambda: False)
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(exist_ok=True)
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
    preprocess = get_qwen_image_21_pre_process_func(
        SimpleNamespace(model=str(tmp_path), enable_cuda_graph_decode=True, enforce_eager=False)
    )

    assert preprocess(make_request("npu", [(64, 64)])).batch_compatibility_key[2] is None


@pytest.mark.parametrize("scheduler_class", [RequestScheduler, StepScheduler])
def test_ragged_prompt_lengths_do_not_share_a_wave(sized_preprocess, scheduler_class):
    short = sized_preprocess(make_request("short", [(64, 64)]))
    long_request = make_request("long", [(64, 64)], color="blue")
    long_request.prompt["prompt"] = "A ceramic teapot on a windowsill " * 8
    long = sized_preprocess(long_request)

    scheduler = scheduler_class()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(short)
    scheduler.add_request(long)

    # Pairing these two would right-pad the short prompt and cost the wave its
    # captured decode graph, so they must not be scheduled together.
    assert scheduler.schedule().scheduled_request_ids == ["short"]
    scheduler.finish_requests("short", DiffusionRequestStatus.FINISHED_COMPLETED)
    assert scheduler.schedule().scheduled_request_ids == ["long"]


@pytest.mark.parametrize("scheduler_class", [RequestScheduler, StepScheduler])
def test_equal_length_prompts_still_share_a_wave(sized_preprocess, scheduler_class):
    scheduler = scheduler_class()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    for request_id, color in (("first", "red"), ("second", "blue")):
        scheduler.add_request(sized_preprocess(make_request(request_id, [(64, 64)], color=color)))

    assert scheduler.schedule().scheduled_request_ids == ["first", "second"]


def test_negative_prompt_length_is_part_of_the_compatibility_key(sized_preprocess):
    def with_negative(request_id, negative):
        request = make_request(request_id, [(64, 64)])
        request.sampling_params.true_cfg_scale = 4.0
        request.prompt["negative_prompt"] = negative
        return sized_preprocess(request)

    short_neg = with_negative("neg-short", "blurry")
    long_neg = with_negative("neg-long", "blurry, low quality, watermark, text")
    same_neg = with_negative("neg-same", "blurry")

    assert short_neg.batch_compatibility_key[:3] == long_neg.batch_compatibility_key[:3]
    assert short_neg.batch_compatibility_key[3] != long_neg.batch_compatibility_key[3]
    assert short_neg.batch_compatibility_key == same_neg.batch_compatibility_key


@pytest.mark.parametrize("scheduler_class", [RequestScheduler, StepScheduler])
def test_omitted_and_empty_negative_prompts_do_not_share_a_wave(sized_preprocess, scheduler_class):
    omitted = make_request("omitted-negative", [(64, 64)])
    omitted.sampling_params.true_cfg_scale = 4.0
    empty = make_request("empty-negative", [(64, 64)], color="blue")
    empty.sampling_params.true_cfg_scale = 4.0
    empty.prompt["negative_prompt"] = ""

    omitted = sized_preprocess(omitted)
    empty = sized_preprocess(empty)

    assert omitted.batch_compatibility_key[:3] == empty.batch_compatibility_key[:3]
    assert omitted.batch_compatibility_key[3] != empty.batch_compatibility_key[3]
    assert empty.batch_compatibility_key[3] == (
        True,
        len(qwen_image_21_prompt_text(" ", num_images=0)),
    )

    scheduler = scheduler_class()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(omitted)
    scheduler.add_request(empty)
    assert scheduler.schedule().scheduled_request_ids == ["omitted-negative"]


@pytest.mark.parametrize("preprocess_kind", ["unsized", "eager"])
@pytest.mark.parametrize("scheduler_class", [RequestScheduler, StepScheduler])
def test_fallback_keys_keep_negative_prompt_presence(
    tmp_path, monkeypatch, preprocess, preprocess_kind, scheduler_class
):
    if preprocess_kind == "unsized":
        preprocess_func = preprocess
    else:
        from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21 as pipeline_mod

        monkeypatch.setitem(pipeline_mod._PROMPT_TOKENIZER_CACHE, str(tmp_path), _stub_tokenizer)
        vae_dir = tmp_path / "vae"
        vae_dir.mkdir(exist_ok=True)
        (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
        preprocess_func = get_qwen_image_21_pre_process_func(SimpleNamespace(model=str(tmp_path), enforce_eager=True))

    omitted = make_request("fallback-omitted", [(64, 64)])
    omitted.sampling_params.true_cfg_scale = 4.0
    empty = make_request("fallback-empty", [(64, 64)], color="blue")
    empty.sampling_params.true_cfg_scale = 4.0
    empty.prompt["negative_prompt"] = ""
    omitted = preprocess_func(omitted)
    empty = preprocess_func(empty)

    assert omitted.batch_compatibility_key[2] is None
    assert empty.batch_compatibility_key[2] is None
    assert omitted.batch_compatibility_key[3] != empty.batch_compatibility_key[3]

    scheduler = scheduler_class()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(omitted)
    scheduler.add_request(empty)
    assert scheduler.schedule().scheduled_request_ids == ["fallback-omitted"]


def test_negative_prompt_ignored_when_cfg_is_off(sized_preprocess):
    request = make_request("neg-off", [(64, 64)])
    request.prompt["negative_prompt"] = "blurry"
    key = sized_preprocess(request).batch_compatibility_key
    assert key[3] is None
