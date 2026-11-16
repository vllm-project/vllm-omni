# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host request preparation and scheduler admission, without model weights."""

from dataclasses import dataclass

import pytest

from vllm_omni.diffusion.mixfusion.request import MixFusionImageRequestConfig
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionCapability, MixFusionRequestInfo, fits_chunk_budget
from vllm_omni.diffusion.mixfusion.setup import prepare_mixfusion_admission
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _Config:
    enable_mixfusion: bool = True
    mixfusion_min_chunk_tokens: int = 256
    mixfusion_max_chunks: int = 128
    max_num_seqs: int = 4
    model_class_name: str = "QwenImagePipeline"


@dataclass
class _Engine:
    od_config: _Config
    mixfusion_capability: MixFusionCapability = MixFusionCapability(MixFusionImageRequestConfig(8, 128, 2, 4.0), "")


def _request(rid, h=1024, w=1024, *, engine=None, **sampling):
    request = OmniDiffusionRequest(
        request_id=rid,
        prompt="a fox",
        sampling_params=OmniDiffusionSamplingParams(height=h, width=w, num_inference_steps=3, **sampling),
    )
    prepare_mixfusion_admission(engine or _Engine(_Config()), request)
    return request


def _scheduler(*requests):
    scheduler = StepScheduler()
    scheduler.initialize(_Config())
    for request in requests:
        scheduler.add_request(request)
    return scheduler


@pytest.mark.parametrize(
    "server,flag,packed", [(True, None, True), (False, True, True), (True, False, False), (False, None, False)]
)
def test_request_flag_precedence(server, flag, packed):
    extra = {} if flag is None else {"enable_mixfusion": flag}
    request = _request("a", engine=_Engine(_Config(enable_mixfusion=server)), extra_args=extra)
    assert (request.mixfusion_info is not None) == packed


def test_mixed_resolutions_share_scheduler_approved_batch():
    a, b = _request("a"), _request("b", 768, 1024)
    scheduler = _scheduler(a, b)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["a", "b"]
    assert output.use_mixfusion


def test_http_image_prompts_share_mixed_resolution_batch():
    a, b = _request("a", 512, 512), _request("b", 768, 768)
    for request in (a, b):
        request.prompt = OmniTextPrompt(prompt="a fox", negative_prompt="blurry", modalities=["image"])
        prepare_mixfusion_admission(_Engine(_Config()), request)
        assert request.mixfusion_info is not None
        assert request.mixfusion_info.condition_key == (True,)
        assert request.prompt["modalities"] == ["image"]
    assert (a.mixfusion_info.image_tokens, b.mixfusion_info.image_tokens) == (1024, 2304)
    output = _scheduler(a, b).schedule()
    assert output.scheduled_request_ids == ["a", "b"]
    assert output.use_mixfusion


@pytest.mark.parametrize("modalities", [None, [], ["video"], ["image", "audio"], "image"])
def test_non_image_modalities_keep_dense_admission(modalities):
    request = _request("a")
    request.prompt = {"prompt": "a fox", "modalities": modalities}
    prepare_mixfusion_admission(_Engine(_Config()), request)
    assert request.mixfusion_info is None


@pytest.mark.parametrize("other,extras", [(992, {}), (768, {"mixfusion_max_chunks": 4})])
def test_joint_chunk_constraint_leaves_candidate_waiting_then_runs_it(other, extras):
    a = _request("a", extra_args=extras)
    b = _request("b", other, 1024, extra_args=extras)
    scheduler = _scheduler(a, b)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["a"]
    assert output.use_mixfusion
    scheduler.finish_requests("a", DiffusionRequestStatus.FINISHED_COMPLETED)
    assert scheduler.schedule().scheduled_request_ids == ["b"]


def test_unfulfillable_single_request_uses_dense_and_retains_geometry():
    extras = {"mixfusion_min_chunk_tokens": 8192}
    a = _request("a", 768, 1024, extra_args=extras)
    b = _request("b", 1024, 768, extra_args=extras)
    assert a.mixfusion_info is None and b.mixfusion_info is None
    output = _scheduler(a, b).schedule()
    assert output.scheduled_request_ids == ["a"]
    assert not output.use_mixfusion


def test_dense_and_packed_requests_do_not_mix_even_with_unspecified_dimensions():
    a = _request("a", None, None)
    b = _request("b", None, None, extra_args={"enable_mixfusion": False})
    assert _scheduler(a, b).schedule().scheduled_request_ids == ["a"]


def test_continuous_admission_checks_cached_and_new_requests_and_completion():
    a, b, c = _request("a"), _request("b", 768, 1024), _request("c", 512, 512)
    scheduler = _scheduler(a, b)
    scheduler.schedule()
    scheduler.add_request(c)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["c", "a", "b"]
    assert output.use_mixfusion
    scheduler.finish_requests("b", DiffusionRequestStatus.FINISHED_ABORTED)
    assert set(scheduler.schedule().scheduled_request_ids) == {"a", "c"}


def test_multi_output_requests_share_packed_admission():
    # Multi-output requests carry their image count in the admission metadata
    # and pack like any other request; the tensor slicing limitation only
    # applies to per-request forward arguments, which travel as lists.
    a = _request("a", num_outputs_per_prompt=2)
    b = _request("b", 768, 1024, num_outputs_per_prompt=2)
    assert a.mixfusion_info.num_images == 2 and b.mixfusion_info.num_images == 2
    output = _scheduler(a, b).schedule()
    assert output.scheduled_request_ids == ["a", "b"]
    assert output.use_mixfusion


def test_cfg_branches_are_part_of_packing_compatibility():
    a, b = _request("a"), _request("b")
    b.prompt = {"prompt": "a fox", "negative_prompt": " "}
    prepare_mixfusion_admission(_Engine(_Config()), b)
    assert _scheduler(a, b).schedule().scheduled_request_ids == ["a"]


def test_geometry_uses_loaded_model_defaults_and_patch_grid():
    request = _request("a", None, None, engine=_Engine(_Config()))
    assert (request.mixfusion_info.num_images, request.mixfusion_info.image_tokens) == (1, 4096)
    assert _request("b", 1031, 777).mixfusion_info.image_tokens == 64 * 48


@pytest.mark.parametrize("default_scale,uses_cfg", [(1.0, False), (4.0, True)])
def test_declared_cfg_default_controls_compatibility(default_scale, uses_cfg):
    engine = _Engine(_Config(), MixFusionCapability(MixFusionImageRequestConfig(8, 128, 2, default_scale), ""))
    request = _request("a")
    request.prompt = {"prompt": "a fox", "negative_prompt": " "}
    prepare_mixfusion_admission(engine, request)
    assert request.mixfusion_info.condition_key == (uses_cfg,)


@pytest.mark.parametrize("modalities", [None, ["image"]])
def test_image_declaration_keeps_extra_prompt_inputs_dense(modalities):
    request = _request("a")
    request.prompt = {"prompt": "a fox", "additional_information": {"reference": "image"}}
    if modalities is not None:
        request.prompt["modalities"] = modalities
    prepare_mixfusion_admission(_Engine(_Config()), request)
    assert request.mixfusion_info is None


@pytest.mark.parametrize(
    "counts,minimum,maximum,accepted",
    [
        ([(1, 8), (1, 12)], 1, 8, True),
        ([(2, 8), (1, 12)], 1, 6, False),
        ([(1, 4)], 8, 8, False),
        ([(1, 8), (1, 12)], 1, 4, False),
    ],
)
def test_budget_predicate_uses_full_batch_and_image_counts(counts, minimum, maximum, accepted):
    requests = [MixFusionRequestInfo(images, tokens, minimum, maximum) for images, tokens in counts]
    assert fits_chunk_budget(requests) is accepted


def test_new_candidate_is_checked_with_new_gcd_and_departed_requests_are_removed():
    # Existing batch: gcd(2048, 4096)=2048, 3 chunks. Adding 3072 changes
    # the GCD to 1024 and raises the complete batch to 9 chunks, over budget.
    a = _request("a", 512, 1024, extra_args={"mixfusion_max_chunks": 8})
    b = _request("b", 1024, 1024, extra_args={"mixfusion_max_chunks": 8})
    c = _request("c", 768, 1024, extra_args={"mixfusion_max_chunks": 8})
    scheduler = _scheduler(a, b)
    assert scheduler.schedule().scheduled_request_ids == ["a", "b"]
    scheduler.add_request(c)
    assert scheduler.schedule().scheduled_request_ids == ["a", "b"]
    scheduler.finish_requests("a", DiffusionRequestStatus.FINISHED_COMPLETED)
    output = scheduler.schedule()
    assert output.scheduled_request_ids == ["c", "b"]
    assert output.use_mixfusion
