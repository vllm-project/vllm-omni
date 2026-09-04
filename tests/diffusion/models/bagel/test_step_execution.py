# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused CPU tests for BAGEL step execution and packed batching."""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel, NaiveCache
from vllm_omni.diffusion.models.bagel.pipeline_bagel import (
    BagelGenParams,
    BagelPipeline,
    get_bagel_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Sampling:
    return_trajectory_latents: bool
    return_trajectory_decoded: bool = False
    true_cfg_scale: float = 4.0
    cfg_normalize: bool = False
    image_latent: torch.Tensor | None = None
    extra_args: dict = field(default_factory=dict)


@dataclass
class _ScheduleConfig:
    _denoise_schedule_extra_step: bool = True


@dataclass
class _ParallelConfig:
    cfg_parallel_size: int


@dataclass
class _LanguageModelOutput:
    packed_query_sequence: torch.Tensor


def _sampling(*, trajectory: bool = False) -> _Sampling:
    return _Sampling(
        return_trajectory_latents=trajectory,
    )


@pytest.mark.parametrize(
    ("prompt", "use_step_execution"),
    [
        ({"prompt": "describe this image", "modalities": ["text"]}, False),
        ({"prompt": "draw a cat", "modalities": ["image"]}, True),
        ("describe this image", True),
    ],
)
def test_bagel_step_preprocessor_only_falls_back_for_explicit_text(prompt, use_step_execution):
    pre_process = get_bagel_pre_process_func(types.SimpleNamespace(step_execution=True))
    request = OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
        request_id="req",
    )

    assert pre_process(request) is request
    assert request.use_step_execution is use_step_execution


def test_bagel_step_preprocessor_defaults_to_non_step_when_omitted():
    pre_process = get_bagel_pre_process_func(types.SimpleNamespace())
    request = OmniDiffusionRequest(
        prompt={"prompt": "describe this image", "modalities": ["text"]},
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
        request_id="req",
    )

    assert pre_process(request) is request
    assert request.use_step_execution is True


@pytest.mark.parametrize(
    ("first_extra_args", "second_extra_args"),
    [
        ({"cfg_text_scale": 1.0}, {"cfg_text_scale": 4.0}),
        ({"cfg_img_scale": 1.0}, {"cfg_img_scale": 1.5}),
        ({"cfg_interval": (0.0, 1.0)}, {"cfg_interval": (0.4, 1.0)}),
        ({"cfg_renorm_type": "global"}, {"cfg_renorm_type": "channel"}),
        ({"cfg_renorm_min": 0.0}, {"cfg_renorm_min": 0.5}),
    ],
)
def test_bagel_step_preprocessor_separates_incompatible_cfg_settings(first_extra_args, second_extra_args):
    pre_process = get_bagel_pre_process_func(types.SimpleNamespace(step_execution=True))
    requests = [
        OmniDiffusionRequest(
            prompt={"prompt": "draw a cat", "modalities": ["image"]},
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, extra_args=extra_args),
            request_id=request_id,
        )
        for request_id, extra_args in (("a", first_extra_args), ("b", second_extra_args))
    ]

    first, second = (pre_process(request) for request in requests)

    assert first.batch_compatibility_key != second.batch_compatibility_key


def test_bagel_step_preprocessor_buckets_effective_img2img_sizes(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"vae_config":{"downsample":8},"latent_patch_size":2,"max_latent_size":32}',
        encoding="utf-8",
    )
    save_file(
        {"latent_pos_embed.pos_embed": torch.zeros(64 * 64, 1)},
        tmp_path / "ema.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"latent_pos_embed.pos_embed":"ema.safetensors"}}',
        encoding="utf-8",
    )
    pre_process = get_bagel_pre_process_func(
        types.SimpleNamespace(model=str(tmp_path), revision=None, step_execution=True)
    )
    requests = [
        OmniDiffusionRequest(
            prompt={
                "prompt": "edit this image",
                "modalities": ["img2img"],
                "multi_modal_data": {"img2img": Image.new("RGB", size)},
            },
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2),
            request_id=request_id,
        )
        for request_id, size in (("a", (800, 400)), ("b", (1024, 512)))
    ]
    first, second = (pre_process(request) for request in requests)

    assert (first.sampling_params.height, first.sampling_params.width) == (400, 800)
    assert (second.sampling_params.height, second.sampling_params.width) == (512, 1024)

    scheduler = StepScheduler()
    scheduler.initialize(types.SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(first)
    scheduler.add_request(second)

    schedule = scheduler.schedule()

    assert schedule.scheduled_request_ids == ["a"]
    assert schedule.num_running_reqs == 1
    assert schedule.num_waiting_reqs == 1


def _cache(length: int, value: float, num_layers: int = 1) -> NaiveCache:
    cache = NaiveCache(num_layers)
    if length:
        for layer in range(num_layers):
            cache.key_cache[layer] = torch.full((length, 1, 2), value)
            cache.value_cache[layer] = torch.full((length, 1, 2), value)
    return cache


def _generation_input(num_latents: int = 4) -> dict[str, torch.Tensor]:
    seq_len = num_latents + 2
    return {
        "packed_text_ids": torch.tensor([10, 11]),
        "packed_text_indexes": torch.tensor([0, seq_len - 1]),
        "packed_init_noises": torch.arange(num_latents * 3, dtype=torch.float32).reshape(num_latents, 3),
        "packed_vae_position_ids": torch.arange(num_latents),
        "packed_vae_token_indexes": torch.arange(1, seq_len - 1),
        "packed_seqlens": torch.tensor([seq_len], dtype=torch.int32),
        "packed_position_ids": torch.arange(seq_len),
    }


def _prepared_context(
    *,
    kv_len: int = 2,
    cfg_text_scale: float = 1.0,
    trajectory: bool = False,
    think_text: str | None = None,
) -> dict:
    del trajectory
    generation_input = _generation_input()
    context = {
        "kv_lens": [kv_len],
        "ropes": [kv_len],
        "past_key_values": _cache(kv_len, float(kv_len)),
    }
    return {
        "generation_input": generation_input,
        "cfg_text_packed_position_ids": generation_input["packed_position_ids"] + 20,
        "cfg_img_packed_position_ids": generation_input["packed_position_ids"] + 40,
        "gen_context": context,
        "cfg_text_context": {
            "kv_lens": [kv_len + 1],
            "ropes": [kv_len + 1],
            "past_key_values": _cache(kv_len + 1, float(kv_len + 1)),
        },
        "cfg_img_context": context,
        "gen_params": BagelGenParams(num_timesteps=5, timestep_shift=2.0, cfg_text_scale=cfg_text_scale),
        "image_shape": (64, 64),
        "think_text": think_text,
    }


def _pipeline_for_prepare(prepared: dict) -> BagelPipeline:
    pipeline = MagicMock()
    pipeline.scheduler = None
    pipeline.scheduler_kwargs = {}
    pipeline.bagel._sp_size = 1
    pipeline._forward_single.return_value = prepared
    pipeline.bagel.prepare_denoise_schedule.side_effect = lambda latents, steps, shift: Bagel.prepare_denoise_schedule(
        _ScheduleConfig(), latents, steps, shift
    )
    pipeline.prepare_encode = types.MethodType(BagelPipeline.prepare_encode, pipeline)
    return pipeline


def _prepare_state(request_id: str, *, kv_len: int, cfg_text_scale: float = 1.0) -> StepRequestState:
    state = StepRequestState(request_id=request_id, sampling=_sampling(trajectory=True), prompt="draw a cat")
    pipeline = _pipeline_for_prepare(_prepared_context(kv_len=kv_len, cfg_text_scale=cfg_text_scale))
    return pipeline.prepare_encode(state)


def test_prepare_encode_populates_request_local_state_and_schedule():
    prepared = _prepared_context(kv_len=3)
    pipeline = _pipeline_for_prepare(prepared)
    state = StepRequestState(request_id="req", sampling=_sampling(trajectory=True), prompt="draw a cat")

    result = pipeline.prepare_encode(state)

    assert result is state
    assert state.latents is prepared["generation_input"]["packed_init_noises"]
    assert state.latents.dtype == torch.float32
    assert state.total_steps == 5
    expected_points = torch.linspace(1, 0, 6)
    expected_points = 2.0 * expected_points / (1 + expected_points)
    assert torch.equal(state.timesteps, expected_points[:-1])
    assert state.timesteps.dtype == torch.float32
    assert state.timesteps.device == state.latents.device
    assert torch.equal(state.extra["bagel_dts"], expected_points[:-1] - expected_points[1:])
    assert state.extra["bagel_dts"].dtype == torch.float32
    assert state.extra["bagel_generation_input"]["packed_text_ids"].dtype == torch.long
    assert state.extra["bagel_generation_input"]["packed_text_indexes"].dtype == torch.long
    assert state.extra["bagel_generation_input"]["packed_vae_position_ids"].dtype == torch.long
    assert state.extra["bagel_generation_input"]["packed_vae_token_indexes"].dtype == torch.long
    assert state.extra["bagel_generation_input"]["packed_position_ids"].dtype == torch.long
    assert state.extra["bagel_generation_input"]["packed_seqlens"].dtype == torch.int32
    assert state.extra["bagel_gen_context"] is prepared["gen_context"]
    assert len(state.extra["bagel_trajectory_latents"]) == 1
    assert "packed_init_noises" not in state.extra["bagel_generation_input"]


def test_prepare_encode_rejects_non_positive_step_schedule():
    prepared = _prepared_context()
    prepared["gen_params"].num_timesteps = 0
    pipeline = _pipeline_for_prepare(prepared)
    state = StepRequestState(request_id="req", sampling=_sampling(), prompt="draw a cat")

    with pytest.raises(ValueError, match="num_inference_steps >= 1"):
        pipeline.prepare_encode(state)


def test_prepare_encode_supports_one_denoising_update():
    prepared = _prepared_context()
    prepared["gen_params"].num_timesteps = 1
    pipeline = _pipeline_for_prepare(prepared)
    state = StepRequestState(request_id="req", sampling=_sampling(), prompt="draw a cat")

    pipeline.prepare_encode(state)

    assert state.total_steps == 1
    assert len(state.timesteps) == 1
    assert len(state.extra["bagel_dts"]) == 1


def test_prepare_encode_rejects_sequence_parallel_before_forward():
    pipeline = _pipeline_for_prepare(_prepared_context())
    pipeline.bagel._sp_size = 2
    state = StepRequestState(request_id="req", sampling=_sampling(), prompt="draw a cat")

    with pytest.raises(NotImplementedError, match="does not currently support sequence parallelism"):
        pipeline.prepare_encode(state)

    pipeline._forward_single.assert_not_called()


def test_step_protocol_is_enabled_for_bagel():
    from vllm_omni.diffusion.models.interface import supports_step_execution

    bagel = object.__new__(BagelPipeline)

    assert supports_step_execution(bagel)


def test_prepare_encode_reuses_image_and_think_generation_context():
    pipeline = _pipeline_for_prepare(_prepared_context(think_text="reasoning"))
    sampling = _sampling()
    sampling.extra_args["think"] = True
    state = StepRequestState(
        request_id="req",
        sampling=sampling,
        prompt={"prompt": "edit", "multi_modal_data": {"image": "input.png"}},
    )

    result = pipeline.prepare_encode(state)

    assert result is state
    pipeline._forward_single.assert_called_once_with(state.prompt, sampling, prepare_only=True)
    assert state.extra["bagel_think_text"] == "reasoning"


def test_two_request_packed_indexes_are_rebased():
    state_a = _prepare_state("a", kv_len=2)
    state_b = _prepare_state("b", kv_len=3)

    packed = BagelPipeline._pack_step_generation_inputs([state_a, state_b])

    assert packed["packed_seqlens"].tolist() == [6, 6]
    assert packed["packed_text_indexes"].tolist() == [0, 5, 6, 11]
    assert packed["packed_vae_token_indexes"].tolist() == [1, 2, 3, 4, 7, 8, 9, 10]


def test_different_packed_sequence_lengths_are_rejected():
    states = []
    for index, num_latents in enumerate((2, 4)):
        state = StepRequestState(request_id=str(index), sampling=_sampling(), prompt="draw a cat")
        state.extra["bagel_generation_input"] = _generation_input(num_latents)
        states.append(state)

    with pytest.raises(ValueError, match="matching packed sequence lengths"):
        BagelPipeline._pack_step_generation_inputs(states)


def test_denoise_step_uses_one_forward_and_preserves_request_kv_lengths():
    states = [
        _prepare_state("a", kv_len=2, cfg_text_scale=4.0),
        _prepare_state("b", kv_len=3, cfg_text_scale=4.0),
    ]
    input_batch = InputBatch.make_batch(states)
    expected = torch.ones_like(input_batch.latents)

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.od_config.dtype = torch.float32
    pipeline.bagel._sp_size = 1
    pipeline.bagel.parallel_config = _ParallelConfig(cfg_parallel_size=1)
    pipeline.bagel.forward.return_value = expected
    pipeline._pack_step_generation_inputs = BagelPipeline._pack_step_generation_inputs
    pipeline._build_denoise_kwargs = types.MethodType(BagelPipeline._build_denoise_kwargs, pipeline)
    pipeline.denoise_step = types.MethodType(BagelPipeline.denoise_step, pipeline)

    actual = pipeline.denoise_step(input_batch, states=states)

    assert actual is expected
    pipeline.bagel.forward.assert_called_once()
    call = pipeline.bagel.forward.call_args.kwargs
    assert call["packed_text_indexes"].tolist() == [0, 5, 6, 11]
    assert call["past_key_values"].key_values_lens == [2, 3]
    assert call["cfg_branch_caches"][1].key_values_lens == [3, 4]
    assert call["cfg_vae_lengths"] == [4, 4]
    assert len(call["cfg_branch_pids"]) == 3


def test_denoise_step_rejects_mixed_cfg_settings_defensively():
    states = [
        _prepare_state("a", kv_len=2, cfg_text_scale=2.0),
        _prepare_state("b", kv_len=3, cfg_text_scale=4.0),
    ]
    input_batch = InputBatch.make_batch(states)
    pipeline = MagicMock()
    pipeline.bagel.parallel_config = _ParallelConfig(cfg_parallel_size=1)
    pipeline._pack_step_generation_inputs = BagelPipeline._pack_step_generation_inputs
    pipeline._build_denoise_kwargs = types.MethodType(BagelPipeline._build_denoise_kwargs, pipeline)

    with pytest.raises(ValueError, match="Mixed BAGEL CFG settings"):
        pipeline._build_denoise_kwargs(input_batch, states)


def test_denoise_step_dispatches_three_cfg_parallel_branches():
    states = [
        _prepare_state("a", kv_len=2, cfg_text_scale=4.0),
        _prepare_state("b", kv_len=3, cfg_text_scale=4.0),
    ]
    input_batch = InputBatch.make_batch(states)
    expected = torch.ones_like(input_batch.latents)

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.od_config.dtype = torch.float32
    pipeline.bagel._sp_size = 1
    pipeline.bagel.parallel_config = _ParallelConfig(cfg_parallel_size=3)
    pipeline.bagel.predict_noise_with_multi_branch_cfg.return_value = expected
    pipeline._pack_step_generation_inputs = BagelPipeline._pack_step_generation_inputs
    pipeline._build_denoise_kwargs = types.MethodType(BagelPipeline._build_denoise_kwargs, pipeline)
    pipeline._denoise_step_cfg_parallel = types.MethodType(BagelPipeline._denoise_step_cfg_parallel, pipeline)
    pipeline.denoise_step = types.MethodType(BagelPipeline.denoise_step, pipeline)
    cfg_group = MagicMock()

    with (
        patch(
            "vllm_omni.diffusion.models.bagel.pipeline_bagel.get_classifier_free_guidance_world_size",
            return_value=3,
        ),
        patch("vllm_omni.diffusion.models.bagel.pipeline_bagel.get_cfg_group", return_value=cfg_group),
    ):
        actual = pipeline.denoise_step(input_batch, states=states)

    assert actual is expected
    assert cfg_group.broadcast.call_args.args[0] is input_batch.latents
    call = pipeline.bagel.predict_noise_with_multi_branch_cfg.call_args.kwargs
    assert call["do_true_cfg"]
    assert len(call["branches_kwargs"]) == 3
    assert call["true_cfg_scale"]["cfg_vae_lengths"] == [4, 4]


def test_denoise_step_allows_idle_cfg_rank_for_two_branches():
    states = [
        _prepare_state("a", kv_len=2, cfg_text_scale=4.0),
        _prepare_state("b", kv_len=3, cfg_text_scale=4.0),
    ]
    for state in states:
        state.extra["bagel_gen_params"].cfg_img_scale = 1.0
    input_batch = InputBatch.make_batch(states)
    expected = torch.ones_like(input_batch.latents)

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.od_config.dtype = torch.float32
    pipeline.bagel.parallel_config = _ParallelConfig(cfg_parallel_size=3)
    pipeline.bagel.predict_noise_with_multi_branch_cfg.return_value = expected
    pipeline._pack_step_generation_inputs = BagelPipeline._pack_step_generation_inputs
    pipeline._build_denoise_kwargs = types.MethodType(BagelPipeline._build_denoise_kwargs, pipeline)
    pipeline._denoise_step_cfg_parallel = types.MethodType(BagelPipeline._denoise_step_cfg_parallel, pipeline)
    pipeline.denoise_step = types.MethodType(BagelPipeline.denoise_step, pipeline)
    cfg_group = MagicMock()

    with (
        patch(
            "vllm_omni.diffusion.models.bagel.pipeline_bagel.get_classifier_free_guidance_world_size",
            return_value=3,
        ),
        patch("vllm_omni.diffusion.models.bagel.pipeline_bagel.get_cfg_group", return_value=cfg_group),
    ):
        actual = pipeline.denoise_step(input_batch, states=states)

    assert actual is expected
    call = pipeline.bagel.predict_noise_with_multi_branch_cfg.call_args.kwargs
    assert len(call["branches_kwargs"]) == 2


def test_denoise_step_rejects_three_cfg_branches_on_two_ranks():
    states = [
        _prepare_state("a", kv_len=2, cfg_text_scale=4.0),
        _prepare_state("b", kv_len=3, cfg_text_scale=4.0),
    ]
    input_batch = InputBatch.make_batch(states)

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.od_config.dtype = torch.float32
    pipeline.bagel.parallel_config = _ParallelConfig(cfg_parallel_size=2)
    pipeline._pack_step_generation_inputs = BagelPipeline._pack_step_generation_inputs
    pipeline._build_denoise_kwargs = types.MethodType(BagelPipeline._build_denoise_kwargs, pipeline)
    pipeline._denoise_step_cfg_parallel = types.MethodType(BagelPipeline._denoise_step_cfg_parallel, pipeline)
    pipeline.denoise_step = types.MethodType(BagelPipeline.denoise_step, pipeline)

    with (
        patch(
            "vllm_omni.diffusion.models.bagel.pipeline_bagel.get_classifier_free_guidance_world_size",
            return_value=2,
        ),
        patch("vllm_omni.diffusion.models.bagel.pipeline_bagel.get_cfg_group", return_value=MagicMock()),
        pytest.raises(ValueError, match="requires cfg_parallel_size=3"),
    ):
        pipeline.denoise_step(input_batch, states=states)


@pytest.mark.parametrize("cfg_text_scale", [1.0, 4.0], ids=["no_cfg", "text_cfg"])
def test_step_scheduler_matches_bagel_euler_update(cfg_text_scale: float):
    state = _prepare_state("req", kv_len=2, cfg_text_scale=cfg_text_scale)
    initial = state.latents.clone()
    velocity = torch.full_like(initial, 0.25)
    expected_trajectory = [initial.clone()]

    pipeline = MagicMock()
    pipeline.step_scheduler = types.MethodType(BagelPipeline.step_scheduler, pipeline)
    for step, dt in enumerate(state.extra["bagel_dts"]):
        pipeline.step_scheduler(state, velocity)
        expected = initial - velocity * state.extra["bagel_dts"][: step + 1].sum()
        expected_trajectory.append(expected)
        assert torch.allclose(state.latents, expected)
        assert state.step_index == step + 1

    assert state.denoise_completed
    assert len(state.extra["bagel_trajectory_latents"]) == state.total_steps + 1
    assert len(state.extra["bagel_trajectory_timesteps"]) == state.total_steps
    for actual, expected in zip(state.extra["bagel_trajectory_latents"], expected_trajectory, strict=True):
        assert torch.allclose(actual, expected)


@pytest.mark.parametrize("cfg_text_scale", [1.0, 4.0], ids=["no_cfg", "text_cfg"])
@pytest.mark.parametrize("velocity_dtype", [torch.float32, torch.bfloat16], ids=["fp32_velocity", "bf16_velocity"])
def test_complete_step_updates_match_full_bagel_loop(cfg_text_scale: float, velocity_dtype: torch.dtype):
    state = _prepare_state("req", kv_len=2, cfg_text_scale=cfg_text_scale)
    initial = state.latents.clone()

    bagel = MagicMock()
    bagel._sp_size = 1
    bagel._denoise_schedule_extra_step = True
    bagel.prepare_denoise_schedule = types.MethodType(Bagel.prepare_denoise_schedule, bagel)
    bagel.forward.side_effect = lambda x_t, **_kwargs: torch.full(
        x_t.shape,
        0.25,
        dtype=velocity_dtype,
        device=x_t.device,
    )
    bagel.generate_image = types.MethodType(Bagel.generate_image, bagel)
    generation_input = dict(state.extra["bagel_generation_input"])
    generation_input["packed_init_noises"] = initial.clone()

    with patch(
        "vllm_omni.diffusion.models.bagel.bagel_transformer.get_classifier_free_guidance_world_size",
        return_value=1,
    ):
        full_latents, full_trajectory, full_timesteps, _ = bagel.generate_image(
            **generation_input,
            past_key_values=state.extra["bagel_gen_context"]["past_key_values"],
            num_timesteps=state.extra["bagel_gen_params"].num_timesteps,
            timestep_shift=state.extra["bagel_gen_params"].timestep_shift,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=1.0,
            cfg_text_packed_position_ids=state.extra["bagel_cfg_text_packed_position_ids"],
            cfg_text_past_key_values=state.extra["bagel_cfg_text_context"]["past_key_values"],
            return_trajectory_latents=True,
        )

    pipeline = MagicMock()
    pipeline.step_scheduler = types.MethodType(BagelPipeline.step_scheduler, pipeline)
    velocity = torch.full(initial.shape, 0.25, dtype=velocity_dtype, device=initial.device)
    while not state.denoise_completed:
        pipeline.step_scheduler(state, velocity)

    assert torch.allclose(state.latents, full_latents[0])
    assert state.latents.dtype == full_latents[0].dtype == torch.float32
    assert len(state.extra["bagel_trajectory_latents"]) == len(full_trajectory)
    assert len(state.extra["bagel_trajectory_timesteps"]) == len(full_timesteps)
    for actual, expected in zip(state.extra["bagel_trajectory_latents"], full_trajectory, strict=True):
        assert torch.allclose(actual, expected)


@dataclass
class _SchedulerOutput:
    prev_sample: torch.Tensor
    log_prob: torch.Tensor


class _Scheduler:
    def step(self, velocity, _timestep, latents, dt, **_kwargs):
        return _SchedulerOutput(latents - velocity * dt, torch.tensor(-0.5))


def test_step_scheduler_records_log_probability():
    state = _prepare_state("req", kv_len=2)
    state.scheduler = _Scheduler()
    pipeline = MagicMock()
    pipeline.step_scheduler = types.MethodType(BagelPipeline.step_scheduler, pipeline)

    pipeline.step_scheduler(state, torch.full_like(state.latents, 0.25))

    assert len(state.extra["bagel_trajectory_log_probs"]) == 1
    assert state.extra["bagel_trajectory_log_probs"][0].item() == -0.5


def test_post_decode_reuses_full_path_output_formatter():
    state = _prepare_state("req", kv_len=2)
    expected = object()
    pipeline = MagicMock()
    pipeline._build_image_output.return_value = expected
    pipeline.post_decode = types.MethodType(BagelPipeline.post_decode, pipeline)

    result = pipeline.post_decode(state)

    assert result is expected
    call = pipeline._build_image_output.call_args
    assert call.args == (state.latents, (64, 64))
    assert call.kwargs["trajectory_latents"] is state.extra["bagel_trajectory_latents"]
    assert call.kwargs["trajectory_timesteps"] is state.extra["bagel_trajectory_timesteps"]
    assert call.kwargs["think_text"] is None


def test_nested_cache_merge_keeps_per_request_branch_lengths():
    gen = NaiveCache.merge([_cache(2, 2.0), _cache(3, 3.0)])
    cfg = NaiveCache.merge([_cache(0, 0.0), _cache(4, 4.0)])

    merged = NaiveCache.merge([gen, cfg])

    assert merged.key_values_lens == [2, 3, 0, 4]
    assert merged.key_cache[0].shape[0] == 9
    assert torch.equal(merged.key_cache[0][:2], gen.key_cache[0][:2])
    assert torch.equal(merged.key_cache[0][-4:], cfg.key_cache[0])


class _PositionAddingLanguageModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.denoise_calls = 0

    def forward(
        self,
        *,
        return_embeddings_only: bool = False,
        packed_text_ids: torch.Tensor | None = None,
        packed_query_sequence: torch.Tensor | None = None,
        packed_query_position_ids: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> _LanguageModelOutput:
        if return_embeddings_only:
            assert packed_text_ids is not None
            count = packed_text_ids.numel()
            return _LanguageModelOutput(packed_query_sequence=torch.zeros(count, self.hidden_size))
        self.denoise_calls += 1
        assert packed_query_sequence is not None
        assert packed_query_position_ids is not None
        positions = packed_query_position_ids.to(packed_query_sequence).unsqueeze(-1)
        return _LanguageModelOutput(packed_query_sequence=packed_query_sequence + positions)


class _ZeroEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(values.shape[0], self.hidden_size, device=values.device)


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        {"cfg_text_scale": 3.0},
        {
            "cfg_text_scale": 1.0,
            "cfg_vae_lengths": [2],
            "cfg_text_scales": [3.0],
            "cfg_img_scales": [1.0],
        },
    ],
    ids=["scalar-scale", "per-request-scales"],
)
def test_bagel_forward_skips_cfg_without_branch_contexts(cfg_kwargs):
    bagel = Bagel.__new__(Bagel)
    nn.Module.__init__(bagel)
    bagel.hidden_size = 2
    bagel.use_moe = False
    bagel.language_model = _PositionAddingLanguageModel(hidden_size=2)
    bagel.latent_pos_embed = _ZeroEmbedding(hidden_size=2)
    bagel.time_embedder = _ZeroEmbedding(hidden_size=2)
    bagel.vae2llm = nn.Identity()
    bagel.llm2vae = nn.Identity()

    result = bagel.forward(
        x_t=torch.zeros(2, 2),
        timestep=torch.zeros(2),
        packed_vae_token_indexes=torch.tensor([1, 2]),
        packed_vae_position_ids=torch.zeros(2, dtype=torch.long),
        packed_text_ids=torch.zeros(2, dtype=torch.long),
        packed_text_indexes=torch.tensor([0, 3]),
        packed_position_ids=torch.ones(4, dtype=torch.long),
        packed_seqlens=torch.tensor([4]),
        past_key_values=NaiveCache(1),
        cfg_renorm_min=1.0,
        **cfg_kwargs,
    )

    assert bagel.language_model.denoise_calls == 1
    assert torch.equal(result, torch.ones(2, 2))


def test_bagel_forward_combines_same_cfg_per_request_in_one_model_call():
    bagel = Bagel.__new__(Bagel)
    nn.Module.__init__(bagel)
    bagel.hidden_size = 2
    bagel.use_moe = False
    bagel.language_model = _PositionAddingLanguageModel(hidden_size=2)
    bagel.latent_pos_embed = _ZeroEmbedding(hidden_size=2)
    bagel.time_embedder = _ZeroEmbedding(hidden_size=2)
    bagel.vae2llm = nn.Identity()
    bagel.llm2vae = nn.Identity()

    result = bagel.forward(
        x_t=torch.zeros(4, 2),
        timestep=torch.zeros(4),
        packed_vae_token_indexes=torch.tensor([1, 2, 5, 6]),
        packed_vae_position_ids=torch.zeros(4, dtype=torch.long),
        packed_text_ids=torch.zeros(4, dtype=torch.long),
        packed_text_indexes=torch.tensor([0, 3, 4, 7]),
        packed_position_ids=torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
        packed_seqlens=torch.tensor([4, 4]),
        past_key_values=NaiveCache(1),
        cfg_renorm_min=1.0,
        cfg_text_scale=3.0,
        cfg_img_scale=1.0,
        cfg_branch_pids=[
            torch.tensor([1, 1, 1, 1, 2, 2, 2, 2]),
            torch.zeros(8, dtype=torch.long),
        ],
        cfg_branch_caches=[NaiveCache(1), NaiveCache(1)],
        cfg_vae_lengths=[2, 2],
        cfg_text_scales=[3.0, 3.0],
        cfg_img_scales=[1.0, 1.0],
    )

    assert bagel.language_model.denoise_calls == 1
    assert torch.equal(result[:2], torch.full((2, 2), 3.0))
    assert torch.equal(result[2:], torch.full((2, 2), 6.0))


def test_multi_branch_cfg_combines_same_cfg_for_each_request_independently():
    bagel = Bagel.__new__(Bagel)
    nn.Module.__init__(bagel)
    positive = torch.tensor([[1.0], [1.0], [2.0], [2.0]])
    negative = torch.zeros_like(positive)

    result = bagel.combine_multi_branch_cfg_noise(
        [positive, negative],
        {
            "cfg_text_scale": 3.0,
            "cfg_img_scale": 1.0,
            "cfg_renorm_type": "global",
            "cfg_renorm_min": 1.0,
            "cfg_vae_lengths": [2, 2],
            "cfg_text_scales": [3.0, 3.0],
            "cfg_img_scales": [1.0, 1.0],
        },
    )

    assert torch.equal(result[:2], torch.full((2, 1), 3.0))
    assert torch.equal(result[2:], torch.full((2, 1), 6.0))
