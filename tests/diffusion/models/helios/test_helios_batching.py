# SPDX-License-Identifier: Apache-2.0
"""Helios request- and step-batching contract tests."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.models.helios.pipeline_helios import (
    HeliosPipeline,
    get_helios_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CountingTransformer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        self.calls += 1
        return (hidden_states + 1.0,)


class _TinyScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0])

    def set_timesteps(self, *args, **kwargs) -> None:
        del args, kwargs
        self.timesteps = torch.tensor([1.0])


def _sampling(**overrides) -> OmniDiffusionSamplingParams:
    params = OmniDiffusionSamplingParams(
        height=32,
        width=32,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
    )
    for name, value in overrides.items():
        setattr(params, name, value)
    return params


def _step_state(request_id: str, value: float) -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=_sampling(seed=int(value)),
        prompt=f"prompt-{request_id}",
        prompt_embeds=torch.full((1, 2, 3), value),
        latents=torch.full((1, 1, 1, 2, 2), value),
        timesteps=torch.tensor([1.0]),
    )
    state.extra.update(
        {
            "batch_size": 1,
            "dtype": torch.float32,
            "attention_kwargs": {},
            "indices_hidden_states": torch.zeros(1, 2, dtype=torch.long),
            "indices_latents_history_short": torch.zeros(1, 2, dtype=torch.long),
            "indices_latents_history_mid": torch.zeros(1, 2, dtype=torch.long),
            "indices_latents_history_long": torch.zeros(1, 2, dtype=torch.long),
            "latents_history_short": torch.zeros(1, 1, 1, 2, 2),
            "latents_history_mid": torch.zeros(1, 1, 1, 2, 2),
            "latents_history_long": torch.zeros(1, 1, 1, 2, 2),
            "is_enable_stage2": False,
            "use_cfg_zero_star": False,
            "guidance_scale": 1.0,
        }
    )
    return state


def _step_pipeline() -> HeliosPipeline:
    pipeline = object.__new__(HeliosPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _CountingTransformer()
    pipeline._current_timestep = None
    pipeline._guidance_scale = 1.0
    return pipeline


def test_helios_step_batch_runs_one_transformer_call_and_preserves_rows() -> None:
    pipeline = _step_pipeline()
    states = [_step_state("request-a", 2.0), _step_state("request-b", 7.0)]

    batch = InputBatch.make_batch(states, idx_mapping=torch.tensor([1, 0]))
    selected_states = [states[1], states[0]]
    noise_pred = pipeline.denoise_step(batch, states=selected_states)

    assert pipeline.transformer.calls == 1
    assert noise_pred.shape == (2, 1, 1, 2, 2)
    assert torch.allclose(noise_pred[0], torch.full_like(noise_pred[0], 8.0))
    assert torch.allclose(noise_pred[1], torch.full_like(noise_pred[1], 3.0))
    assert batch.request_ids == ["request-b", "request-a"]


def test_helios_step_single_request_regression_uses_same_path() -> None:
    pipeline = _step_pipeline()
    state = _step_state("single", 4.0)
    batch = InputBatch.make_batch([state])

    noise_pred = pipeline.denoise_step(batch, states=[state])

    assert pipeline.transformer.calls == 1
    assert torch.allclose(noise_pred, torch.full_like(noise_pred, 5.0))


def test_helios_step_request_churn_keeps_state_and_output_identity() -> None:
    pipeline = _step_pipeline()
    request_a = _step_state("request-a", 2.0)
    request_b = _step_state("request-b", 7.0)
    request_c = _step_state("request-c", 11.0)

    first_batch = InputBatch.make_batch([request_a, request_b])
    first_pred = pipeline.denoise_step(first_batch, states=[request_a, request_b])

    request_c.timesteps = torch.tensor([0.5])
    second_batch = InputBatch.make_batch([request_b, request_c], cached_batch=first_batch)
    second_pred = pipeline.denoise_step(second_batch, states=[request_b, request_c])

    # B and C have different current timesteps, so compatibility grouping
    # correctly executes two groups on the second tick.
    assert pipeline.transformer.calls == 3
    assert second_batch.request_ids == ["request-b", "request-c"]
    assert torch.allclose(first_pred[0], torch.full_like(first_pred[0], 3.0))
    assert torch.allclose(first_pred[1], torch.full_like(first_pred[1], 8.0))
    assert torch.allclose(second_pred[0], torch.full_like(second_pred[0], 8.0))
    assert torch.allclose(second_pred[1], torch.full_like(second_pred[1], 12.0))


def test_helios_step_groups_mixed_stage_and_restores_request_order() -> None:
    pipeline = _step_pipeline()
    stage1 = _step_state("stage-1", 2.0)
    stage2 = _step_state("stage-2", 7.0)
    stage2.extra["is_enable_stage2"] = True
    stage2.extra["stage_index"] = 1

    prediction = pipeline.denoise_step(
        InputBatch.make_batch([stage1, stage2]),
        states=[stage1, stage2],
    )

    assert pipeline.transformer.calls == 2
    assert torch.allclose(prediction[0], torch.full_like(prediction[0], 3.0))
    assert torch.allclose(prediction[1], torch.full_like(prediction[1], 8.0))


def test_helios_step_groups_mixed_cfg_without_input_batch_failure() -> None:
    pipeline = _step_pipeline()
    cfg = _step_state("cfg", 2.0)
    no_cfg = _step_state("no-cfg", 7.0)
    cfg.do_true_cfg = True
    cfg.negative_prompt_embeds = torch.full((1, 2, 3), -1.0)

    def fake_stage1(self, states, latents, timesteps):
        del states, timesteps
        return latents + 1.0

    pipeline._denoise_stage1_step = MethodType(fake_stage1, pipeline)

    prediction = pipeline.denoise_step(
        # InputBatch intentionally rejects mixed CFG scalar settings.  The
        # pipeline grouping contract is exercised directly here so that
        # mixed CFG requests are still isolated when presented by a caller.
        InputBatch.make_batch([cfg]),
        states=[cfg, no_cfg],
    )

    assert prediction.shape[0] == 2
    assert torch.allclose(prediction[:, 0, 0, 0, 0], torch.tensor([3.0, 8.0]))


def test_helios_request_noise_uses_one_generator_per_sample() -> None:
    generators = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(22)]
    batched = HeliosPipeline._rand_per_sample((2,), generator=generators, device=torch.device("cpu"))
    expected = torch.cat(
        [
            torch.rand(1, generator=torch.Generator().manual_seed(11)),
            torch.rand(1, generator=torch.Generator().manual_seed(22)),
        ]
    )

    torch.testing.assert_close(batched, expected)
    assert not torch.equal(batched[0], batched[1])


def test_helios_zero_star_groups_by_stage1_progress() -> None:
    pipeline = _step_pipeline()
    first = _step_state("first", 2.0)
    second = _step_state("second", 2.0)
    for state in (first, second):
        state.do_true_cfg = True
        state.negative_prompt_embeds = torch.full((1, 2, 3), -1.0)
        state.extra.update({"use_cfg_zero_star": True, "use_zero_init": True, "zero_steps": 1})
    second.step_in_chunk = 2

    groups = pipeline._split_step_groups([first, second])

    assert [[state.request_id for state in group] for group in groups] == [["first"], ["second"]]


def test_helios_zero_star_groups_by_stage2_progress() -> None:
    pipeline = _step_pipeline()
    first = _step_state("first", 2.0)
    second = _step_state("second", 2.0)
    for state in (first, second):
        state.extra.update(
            {
                "is_enable_stage2": True,
                "stage_index": 0,
                "stage_step_index": 0,
                "use_cfg_zero_star": True,
                "use_zero_init": True,
                "zero_steps": 1,
            }
        )
    second.extra["stage_step_index"] = 2

    groups = pipeline._split_step_groups([first, second])

    assert [[state.request_id for state in group] for group in groups] == [["first"], ["second"]]


def test_helios_step_batch_uses_production_runner_path(monkeypatch) -> None:
    """Exercise scheduler output -> runner state -> InputBatch -> Helios denoise."""
    pipeline = _step_pipeline()

    def prepare_encode(self, state):
        value = 2.0 if state.request_id == "request-a" else 7.0
        template = _step_state(state.request_id, value)
        state.latents = template.latents
        state.timesteps = template.timesteps
        state.prompt_embeds = template.prompt_embeds
        state.extra.update(template.extra)
        state.total_chunks = 1
        return state

    def step_scheduler(self, state, noise_pred, **kwargs):
        del kwargs
        state.latents = noise_pred
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        return DiffusionOutput(output=state.latents.clone())

    pipeline.prepare_encode = MethodType(prepare_encode, pipeline)
    pipeline.step_scheduler = MethodType(step_scheduler, pipeline)
    pipeline.post_decode = MethodType(post_decode, pipeline)
    pipeline.supports_step_execution = True

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.od_config = SimpleNamespace(
        cache_backend=None,
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        parallel_config=SimpleNamespace(use_hsdp=False),
        streaming_output=False,
    )
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.input_batch = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner._interaction_coordinator = None
    runner.state_cache = {}
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda *args, **kwargs: None,
    )
    runner._sample_peak_memory_mb = lambda: 0.0
    runner._maybe_send_stage_payload = lambda *args, **kwargs: None
    monkeypatch.setattr(model_runner_module, "set_forward_context", lambda **kwargs: _noop_context())
    monkeypatch.setattr(model_runner_module, "supports_interaction_apply", lambda _pipeline: False)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "is_available", lambda: False)

    requests = [_request("request-a"), _request("request-b")]
    scheduler_output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=req.request_id, req=req) for req in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=2,
        num_waiting_reqs=0,
    )

    result = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)

    assert pipeline.transformer.calls == 1
    assert result.request_ids == ["request-a", "request-b"]
    assert torch.allclose(result.get_request_output("request-a").result.output, torch.full((1, 1, 1, 2, 2), 3.0))
    assert torch.allclose(result.get_request_output("request-b").result.output, torch.full((1, 1, 1, 2, 2), 8.0))


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _noop_context():
    return _NoopContext()


def _request(
    request_id: str,
    *,
    prompt: str | dict[str, object] | None = None,
    extra_args: dict | None = None,
    guidance_scale: float | None = 1.0,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt=prompt if prompt is not None else f"prompt-{request_id}",
        sampling_params=_sampling(
            seed=len(request_id),
            guidance_scale=guidance_scale,
            extra_args=extra_args or {},
        ),
    )


def test_helios_preprocess_key_separates_structural_request_options() -> None:
    preprocess = get_helios_pre_process_func(SimpleNamespace())
    request_a = preprocess(_request("a", extra_args={"num_latent_frames_per_chunk": 9}))
    request_b = preprocess(_request("b", extra_args={"num_latent_frames_per_chunk": 5}))

    assert request_a.batch_compatibility_key != request_b.batch_compatibility_key


def test_helios_preprocess_key_separates_effective_cfg_modes() -> None:
    preprocess = get_helios_pre_process_func(SimpleNamespace())
    cfg_request = preprocess(
        _request(
            "cfg",
            prompt={"prompt": "prompt-cfg", "negative_prompt": "bad quality"},
            guidance_scale=None,
        )
    )
    no_cfg_request = preprocess(
        _request(
            "no-cfg",
            prompt={"prompt": "prompt-no-cfg", "negative_prompt": None},
            guidance_scale=None,
        )
    )

    assert cfg_request.batch_compatibility_key != no_cfg_request.batch_compatibility_key


def _batch_pipeline() -> HeliosPipeline:
    pipeline = object.__new__(HeliosPipeline)
    pipeline.device = torch.device("cpu")
    pipeline._guidance_scale = None
    pipeline._current_timestep = None
    pipeline.is_distilled = False
    pipeline.vae_scale_factor_temporal = 1
    pipeline.vae_scale_factor_spatial = 1
    pipeline.transformer = SimpleNamespace(
        dtype=torch.float32,
        config=SimpleNamespace(in_channels=1, patch_size=(1, 1, 1)),
    )
    pipeline.vae = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=SimpleNamespace(latents_mean=[0.0], latents_std=[1.0], z_dim=1),
    )
    pipeline.scheduler = _TinyScheduler()

    def encode_prompt(self, prompt, **kwargs):
        del kwargs
        return torch.arange(len(prompt), dtype=torch.float32).view(len(prompt), 1, 1), None

    def prepare_latents(self, batch_size, *args, **kwargs):
        del args, kwargs
        return torch.zeros(batch_size, 1, 1, 32, 32)

    def stage1_sample(self, latents, prompt_embeds, **kwargs):
        del kwargs
        return latents + prompt_embeds.view(prompt_embeds.shape[0], 1, 1, 1, 1)

    def decode(latents, **kwargs):
        del kwargs
        return (latents,)

    pipeline.encode_prompt = MethodType(encode_prompt, pipeline)
    pipeline.prepare_latents = MethodType(prepare_latents, pipeline)
    pipeline._stage1_sample = MethodType(stage1_sample, pipeline)
    pipeline.vae.decode = decode
    return pipeline


def test_helios_request_batch_forward_is_fused_and_keeps_output_order(monkeypatch) -> None:
    monkeypatch.setattr(current_omni_platform, "empty_cache", lambda: None)
    pipeline = _batch_pipeline()
    request_batch = DiffusionRequestBatch([_request("a"), _request("b")])

    result = pipeline.forward(request_batch, output_type="latent")

    assert [output.output.shape[0] for output in result] == [1, 1]
    assert torch.allclose(result[0].output, torch.zeros_like(result[0].output))
    assert torch.allclose(result[1].output, torch.ones_like(result[1].output))


def test_helios_request_batch_matches_single_request_output(monkeypatch) -> None:
    monkeypatch.setattr(current_omni_platform, "empty_cache", lambda: None)
    single = _batch_pipeline().forward(DiffusionRequestBatch([_request("a")]), output_type="latent")[0]
    batched = _batch_pipeline().forward(DiffusionRequestBatch([_request("a"), _request("b")]), output_type="latent")[0]

    assert torch.allclose(single.output, batched.output)


def test_helios_declares_both_batch_capabilities() -> None:
    assert HeliosPipeline.supports_step_execution is True
    assert HeliosPipeline.supports_request_batch is True
