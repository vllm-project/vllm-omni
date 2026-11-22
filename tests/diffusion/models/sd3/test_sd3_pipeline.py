# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn

from tests.diffusion.test_diffusion_step_pipeline import _make_runner, _noop_forward_context
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.models.interface import allows_streaming_step_execution, supports_step_execution
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.utils.pipeline_utils import resolve_pipeline_cls
from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_sd3_sampling(**overrides):
    values = {
        "height": 32,
        "width": 32,
        "num_inference_steps": 2,
        "sigmas": None,
        "max_sequence_length": None,
        "num_outputs_per_prompt": 0,
        "generator": None,
        "latents": None,
        "guidance_scale": 4.0,
        "output_type": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_sd3_pipeline():
    pipeline = object.__new__(StableDiffusion3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._interrupt = False
    pipeline.vae_scale_factor = 8
    pipeline.patch_size = 2
    pipeline.default_sample_size = 128
    pipeline.transformer = SimpleNamespace(in_channels=1)
    return pipeline


def test_sd3_algorithm_adapter_can_disable_inherited_step_execution():
    class RequestOnlySD3(StableDiffusion3Pipeline):
        supports_step_execution = False

    assert supports_step_execution(StableDiffusion3Pipeline)
    assert not allows_streaming_step_execution(StableDiffusion3Pipeline)
    for candidate in (RequestOnlySD3, object.__new__(RequestOnlySD3)):
        assert not supports_step_execution(candidate)
        assert not allows_streaming_step_execution(candidate)


def test_forward_collates_request_prompt_tensors_for_sd3():
    pipeline = _make_sd3_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    encode_calls = []
    diffuse_call = {}

    def _fake_encode_prompt(**kwargs):
        encode_calls.append(kwargs)
        prompt_embeds = kwargs["prompt_embeds"]
        if prompt_embeds is None:
            prompt_embeds = torch.empty(2, 2, 3)
        return prompt_embeds, kwargs.get("pooled_prompt_embeds")

    def _fake_diffuse(**kwargs):
        diffuse_call.update(kwargs)
        raise StopAfterDiffuseError

    pipeline.encode_prompt = _fake_encode_prompt
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(2, 1, 1, 1)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0]), 1)
    pipeline.diffuse = _fake_diffuse

    prompt_embeds_a = torch.zeros(2, 3)
    prompt_embeds_b = torch.ones(2, 3)
    pooled_prompt_embeds_a = torch.full((4,), 2.0)
    pooled_prompt_embeds_b = torch.full((4,), 3.0)
    negative_prompt_embeds_a = torch.full((2, 3), 4.0)
    negative_prompt_embeds_b = torch.full((2, 3), 5.0)
    negative_pooled_prompt_embeds_a = torch.full((4,), 6.0)
    negative_pooled_prompt_embeds_b = torch.full((4,), 7.0)

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="sd3-prompt-a",
                prompt={
                    "prompt": "prompt-a",
                    "negative_prompt": "negative-a",
                    "prompt_embeds": prompt_embeds_a,
                    "pooled_prompt_embeds": pooled_prompt_embeds_a,
                    "negative_prompt_embeds": negative_prompt_embeds_a,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds_a,
                },
                sampling_params=_make_sd3_sampling(),
            ),
            SimpleNamespace(
                request_id="sd3-prompt-b",
                prompt={
                    "prompt": "prompt-b",
                    "negative_prompt": "negative-b",
                    "additional_information": {
                        "prompt_embeds": [prompt_embeds_b],
                        "pooled_prompt_embeds": [pooled_prompt_embeds_b],
                        "negative_prompt_embeds": [negative_prompt_embeds_b],
                        "negative_pooled_prompt_embeds": [negative_pooled_prompt_embeds_b],
                    },
                },
                sampling_params=_make_sd3_sampling(),
            ),
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(batch)

    assert encode_calls[0]["prompt"] is None
    assert encode_calls[0]["prompt_2"] is None
    assert encode_calls[0]["prompt_3"] is None
    torch.testing.assert_close(
        encode_calls[0]["prompt_embeds"],
        torch.stack([prompt_embeds_a, prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        encode_calls[0]["pooled_prompt_embeds"],
        torch.stack([pooled_prompt_embeds_a, pooled_prompt_embeds_b], dim=0),
    )

    assert encode_calls[1]["prompt"] is None
    assert encode_calls[1]["prompt_2"] is None
    assert encode_calls[1]["prompt_3"] is None
    torch.testing.assert_close(
        encode_calls[1]["prompt_embeds"],
        torch.stack([negative_prompt_embeds_a, negative_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        encode_calls[1]["pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        diffuse_call["pooled_prompt_embeds"],
        torch.stack([pooled_prompt_embeds_a, pooled_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        diffuse_call["negative_pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )


def test_encode_prompt_preserves_direct_pooled_prompt_embeds():
    pipeline = _make_sd3_pipeline()
    prompt_embeds = torch.zeros(1, 2, 3)
    pooled_prompt_embeds = torch.ones(1, 4)

    actual_prompt_embeds, actual_pooled_prompt_embeds = pipeline.encode_prompt(
        prompt=None,
        prompt_2=None,
        prompt_3=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
    )

    assert actual_prompt_embeds is prompt_embeds
    assert actual_pooled_prompt_embeds is pooled_prompt_embeds


@pytest.mark.parametrize(
    ("guidance_scale", "expected"),
    [
        pytest.param(4.0, True, id="cfg"),
        pytest.param(1.0, False, id="no-cfg"),
    ],
)
def test_forward_tells_cache_dit_whether_the_request_runs_cfg(
    monkeypatch: pytest.MonkeyPatch,
    guidance_scale: float,
    expected: bool,
):
    """Regression test for https://github.com/vllm-project/vllm-omni/issues/8734."""
    pipeline = _make_sd3_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    def _fake_diffuse(**kwargs):
        raise StopAfterDiffuseError

    pipeline.encode_prompt = lambda **kwargs: (torch.empty(1, 2, 3), torch.empty(1, 4))
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 1, 1)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0]), 1)
    pipeline.diffuse = _fake_diffuse

    calls = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.sd3.pipeline_sd3.set_separate_cfg",
        lambda transformer, do_true_cfg: calls.append((transformer, do_true_cfg)),
    )

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="sd3-cache-dit-cfg",
                prompt="a prompt",
                sampling_params=_make_sd3_sampling(guidance_scale=guidance_scale),
            )
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(batch)

    assert calls == [(pipeline.transformer, expected)]


def _make_step_sd3_pipeline():
    pipeline = _make_sd3_pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.od_config = OmniDiffusionConfig(dtype=torch.float32)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler()

    def predict(do_cfg, scale, positive, negative, normalize):
        def branch(values):
            condition = values["encoder_hidden_states"].mean(dim=(1, 2))
            pooled = values["pooled_projections"].mean(dim=1)
            timestep = values["timestep"].reshape(-1, 1, 1, 1)
            return values["hidden_states"] * 0.1 + timestep * 0.001 + (condition + pooled).reshape(-1, 1, 1, 1)

        conditional = branch(positive)
        if not do_cfg:
            return conditional
        unconditional = branch(negative)
        return unconditional + scale * (conditional - unconditional)

    pipeline.predict_noise_maybe_with_cfg = predict
    return pipeline


def _make_step_sd3_requests(guidance_scale):
    requests = []
    for index in range(2):
        sampling = OmniDiffusionSamplingParams(
            height=16,
            width=16,
            num_inference_steps=3,
            guidance_scale=guidance_scale,
            output_type="latent",
            latents=torch.full((1, 1, 2, 2), float(index)),
        )
        requests.append(
            OmniDiffusionRequest(
                request_id=f"step-{index}",
                sampling_params=sampling,
                prompt={
                    "prompt_embeds": torch.full((2, 3), float(index + 1)),
                    "pooled_prompt_embeds": torch.full((4,), float(index + 2)),
                    "negative_prompt_embeds": torch.full((2, 3), -float(index + 1)),
                    "negative_pooled_prompt_embeds": torch.full((4,), -float(index + 2)),
                },
            )
        )
    return requests


@pytest.mark.parametrize("guidance_scale", [1.0, 4.0])
def test_step_execution_matches_full_loop_with_reordered_requests(monkeypatch, guidance_scale):
    pipeline = _make_step_sd3_pipeline()
    monkeypatch.setattr("vllm_omni.diffusion.models.sd3.pipeline_sd3.set_separate_cfg", lambda *args: None)
    requests = _make_step_sd3_requests(guidance_scale)
    expected = pipeline.forward(DiffusionRequestBatch(requests=copy.deepcopy(requests)))
    states = [
        StepRequestState(request_id=req.request_id, sampling=req.sampling_params, prompt=req.prompt)
        for req in copy.deepcopy(requests)
    ]

    def advance(step_states):
        order = torch.arange(len(step_states) - 1, -1, -1)
        batch = InputBatch.make_batch(step_states, idx_mapping=order)
        predictions = pipeline.denoise_step(batch)
        for state, prediction in zip(batch.states, predictions.split(1), strict=True):
            pipeline.step_scheduler(state, prediction)

    pipeline.prepare_encode(states[0])
    advance(states[:1])
    pipeline.prepare_encode(states[1])
    assert states[0].scheduler is not states[1].scheduler
    assert all(state.scheduler is not pipeline.scheduler for state in states)
    for _ in range(2):
        advance(states)
    advance(states[1:])
    for state, output in zip(states, expected, strict=True):
        assert state.denoise_completed and state.step_index == 3
        torch.testing.assert_close(pipeline.post_decode(state).output, output.output, rtol=0, atol=0)


def _sd3_step_runtime(monkeypatch, *, step=True, cache_backend=None):
    config = OmniDiffusionConfig(
        model_class_name="StableDiffusion3Pipeline",
        step_execution=step,
        max_num_seqs=2,
        output_type="latent",
        cache_backend=cache_backend,
    )
    engine = object.__new__(DiffusionEngine)
    engine.od_config = config
    engine._init_process_hooks(config)
    engine.execution_mode = engine._resolve_execution_mode(config)
    engine._init_scheduler(config)
    engine._init_runtime_state()

    runner = _make_runner()
    runner.od_config = config
    pipeline = runner.pipeline = _make_step_sd3_pipeline()
    pipeline._get_clip_prompt_embeds = lambda prompt, **_: (torch.ones(len(prompt), 4, 2), torch.ones(len(prompt), 2))
    pipeline._get_t5_prompt_embeds = lambda prompt, **kw: torch.ones(len(prompt), kw["max_sequence_length"], 4)
    monkeypatch.setattr("vllm_omni.diffusion.models.sd3.pipeline_sd3.set_separate_cfg", lambda *_: None)
    monkeypatch.setattr(runner_module, "set_forward_context", _noop_forward_context)
    for name in ("reset_peak_memory_stats", "max_memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(runner_module.current_omni_platform, name, lambda: 0)
    engine.executor = SimpleNamespace(
        collective_rpc=lambda method, args: getattr(runner, method)(*args),
    )
    return engine, runner


@pytest.mark.parametrize(
    ("conditioning", "lengths", "guidance", "shares_wave"),
    [
        ("text", (64, 128), 4.0, False),
        ("text", (64, 128), 1.0, False),
        ("text", (64, 64), 4.0, True),
        ("text", (None, 256), 1.0, True),
        ("embeddings", (2, 3), 4.0, False),
        ("embeddings", (2, 2), 1.0, True),
        ("negative", (2, 3), 4.0, False),
        ("negative", (2, 3), 1.0, True),
    ],
)
def test_sd3_admission_batches_compatible_conditions(monkeypatch, conditioning, lengths, guidance, shares_wave):
    engine, runner = _sd3_step_runtime(monkeypatch)
    requests = _make_step_sd3_requests(guidance)
    for index, (request, length) in enumerate(zip(requests, lengths, strict=True)):
        if conditioning == "text":
            # Keep text identical for the differing-length case. Different text
            # at the same configured length still has the same padded layout.
            request.prompt = f"prompt {index}" if shares_wave else "same text"
            request.sampling_params.max_sequence_length = length
        elif conditioning == "negative":
            request.prompt = {
                "prompt": "same text",
                "negative_prompt_embeds": torch.full((length, 4), float(-index - 1)),
                "negative_pooled_prompt_embeds": request.prompt["negative_pooled_prompt_embeds"],
            }
        else:
            for field in ("prompt_embeds", "negative_prompt_embeds"):
                request.prompt[field] = request.prompt[field][:1].expand(length, -1)
            if index:
                request.prompt = {"additional_information": {k: [v] for k, v in request.prompt.items()}}

    expected = {
        req.request_id: runner.pipeline.forward(DiffusionRequestBatch(requests=[copy.deepcopy(req)]))[0].output
        for req in requests
    }
    a, b = [req.request_id for req in requests]
    traces: dict[str, list[int]] = {a: [], b: []}

    def advance():
        wave = engine.scheduler.schedule()
        outputs = runner.execute_stepwise(wave)
        engine.scheduler.update_from_output(wave, outputs)
        for rid in wave.scheduled_request_ids:
            output = outputs.get_request_output(rid)
            traces[rid].append(output.step_index)
            if output.finished:
                assert engine.scheduler.get_request_state(rid).status == DiffusionRequestStatus.FINISHED_COMPLETED
                result = engine._finalize_finished_request(rid, output)
                torch.testing.assert_close(result.output, expected[rid], atol=0, rtol=0)
        return set(wave.scheduled_request_ids)

    assert engine.add_request(requests[0]) == a
    assert advance() == {a}
    assert engine.add_request(requests[1]) == b
    assert advance() == ({a, b} if shares_wave else {a})
    for _ in range(4 if not shares_wave else 2):
        advance()
    assert traces == {a: [1, 2, 3], b: [1, 2, 3]}


def test_sd3_invalid_pooled_condition_is_rejected_without_affecting_active_request(monkeypatch):
    engine, runner = _sd3_step_runtime(monkeypatch)
    a, b = _make_step_sd3_requests(4.0)
    aid = engine.add_request(a)
    wave = engine.scheduler.schedule()
    engine.scheduler.update_from_output(wave, runner.execute_stepwise(wave))

    b.prompt.pop("negative_pooled_prompt_embeds")
    with pytest.raises(OmniClientError, match="negative_pooled_prompt_embeds"):
        engine.add_request(b)
    assert engine.scheduler.get_request_state(b.request_id) is None

    for step in (2, 3):
        wave = engine.scheduler.schedule()
        assert wave.scheduled_request_ids == [aid]
        outputs = runner.execute_stepwise(wave)
        assert outputs.get_request_output(aid).step_index == step
        engine.scheduler.update_from_output(wave, outputs)
    assert engine.scheduler.get_request_state(aid).status == DiffusionRequestStatus.FINISHED_COMPLETED


def test_sd3_request_scheduler_keeps_its_existing_batch_key(monkeypatch):
    engine, _ = _sd3_step_runtime(monkeypatch, step=False)
    requests = _make_step_sd3_requests(4.0)
    for request in requests:
        engine.add_request(request)
        assert request.batch_compatibility_key is None
    assert len(engine.scheduler.schedule().scheduled_request_ids) == 2


def test_sd3_explicit_step_keeps_unsupported_config_guards(monkeypatch):
    engine, runner = _sd3_step_runtime(monkeypatch, cache_backend="tea_cache")
    engine.add_request(_make_step_sd3_requests(1.0)[0])
    with pytest.raises(ValueError, match="Step mode does not support cache_backend"):
        runner.execute_stepwise(engine.scheduler.schedule())

    config = OmniDiffusionConfig(
        model_class_name="StableDiffusion3Pipeline",
        step_execution=True,
        parallel_config={"use_hsdp": True, "hsdp_data_parallel": True, "hsdp_shard_size": 2},
    )
    with pytest.raises(ValueError, match="hsdp_data_parallel currently supports request execution only"):
        engine._resolve_execution_mode(config)


def test_sd3_abort_removes_only_the_aborted_request(monkeypatch):
    engine, runner = _sd3_step_runtime(monkeypatch)
    a, b = _make_step_sd3_requests(4.0)
    first = engine.add_request(a)
    wave = engine.scheduler.schedule()
    engine.scheduler.update_from_output(wave, runner.execute_stepwise(wave))
    second = engine.add_request(b)
    wave = engine.scheduler.schedule()
    assert set(wave.scheduled_request_ids) == {first, second}
    engine.scheduler.update_from_output(wave, runner.execute_stepwise(wave))

    engine.abort(first)
    engine._process_aborts_queue()
    assert engine.scheduler.get_request_state(first).status == DiffusionRequestStatus.FINISHED_ABORTED
    assert first not in runner.state_cache
    for _ in range(2):
        wave = engine.scheduler.schedule()
        assert wave.scheduled_request_ids == [second]
        outputs = runner.execute_stepwise(wave)
        engine.scheduler.update_from_output(wave, outputs)
    output = outputs.get_request_output(second)
    assert output.finished and output.step_index == 3 and output.result.error is None
    assert engine.scheduler.get_request_state(second).status == DiffusionRequestStatus.FINISHED_COMPLETED


@pytest.mark.parametrize(
    ("model", "overrides", "step"),
    [
        ("StableDiffusion3Pipeline", {}, False),
        ("StableDiffusion3Pipeline", {"step_execution": False, "cache_backend": "tea_cache"}, False),
        (
            "StableDiffusion3Pipeline",
            {
                "step_execution": False,
                "parallel_config": {"use_hsdp": True, "hsdp_data_parallel": True, "hsdp_shard_size": 2},
            },
            False,
        ),
        ("StableDiffusion3Pipeline", {"step_execution": True}, True),
        ("QwenImagePipeline", {}, True),
    ],
)
@pytest.mark.parametrize("scope", ["engine", "runner"])
def test_streaming_engine_and_runner_agree_on_execution_mode(monkeypatch, model, overrides, step, scope):
    def new_config():
        return OmniDiffusionConfig(model_class_name=model, streaming_output=True, enforce_eager=True, **overrides)

    if scope == "engine":
        engine = object.__new__(DiffusionEngine)
        mode = engine._resolve_execution_mode(new_config())
        assert mode == (DiffusionExecutionMode.STEP_BATCH if step else DiffusionExecutionMode.REQUEST_BATCH)
        return

    # Give the runner an independent config so the engine's mutation cannot
    # hide a different policy at model load time.
    runner = _make_runner()
    runner.od_config = new_config()
    pipeline = object.__new__(resolve_pipeline_cls(runner.od_config))
    nn.Module.__init__(pipeline)
    monkeypatch.setattr(
        runner_module, "DiffusersPipelineLoader", lambda *a, **kw: SimpleNamespace(load_model=lambda **_: pipeline)
    )
    monkeypatch.setattr(runner_module, "DeviceMemoryProfiler", lambda: nullcontext(SimpleNamespace(consumed_memory=0)))
    monkeypatch.setattr(runner_module.current_omni_platform, "init_diffusion_model_runner_runtime", lambda **_: None)
    monkeypatch.setattr(runner_module, "enable_offload_backend", lambda cfg, pipe, **_: (pipe, None))
    cache = SimpleNamespace(enable=lambda model: None)
    monkeypatch.setattr(runner_module, "get_cache_backend", lambda name, _: cache if name == "tea_cache" else None)

    runner.load_model()
    assert runner.od_config.step_execution is step
    if runner.od_config.cache_backend == "tea_cache":
        assert runner.cache_backend is cache
