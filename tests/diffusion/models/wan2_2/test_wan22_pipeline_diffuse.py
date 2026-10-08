# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.attention.layer as layer_mod
from tests.diffusion.attention.test_attention_schedule_candidates import _fake_resolve, _FakeImpl, _make_config
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    AttentionSigmaWindow,
    InvalidAttentionScheduleError,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionScheduleConfig, AttentionSpec
from vllm_omni.diffusion.forward_context import (
    bind_attention_schedule,
    bind_attention_sigma_schedule,
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm_omni.diffusion.media import VideoTensorEncoding, VideoTensorLayout, VideoValueRange
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline, build_wan_scheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanSelfAttention
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _cpu_pipeline_runtime(monkeypatch):
    # These lightweight pipelines have no accelerator or distributed groups.
    module = importlib.import_module("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2")
    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: False)
    monkeypatch.setattr("vllm_omni.diffusion.distributed.parallel_state._PP", SimpleNamespace(world_size=1))


class _StubTransformer(nn.Module):
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


class _StubTextEncoder(nn.Module):
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


class _StubVaeConfig:
    latents_mean = [0.0, 0.0, 0.0, 0.0]
    latents_std = [1.0, 1.0, 1.0, 1.0]
    z_dim = 4


class _StubVae(nn.Module):
    dtype = torch.float32
    config = _StubVaeConfig()

    def decode(self, latents, return_dict=False):
        del return_dict
        batch, _, frames, height, width = latents.shape
        return (torch.zeros(batch, 3, frames, height, width),)


class _StubScheduler:
    def __init__(self, timesteps: list[int]) -> None:
        self.timesteps = torch.tensor(timesteps, dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.set_timesteps_calls: list[tuple[int, torch.device, float | None]] = []

    def set_timesteps(self, num_steps: int, device: torch.device, shift: float | None = None) -> None:
        self.set_timesteps_calls.append((num_steps, device, shift))


@contextmanager
def _noop_progress_bar(*args, **kwargs):
    del args, kwargs

    class _Bar:
        def update(self) -> None:
            return None

    yield _Bar()


def _stub_encode_prompt(
    prompt,
    negative_prompt=None,
    do_classifier_free_guidance=True,
    num_videos_per_prompt=1,
    max_sequence_length=512,
    device=None,
    dtype=None,
):
    del negative_prompt, do_classifier_free_guidance, device, dtype
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    n = batch_size * num_videos_per_prompt
    hidden_size = 8
    prompt_embeds = torch.zeros(n, max_sequence_length, hidden_size)
    return prompt_embeds, None


def _make_pipeline() -> Wan22Pipeline:
    pipeline = object.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _StubTransformer()
    pipeline.transformer_2 = None
    pipeline.text_encoder = _StubTextEncoder()
    pipeline.vae = _StubVae()
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2, 2), in_channels=4, out_channels=4)
    pipeline.scheduler = _StubScheduler([9, 5])
    pipeline.od_config = SimpleNamespace(flow_shift=5.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.boundary_ratio = 0.875
    pipeline.expand_timesteps = False
    pipeline.is_dmd = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.encode_prompt = _stub_encode_prompt  # type: ignore[method-assign]
    pipeline.prepare_latents = lambda **kwargs: torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
    pipeline.progress_bar = _noop_progress_bar
    return pipeline


def _make_sampling(**overrides):
    values: dict[str, object] = {
        "height": None,
        "width": None,
        "num_frames": 1,
        "num_inference_steps": 2,
        "guidance_scale_provided": True,
        "guidance_scale": 1.0,
        "guidance_scale_2": None,
        "guidance_scale_2_provided": False,
        "boundary_ratio": None,
        "generator": None,
        "seed": None,
        "num_outputs_per_prompt": 1,
        "max_sequence_length": 32,
        "latents": None,
        "output_type": "latent",
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("sampling_params_kwargs", "expected_low", "expected_high"),
    [
        ({}, 4.0, 4.0),
        ({"guidance_scale": 0.0}, 0.0, 0.0),
        ({"guidance_scale": 1.0}, 1.0, 1.0),
        ({"guidance_scale": 3.0, "guidance_scale_2": 5.0}, 3.0, 5.0),
    ],
)
def test_forward_delegates_denoising_to_diffuse(
    sampling_params_kwargs: dict[str, float],
    expected_low: float,
    expected_high: float,
) -> None:
    pipeline = _make_pipeline()
    captured: dict[str, object] = {}

    def _fake_diffuse(**kwargs):
        captured.update(kwargs)
        return kwargs["latents"] + 1

    pipeline.diffuse = _fake_diffuse  # type: ignore[method-assign]

    mock_req = OmniDiffusionRequest(
        prompt="prompt",
        request_id="test-req",
        sampling_params=OmniDiffusionSamplingParams(
            num_frames=1,
            num_inference_steps=2,
            max_sequence_length=32,
            output_type="latent",
            **sampling_params_kwargs,
        ),
    )
    batch = DiffusionRequestBatch(requests=[mock_req])

    outputs = pipeline.forward(batch)

    assert len(outputs) == 1
    assert torch.equal(outputs[0].output, torch.ones((1, 4, 1, 8, 8)))
    assert torch.equal(captured["prompt_embeds"], torch.zeros(1, 32, 8))
    assert torch.equal(captured["timesteps"], pipeline.scheduler.timesteps)
    assert captured["guidance_low"] == expected_low
    assert captured["guidance_high"] == expected_high
    assert captured["boundary_timestep"] == pytest.approx(875.0)
    assert captured["latent_condition"] is None
    assert captured["first_frame_mask"] is None
    assert pipeline.scheduler.set_timesteps_calls == [(2, torch.device("cpu"), 5.0)]


@pytest.mark.parametrize("solver", ["unipc", "euler"])
def test_forward_passes_request_shift_without_mutating_scheduler_config(solver: str) -> None:
    pipeline = _make_pipeline()
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]
    for shift in (3.0, 12.0, 5.0):
        sampling = _make_sampling(num_inference_steps=5, extra_args={"sample_solver": solver, "flow_shift": shift})
        request = OmniDiffusionRequest(prompt="prompt", request_id="schedule", sampling_params=sampling)
        pipeline.forward(DiffusionRequestBatch(requests=[request]))
        reference = build_wan_scheduler(solver, shift)
        if solver == "unipc":
            reference.set_timesteps(5, device="cpu", shift=shift)
            assert pipeline.scheduler.config.shift == pipeline.scheduler.config["shift"] == 1.0
        else:
            reference.set_timesteps(5, device="cpu")
        torch.testing.assert_close(pipeline.scheduler.sigmas, reference.sigmas, rtol=0, atol=0)


def test_forward_batches_text_generators_latents_and_splits_outputs() -> None:
    pipeline = _make_pipeline()
    encode_call = {}
    prepare_call = {}

    def _fake_encode_prompt(**kwargs):
        encode_call.update(kwargs)
        batch_size = len(kwargs["prompt"])
        n = batch_size * kwargs["num_videos_per_prompt"]
        return torch.arange(n, dtype=torch.float32).view(n, 1, 1), torch.zeros(n, 1, 1)

    def _fake_prepare_latents(**kwargs):
        prepare_call.update(kwargs)
        return kwargs["latents"]

    pipeline.encode_prompt = _fake_encode_prompt  # type: ignore[method-assign]
    pipeline.prepare_latents = _fake_prepare_latents  # type: ignore[method-assign]
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]  # type: ignore[method-assign]

    gen_a = torch.Generator(device="cpu").manual_seed(1)
    gen_b = torch.Generator(device="cpu").manual_seed(2)
    latents_a = torch.zeros(2, 4, 1, 2, 2)
    latents_b = torch.ones(2, 4, 1, 2, 2)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt": "first", "negative_prompt": "bad first"},
                sampling_params=_make_sampling(
                    generator=gen_a,
                    latents=latents_a,
                    num_outputs_per_prompt=2,
                ),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "negative_prompt": "bad second"},
                sampling_params=_make_sampling(
                    generator=gen_b,
                    latents=latents_b,
                    num_outputs_per_prompt=2,
                ),
            ),
        ]
    )

    outputs = pipeline.forward(batch)

    assert encode_call["prompt"] == ["first", "second"]
    assert encode_call["negative_prompt"] == ["bad first", "bad second"]
    assert prepare_call["batch_size"] == 4
    assert prepare_call["generator"] == [gen_a, gen_a, gen_b, gen_b]
    torch.testing.assert_close(prepare_call["latents"], torch.cat([latents_a, latents_b]))
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].output, latents_a)
    torch.testing.assert_close(outputs[1].output, latents_b)


def test_forward_emits_request_local_typed_media_after_vae_decode() -> None:
    pipeline = _make_pipeline()

    def _fake_diffuse(
        *,
        latents,
        timesteps,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_low,
        guidance_high,
        boundary_timestep,
        dtype,
        attention_kwargs,
        latent_condition,
        first_frame_mask,
        generator,
    ):
        del (
            timesteps,
            prompt_embeds,
            negative_prompt_embeds,
            guidance_low,
            guidance_high,
            boundary_timestep,
            dtype,
            attention_kwargs,
            latent_condition,
            first_frame_mask,
            generator,
        )
        return torch.zeros_like(latents)

    pipeline.diffuse = _fake_diffuse  # type: ignore[method-assign]
    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="prompt",
                request_id="request-0",
                sampling_params=OmniDiffusionSamplingParams(
                    num_frames=1,
                    num_inference_steps=2,
                    max_sequence_length=32,
                    output_type="np",
                ),
            )
        ]
    )

    outputs = pipeline.forward(batch)

    assert len(outputs) == 1
    assert outputs[0].output is None
    assert outputs[0].media is not None
    assert outputs[0].media.prepared_for_transport is False
    assert outputs[0].media.video.tensor.shape == (1, 3, 1, 8, 8)
    assert outputs[0].media.video.spec.layout is VideoTensorLayout.BCTHW
    assert outputs[0].media.video.spec.encoding is VideoTensorEncoding.NORMALIZED_FLOAT
    assert outputs[0].media.video.spec.value_range is VideoValueRange.NEGATIVE_ONE_TO_ONE


def test_forward_keeps_legacy_output_on_non_owner_vae_rank() -> None:
    # Distributed VAE decode uses broadcast_result=False, so non-owner ranks get
    # an empty placeholder instead of the full video. Wrapping that as typed media
    # would fail split_diffusion_output_by_request's batch check on every non-owner
    # rank, so the pipeline must keep the placeholder on the legacy output field.
    pipeline = _make_pipeline()
    pipeline.vae.decode = lambda latents, return_dict=False: (torch.empty(0),)  # type: ignore[assignment]
    pipeline.diffuse = lambda **kwargs: torch.zeros_like(kwargs["latents"])  # type: ignore[method-assign]

    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="prompt",
                request_id="request-0",
                sampling_params=OmniDiffusionSamplingParams(
                    num_frames=1,
                    num_inference_steps=2,
                    max_sequence_length=32,
                    output_type="np",
                ),
            )
        ]
    )

    outputs = pipeline.forward(batch)

    assert len(outputs) == 1
    assert outputs[0].media is None
    assert outputs[0].output is not None
    assert outputs[0].output.numel() == 0


def test_forward_batches_precomputed_prompt_embeddings() -> None:
    pipeline = _make_pipeline()
    diffuse_call = {}
    pipeline.encode_prompt = lambda **kwargs: pytest.fail("text encoder must not run")  # type: ignore[method-assign]
    pipeline.prepare_latents = lambda **kwargs: torch.zeros(kwargs["batch_size"], 4, 1, 2, 2)  # type: ignore[method-assign]

    def _fake_diffuse(**kwargs):
        diffuse_call.update(kwargs)
        return kwargs["latents"]

    pipeline.diffuse = _fake_diffuse  # type: ignore[method-assign]
    embeds_a = torch.zeros(3, 4)
    embeds_b = torch.ones(3, 4)
    negative_a = torch.full((3, 4), 2.0)
    negative_b = torch.full((3, 4), 3.0)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt_embeds": embeds_a, "negative_prompt_embeds": negative_a},
                sampling_params=_make_sampling(guidance_scale=4.0),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt_embeds": embeds_b, "negative_prompt_embeds": negative_b},
                sampling_params=_make_sampling(guidance_scale=4.0),
            ),
        ]
    )

    outputs = pipeline.forward(batch)

    torch.testing.assert_close(diffuse_call["prompt_embeds"], torch.stack([embeds_a, embeds_b]))
    torch.testing.assert_close(diffuse_call["negative_prompt_embeds"], torch.stack([negative_a, negative_b]))
    assert len(outputs) == 2


def test_prepare_latents_with_request_generators_matches_single_generation() -> None:
    pipeline = _make_pipeline()
    kwargs = {
        "num_channels_latents": 4,
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }

    batched = Wan22Pipeline.prepare_latents(
        pipeline,
        batch_size=2,
        generator=[torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)],
        **kwargs,
    )
    singles = torch.cat(
        [
            Wan22Pipeline.prepare_latents(
                pipeline,
                batch_size=1,
                generator=torch.Generator().manual_seed(seed),
                **kwargs,
            )
            for seed in (1, 2)
        ]
    )

    torch.testing.assert_close(batched, singles)
    assert not torch.equal(batched[0], batched[1])


def test_diffuse_runs_prediction_and_scheduler_for_each_timestep() -> None:
    pipeline = _make_pipeline()
    latents = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    timesteps = torch.tensor([7, 3], dtype=torch.int64)
    prompt_embeds = torch.randn(1, 8)

    predict_calls: list[dict[str, object]] = []
    scheduler_calls: list[tuple[float, int, float, bool]] = []

    def _fake_predict_noise_maybe_with_cfg(**kwargs):
        predict_calls.append(kwargs)
        timestep = kwargs["positive_kwargs"]["timestep"]
        assert isinstance(timestep, torch.Tensor)
        return torch.full_like(latents, float(timestep[0].item()))

    def _fake_scheduler_step_maybe_with_cfg(noise_pred, t, current_latents, do_true_cfg):
        scheduler_calls.append(
            (float(noise_pred[0, 0, 0, 0, 0]), int(t.item()), float(current_latents.sum()), do_true_cfg)
        )
        return current_latents + noise_pred

    pipeline.predict_noise_maybe_with_cfg = _fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = _fake_scheduler_step_maybe_with_cfg  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        guidance_low=1.0,
        guidance_high=2.0,
        boundary_timestep=5.0,
        dtype=torch.float32,
        attention_kwargs={},
    )

    assert len(predict_calls) == 2
    assert predict_calls[0]["true_cfg_scale"] == 1.0
    assert predict_calls[1]["true_cfg_scale"] == 2.0
    assert scheduler_calls == [
        (7.0, 7, 0.0, False),
        (3.0, 3, 28.0, False),
    ]
    assert torch.equal(result, torch.full_like(latents, 10.0))


@pytest.mark.parametrize("fail_first_request", [False, True])
def test_runner_publishes_wan_step_context_across_requests(fail_first_request, monkeypatch):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        get_forward_context,
        is_forward_context_available,
        override_forward_context,
    )
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from vllm_omni.quantization.mxfp4_config import _is_w4a8_fallback_step

    contexts: list[ForwardContext] = []
    events = []

    class InjectedPredictionError(Exception):
        pass

    class RecordingTransformer(_StubTransformer):
        def __init__(self, expert):
            super().__init__()
            self.expert = expert

        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            context = get_forward_context()
            assert context is contexts[-1]
            request_index = len(contexts) - 1
            step = context.denoise_step_idx
            positive = bool(encoder_hidden_states[0, 0, 0] > 0)
            events.append((request_index, step, self.expert, positive, _is_w4a8_fallback_step([0, 2])))
            if fail_first_request and request_index == 0 and step == 1 and not positive:
                raise InjectedPredictionError
            return (torch.ones_like(hidden_states) * (1 if positive else -1),)

    def encode_prompt(**kwargs):
        context = get_forward_context()
        assert context.denoise_step_idx is None
        contexts.append(context)
        embeds = torch.ones(1, 8, 8)
        return embeds, -embeds

    pipeline = _make_pipeline()
    pipeline.transformer = RecordingTransformer("high")
    pipeline.transformer_2 = RecordingTransformer("low")
    pipeline.scheduler = _StubScheduler([900, 500, 100])
    monkeypatch.setattr(pipeline, "encode_prompt", encode_prompt)
    monkeypatch.setattr(pipeline, "scheduler_step_maybe_with_cfg", lambda pred, t, latents, cfg: latents)

    # Use the real runner context manager, Wan forward/diffuse, and CFG
    # dispatch. Only model computation, scheduler math and unrelated KV I/O
    # are replaced; no test code publishes a denoise step.
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.od_config = OmniDiffusionConfig(model="", dtype=torch.float32)
    runner.vllm_config = None
    runner.cache_backend = None
    monkeypatch.setattr(runner, "_prepare_request_for_forward", lambda *args, **kwargs: None)
    with override_forward_context(None):
        for request_index in range(2):
            request = OmniDiffusionRequest(
                prompt="a fox walks",
                request_id=f"request-{request_index}",
                sampling_params=OmniDiffusionSamplingParams(
                    num_frames=1, num_inference_steps=3, guidance_scale=2.0, output_type="latent"
                ),
            )

            def execute():
                return runner._execute_request_list(
                    [request],
                    od_config=runner.od_config,
                    allow_single_output=True,
                    require_request_batch_support=False,
                    record_name="test_wan_step_context",
                    record_output_peak_memory=False,
                )

            if fail_first_request and request_index == 0:
                with pytest.raises(InjectedPredictionError):
                    execute()
            else:
                execute()
            assert not is_forward_context_available()

    assert len(contexts) == 2 and contexts[0] is not contexts[1]
    expected = [
        (request, step, "high" if step == 0 else "low", positive, step in (0, 2))
        for request in range(2)
        for step in range(2 if fail_first_request and request == 0 else 3)
        for positive in (True, False)
    ]
    assert events == expected


@pytest.fixture
def fake_attention_backends(monkeypatch):
    # Attention layers resolve fake backends from the supplied config, as in tests/diffusion/attention.
    monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", lambda **kwargs: _fake_resolve(**kwargs))


def _candidate_from_step_2() -> AttentionScheduleConfig:
    return AttentionScheduleConfig(
        profiles={"candidate": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 2, "end": None, "profile": "candidate"}],
    )


def _make_scheduled_runner(pipeline, monkeypatch, schedule_config):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.od_config = OmniDiffusionConfig(model="", dtype=torch.float32)
    # Set after construction so the config does not look up calibration for a model.
    runner.od_config.diffusion_attention_schedule = schedule_config
    runner.vllm_config = None
    runner.cache_backend = None
    monkeypatch.setattr(runner, "_prepare_request_for_forward", lambda *args, **kwargs: None)
    return runner


def _execute_wan_request(runner, request):
    return runner._execute_request_list(
        [request],
        od_config=runner.od_config,
        allow_single_output=True,
        require_request_batch_support=False,
        record_name="test_wan_attention_schedule",
        record_output_peak_memory=False,
    )


@pytest.mark.parametrize("scheduled", [True, False], ids=["inherited-default", "disabled"])
def test_runner_selects_wan_attention_by_actual_denoise_step(scheduled, monkeypatch, fake_attention_backends):
    from vllm_omni.diffusion.forward_context import override_forward_context

    schedule_config = _candidate_from_step_2()
    with set_current_diffusion_config(_make_config(schedule=schedule_config)):
        layer = Attention(num_heads=4, head_size=64, causal=False, softmax_scale=1.0)
    names = {id(layer.attention): "base", id(layer._schedule_candidates["candidate"].impl): "candidate"}

    def selection():
        context = get_forward_context()
        return context.denoise_step_idx, context.total_denoise_steps, names[id(layer.effective_attention()[0])]

    events = []

    class RecordingTransformer(_StubTransformer):
        def __init__(self, expert):
            super().__init__()
            self.expert = expert

        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            positive = bool(encoder_hidden_states[0, 0, 0] > 0)
            events.append((*selection(), self.expert, positive))
            return (torch.zeros_like(hidden_states),)

    def encode_prompt(**kwargs):
        embeds = torch.ones(1, 8, 8)
        return embeds, -embeds

    pipeline = _make_pipeline()
    pipeline.transformer = RecordingTransformer("high")
    pipeline.transformer_2 = RecordingTransformer("low")
    # The request asks for 4 steps, but the scheduler builds 3; the schedule must follow the 3.
    pipeline.scheduler = _StubScheduler([900, 500, 100])
    monkeypatch.setattr(pipeline, "encode_prompt", encode_prompt)
    monkeypatch.setattr(pipeline, "scheduler_step_maybe_with_cfg", lambda pred, t, latents, cfg: latents)
    after_loop = []
    diffuse = pipeline.diffuse

    def diffuse_then_record(**kwargs):
        latents = diffuse(**kwargs)
        after_loop.append(selection())
        return latents

    monkeypatch.setattr(pipeline, "diffuse", diffuse_then_record)
    runner = _make_scheduled_runner(pipeline, monkeypatch, schedule_config)
    request = OmniDiffusionRequest(
        prompt="a fox walks",
        request_id="request-0",
        sampling_params=OmniDiffusionSamplingParams(
            num_frames=1,
            num_inference_steps=4,
            guidance_scale=2.0,
            output_type="latent",
            attention_schedule=None if scheduled else [],
        ),
    )

    with override_forward_context(None):
        _execute_wan_request(runner, request)

    total = 3 if scheduled else None
    last = "candidate" if scheduled else "base"
    # Both CFG passes of a step see the same step and selection. The expert changes at the numeric
    # boundary (875) between steps 0 and 1; the step index does not, so step 1 still uses the baseline.
    assert events == [
        (0, total, "base", "high", True),
        (0, total, "base", "high", False),
        (1, total, "base", "low", True),
        (1, total, "base", "low", False),
        (2, total, last, "low", True),
        (2, total, last, "low", False),
    ]
    # A scheduled run clears the step after the loop, so VAE decode uses the baseline. An unscheduled
    # run leaves the last step in place.
    assert after_loop == [(None, None, "base") if scheduled else (2, None, "base")]


def test_runner_rejects_wan_schedule_past_actual_total_before_denoise(monkeypatch):
    from vllm_omni.diffusion.forward_context import is_forward_context_available, override_forward_context

    calls = []
    pipeline = _make_pipeline()
    pipeline.scheduler = _StubScheduler([900, 500, 100])
    monkeypatch.setattr(pipeline, "predict_noise_maybe_with_cfg", lambda **kwargs: calls.append(kwargs))
    runner = _make_scheduled_runner(pipeline, monkeypatch, _candidate_from_step_2())
    # Steps 3 to 4 fit the 4 requested steps but not the 3 the scheduler builds.
    request = OmniDiffusionRequest(
        prompt="a fox walks",
        request_id="request-0",
        sampling_params=OmniDiffusionSamplingParams(
            num_frames=1,
            num_inference_steps=4,
            output_type="latent",
            attention_schedule=[{"start": 3, "end": 4, "profile": "candidate"}],
        ),
    )

    with override_forward_context(None):
        with pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=3"):
            _execute_wan_request(runner, request)
        assert not is_forward_context_available()

    assert calls == []


class _StubDMDScheduler:
    def __init__(self) -> None:
        self.predict_clean_calls: list[tuple[float, float, float]] = []
        self.add_noise_calls: list[tuple[float, float, float]] = []

    def predict_clean(self, model_output, sample, timestep):
        self.predict_clean_calls.append((float(model_output.mean()), float(sample.mean()), float(timestep)))
        return sample - model_output

    def add_noise(self, clean_sample, noise, timestep):
        self.add_noise_calls.append((float(clean_sample.mean()), float(noise.mean()), float(timestep)))
        return clean_sample + 10.0


@pytest.mark.parametrize("scope", ["no-context", "unbound", "disabled", "step-only", "sigma"])
def test_diffuse_dmd_predicts_clean_and_renoises_between_steps(monkeypatch, scope) -> None:
    pipeline = _make_pipeline()
    pipeline.is_dmd = True
    pipeline.scheduler = _StubDMDScheduler()
    # Only sigma scheduling needs a training-time scale. All other paths must
    # retain C1 behavior with this scheduler, which deliberately has no config.
    if scope == "sigma":
        pipeline.scheduler.config = SimpleNamespace(num_train_timesteps=1000)  # type: ignore[attr-defined]
    latents = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)
    timesteps = torch.tensor([1000.0, 757.0, 522.0])

    pipeline.predict_noise_maybe_with_cfg = lambda **kwargs: torch.ones_like(latents)  # type: ignore[method-assign]
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.randn_tensor",
        lambda *args, **kwargs: torch.full(args[0], 2.0, dtype=kwargs["dtype"]),
    )

    published = []

    def predict(**kwargs):
        published.append(get_forward_context().denoise_sigma if is_forward_context_available() else None)
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = predict
    with (
        set_forward_context() if scope != "no-context" else nullcontext(),
        bind_attention_sigma_schedule((AttentionSigmaWindow(0.0, 1.0, "p"),))
        if scope == "sigma"
        else (bind_attention_sigma_schedule(()) if scope == "disabled" else nullcontext()),
        bind_attention_schedule((AttentionScheduleRange(0, None, "p"),)) if scope == "step-only" else nullcontext(),
    ):
        result = pipeline.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=torch.zeros(1, 8),
            negative_prompt_embeds=None,
            guidance_low=1.0,
            guidance_high=1.0,
            boundary_timestep=None,
            dtype=torch.float32,
            attention_kwargs={},
            generator=torch.Generator(device="cpu").manual_seed(1),
        )
    expected_sigmas: list[float | None] = [1.0, 0.757, 0.522] if scope == "sigma" else [None] * 3
    assert published == expected_sigmas

    assert pipeline.scheduler.predict_clean_calls == [
        (1.0, 0.0, 1000.0),
        (1.0, 9.0, 757.0),
        (1.0, 18.0, 522.0),
    ]
    assert pipeline.scheduler.add_noise_calls == [
        (-1.0, 2.0, 757.0),
        (8.0, 2.0, 522.0),
    ]
    torch.testing.assert_close(result, torch.tensor([[[[[17.0]]]]]))


def _denoise_progress():
    context = get_forward_context()
    return context.denoise_step_idx, context.total_denoise_steps, context.attention_schedule_denoise_active


@pytest.mark.parametrize(
    ("schedule", "fits"),
    [
        ((AttentionScheduleRange(start=2, end=3, profile="candidate"),), True),
        ((AttentionScheduleRange(start=3, end=4, profile="candidate"),), False),
    ],
    ids=["fits-three-steps", "past-three-steps"],
)
def test_dmd_forward_checks_and_publishes_the_fixed_three_step_total(schedule, fits, monkeypatch) -> None:
    pipeline = _make_pipeline()
    pipeline.is_dmd = True
    pipeline.scheduler = _StubDMDScheduler()
    # forward reads num_train_timesteps for the expert boundary.
    monkeypatch.setattr(pipeline.scheduler, "config", SimpleNamespace(num_train_timesteps=1000), raising=False)
    calls: list[tuple[int | None, int | None, bool]] = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        calls.append(_denoise_progress())
        return torch.zeros_like(kwargs["positive_kwargs"]["hidden_states"])

    monkeypatch.setattr(pipeline, "predict_noise_maybe_with_cfg", fake_predict_noise_maybe_with_cfg)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.randn_tensor",
        lambda *args, **kwargs: torch.zeros(args[0], dtype=kwargs["dtype"]),
    )
    # DMD ignores the 40 requested steps and runs its 3 fixed timesteps.
    request = OmniDiffusionRequest(
        prompt="prompt",
        request_id="dmd",
        sampling_params=OmniDiffusionSamplingParams(
            num_frames=1, num_inference_steps=40, max_sequence_length=32, output_type="latent"
        ),
    )

    with set_forward_context(), bind_attention_schedule(schedule):
        if fits:
            pipeline.forward(DiffusionRequestBatch(requests=[request]))
        else:
            with pytest.raises(InvalidAttentionScheduleError, match="exceeds total_steps=3"):
                pipeline.forward(DiffusionRequestBatch(requests=[request]))
        after = _denoise_progress()

    assert calls == ([(0, 3, True), (1, 3, True), (2, 3, True)] if fits else [])
    assert after == (None, None, False)


def _make_gate_loading_pipeline():
    pipeline = Wan22Pipeline.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    gate = WanSelfAttention.__new__(WanSelfAttention)
    nn.Module.__init__(gate)
    gate.to_gate_compress = nn.Linear(1, 1)
    pipeline.gate_holder = gate
    return pipeline, gate


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("pipeline_wan2_2", "Wan22Pipeline"),
        ("pipeline_wan2_2_i2v", "Wan22I2VPipeline"),
        ("pipeline_wan2_2_s2v", "Wan22S2VPipeline"),
        ("pipeline_wan2_2_vace", "Wan22VACEPipeline"),
    ],
)
def test_wan_pipeline_loaders_share_optional_gate_cleanup(monkeypatch, module_name, class_name) -> None:
    module = importlib.import_module(f"vllm_omni.diffusion.models.wan2_2.{module_name}")
    pipeline_cls = getattr(module, class_name)
    pipeline = pipeline_cls.__new__(pipeline_cls)
    expected = {"loaded"}

    def fake_loader(model, weights):
        assert model is pipeline
        assert list(weights) == [("weight", torch.ones(1))]
        return expected

    monkeypatch.setattr(module, "load_wan_weights_with_optional_gate", fake_loader)

    assert pipeline_cls.load_weights(pipeline, iter((("weight", torch.ones(1)),))) is expected


def test_load_weights_removes_unloaded_vsa_gate(monkeypatch) -> None:
    pipeline, gate = _make_gate_loading_pipeline()

    class _Loader:
        def __init__(self, model):
            del model

        def load_weights(self, weights):
            return {name for name, _ in weights}

    monkeypatch.setattr("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.AutoWeightsLoader", _Loader)
    pipeline.load_weights(iter((("other.weight", torch.ones(1)),)))

    assert pipeline.has_gate_compress_weights is False
    assert gate.to_gate_compress is None


def test_load_weights_keeps_trained_vsa_gate(monkeypatch) -> None:
    pipeline, gate = _make_gate_loading_pipeline()
    original_gate = gate.to_gate_compress

    class _Loader:
        def __init__(self, model):
            del model

        def load_weights(self, weights):
            return {name for name, _ in weights}

    monkeypatch.setattr("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.AutoWeightsLoader", _Loader)
    pipeline.load_weights(iter((("gate_holder.to_gate_compress.weight", torch.ones(1)),)))

    assert pipeline.has_gate_compress_weights is True
    assert gate.to_gate_compress is original_gate


class _StubParallelLinear(nn.Module):
    """Stands in for vLLM parallel linear layers, which need a tensor-parallel group."""

    def __init__(self, *args, total_num_heads=None, **kwargs):
        del args, kwargs
        super().__init__()
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_heads
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))


@pytest.mark.parametrize(
    ("baseline", "candidate", "has_gate", "rejected"),
    [
        (None, "FASTVIDEO_VSA", False, True),
        (None, "SDPA", False, False),
        ("FASTVIDEO_VSA", "SDPA", True, False),
        ("FASTVIDEO_VSA", "FASTVIDEO_VSA", True, False),
    ],
    ids=["default-baseline-vsa", "default-baseline-sdpa", "vsa-baseline-sdpa", "vsa-baseline-vsa"],
)
def test_startup_validation_rejects_vsa_candidate_on_self_attention_without_gate(
    baseline, candidate, has_gate, rejected, monkeypatch, fake_attention_backends
) -> None:
    module = importlib.import_module("vllm_omni.diffusion.models.wan2_2.wan2_2_transformer")
    for name in ("QKVParallelLinear", "ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(module, name, _StubParallelLinear)
    monkeypatch.setattr(module, "RMSNorm", nn.Identity)
    monkeypatch.setattr(module, "RotaryEmbeddingWan", nn.Identity)
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    schedule = AttentionScheduleConfig(
        profiles={"candidate": AttentionConfig(default=AttentionSpec(backend=candidate))},
        default=[{"start": 0, "end": None, "profile": "candidate"}],
    )
    baseline_config = None if baseline is None else AttentionConfig(default=AttentionSpec(backend=baseline))
    config = _make_config(baseline=baseline_config, schedule=schedule)
    with set_current_diffusion_config(config):
        attention = WanSelfAttention(dim=256, num_heads=4, head_dim=64, prefix="blocks.0.attn1")

    # The gate is built from the baseline backend only, so a FASTVIDEO_VSA candidate on a layer without
    # it is rejected at startup. A candidate on a layer with the gate, or a dense candidate, is accepted.
    assert (attention.to_gate_compress is not None) is has_gate
    if rejected:
        with pytest.raises(ValueError, match="profile 'candidate' selects backend FASTVIDEO_VSA.*to_gate_compress"):
            layer_mod.validate_attention_schedule_candidates(attention, config)
    else:
        assert layer_mod.validate_attention_schedule_candidates(attention, config) == 1
