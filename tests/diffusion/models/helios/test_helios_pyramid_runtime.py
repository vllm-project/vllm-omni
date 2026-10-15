# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real scheduler/runner/pyramid transitions with heavy compute isolated."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.models.helios.pipeline_helios import (
    HeliosPipeline,
    get_helios_pre_process_func,
)
from vllm_omni.diffusion.models.helios.scheduling_helios import HeliosScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SmallTransformer:
    """Only transformer arithmetic is isolated, not pipeline state handling."""

    dtype = torch.float32
    config = SimpleNamespace(in_channels=1, patch_size=(1, 2, 2))

    def __init__(self):
        self.batch_rows = []
        self.spatial_shapes = []

    def __call__(self, *, hidden_states, **kwargs):
        self.batch_rows.append(hidden_states.shape[0])
        self.spatial_shapes.append(tuple(hidden_states.shape[-2:]))
        prompt = kwargs["encoder_hidden_states"].mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        history = kwargs["latents_history_short"].mean(dim=(1, 2, 3, 4)).view(-1, 1, 1, 1, 1)
        return (hidden_states * 0.1 + prompt * 0.01 + history * 0.01,)


class _SmallVAE:
    dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.config = SimpleNamespace(z_dim=1, latents_mean=[0.0], latents_std=[1.0])

    def decode(self, latents, **kwargs):
        return (latents,)

    def encode(self, video):
        # Only VAE arithmetic is isolated; use its real posterior/RNG contract.
        mean = video[:, :1, ::4, ::8, ::8]
        posterior = DiagonalGaussianDistribution(torch.cat([mean, torch.zeros_like(mean)], dim=1))
        return SimpleNamespace(latent_dist=posterior)


def _runtime(monkeypatch, max_num_seqs=2):
    config = OmniDiffusionConfig(
        model="helios-runtime-test",
        model_class_name="HeliosPipeline",
        max_num_seqs=max_num_seqs,
        dtype=torch.float32,
        streaming_output=True,
    )
    pipeline = object.__new__(HeliosPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = config
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _SmallTransformer()
    pipeline.vae = _SmallVAE()
    pipeline.scheduler = HeliosScheduler()
    pipeline.is_distilled = False
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8

    def encode_prompt(*, prompt, num_videos_per_prompt=1, **kwargs):
        count = len(prompt) if isinstance(prompt, list) else 1
        return torch.ones((count * num_videos_per_prompt, 2, 3)), None

    monkeypatch.setattr(pipeline, "encode_prompt", encode_prompt)

    runner = DiffusionModelRunner(None, config, torch.device("cpu"))
    runner.pipeline = pipeline
    coordinator = InteractionCoordinator({}, model_class_name="HeliosPipeline")
    runner._interaction_coordinator = coordinator
    pipeline._interaction_coordinator = coordinator
    scheduler = StepScheduler()
    scheduler.initialize(config)
    return config, scheduler, runner


def _request(config, request_id, seed, stage_steps, num_frames=9):
    return get_helios_pre_process_func(config)(
        OmniDiffusionRequest(
            request_id=request_id,
            prompt="a cat",
            sampling_params=OmniDiffusionSamplingParams(
                height=64,
                width=64,
                num_frames=num_frames,
                num_inference_steps=4,
                guidance_scale=1.0,
                seed=seed,
                extra_args={
                    "output_type": "latent",
                    "history_sizes": [2, 1, 1],
                    "num_latent_frames_per_chunk": 3,
                    "is_enable_stage2": True,
                    "pyramid_num_inference_steps_list": stage_steps,
                },
            ),
        )
    )


def _run(monkeypatch, stage_steps, *, delayed_batch, admit_tick=1, seed=65):
    config, scheduler, runner = _runtime(monkeypatch)
    requests = [_request(config, "a", seed, stage_steps)]
    if delayed_batch:
        requests.append(_request(config, "b", 66, stage_steps))
        assert requests[0].batch_compatibility_key == requests[1].batch_compatibility_key
    scheduler.add_request(requests[0])
    completed = {}
    for tick in range(16):
        if delayed_batch and tick == admit_tick:
            scheduler.add_request(requests[1])
        wave = scheduler.schedule()
        if delayed_batch and tick == admit_tick:
            # This is emitted by the real scheduler, not a hand-built wave.
            assert set(wave.scheduled_request_ids) == {"a", "b"}
        result = runner.execute_stepwise(wave)
        assert result.request_ids == wave.scheduled_request_ids
        assert len(result.runner_outputs) == len(wave.scheduled_request_ids)
        if delayed_batch and tick == admit_tick:
            first, second = runner.state_cache["a"], runner.state_cache["b"]
            assert first.scheduler is not second.scheduler
            assert first.extra["generator"] is not second.extra["generator"]
        for output in result.runner_outputs:
            assert output.result is None or output.result.error is None
            if output.finished:
                assert output.request_id not in completed
                assert output.result is not None
                # One 3-frame latent chunk at full 64/8 spatial resolution.
                assert output.result.output.shape == (1, 1, 3, 8, 8)
                assert torch.isfinite(output.result.output).all()
                completed[output.request_id] = output.result.output.clone()
        scheduler.update_from_output(wave, result)
        if not scheduler.has_requests():
            break
    assert set(completed) == {request.request_id for request in requests}
    assert not runner.state_cache
    if delayed_batch and admit_tick == 0:
        assert runner.pipeline.transformer.batch_rows == [2, 2, 2]
        assert runner.pipeline.transformer.spatial_shapes == [(2, 2), (4, 4), (8, 8)]
    return completed


def test_standard_latent_output_skips_vae_in_real_step_runtime(monkeypatch, mocker):
    config, scheduler, runner = _runtime(monkeypatch)
    request = _request(config, "latent", 65, [1, 1, 1])
    request.sampling_params.output_type = "latent"
    request.sampling_params.extra_args.pop("output_type")
    runner.pipeline.vae.config.latents_mean = [3.0]
    runner.pipeline.vae.config.latents_std = [2.0]

    def unexpected_decode(*args, **kwargs):
        pytest.fail("latent output must not invoke VAE.decode")

    monkeypatch.setattr(runner.pipeline.vae, "decode", unexpected_decode)
    step = mocker.spy(runner.pipeline, "step_scheduler")
    scheduler.add_request(request)
    result = None
    for _ in range(3):
        wave = scheduler.schedule()
        result = runner.execute_stepwise(wave)
        scheduler.update_from_output(wave, result)
    assert result is not None
    output = result.get_request_output("latent")
    assert output.finished
    assert output.result.output.shape == (1, 1, 3, 8, 8)
    torch.testing.assert_close(output.result.output, step.call_args.args[0].latents)
    assert not scheduler.has_requests()
    assert not runner.state_cache


def _request_mode_run(
    monkeypatch, seeds, *, num_outputs=1, modality=None, unbatched=False, guidance=1.0, per_output_generators=False
):
    config, _, runner = _runtime(monkeypatch, max_num_seqs=len(seeds))
    config.streaming_output = False
    scheduler = RequestScheduler()
    scheduler.initialize(config)
    if guidance > 1:
        _use_real_prompt_encoding(monkeypatch, runner.pipeline)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("request-mode latent output must not invoke VAE.decode")

    monkeypatch.setattr(runner.pipeline.vae, "decode", unexpected_decode)
    for index, seed in enumerate(seeds):
        request = _request(config, str(index), seed, [1, 1, 1], num_frames=17)
        request.sampling_params.output_type = "latent"
        request.sampling_params.num_outputs_per_prompt = num_outputs
        request.sampling_params.guidance_scale = guidance
        if per_output_generators:
            request.sampling_params.generator = [
                torch.Generator().manual_seed(seed + output_index * 100) for output_index in range(num_outputs)
            ]
        if modality is not None:
            shape = (1, 3, 64, 64) if modality == "image" else (1, 3, 9, 64, 64)
            request.sampling_params.extra_args[modality] = torch.full(shape, float(seed) / 100)
            if unbatched:
                request.sampling_params.extra_args[modality] = request.sampling_params.extra_args[modality].squeeze(0)
        request = get_helios_pre_process_func(config)(request)
        scheduler.add_request(request)
    # Isolate only CPU-unavailable memory telemetry, never the execution path.
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(runner, "_sample_peak_memory_mb", lambda: 0.0)
    wave = scheduler.schedule()
    assert wave.scheduled_request_ids == [str(index) for index in range(len(seeds))]
    result = runner.execute_model_batch(wave, config)
    assert result.request_ids == wave.scheduled_request_ids
    assert all(output.finished and output.result.error is None for output in result.runner_outputs)
    scheduler.update_from_output(wave, result)
    assert not scheduler.has_requests()
    assert runner.pipeline.transformer.batch_rows == [len(seeds) * num_outputs] * (12 if guidance > 1 else 6)
    return result


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("num_outputs", [1, 2])
@pytest.mark.parametrize("unbatched", [False, True], ids=["with_batch_axis", "without_batch_axis"])
def test_conditioned_request_batch_preserves_posterior_seed_and_output_rows(
    monkeypatch, mocker, modality, num_outputs, unbatched
):
    sample = mocker.spy(DiagonalGaussianDistribution, "sample")
    batch = _request_mode_run(monkeypatch, [65, 66], num_outputs=num_outputs, modality=modality, unbatched=unbatched)
    # Two real posterior draws: first-frame plus repeated-image/video chunk.
    assert sample.call_count == 2
    for call in sample.call_args_list:
        posterior = call.args[0]
        assert posterior.mean.shape[0] == 2 * num_outputs
        torch.testing.assert_close(
            posterior.mean[:, 0, 0, 0, 0],
            torch.tensor([0.65] * num_outputs + [0.66] * num_outputs),
        )
        generators = call.kwargs["generator"]
        assert [generator.initial_seed() for generator in generators] == [65] * num_outputs + [66] * num_outputs
    serial_a = _request_mode_run(monkeypatch, [65], num_outputs=num_outputs, modality=modality)
    serial_b = _request_mode_run(monkeypatch, [66], num_outputs=num_outputs, modality=modality)
    frames = 7 if modality == "image" else 9
    for index, serial in enumerate((serial_a, serial_b)):
        actual = batch.get_request_output(str(index)).result.output
        assert actual.shape == (num_outputs, 1, frames, 8, 8)
        torch.testing.assert_close(actual, serial.get_request_output("0").result.output)
    assert not torch.equal(batch.runner_outputs[0].result.output, batch.runner_outputs[1].result.output)


@pytest.mark.parametrize("mode", ["request", "step"])
def test_single_request_multi_output_uses_output_row_generators(mocker, monkeypatch, mode):
    # Qwen-Image collates output generators even for one logical request.
    # CUDA normal draws for shape (2, ...) differ from two (1, ...) draws;
    # preserve that same sampling granularity before real posterior sampling.
    sample = mocker.spy(DiagonalGaussianDistribution, "sample")
    if mode == "request":
        _request_mode_run(monkeypatch, [65], num_outputs=2, modality="image")
    else:
        config, scheduler, runner = _runtime(monkeypatch)
        request = _request(config, "conditioned", 65, [1, 1, 1])
        request.sampling_params.num_outputs_per_prompt = 2
        request.sampling_params.extra_args["image"] = torch.full((3, 64, 64), 0.65)
        scheduler.add_request(get_helios_pre_process_func(config)(request))
        wave = scheduler.schedule()
        result = runner.execute_stepwise(wave)
        assert result.request_ids == ["conditioned"]
        scheduler.update_from_output(wave, result)
    assert sample.call_count == 2
    for call in sample.call_args_list:
        generators = call.kwargs["generator"]
        assert isinstance(generators, list)
        assert len(generators) == 2
        assert generators[0] is generators[1]
        assert [generator.initial_seed() for generator in generators] == [65, 65]


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("per_output_generators", [False, True], ids=["request_seed", "output_generator_list"])
def test_conditioned_step_multi_output_matches_request_mode(monkeypatch, mocker, modality, per_output_generators):
    reference = _request_mode_run(
        monkeypatch, [65], num_outputs=2, modality=modality, per_output_generators=per_output_generators
    )
    config, scheduler, runner = _runtime(monkeypatch)
    config.streaming_output = False
    request = _request(config, "conditioned", 65, [1, 1, 1], num_frames=17)
    request.sampling_params.output_type = "latent"
    request.sampling_params.num_outputs_per_prompt = 2
    if per_output_generators:
        request.sampling_params.generator = [torch.Generator().manual_seed(seed) for seed in (65, 165)]
    shape = (3, 64, 64) if modality == "image" else (3, 9, 64, 64)
    request.sampling_params.extra_args[modality] = torch.full(shape, 0.65)
    request = get_helios_pre_process_func(config)(request)
    sample = mocker.spy(DiagonalGaussianDistribution, "sample")
    scheduler.add_request(request)
    for _ in range(6):
        wave = scheduler.schedule()
        result = runner.execute_stepwise(wave)
        scheduler.update_from_output(wave, result)
    assert sample.call_count == 2
    assert all(call.args[0].mean.shape[0] == 2 for call in sample.call_args_list)
    output = result.get_request_output("conditioned")
    assert output.finished and output.result.error is None
    torch.testing.assert_close(output.result.output, reference.get_request_output("0").result.output)
    assert output.result.output.shape == (2, 1, 7 if modality == "image" else 9, 8, 8)
    assert runner.pipeline.transformer.batch_rows == [2] * 6
    assert not scheduler.has_requests() and not runner.state_cache


def test_fused_string_prompt_cfg_preserves_each_request(monkeypatch):
    serial_a = _request_mode_run(monkeypatch, [65], guidance=5.0)
    serial_b = _request_mode_run(monkeypatch, [66], guidance=5.0)
    batch = _request_mode_run(monkeypatch, [65, 66], guidance=5.0)
    for index, serial in enumerate((serial_a, serial_b)):
        actual = batch.get_request_output(str(index)).result.output
        assert actual.shape == (1, 1, 6, 8, 8)
        torch.testing.assert_close(actual, serial.get_request_output("0").result.output)
    assert not torch.equal(batch.runner_outputs[0].result.output, batch.runner_outputs[1].result.output)


class _SmallTextEncoder:
    dtype = torch.float32

    def __call__(self, ids, mask):
        return SimpleNamespace(last_hidden_state=ids.float().unsqueeze(-1).expand(-1, -1, 3))


def _use_real_prompt_encoding(monkeypatch, pipeline):
    def tokenize(texts, *, max_length, **kwargs):
        # Heavy text components only; production prompt/negative/output expansion stays real.
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        ids[:, 0] = torch.tensor([int(bool(text)) for text in texts])
        mask = torch.zeros_like(ids)
        mask[:, 0] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)

    pipeline.tokenizer = tokenize
    pipeline.text_encoder = _SmallTextEncoder()
    monkeypatch.setattr(pipeline, "encode_prompt", HeliosPipeline.encode_prompt.__get__(pipeline))


@pytest.mark.parametrize("guidance", [1.0, 5.0])
def test_string_prompt_cfg_matches_request_mode_through_real_runtime(monkeypatch, mocker, guidance):
    config, scheduler, runner = _runtime(monkeypatch)
    config.streaming_output = False
    _use_real_prompt_encoding(monkeypatch, runner.pipeline)
    request = _request(config, "cfg", 65, [1, 1, 1])
    request.sampling_params.guidance_scale = guidance
    request.sampling_params.output_type = "latent"
    request = get_helios_pre_process_func(config)(request)
    encode = mocker.spy(runner.pipeline, "encode_prompt")
    prepare = mocker.spy(runner.pipeline, "prepare_encode")
    scheduler.add_request(request)
    for _ in range(3):
        wave = scheduler.schedule()
        step_result = runner.execute_stepwise(wave)
        scheduler.update_from_output(wave, step_result)
    state = prepare.spy_return
    assert state.do_true_cfg is (guidance > 1)
    assert (state.negative_prompt_embeds is not None) is (guidance > 1)
    assert encode.call_args.kwargs["do_classifier_free_guidance"] is (guidance > 1)
    assert runner.pipeline.transformer.batch_rows == [1] * (6 if guidance > 1 else 3)
    assert not scheduler.has_requests() and not runner.state_cache

    request_config, _, request_runner = _runtime(monkeypatch)
    request_config.streaming_output = False
    _use_real_prompt_encoding(monkeypatch, request_runner.pipeline)
    request_scheduler = RequestScheduler()
    request_scheduler.initialize(request_config)
    request = _request(request_config, "cfg", 65, [1, 1, 1])
    request.sampling_params.guidance_scale = guidance
    request.sampling_params.output_type = "latent"
    request = get_helios_pre_process_func(request_config)(request)
    request_scheduler.add_request(request)
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(request_runner, "_sample_peak_memory_mb", lambda: 0.0)
    wave = request_scheduler.schedule()
    request_result = request_runner.execute_model_batch(wave, request_config)
    request_scheduler.update_from_output(wave, request_result)
    assert not request_scheduler.has_requests()
    assert request_runner.pipeline.transformer.batch_rows == [1] * (6 if guidance > 1 else 3)
    torch.testing.assert_close(
        step_result.get_request_output("cfg").result.output,
        request_result.get_request_output("cfg").result.output,
    )


def test_request_scheduler_fuses_helios_forward_and_preserves_each_seed(monkeypatch):
    serial_a = _request_mode_run(monkeypatch, [65], num_outputs=2)
    serial_b = _request_mode_run(monkeypatch, [66], num_outputs=2)
    batch = _request_mode_run(monkeypatch, [65, 66], num_outputs=2)
    for index, serial in enumerate((serial_a, serial_b)):
        actual = batch.get_request_output(str(index)).result.output
        assert actual.shape == (2, 1, 6, 8, 8)
        torch.testing.assert_close(actual, serial.get_request_output("0").result.output)
    assert not torch.equal(batch.runner_outputs[0].result.output, batch.runner_outputs[1].result.output)


def test_nonstreaming_step_completion_returns_all_latent_chunks(monkeypatch):
    serial = _request_mode_run(monkeypatch, [65])
    config, scheduler, runner = _runtime(monkeypatch)
    config.streaming_output = False
    runner.pipeline.vae.config.latents_mean = [3.0]
    runner.pipeline.vae.config.latents_std = [2.0]
    request = _request(config, "all-chunks", 65, [1, 1, 1], num_frames=17)
    request.sampling_params.output_type = "latent"
    scheduler.add_request(request)
    completed: list[RunnerOutput] = []
    for _ in range(6):
        wave = scheduler.schedule()
        result = runner.execute_stepwise(wave)
        assert result.request_ids == ["all-chunks"]
        completed.extend(output for output in result.runner_outputs if output.finished)
        scheduler.update_from_output(wave, result)
    assert len(completed) == 1
    assert completed[0].result.output.shape == (1, 1, 6, 8, 8)
    torch.testing.assert_close(completed[0].result.output, serial.get_request_output("0").result.output)
    assert not scheduler.has_requests()
    assert not runner.state_cache


def test_nonstreaming_video_decodes_each_chunk_and_returns_full_history(monkeypatch, mocker):
    config, scheduler, runner = _runtime(monkeypatch)
    config.streaming_output = False
    runner.pipeline.vae.config.latents_mean = [3.0]
    runner.pipeline.vae.config.latents_std = [2.0]
    request = _request(config, "video", 65, [1, 1, 1], num_frames=17)
    # The standard field takes precedence over the legacy extra_args latent field.
    request.sampling_params.output_type = "np"
    decode = mocker.spy(runner.pipeline.vae, "decode")
    step = mocker.spy(runner.pipeline, "step_scheduler")
    scheduler.add_request(request)
    chunks: list[torch.Tensor] = []
    for _ in range(6):
        wave = scheduler.schedule()
        result = runner.execute_stepwise(wave)
        if decode.call_count > len(chunks):
            model_latents = step.call_args.args[0].extra["history_latents"][:, :, -3:]
            # Wan's checkpoint normalization is x / (1/std) + mean.
            torch.testing.assert_close(decode.call_args.args[0], model_latents * 2 + 3)
            chunks.append(decode.spy_return[0].clone())
        scheduler.update_from_output(wave, result)
    assert decode.call_count == 2
    output = result.get_request_output("video")
    assert output.finished
    torch.testing.assert_close(output.result.output, torch.cat(chunks, dim=2))
    assert output.result.output.shape == (1, 1, 6, 8, 8)
    assert not scheduler.has_requests()
    assert not runner.state_cache


@pytest.mark.parametrize("stage_steps", [[1, 1, 1], [2, 2, 2]], ids=["different_at_admission", "diverge_after_step"])
def test_delayed_pyramid_request_preserves_existing_request(monkeypatch, stage_steps):
    """Admission key equality must not force dynamically different shapes into one tensor.

    Real preparation and scheduler steps create the shapes. The consumer is
    the real runner; an exception, missing result, wrong request ID, or lost
    completion fails this test. Serial parity is a secondary oracle and shares
    pipeline code; explicit request/count/full-resolution invariants supplement it.
    """
    serial = _run(monkeypatch, stage_steps, delayed_batch=False)
    serial_b = _run(monkeypatch, stage_steps, delayed_batch=False, seed=66)
    batched = _run(monkeypatch, stage_steps, delayed_batch=True)
    torch.testing.assert_close(batched["a"], serial["a"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched["b"], serial_b["a"], rtol=1e-5, atol=1e-6)
    assert not torch.equal(batched["a"], batched["b"])


def test_synchronized_pyramid_requests_still_fuse_transformer_calls(monkeypatch):
    """Separating staggered states must preserve fusion for compatible requests."""
    _run(monkeypatch, [1, 1, 1], delayed_batch=True, admit_tick=0)


def test_pyramid_chunk_churn_preserves_wave_order_across_execution_groups(monkeypatch):
    config, scheduler, runner = _runtime(monkeypatch, max_num_seqs=3)
    requests = [_request(config, rid, seed, [2, 2, 2], num_frames=17) for rid, seed in zip("abc", [65, 66, 67])]
    completed = set()
    decoded_chunks: dict[str, list[int]] = {rid: [] for rid in "abc"}
    for tick in range(24):
        if tick in (0, 1, 6):
            scheduler.add_request(requests[{0: 0, 1: 1, 6: 2}[tick]])
        wave = scheduler.schedule()
        previous_calls = len(runner.pipeline.transformer.batch_rows)
        result = runner.execute_stepwise(wave)
        assert result.request_ids == wave.scheduled_request_ids
        assert len(result.runner_outputs) == len(wave.scheduled_request_ids)
        if tick == 7:
            # A and C now share the new chunk's progress, with B between them
            # in Scheduler order. One physical group is [A,C], the other [B].
            assert wave.scheduled_request_ids == ["a", "b", "c"]
            assert runner.pipeline.transformer.batch_rows[previous_calls:] == [2, 1]
        for output in result.runner_outputs:
            assert output.result is None or output.result.error is None
            if output.result is not None:
                decoded_chunks[output.request_id].append(output.result.chunk_index)
            if output.finished:
                assert output.request_id not in completed
                completed.add(output.request_id)
        scheduler.update_from_output(wave, result)
        if tick >= 6 and not scheduler.has_requests():
            break
    assert completed == {"a", "b", "c"}
    assert decoded_chunks == {"a": [0, 1], "b": [0, 1], "c": [0, 1]}
    assert not runner.state_cache
