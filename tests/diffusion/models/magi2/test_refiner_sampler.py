# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import inspect

import pytest
import torch

from vllm_omni.diffusion.distributed import cfg_parallel
from vllm_omni.diffusion.models.magi2 import sampler_magi2_refiner
from vllm_omni.diffusion.models.magi2.configuration_magi2 import Magi2RefinerConfig
from vllm_omni.diffusion.models.magi2.modeling_magi2_refiner import Magi2RefinerTransformer
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.diffusion.models.magi2.refiner_data_proxy import Magi2RefinerDataProxy
from vllm_omni.diffusion.models.magi2.sampler_magi2_refiner import (
    Magi2RefinerSampler,
    build_magi2_refiner_scheduler,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_GRID = (9, 4, 8)
_COND_LEN, _UNCOND_LEN = 7, 3
_TEXT_CHANNELS = 16
_CONFIG = Magi2RefinerConfig(
    num_layers=2,
    hidden_size=64,
    head_dim=32,
    num_query_groups=1,
    multimodal_layers=(0,),
    text_in_channels=_TEXT_CHANNELS,
)
_TIMESTEPS = [999, 952, 882, 768, 555]
_MODEL_ARGUMENTS = set(inspect.signature(Magi2RefinerTransformer.forward).parameters) - {"self"}


class _FakeRefiner:
    """Elementwise velocity whose offset depends on the text branch; records every call."""

    def __init__(self, text_dependent: bool = True) -> None:
        self.text_dependent = text_dependent
        self.calls: list[dict] = []

    def __call__(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        video, text = kwargs["video"], kwargs["text"]
        offset = text.mean() if self.text_dependent and text.numel() else 0.0
        return torch.tanh(video) * 0.5 + offset


def _latent(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 48, *_GRID, generator=generator)


def _contexts(uncond_len: int = _UNCOND_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    context = torch.randn(1, _COND_LEN, _TEXT_CHANNELS, generator=generator) + 1.0
    context_null = torch.randn(1, uncond_len, _TEXT_CHANNELS, generator=generator) - 1.0
    return context.to(torch.bfloat16), context_null.to(torch.bfloat16)


def _sampler(model) -> Magi2RefinerSampler:
    return Magi2RefinerSampler(model, Magi2RefinerDataProxy(_CONFIG))


def _refiner_model() -> Magi2RefinerTransformer:
    model = Magi2RefinerTransformer(_CONFIG)
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * parameter.shape[-1] ** -0.5)
    return model


class _RecordingScheduler:
    def __init__(self, num_inference_steps: int = 5, shift: float = 5.0) -> None:
        self.inner = build_magi2_refiner_scheduler(num_inference_steps, shift)
        self.timesteps = self.inner.timesteps
        self.steps: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def step(self, velocity, timestep, latent, return_dict=True):
        self.steps.append((velocity, timestep, latent))
        return self.inner.step(velocity, timestep, latent, return_dict=return_dict)


def _branch_velocities(sampler, latent, context, context_null):
    return sampler.predict_noise(latent, context), sampler.predict_noise(latent, context_null)


@pytest.mark.parametrize("uncond_len", [_UNCOND_LEN, 0], ids=["negative-prompt", "empty-negative-prompt"])
def test_each_step_runs_cond_then_uncond_without_a_timestep(monkeypatch: pytest.MonkeyPatch, uncond_len: int) -> None:
    model = _FakeRefiner()
    sampler = _sampler(model)
    context, context_null = _contexts(uncond_len)
    scheduler = _RecordingScheduler()
    monkeypatch.setattr(sampler_magi2_refiner, "build_magi2_refiner_scheduler", lambda *a, **k: scheduler)

    result = sampler.sample(_latent(), context, context_null)

    assert len(model.calls) == 10
    assert [call["text"].shape[0] for call in model.calls] == [_COND_LEN, uncond_len] * 5
    assert [call["window_plan"].text_len for call in model.calls] == [_COND_LEN, uncond_len] * 5
    assert [call["coords"].shape[0] - call["video"].shape[0] for call in model.calls] == [_COND_LEN, uncond_len] * 5
    assert all(set(call) == _MODEL_ARGUMENTS for call in model.calls)
    assert all(call["text"].dtype == torch.float32 for call in model.calls)
    for step, (cond, uncond) in enumerate(zip(model.calls[::2], model.calls[1::2])):
        assert torch.equal(cond["video"], uncond["video"])
        assert torch.equal(cond["text"], context[0].float()) and torch.equal(uncond["text"], context_null[0].float())
        packed = sampler.data_proxy.pack(scheduler.steps[step][2], context)
        assert torch.equal(cond["video"], packed.video)
    assert [int(t) for _, t, _ in scheduler.steps] == _TIMESTEPS
    assert all(v.dtype == torch.float32 and x.dtype == torch.float32 for v, _, x in scheduler.steps)
    assert result.dtype == torch.float32 and result.shape == (1, 48, *_GRID)


@pytest.mark.parametrize("uncond_len", [_UNCOND_LEN, 0], ids=["negative-prompt", "empty-negative-prompt"])
def test_guidance_is_u_plus_two_c_minus_u_without_rescaling(uncond_len: int) -> None:
    sampler = _sampler(_FakeRefiner())
    context, context_null = _contexts(uncond_len)
    latent = _latent()
    cond, uncond = _branch_velocities(sampler, latent, context, context_null)
    scheduler = _RecordingScheduler()

    next_latent = sampler.denoise_step(latent, scheduler.timesteps[0], context, context_null, scheduler)

    ((velocity, timestep, stepped),) = scheduler.steps
    assert sampler.guidance_scale == 2.0
    assert torch.equal(velocity, uncond + 2.0 * (cond - uncond))
    assert stepped is latent and int(timestep) == 999
    # The inputs are chosen so that the shared helper's norm matching would change the result.
    assert not torch.equal(velocity, sampler.cfg_normalize_function(cond, velocity))
    reference = build_magi2_refiner_scheduler()
    assert torch.equal(next_latent, reference.step(uncond + 2.0 * (cond - uncond), 999, latent, return_dict=False)[0])


@pytest.mark.parametrize("uncond_len", [_UNCOND_LEN, 0], ids=["negative-prompt", "empty-negative-prompt"])
def test_sampler_matches_a_hand_written_loop_with_the_refiner_model(uncond_len: int) -> None:
    model = _refiner_model()
    proxy = Magi2RefinerDataProxy(_CONFIG)
    sampler = Magi2RefinerSampler(model, proxy)
    context, context_null = _contexts(uncond_len)
    latent = _latent()

    actual = sampler.sample(latent, context, context_null)

    def velocity(x, text):
        packed = proxy.pack(x, text)
        return proxy.unpack(model(**packed.model_kwargs()), packed.grid)

    expected = latent.clone()
    scheduler = build_magi2_refiner_scheduler()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            c, u = velocity(expected, context), velocity(expected, context_null)
            expected = scheduler.step(u + 2.0 * (c - u), timestep, expected, return_dict=False)[0]
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_predict_noise_runs_in_inference_mode_after_sample() -> None:
    model = _refiner_model()
    sampler = _sampler(model)
    context, context_null = _contexts()
    latent = _latent()
    # sample() runs in inference mode and fills the proxy's caches first.
    sampler.sample(latent, context, context_null, num_inference_steps=1)

    velocity = sampler.predict_noise(latent, context)

    assert torch.is_grad_enabled() and any(p.requires_grad for p in model.parameters())
    assert not velocity.requires_grad
    with torch.no_grad():
        packed = sampler.data_proxy.pack(latent, context)
        expected = sampler.data_proxy.unpack(model(**packed.model_kwargs()), packed.grid)
    assert torch.equal(velocity, expected)


def test_sampler_draws_no_random_numbers_and_keeps_its_input() -> None:
    sampler = _sampler(_FakeRefiner())
    context, context_null = _contexts()
    latent = _latent()
    original = latent.clone()
    state = torch.get_rng_state()

    first = sampler.sample(latent, context, context_null)

    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(latent, original)
    # Each call builds a fresh multistep scheduler, so a repeated request repeats its result.
    assert torch.equal(sampler.sample(latent, context, context_null), first)


def test_sampler_steps_an_upcast_latent_with_the_requested_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeRefiner()
    sampler = _sampler(model)
    context, context_null = _contexts()
    built: list[tuple[int, float]] = []
    schedulers: list[_RecordingScheduler] = []

    def build(num_inference_steps, shift, device=None):
        built.append((num_inference_steps, shift))
        schedulers.append(_RecordingScheduler(num_inference_steps, shift))
        return schedulers[-1]

    monkeypatch.setattr(sampler_magi2_refiner, "build_magi2_refiner_scheduler", build)
    result = sampler.sample(_latent().to(torch.bfloat16), context, context_null, num_inference_steps=3, shift=2.0)

    assert built == [(3, 2.0)]
    (scheduler,) = schedulers
    assert len(scheduler.steps) == 3 and len(model.calls) == 6
    assert all(v.dtype == x.dtype == torch.float32 for v, _, x in scheduler.steps)
    assert torch.equal(scheduler.steps[0][2], _latent().to(torch.bfloat16).float())
    assert result.dtype == torch.float32


def test_denoise_step_is_the_per_step_boundary() -> None:
    sampler = _sampler(_FakeRefiner())
    context, context_null = _contexts()
    steps: list[int] = []
    denoise_step = sampler.denoise_step

    def recording(latent, timestep, *args):
        steps.append(int(timestep))
        return denoise_step(latent, timestep, *args)

    # The pipeline profiler wraps the instance attribute in the same way.
    sampler.denoise_step = recording
    sampler.sample(_latent(), context, context_null)

    assert steps == _TIMESTEPS


class _FakeCfgGroup:
    def __init__(self, rank_tensors: list[torch.Tensor]) -> None:
        self.rank_tensors = rank_tensors
        self.gathered: list[torch.Tensor] = []

    def all_gather(self, input_, dim=0, separate_tensors=False):
        assert separate_tensors
        self.gathered.append(input_)
        return [tensor.clone() for tensor in self.rank_tensors]


@pytest.mark.parametrize("uncond_len", [_UNCOND_LEN, 0], ids=["negative-prompt", "empty-negative-prompt"])
def test_cfg_parallel_ranks_run_one_branch_each_and_agree(monkeypatch: pytest.MonkeyPatch, uncond_len: int) -> None:
    context, context_null = _contexts(uncond_len)
    latent = _latent()
    sequential = _sampler(_FakeRefiner())
    cond, uncond = _branch_velocities(sequential, latent, context, context_null)
    expected = sequential.denoise_step(
        latent, torch.tensor(999), context, context_null, build_magi2_refiner_scheduler()
    )

    for rank in (0, 1):
        model = _FakeRefiner()
        sampler = _sampler(model)
        group = _FakeCfgGroup([cond, uncond])
        monkeypatch.setattr(cfg_parallel, "is_cfg_group_initialized", lambda: True)
        monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_rank", lambda rank=rank: rank)
        monkeypatch.setattr(cfg_parallel, "get_cfg_group", lambda group=group: group)

        actual = sampler.denoise_step(latent, torch.tensor(999), context, context_null, build_magi2_refiner_scheduler())

        assert [call["text"].shape[0] for call in model.calls] == [(_COND_LEN, uncond_len)[rank]]
        assert len(group.gathered) == 1
        assert torch.equal(group.gathered[0], (cond, uncond)[rank])
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("text_len", [_COND_LEN, 0])
def test_ulysses_ranks_run_their_slice_and_gather_the_video(monkeypatch: pytest.MonkeyPatch, text_len: int) -> None:
    context = _contexts()[0][:, :text_len]
    latent = _latent()
    proxy = Magi2RefinerDataProxy(_CONFIG)
    packed = proxy.pack(latent, context)
    full = _FakeRefiner(text_dependent=False)(video=packed.video, text=packed.text)
    world_size = 4

    for rank in range(world_size):
        local = packed.shard(rank, world_size)
        model = _FakeRefiner(text_dependent=False)
        gathered: list[tuple[torch.Tensor, list[int]]] = []

        def fake_gather(tensor, split_sizes, group):
            gathered.append((tensor, list(split_sizes)))
            return full

        group = Magi2ParallelGroup(None, world_size, rank)
        monkeypatch.setattr(sampler_magi2_refiner, "get_magi2_ulysses_group", lambda group=group: group)
        monkeypatch.setattr(sampler_magi2_refiner, "gather_sequence", fake_gather)

        velocity = Magi2RefinerSampler(model, proxy).predict_noise(latent, context)

        (call,) = model.calls
        assert call["split_sizes"] == list(local.split_sizes)
        assert torch.equal(call["video"], local.video) and torch.equal(call["text"], local.text)
        assert torch.equal(call["coords"], local.coords) and torch.equal(call["modality"], local.modality)
        ((local_velocity, video_split_sizes),) = gathered
        assert video_split_sizes == local.video_split_sizes
        if not text_len:
            assert video_split_sizes == list(local.split_sizes)
        assert torch.equal(local_velocity, full[slice(*_video_range(local))])
        assert torch.equal(velocity, proxy.unpack(full, _GRID))


def _video_range(local) -> tuple[int, int]:
    start = sum(local.video_split_sizes[: local.rank])
    return start, start + local.video_split_sizes[local.rank]


def test_guidance_scale_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        Magi2RefinerSampler(_FakeRefiner(), guidance_scale=float("nan"))
