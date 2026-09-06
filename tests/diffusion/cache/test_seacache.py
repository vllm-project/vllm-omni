# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.seacache import (
    SeaCacheBackend,
    SeaCacheConfig,
    SeaCacheRootHook,
    apply_sea_cache_hook,
)
from vllm_omni.diffusion.cache.seacache.sea_filter import (
    apply_sea_filter,
    extrapolate_residual,
    indicator_distance,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TinyCosmos3Transformer(torch.nn.Module):
    """Small model that implements the SeaCache forward-control contract."""

    def _run_gen_layers(self, hidden_gen: torch.Tensor) -> torch.Tensor:
        return hidden_gen * 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        video_shape: tuple[int, int, int] | None = None,
        noisy_frame_mask: torch.Tensor | None = None,
        control_latents: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, text_ids, text_mask, video_shape, noisy_frame_mask
        controls = (
            []
            if control_latents is None
            else [control_latents]
            if isinstance(control_latents, torch.Tensor)
            else list(control_latents)
        )
        inputs = [*controls, hidden_states]
        gen_input = torch.cat(
            [value.movedim(1, -1).flatten(1, 3) for value in inputs],
            dim=1,
        )
        residual = getattr(self, "_seacache_residual", None)
        if getattr(self, "_seacache_skip", False) and isinstance(residual, torch.Tensor):
            return gen_input + residual
        output = self._run_gen_layers(gen_input)
        if getattr(self, "_seacache_record", False):
            self._seacache_last_residual = output - gen_input
        return output


class Cosmos3OmniDiffusersPipeline:
    def __init__(self) -> None:
        self.transformer = TinyCosmos3Transformer()
        self._current_step_index: int | None = None
        self._current_sigma: float | None = None
        self._num_timesteps: int | None = None

    @property
    def current_step_index(self) -> int | None:
        return self._current_step_index

    @property
    def current_sigma(self) -> float | None:
        return self._current_sigma

    @property
    def num_timesteps(self) -> int | None:
        return self._num_timesteps


def _latent(value: float) -> torch.Tensor:
    return torch.full((1, 2, 2, 2, 2), value)


def _run_step(
    transformer: TinyCosmos3Transformer,
    timestep: int,
    value: float,
    *,
    hook: SeaCacheRootHook | None = None,
    context: str = "cond",
    control: float | None = None,
    noisy_frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    controls = None if control is None else [_latent(control)]
    cache_context = hook.cache_context(context) if hook is not None else nullcontext()
    with torch.inference_mode(), cache_context:
        return transformer(
            hidden_states=_latent(value),
            timestep=torch.tensor([timestep]),
            noisy_frame_mask=noisy_frame_mask,
            control_latents=controls,
        )


def _apply_test_hook(
    transformer: TinyCosmos3Transformer,
    metadata: SimpleNamespace,
    config: SeaCacheConfig | None = None,
) -> SeaCacheRootHook:
    return apply_sea_cache_hook(
        transformer,
        config or SeaCacheConfig(threshold=100.0),
        current_step_callback=lambda: metadata.step,
        current_sigma_callback=lambda: metadata.sigma,
        num_inference_steps_callback=lambda: metadata.num_steps,
    )


def test_config_validation() -> None:
    assert SeaCacheConfig().threshold == 0.25
    with pytest.raises(ValueError, match="residual_order"):
        SeaCacheConfig(residual_order=-1)
    with pytest.raises(ValueError, match="max_consecutive_cached"):
        SeaCacheConfig(max_consecutive_cached=-1)


def test_sea_filter_matches_reference_equation() -> None:
    hidden = torch.randn(3, 4, 5, 2, dtype=torch.float32)
    sigma = 0.4
    power_exp = 3.0

    spectrum = torch.fft.fftn(hidden, dim=(0, 1, 2))
    gain = None
    for axis in (0, 1, 2):
        frequencies = torch.fft.fftfreq(hidden.shape[axis], dtype=torch.float32)
        clean_power = 1.0 / (frequencies.abs().pow(power_exp) + 1e-16)
        axis_gain = (1.0 - sigma) * clean_power / ((1.0 - sigma) ** 2 * clean_power + sigma**2 + 1e-16)
        shape = [1] * hidden.ndim
        shape[axis] = hidden.shape[axis]
        gain = axis_gain.reshape(shape) if gain is None else gain * axis_gain.reshape(shape)
    assert gain is not None
    gain = gain / gain.mean()
    expected = torch.fft.ifftn(spectrum * gain, dim=(0, 1, 2)).real

    torch.testing.assert_close(
        apply_sea_filter(hidden, sigma=sigma, power_exp=power_exp),
        expected,
    )


def test_indicator_distance_and_linear_extrapolation() -> None:
    previous = [torch.ones(2, 2)]
    current = [torch.full((2, 2), 1.5)]
    assert indicator_distance(current, previous) == pytest.approx(0.5)
    assert indicator_distance([torch.ones(3)], previous) == float("inf")

    history = [
        (0, torch.full((2, 2), 2.0)),
        (2, torch.full((2, 2), 6.0)),
    ]
    torch.testing.assert_close(
        extrapolate_residual(history, step=3, order=1),
        torch.full((2, 2), 8.0),
    )
    torch.testing.assert_close(
        extrapolate_residual(history, step=3, order=0),
        torch.full((2, 2), 6.0),
    )
    quadratic_history = [
        (0, torch.zeros(2, 2)),
        (1, torch.ones(2, 2)),
        (2, torch.full((2, 2), 4.0)),
    ]
    torch.testing.assert_close(
        extrapolate_residual(quadratic_history, step=3, order=2),
        torch.full((2, 2), 9.0),
    )


def test_hook_skips_middle_steps_and_forces_endpoints() -> None:
    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, sigma=1.0, num_steps=4)
    hook = _apply_test_hook(
        transformer,
        metadata,
        SeaCacheConfig(threshold=100.0, max_consecutive_cached=2),
    )
    hook.refresh(transformer)

    for step, timestep in enumerate((1000, 750, 500, 250)):
        metadata.step = step
        metadata.sigma = timestep / 1000
        _run_step(transformer, timestep, 1.0 - step * 0.01, hook=hook)

    assert hook.full_count == 2
    assert hook.skip_count == 2
    assert [step for step, _ in hook.state_manager._states["cond"].history] == [0, 3]


def test_parameter_sharded_hook_skips_when_all_world_ranks_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.distributed import parallel_state

    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, sigma=1.0, num_steps=3)
    hook = _apply_test_hook(transformer, metadata)
    hook._parameter_sharded = True
    world_group = object()
    reduced_groups: list[object] = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(world_size=2, device_group=world_group),
    )

    def all_reduce(decision, *, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        reduced_groups.append(group)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    _run_step(transformer, 1000, 1.0, hook=hook)
    metadata.step = 1
    metadata.sigma = 0.5
    _run_step(transformer, 500, 0.99, hook=hook)

    assert hook.full_count == 1
    assert hook.skip_count == 1
    assert reduced_groups == [world_group, world_group]


def test_parameter_sharded_peer_forces_full_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.distributed import parallel_state

    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, sigma=1.0, num_steps=3)
    hook = _apply_test_hook(transformer, metadata)
    hook._parameter_sharded = True

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(world_size=2, device_group=object()),
    )
    reduce_count = 0

    def all_reduce(decision, *, op, group):
        nonlocal reduce_count
        del op, group
        reduce_count += 1
        if reduce_count == 2:
            decision.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    _run_step(transformer, 1000, 1.0, hook=hook)
    metadata.step = 1
    metadata.sigma = 0.5
    _run_step(transformer, 500, 0.99, hook=hook)

    assert hook.full_count == 2
    assert hook.skip_count == 0
    assert hook.state_manager._states["cond"].accumulated_distance == 0.0


def test_parameter_sharded_hook_fails_open_without_distributed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = SeaCacheRootHook(SeaCacheConfig())
    hook._parameter_sharded = True
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    assert hook._synchronize_compute(False, torch.device("cpu")) is True


def test_hook_keeps_three_transfer_branches_separate() -> None:
    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, sigma=1.0, num_steps=3)
    hook = _apply_test_hook(transformer, metadata)
    hook.refresh(transformer)

    contexts = ("cond", "cond_no_control", "uncond")
    for step, (timestep, value) in enumerate(((1000, 1.0), (500, 0.9))):
        metadata.step = step
        metadata.sigma = timestep / 1000
        _run_step(transformer, timestep, value, hook=hook, context=contexts[0], control=0.5)
        _run_step(transformer, timestep, value, hook=hook, context=contexts[1])
        _run_step(transformer, timestep, value, hook=hook, context=contexts[2], control=0.5)

    assert set(hook.state_manager._states) == set(contexts)
    assert hook.full_count == 3
    assert hook.skip_count == 3
    assert len(hook.state_manager._states["cond"].previous_indicator) == 2
    assert len(hook.state_manager._states["cond_no_control"].previous_indicator) == 1


def test_hook_fails_open_without_noisy_vision() -> None:
    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, sigma=1.0, num_steps=3)
    hook = _apply_test_hook(transformer, metadata)
    hook.refresh(transformer)
    all_clean = torch.zeros(1, 1, 2, 1, 1)

    _run_step(transformer, 1000, 1.0, hook=hook, noisy_frame_mask=all_clean)
    metadata.step = 1
    metadata.sigma = 0.5
    _run_step(transformer, 500, 0.9, hook=hook, noisy_frame_mask=all_clean)

    assert hook.full_count == 0
    assert hook.skip_count == 0


def test_hook_fails_open_without_explicit_context() -> None:
    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, sigma=1.0, num_steps=2)
    hook = _apply_test_hook(transformer, metadata)
    hook.refresh(transformer)

    _run_step(transformer, 1000, 1.0)

    assert hook.full_count == 0
    assert hook.skip_count == 0
    assert hook.state_manager._states == {}


def test_hook_uses_exact_sigma_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.cache.seacache import hook as hook_module

    transformer = TinyCosmos3Transformer()
    metadata = SimpleNamespace(step=0, sigma=0.37, num_steps=2)
    observed_sigmas: list[float] = []
    original_filter = hook_module.apply_sea_filter

    def recording_filter(hidden_states: torch.Tensor, sigma: float, power_exp: float) -> torch.Tensor:
        observed_sigmas.append(sigma)
        return original_filter(hidden_states, sigma, power_exp)

    monkeypatch.setattr(hook_module, "apply_sea_filter", recording_filter)
    hook = _apply_test_hook(transformer, metadata)
    hook.refresh(transformer)
    _run_step(transformer, timestep=999, value=1.0, hook=hook)

    assert observed_sigmas
    assert all(sigma == pytest.approx(0.37) for sigma in observed_sigmas)


def test_backend_selector_and_refresh() -> None:
    backend = get_cache_backend(
        "sea_cache",
        {
            "sea_threshold": 0.4,
            "sea_residual_order": 0,
        },
    )
    assert isinstance(backend, SeaCacheBackend)
    assert backend.config.sea_threshold == 0.4

    pipeline = Cosmos3OmniDiffusersPipeline()
    backend.enable(pipeline)
    hook = pipeline.transformer._hook_registry.get_hook(SeaCacheRootHook._HOOK_NAME)
    assert isinstance(hook, SeaCacheRootHook)
    pipeline._current_step_index = 0
    pipeline._current_sigma = 1.0
    pipeline._num_timesteps = 7
    _run_step(pipeline.transformer, 1000, 1.0, hook=hook)
    assert hook.full_count == 1

    backend.refresh(pipeline, num_inference_steps=7)
    assert hook.full_count == 0
    assert hook.skip_count == 0
    assert hook.state_manager._states == {}


def test_backend_uses_resolved_pipeline_metadata_not_refresh_argument() -> None:
    pipeline = Cosmos3OmniDiffusersPipeline()
    backend = SeaCacheBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    hook = pipeline.transformer._hook_registry.get_hook(SeaCacheRootHook._HOOK_NAME)
    assert isinstance(hook, SeaCacheRootHook)
    backend.refresh(pipeline, num_inference_steps=35)
    pipeline._num_timesteps = 50
    pipeline._current_step_index = 0
    pipeline._current_sigma = 1.0
    _run_step(pipeline.transformer, 1000, 1.0, hook=hook)
    assert hook.full_count == 1
    assert hook.num_inference_steps_callback is not None
    assert hook.num_inference_steps_callback() == 50

    pipeline._current_step_index = 1
    pipeline._current_sigma = 0.98
    _run_step(pipeline.transformer, 980, 0.99, hook=hook)
    assert hook.full_count == 1
    assert hook.skip_count == 1


def test_shared_config_defaults() -> None:
    config = DiffusionCacheConfig()
    assert config.sea_threshold == 0.25
    assert config.sea_residual_order == 1
    assert config.sea_max_consecutive_cached == 2
    assert config.sea_power_exp == 3.0
