# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the framework-side RefHintCacheBackend."""

from typing import cast

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm")

from vllm_omni.diffusion.cache.ref_hint_cache import RefHintCacheBackend  # noqa: E402
from vllm_omni.diffusion.data import DiffusionCacheConfig  # noqa: E402
from vllm_omni.diffusion.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
)
from vllm_omni.diffusion.model_region import ModelRegion  # noqa: E402


class _VaceLikeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vace_blocks = nn.ModuleList([nn.Identity()])


class _PlainTransformer(nn.Module):
    pass


class _FakePipeline:
    def __init__(self, transformer=None, transformer_2=None):
        if transformer is not None:
            self.transformer = transformer
        if transformer_2 is not None:
            self.transformer_2 = transformer_2


def _cfg(**kw):
    return DiffusionCacheConfig(**kw)


def test_quality_validated_strategy_is_default():
    backend = RefHintCacheBackend(_cfg())
    assert backend._strategy() == "forecast50"


def test_lossy_interval_requires_acknowledgement():
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=2))
    with pytest.raises(ValueError, match="acknowledge_lossy"):
        backend.enable(_FakePipeline(_VaceLikeTransformer()))


def test_lossless_interval_is_exempt_and_exposes_handler():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(_FakePipeline(owner))
    assert backend.enabled
    assert backend.get_model_region_handler() is backend

    sentinel = [torch.tensor([1.0])]
    context = ForwardContext()
    with override_forward_context(context):
        context.denoise_step_idx = 0
        assert backend.execute(ModelRegion.REFERENCE_HINTS, owner, lambda: sentinel) is sentinel

    state = backend._states[id(owner)]
    assert state._history == {}
    assert state.misses == 0


def test_both_experts_get_isolated_state_and_reset():
    first, second = _VaceLikeTransformer(), _VaceLikeTransformer()
    pipeline = _FakePipeline(first, second)
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(pipeline)
    assert set(backend._states) == {id(first), id(second)}

    for state in backend._states.values():
        branch, _ = state.begin_call(0)
        state.store(branch, 0, [torch.tensor([1.0])])
    backend.refresh(pipeline, num_inference_steps=30)
    assert all(state.misses == 0 and state._history == {} for state in backend._states.values())


def test_second_expert_only_config():
    second = _VaceLikeTransformer()
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(_FakePipeline(transformer_2=second))
    assert set(backend._states) == {id(second)}


def test_no_transformer_raises():
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    with pytest.raises(ValueError, match="transformer_2"):
        backend.enable(_FakePipeline())


def test_unsupported_model_raises():
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    with pytest.raises(ValueError, match="does not expose"):
        backend.enable(_FakePipeline(_PlainTransformer()))


def test_reuse_strategy_skips_compute_on_second_step():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="reuse",
            ref_hint_acknowledge_lossy=True,
        )
    )
    backend.enable(_FakePipeline(owner))
    context = ForwardContext()
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return [torch.tensor([float(calls)])]

    with override_forward_context(context):
        context.denoise_step_idx = 0
        first = backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        context.denoise_step_idx = 1
        second = backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)

    assert calls == 1
    assert torch.equal(first[0], second[0])


def test_forecast50_uses_two_fresh_values_and_damped_prediction():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="forecast50",
            ref_hint_acknowledge_lossy=True,
        )
    )
    backend.enable(_FakePipeline(owner))
    context = ForwardContext()
    values = iter((0.0, 2.0))

    def compute():
        return [torch.tensor([next(values)])]

    with override_forward_context(context):
        context.denoise_step_idx = 0
        backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        context.denoise_step_idx = 1
        backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)
        context.denoise_step_idx = 2
        forecast = backend.execute(ModelRegion.REFERENCE_HINTS, owner, compute)

    # Nominal gain 0.5 is limited by the 0.25 trust region:
    # 2 + 0.25 * (2 - 0) = 2.5.
    assert torch.equal(forecast[0], torch.tensor([2.5]))


def test_finish_request_releases_retained_hints():
    owner = _VaceLikeTransformer()
    pipeline = _FakePipeline(owner)
    backend = RefHintCacheBackend(
        _cfg(
            ref_hint_refresh_interval=2,
            ref_hint_strategy="reuse",
            ref_hint_acknowledge_lossy=True,
        )
    )
    backend.enable(pipeline)
    context = ForwardContext()

    with override_forward_context(context):
        context.denoise_step_idx = 0
        backend.execute(
            ModelRegion.REFERENCE_HINTS,
            owner,
            lambda: [torch.tensor([1.0])],
        )

    state = backend._states[id(owner)]
    assert state._history
    backend.finish_request(pipeline)
    assert state._history == {}
    assert state.misses == 0


def test_unrelated_region_behavior_is_direct_compute():
    owner = _VaceLikeTransformer()
    backend = RefHintCacheBackend(_cfg(ref_hint_refresh_interval=1))
    backend.enable(_FakePipeline(owner))
    sentinel = [torch.tensor([7.0])]
    assert backend.execute(cast(ModelRegion, "other"), owner, lambda: sentinel) is sentinel
