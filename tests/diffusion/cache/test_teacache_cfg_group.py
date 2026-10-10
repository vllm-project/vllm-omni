# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import EXTRACTOR_REGISTRY, CacheContext
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.distributed import parallel_state
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.hooks import HookRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _TeaCacheProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.computed_residuals: list[float] = []

    def forward(self, hidden_states: torch.Tensor, residual: float) -> tuple[torch.Tensor]:
        raise AssertionError("TeaCache hook must intercept this forward")


def _extract_probe_context(module: _TeaCacheProbe, hidden_states: torch.Tensor, residual: float) -> CacheContext:
    def run_blocks() -> tuple[torch.Tensor]:
        module.computed_residuals.append(residual)
        return (hidden_states + residual,)

    return CacheContext(
        modulated_input=hidden_states.clone(),
        hidden_states=hidden_states.clone(),
        encoder_hidden_states=None,
        temb=torch.zeros_like(hidden_states),
        run_transformer_blocks=run_blocks,
        postprocess=lambda output: (output,),
    )


class _ProbePipeline(CFGParallelMixin):
    def __init__(self) -> None:
        self.transformer = _TeaCacheProbe()
        apply_teacache_hook(
            self.transformer,
            TeaCacheConfig(transformer_type="_TeaCacheProbe", coefficients=[0, 0, 0, 0, 0], rel_l1_thresh=0.2),
        )


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> _ProbePipeline:
    monkeypatch.setattr(parallel_state, "_CFG", None)
    monkeypatch.setitem(EXTRACTOR_REGISTRY, "_TeaCacheProbe", _extract_probe_context)
    return _ProbePipeline()


@pytest.mark.parametrize("group_initialized", [False, True])
def test_sequential_cfg_with_teacache_keeps_branch_residuals_separate(
    pipeline: _ProbePipeline, monkeypatch: pytest.MonkeyPatch, group_initialized: bool
) -> None:
    """Direct pipelines without a CFG group must support real TeaCache hits."""
    if group_initialized:
        monkeypatch.setattr(parallel_state, "_CFG", SimpleNamespace(world_size=1, rank_in_group=0))

    hidden_states = torch.ones(1, 1)
    positive_kwargs = {"hidden_states": hidden_states, "residual": 2.0}
    negative_kwargs = {"hidden_states": hidden_states, "residual": 10.0}

    for _ in range(2):
        prediction = pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=4.0,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )
        torch.testing.assert_close(prediction, torch.full_like(hidden_states, -21.0))

    # Each branch computes once, then reuses its own residual on the next step.
    assert pipeline.transformer.computed_residuals == [2.0, 10.0]
    hook = HookRegistry.get_or_create(pipeline.transformer).get_hook("teacache")
    assert isinstance(hook, TeaCacheHook)
    assert set(hook.state_manager._states) == {"teacache_positive", "teacache_negative"}
    assert all(state.cnt == 2 for state in hook.state_manager._states.values())


@pytest.mark.parametrize("rank", [0, 1])
def test_teacache_uses_initialized_cfg_rank(
    pipeline: _ProbePipeline, monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    monkeypatch.setattr(parallel_state, "_CFG", SimpleNamespace(world_size=2, rank_in_group=rank))
    pipeline.transformer.do_true_cfg = True
    hidden_states = torch.ones(1, 1)
    for _ in range(2):
        output = pipeline.transformer(hidden_states=hidden_states, residual=2.0)[0]
        torch.testing.assert_close(output, hidden_states + 2.0)

    hook = HookRegistry.get_or_create(pipeline.transformer).get_hook("teacache")
    assert isinstance(hook, TeaCacheHook)
    expected_branch = "negative" if rank == 1 else "positive"
    assert set(hook.state_manager._states) == {f"teacache_{expected_branch}"}
    assert pipeline.transformer.computed_residuals == [2.0]
