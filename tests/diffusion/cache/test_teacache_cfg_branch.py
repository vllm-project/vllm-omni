# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TeaCacheHook takes the CFG branch from the caller's stamp (#8482).

Under step execution the TeaCache hook is not refreshed between scheduler
waves, so a CFG wave followed by a non-CFG wave with an odd number of forwards
used to flip the forward-count parity: the next CFG step's positive call landed
in the negative state. CFGParallelMixin now stamps ``transformer.cfg_branch`` on
every call and the hook prefers it.
"""

from typing import Any

import pytest
import torch

import vllm_omni.diffusion.cache.teacache.hook as teacache_hook
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import EXTRACTOR_REGISTRY, CacheContext
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_POS, _NEG = "teacache_positive", "teacache_negative"


class _BranchProbeDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.do_true_cfg = False
        self.cfg_branch: str | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:  # replaced by the hook
        return (x,)


def _extract(module: torch.nn.Module, x: torch.Tensor, **_: Any) -> CacheContext:
    return CacheContext(
        modulated_input=x,
        hidden_states=x,
        encoder_hidden_states=None,
        temb=x,
        run_transformer_blocks=lambda: (x + 1.0,),
        postprocess=lambda h: (h,),
    )


class _Pipeline(CFGParallelMixin):
    def __init__(self, transformer: torch.nn.Module) -> None:
        self.transformer = transformer


@pytest.fixture
def hooked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(EXTRACTOR_REGISTRY, "_BranchProbeDiT", _extract)
    monkeypatch.setattr(teacache_hook, "get_classifier_free_guidance_world_size", lambda: 1)
    module = _BranchProbeDiT()
    teacache_hook.apply_teacache_hook(
        module,
        TeaCacheConfig(transformer_type="_BranchProbeDiT", coefficients=[0.0, 0.0, 0.0, 1.0, 0.0], rel_l1_thresh=0.2),
    )
    hook = module._hook_registry.get_hook(teacache_hook.TeaCacheHook._HOOK_NAME)
    contexts: list[str] = []
    set_context = hook.state_manager.set_context

    def record(name: str) -> None:
        contexts.append(name)
        set_context(name)

    monkeypatch.setattr(hook.state_manager, "set_context", record)
    return module, contexts


def _step(pipeline: _Pipeline, x: torch.Tensor, *, cfg: bool) -> None:
    with torch.no_grad():
        pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=cfg,
            true_cfg_scale=4.0,
            positive_kwargs={"x": x},
            negative_kwargs={"x": x} if cfg else None,
            cfg_normalize=False,
        )


def test_interleaved_non_cfg_wave_keeps_branch_states_aligned(hooked) -> None:
    module, contexts = hooked
    pipeline = _Pipeline(module)
    x = torch.ones(1, 4)

    _step(pipeline, x, cfg=True)  # CFG wave: positive, negative
    _step(pipeline, x, cfg=False)  # non-CFG wave: one forward, odd count
    _step(pipeline, x, cfg=True)  # next CFG wave must start positive again

    assert contexts == [_POS, _NEG, _POS, _POS, _NEG]
    assert module.cfg_branch is None


def test_unstamped_callers_keep_the_parity_fallback(hooked) -> None:
    """Callers outside CFGParallelMixin are unchanged: branch inferred from parity."""
    module, contexts = hooked
    x = torch.ones(1, 4)

    with torch.no_grad():
        for cfg in (True, True, False, True, True):
            module.do_true_cfg = cfg
            module(x)

    assert contexts == [_POS, _NEG, _POS, _NEG, _POS]


def _extract_with_hint(module: torch.nn.Module, x: torch.Tensor, teacache_branch: str | None = None, **_: Any):
    ctx = _extract(module, x)
    ctx.extra_states = {"teacache_branch": teacache_branch} if teacache_branch is not None else None
    return ctx


def test_extractor_hint_takes_precedence_over_stamp(hooked, monkeypatch: pytest.MonkeyPatch) -> None:
    """An extractor's per-call hint (e.g. MammothModa2) is more specific than the mixin stamp."""
    module, contexts = hooked
    hook = module._hook_registry.get_hook(teacache_hook.TeaCacheHook._HOOK_NAME)
    monkeypatch.setattr(hook, "extractor_fn", _extract_with_hint)
    x = torch.ones(1, 4)

    with torch.no_grad():
        module.cfg_branch = "negative"
        module(x, teacache_branch="positive")
        module(x)  # no hint: the stamp applies
    module.cfg_branch = None

    assert contexts == [_POS, _NEG]


def test_invalid_stamp_is_rejected(hooked) -> None:
    module, _ = hooked
    module.cfg_branch = "uncond"

    with pytest.raises(ValueError, match="cfg_branch"), torch.no_grad():
        module(torch.ones(1, 4))
