# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TeaCache support for Bagel (an LLM + KV-cache model with internal CFG).

``Bagel.forward`` fans out the CFG branches and calls ``_combine_cfg`` *inside*
the forward, so the generic hidden-residual TeaCache hook cannot wrap it without
silently dropping CFG (the conditional-only extractor path). Bagel therefore
uses direct, CFG-preserving caching in its denoise loop instead.

These weight-free tests guard that contract:
- ``enable_bagel_teacache`` attaches a config (direct path), not a generic hook;
- the cache decision follows the standard TeaCache accumulate-vs-threshold rule;
- a cached prediction is never reused when its CFG branch layout doesn't match
  the current step (``cfg_interval`` can toggle CFG mid-loop).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.cache.teacache.backend import enable_bagel_teacache
from vllm_omni.diffusion.cache.teacache.config import _MODEL_COEFFICIENTS
from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_enable_bagel_attaches_config_not_hook():
    """Bagel must take the direct path: a config is attached, no hook installed."""
    transformer = SimpleNamespace()
    pipeline = SimpleNamespace(bagel=transformer)
    enable_bagel_teacache(pipeline, SimpleNamespace(rel_l1_thresh=0.2, coefficients=None))

    assert getattr(transformer, "_tea_cache_config", None) is not None
    assert transformer._tea_cache_config.transformer_type == "Bagel"
    assert abs(transformer._tea_cache_config.rel_l1_thresh - 0.2) < 1e-9
    # Coefficients resolve to the Bagel-specific defaults.
    assert list(transformer._tea_cache_config.coefficients) == _MODEL_COEFFICIENTS["Bagel"]
    # The generic hidden-residual hook (which would drop CFG) must NOT be applied.
    assert not hasattr(transformer, "_hook_registry")
    assert pipeline.transformer is transformer


def _stub(thresh=0.2, coefficients=(1.0, 0.0)):
    """Minimal stand-in carrying just the attributes the decision touches.

    Default coefficients ``(1.0, 0.0)`` make ``np.poly1d`` the identity, so the
    rescaled distance equals the raw relative-L1 — letting us test the
    accumulate-vs-threshold logic independently of the real Bagel polynomial.
    """
    cfg = SimpleNamespace(rel_l1_thresh=thresh, coefficients=list(coefficients))
    return SimpleNamespace(
        _tea_cache_config=cfg,
        _tc_rescale=np.poly1d(cfg.coefficients),
        _tc_acc=0.0,
        _tc_prev_mod=None,
        _tc_prev_pred=None,
        _tc_cnt=0,
    )


def _decide(stub, cur_mod, need_text=True, need_img=False):
    return Bagel._tea_cache_should_compute(stub, cur_mod, need_text, need_img)


def test_disabled_always_computes():
    assert Bagel._tea_cache_should_compute(SimpleNamespace(_tea_cache_config=None), torch.ones(4, 8), True, False)


def test_first_step_always_computes():
    stub = _stub()
    assert _decide(stub, torch.zeros(4, 8)) is True
    assert stub._tc_cnt == 1


def test_stable_input_reuses_cache():
    stub = _stub(thresh=0.2)
    stub._tc_prev_pred = (torch.ones(4, 8), torch.ones(4, 8), None)  # text-only layout
    x = torch.ones(4, 8)
    assert _decide(stub, x) is True  # first step computes
    assert _decide(stub, x.clone()) is False  # no change -> reuse cache


def test_large_change_forces_compute():
    stub = _stub(thresh=0.2)
    stub._tc_prev_pred = (torch.ones(4, 8), torch.ones(4, 8), None)
    assert _decide(stub, torch.zeros(4, 8)) is True
    assert _decide(stub, torch.ones(4, 8) * 1000.0) is True  # huge jump -> recompute


def test_branch_layout_mismatch_forces_compute():
    """A cached text-only prediction must not be reused when img CFG is needed."""
    stub = _stub(thresh=0.2)
    x = torch.ones(4, 8)
    stub._tc_prev_pred = (torch.ones(4, 8), torch.ones(4, 8), None)  # no img branch
    stub._tc_cnt = 1
    stub._tc_prev_mod = x.clone()
    assert _decide(stub, x.clone(), need_text=True, need_img=True) is True
