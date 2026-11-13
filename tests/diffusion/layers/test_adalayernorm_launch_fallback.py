# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Direct tests for the synchronous launch-failure fallback (C3a).

Simulates a backend-specific Triton launch failure (the launcher raises
synchronously) and verifies the two-layer response: the failed config is
recorded, the caller gets the correct native result, the same config is not
retried, and a different constexpr variant stays eligible.

The failed-key set is module-global; the autouse fixture snapshots and
restores it so tests are self-contained and independent of execution order.
Collection is safe on non-Triton environments: the Triton-only kernel symbol
is resolved lazily inside each test and skips if unavailable.
"""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers import adalayernorm as adaln_mod
from vllm_omni.diffusion.layers.adalayernorm import (
    _FAILED_ADALN_KEYS,
    AdaLayerNorm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]

# Unique channels + eps so the failed keys used here cannot collide with
# other tests in the same session.
_HIDDEN = 2048
_EPS = 1e-5


@pytest.fixture(autouse=True)
def _isolate_failed_keys():
    saved = set(_FAILED_ADALN_KEYS)
    _FAILED_ADALN_KEYS.clear()
    try:
        yield
    finally:
        _FAILED_ADALN_KEYS.clear()
        _FAILED_ADALN_KEYS.update(saved)


def _make(bs=1):
    m = AdaLayerNorm(_HIDDEN, eps=_EPS).to(device="cuda", dtype=torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(31)
    x = torch.randn(bs, 512, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    return m, x, scale, shift


def _get_kernel():
    kernel = getattr(adaln_mod, "_adaln_scale_shift_layernorm_kernel", None)
    if kernel is None:
        pytest.skip("Triton AdaLayerNorm kernel unavailable")
    return kernel


def test_shared_launch_failure_fallback_and_cache(monkeypatch):
    kernel = _get_kernel()
    m, x, scale, shift = _make(1)
    native = m.forward_native(x, scale, shift)

    calls = {"n": 0}

    def raising_run(self, *args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("simulated synchronous launch failure")

    monkeypatch.setattr(type(kernel), "run", raising_run)

    # Phase 1: the launch raises synchronously; the caller must receive the
    # correct native result and the failed key must be recorded.
    out = m.forward_cuda(x, scale, shift)
    torch.testing.assert_close(out.float(), native.float())
    expected_key = (x.device.index, _HIDDEN, x.dtype, False, False, False, False)
    assert expected_key in _FAILED_ADALN_KEYS
    assert calls["n"] == 1  # the fused launcher was attempted exactly once

    # Phase 2: the same config must NOT retry the launcher (failed-key
    # cache) and still return the correct native result.
    calls["n"] = 0
    out2 = m.forward_cuda(x, scale, shift)
    assert calls["n"] == 0
    torch.testing.assert_close(out2.float(), native.float())


def test_shared_failure_then_per_sample_variant_eligible(monkeypatch):
    # The failed-key cache is per constexpr variant: after the shared
    # (False, False) variant fails, the per-sample (True, True) variant must
    # still attempt the launcher and succeed.
    kernel = _get_kernel()
    m, x, shared_scale, shared_shift = _make(2)
    shared_native = m.forward_native(x, shared_scale, shared_shift)
    shared_key = (x.device.index, _HIDDEN, x.dtype, False, False, False, False)
    per_sample_key = (x.device.index, _HIDDEN, x.dtype, False, False, True, True)
    calls = {"n": 0}

    def raising_run(self, *args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("simulated synchronous shared launch failure")

    # Establish the shared failure in this test; the autouse fixture clears
    # failures from other tests, so their cache entries cannot prove isolation.
    with monkeypatch.context() as shared_patch:
        shared_patch.setattr(type(kernel), "run", raising_run)
        shared_out = m.forward_cuda(x, shared_scale, shared_shift)
        torch.testing.assert_close(shared_out.float(), shared_native.float())
        assert calls["n"] == 1
        assert shared_key in _FAILED_ADALN_KEYS
        assert per_sample_key not in _FAILED_ADALN_KEYS

        m.forward_cuda(x, shared_scale, shared_shift)
        assert calls["n"] == 1, "the cached shared failure must not retry"

    g = torch.Generator(device="cuda").manual_seed(37)
    scale = torch.randn(2, 1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(2, 1, _HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
    calls["n"] = 0

    def counting_run(self, *args, **kwargs):
        calls["n"] += 1
        return real_run(self, *args, **kwargs)

    real_run = type(kernel).run
    monkeypatch.setattr(type(kernel), "run", counting_run)

    out = adaln_mod._adaln_fused_forward(m, x, scale, shift)
    assert out is not None, "the per-sample variant must remain eligible after a shared failure"
    native = m.forward_native(x, scale, shift)
    torch.testing.assert_close(out.float(), native.float(), atol=2e-2, rtol=2e-2)
    assert calls["n"] == 1
    assert shared_key in _FAILED_ADALN_KEYS
    assert per_sample_key not in _FAILED_ADALN_KEYS
