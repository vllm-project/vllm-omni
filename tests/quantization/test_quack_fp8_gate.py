# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the quack FP8 auto-enable gate.

quack's CuteDSL FP8 GEMM uses the 5th-gen ``tcgen05`` tensor-core MMA, which
exists only on datacenter Blackwell (``sm_100a`` / ``sm_101a`` / ``sm_103a``,
compute capability ``10.x``). It must NOT auto-enable on workstation/consumer
Blackwell (``sm_120`` / ``sm_121``, cc ``12.x``, e.g. RTX PRO 6000 / RTX
50-series), where ``tcgen05`` is absent and every FP8 GEMM would fall back to
FlashInfer one call at a time (catastrophically slow). See
``vllm_omni/quantization/quack_fp8.py``.

Regression guard for the ``>= 10`` gate that matched cc ``12.x`` too. These
tests are hardware-free: the ``torch.cuda`` probes are monkeypatched, so no GPU
is required.
"""

import pytest
import torch

from vllm_omni.quantization import quack_fp8

_ENV = "VLLM_OMNI_USE_QUACK_FP8"


def _fake_cuda(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> None:
    """Pretend a CUDA device with the given (major, minor) capability is present."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)


@pytest.mark.parametrize(
    "capability, expected",
    [
        ((10, 0), True),  # sm_100 datacenter Blackwell — tcgen05 present
        ((10, 3), True),  # sm_103 datacenter Blackwell
        ((12, 0), False),  # sm_120 workstation Blackwell — the regression
        ((12, 1), False),  # sm_121 consumer Blackwell
        ((9, 0), False),  # Hopper — CUTLASS already fuses bias, quack unused
        ((8, 9), False),  # Ada
    ],
)
def test_auto_enable_only_on_datacenter_blackwell(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected: bool,
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    _fake_cuda(monkeypatch, capability)
    assert quack_fp8._is_quack_capable() is expected
    assert quack_fp8.quack_enabled() is expected


def test_no_cuda_disables_quack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert quack_fp8._is_quack_capable() is False
    assert quack_fp8.quack_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_env_override_forces_on_even_on_sm120(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # sm_120 auto-disables, but an explicit truthy override forces quack on
    # (e.g. once CuteDSL ships sm_120a support).
    _fake_cuda(monkeypatch, (12, 0))
    monkeypatch.setenv(_ENV, value)
    assert quack_fp8.quack_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_env_override_forces_off_even_on_datacenter_blackwell(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # sm_100 auto-enables, but an explicit non-truthy override forces quack off.
    _fake_cuda(monkeypatch, (10, 0))
    monkeypatch.setenv(_ENV, value)
    assert quack_fp8.quack_enabled() is False
