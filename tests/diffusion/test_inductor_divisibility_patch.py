# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the inductor factorable-divisibility patch.

torch 2.13 proves symbolic divisibility with torch's own unevaluated Mod,
so ``15360*s31 + 15360*s87`` is no longer provably divisible by
``s31 + s87`` and inductor raises CantSplit on FLUX's fp8 graph (nightly
builds 2953/2954). The patch adds an exact-polynomial-division proof that
is sound for every symbol assignment; on torch versions whose original
check already proves the case (<= 2.11, via sympy.Mod), the patch
self-extinguishes.
"""

import pytest
import sympy
from torch._inductor.sizevars import SizeVarAllocator

import vllm_omni.patch as patch_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_S1, _S2 = sympy.symbols("s31 s87", positive=True, integer=True)


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (15360 * _S1 + 15360 * _S2, _S1 + _S2, True),  # the CantSplit expression
        (7 * _S1 + 7 * _S2, _S1 + _S2, True),
        (_S1 * _S2, _S2, True),
        (_S1 * _S2 + _S2, _S2, True),  # s2*(s1 + 1)
        (3 * _S1, 2 * _S1, False),
        (_S1 + 1, _S1, False),
        (15360 * _S1 + 15359 * _S2, _S1 + _S2, False),
        (sympy.Integer(12), sympy.Integer(4), True),
        (sympy.Integer(12), sympy.Integer(5), False),
    ],
)
def test_provably_factorable_multiple(numerator, denominator, expected):
    assert patch_module._provably_factorable_multiple(numerator, denominator) is expected


def test_non_polynomial_atoms_never_reach_cancel():
    from torch.utils._sympy.functions import FloorDiv, ModularIndexing

    assert patch_module._provably_factorable_multiple(FloorDiv(_S1, 2) * 4, _S1) is False
    assert patch_module._provably_factorable_multiple(ModularIndexing(_S1, 1, 8) * _S2, _S2) is False


def test_divisibility_holds_after_patch_regardless_of_torch_version():
    """The end contract on any torch: the CantSplit expression proves divisible.

    On torch <= 2.11 the original method proves it (patch self-extinguished);
    on torch >= 2.13 the installed wrapper proves it via exact division.
    """
    sv = SizeVarAllocator()
    method = SizeVarAllocator.statically_known_multiple_of
    patched = getattr(method, "_vllm_omni_factorable_divisibility", False)
    assert method(sv, 15360 * _S1 + 15360 * _S2, _S1 + _S2) is True
    # Negative control must stay non-divisible either way.
    assert method(sv, 3 * _S1, 2 * _S1) is False
    if patched:
        # Wrapper installed: the original alone must have failed the probe.
        assert patch_module._provably_factorable_multiple(15360 * _S1 + 15360 * _S2, _S1 + _S2) is True


def test_patch_installs_when_original_cannot_prove(monkeypatch):
    """Simulate torch 2.13's unevaluated-Mod prover: the wrapper must install."""

    def weak(self, numerator, denominator):
        return False

    monkeypatch.setattr(SizeVarAllocator, "statically_known_multiple_of", weak)
    patch_module._patch_inductor_factorable_divisibility()
    method = SizeVarAllocator.statically_known_multiple_of
    assert getattr(method, "_vllm_omni_factorable_divisibility", False) is True

    sv = SizeVarAllocator()
    assert method(sv, 15360 * _S1 + 15360 * _S2, _S1 + _S2) is True
    assert method(sv, 3 * _S1, 2 * _S1) is False
    # Idempotency: a second install call must not wrap the wrapper.
    patch_module._patch_inductor_factorable_divisibility()
    assert SizeVarAllocator.statically_known_multiple_of is method
