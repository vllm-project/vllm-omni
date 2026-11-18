# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only Flux3 text encoder checks: compiled language model versus eager."""

import pytest
import torch

from tests.diffusion.models.flux3.test_flux3_text_encoder import _make_encoder, _token_inputs
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1)
def test_compiled_language_model_matches_eager() -> None:
    torch.manual_seed(0)
    enc = _make_encoder().eval().to(device="cuda", dtype=torch.bfloat16)
    enc.finalize_after_loading()
    inputs = _token_inputs("cuda")

    expected = enc._language_model_forward(*inputs)
    assert all(tensor.dtype == torch.bfloat16 for tensor in expected)
    enc.setup_compile(dynamic=False)
    assert enc._compiled_language_model_forward is not None
    actual = enc._compiled_language_model_forward(*inputs)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=5e-3, rtol=5e-3)
