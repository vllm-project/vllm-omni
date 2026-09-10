# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sage_attention_rejects_mask_instead_of_ignoring_it(monkeypatch):
    fake_package = types.ModuleType("sageattention")
    fake_package.sageattn = lambda *args, **kwargs: pytest.fail("unexpected Sage kernel call")
    monkeypatch.setitem(sys.modules, "sageattention", fake_package)
    module_name = "vllm_omni.diffusion.attention.backends.sage_attn"
    sys.modules.pop(module_name, None)
    try:
        backend_module = importlib.import_module(module_name)
        impl = backend_module.SageAttentionImpl(
            num_heads=4,
            head_size=64,
            softmax_scale=1.0 / 8.0,
            causal=False,
        )
        query = torch.randn(1, 2, 4, 64)
        metadata = AttentionMetadata(attn_mask=torch.ones(1, 2, dtype=torch.bool))

        with pytest.raises(ValueError, match="does not support attn_mask"):
            impl.forward_cuda(query, query, query, metadata)
    finally:
        sys.modules.pop(module_name, None)
