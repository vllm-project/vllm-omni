# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate pinned FA versions without requiring CUDA kernels."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("version", [None, 0, -1, 1, 5, 2, 3, 4])
def test_resolved_varlen_version_is_validated_without_rediscovery(monkeypatch, version):
    query = torch.zeros(2, 1, 4)
    lse = torch.zeros(1, 2)
    kernel = Mock(return_value=(query, lse))
    wrapper = ModuleType("vllm.vllm_flash_attn")
    wrapper.flash_attn_varlen_func = kernel
    monkeypatch.setitem(sys.modules, wrapper.__name__, wrapper)
    resolver = Mock(side_effect=AssertionError("Pinned FA versions must not be rediscovered"))
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolver)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    kwargs = dict(
        cu_seqlens_q=offsets,
        cu_seqlens_k=offsets,
        max_seqlen_q=2,
        max_seqlen_k=2,
        fa_version=version,
        fa_version_is_resolved=True,
    )
    if version in (2, 3, 4):
        out, actual_lse = fa.vllm_flash_attn_varlen_with_lse(query, query, query, **kwargs)
        assert out is query and actual_lse is lse
        kernel.assert_called_once()
        assert kernel.call_args.kwargs["fa_version"] == version
    else:
        with pytest.raises(ValueError, match="resolved FlashAttention version"):
            fa.vllm_flash_attn_varlen_with_lse(query, query, query, **kwargs)
        kernel.assert_not_called()
    resolver.assert_not_called()
