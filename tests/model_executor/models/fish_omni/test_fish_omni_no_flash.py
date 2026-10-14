# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Drama 3 tower works on builds without vLLM's flash-attention extension."""

import sys

import pytest

from vllm_omni.model_executor.models.fish_omni import codebook_tower

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_missing_flash_attention_selects_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes ``import vllm.vllm_flash_attn`` raise ImportError.
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)
    codebook_tower.paged_flash_attention.cache_clear()
    try:
        assert codebook_tower.paged_flash_attention() is None
    finally:
        codebook_tower.paged_flash_attention.cache_clear()
