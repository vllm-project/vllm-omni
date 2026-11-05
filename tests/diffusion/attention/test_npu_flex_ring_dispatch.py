# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the optional NPU Ring provider at the shared attention boundary."""

import pytest
import torch

from vllm_omni.diffusion.attention import layer as attention_layer
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.backends.utils import npu_flex_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


@pytest.mark.parametrize("is_npu,flex", [(True, True), (True, False), (False, False), (False, True)])
def test_ring_provider_changes_only_for_explicit_npu_flex(monkeypatch, is_npu, flex):
    class Platform:
        @staticmethod
        def is_npu():
            return is_npu

    calls = []

    class ExistingRing:
        @staticmethod
        def run_attention(q, k, v, metadata, **kwargs):
            calls.append("existing")
            return q

    def run_flex(q, k, v, metadata, scale, *, ring_group):
        assert ring_group is layer.ring_pg
        assert scale == layer.softmax_scale
        calls.append("flex")
        return q

    monkeypatch.setattr(attention_layer, "current_omni_platform", Platform())
    monkeypatch.setattr(npu_flex_attention, "try_npu_flex_attention", run_flex)
    layer = attention_layer.Attention.__new__(attention_layer.Attention)
    torch.nn.Module.__init__(layer)
    layer.attn_backend = FlashAttentionBackend
    layer.ring_pg = object()
    layer.ring_runner = ExistingRing()
    layer.softmax_scale = 0.25
    layer.causal = False
    layer.attention = torch.nn.Identity()
    q = torch.zeros(1, 4, 2, 16)
    metadata = AttentionMetadata(full_attn_spans=[[(1, 4)]], extra={"npu_flex_attention": True} if flex else {})
    assert layer._run_ring_attention(q, q, q, metadata) is q
    assert calls == (["flex"] if is_npu and flex else ["existing"])


@pytest.mark.parametrize("decode", [False, True])
@torch.no_grad()
def test_ring_coordinates_keep_prefix_once(monkeypatch, decode):
    captured = []

    class Runtime:
        @staticmethod
        def make_block_causal_mask_mod(*args, **kwargs):
            return object()

        class FlexAttentionMask:
            def __init__(self, *args, **kwargs):
                pass

        @staticmethod
        def parallel_flex_attention(q, k, v, **kwargs):
            captured.append(kwargs)
            return q

    monkeypatch.setattr(npu_flex_attention, "_can_use_flex", lambda *args, **kwargs: True)
    monkeypatch.setattr(npu_flex_attention, "load_mindie_flex_runtime", lambda: Runtime)
    monkeypatch.setattr(npu_flex_attention.dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(npu_flex_attention.dist, "get_world_size", lambda group: 2)
    kv = torch.zeros(1, 3, 2, 16)
    prefix = kv[:, :2]
    q = kv if decode else torch.cat((prefix, kv), 1)
    metadata = AttentionMetadata(
        full_attn_spans=[[(0 if decode else 2, 8)]],
        joint_query=prefix[:, :0] if decode else prefix, joint_key=prefix, joint_value=prefix,
        extra={"npu_flex_seq_len": 8},
    )
    npu_flex_attention.try_npu_flex_attention(q, kv, kv, metadata, 0.25, ring_group=object())
    expected_query = torch.tensor([5, 6, 7]) if decode else torch.tensor([0, 1, 5, 6, 7])
    torch.testing.assert_close(captured[0]["query_positions"], expected_query)
    torch.testing.assert_close(captured[0]["key_positions"], torch.tensor([5, 6, 7]))
    torch.testing.assert_close(captured[0]["joint_key_positions"], torch.tensor([0, 1]))
    assert captured[0]["joint_key"] is prefix
    assert captured[0]["ulysses_group"] is None
