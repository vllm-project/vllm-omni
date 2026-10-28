# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import factory, ulysses
from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("npu", [False, True])
@pytest.mark.parametrize("joint", [None, "front", "rear"])
@pytest.mark.parametrize("use_sync", [False, True])
def test_shared_a2a_routes_qkv_gate_and_reverse(monkeypatch, npu, joint, use_sync):
    """Both platforms share attention routing, including joint slicing."""
    calls = []
    platform = SimpleNamespace(is_npu=lambda: npu, synchronize=lambda: calls.append("sync"))
    monkeypatch.setattr(ulysses, "current_omni_platform", platform)
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: None)
    group = SimpleNamespace(
        ulysses_group=SimpleNamespace(group_name="ulysses"),
        ulysses_world_size=2,
        ulysses_rank=0,
        ring_world_size=1,
    )
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda pg: 2)
    strategy = ulysses.UlyssesParallelAttention(group, 2, 1, use_sync, ulysses_a2a_permute=True)
    query = torch.randn(1, 3, 4, 16)
    key = torch.randn(1, 3, 2, 16)
    value = torch.randn_like(key)
    gate = torch.randn_like(query)

    def qkv(q, k, v, group_name, world_size):
        assert q is query and k is key and v is value
        assert (group_name, world_size) == ("ulysses", 2)
        calls.append("qkv")
        return tuple(t[:, :, : t.shape[2] // 2].repeat(1, 2, 1, 1) for t in (q, k, v))

    def forward(tensor, group_name, world_size):
        calls.append("gate")
        assert tensor is gate
        return tensor[:, :, :2].repeat(1, 2, 1, 1)

    def reverse(tensor, group_name, world_size):
        calls.append("reverse")
        assert tensor.shape == (1, 6, 2, 16)
        return tensor[:, :3].repeat(1, 1, 2, 1)

    monkeypatch.setattr(a2a_permute, "ulysses_qkv_fwd_group", qkv)
    monkeypatch.setattr(a2a_permute, "ulysses_qkv_fwd", forward)
    monkeypatch.setattr(a2a_permute, "ulysses_o_rev", reverse)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "strict")
    metadata = AttentionMetadata(extra={"gate_compress": gate})
    if joint:
        metadata.joint_strategy = joint
        metadata.joint_query = torch.randn(1, 2, 4, 16)
        metadata.joint_key = torch.randn(1, 2, 2, 16)
        metadata.joint_value = torch.randn(1, 2, 2, 16)

        def gather(outputs, tensor, **kwargs):
            for output in outputs:
                output.copy_(tensor)

        monkeypatch.setattr(ulysses.dist, "all_gather", gather)
    q, k, v, metadata, context = strategy.pre_attention(query, key, value, metadata)
    assert q.shape == (1, 8 if joint else 6, 2, 16)
    assert k.shape == v.shape == (1, 8 if joint else 6, 1, 16)
    assert metadata.extra["gate_compress"].shape == (1, 6, 2, 16)
    output = strategy.post_attention(q, context)
    assert output.shape == (1, 5 if joint else 3, 4, 16)
    assert calls == (["qkv", "gate", "sync", "reverse", "sync"] if use_sync else ["qkv", "gate", "reverse"])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"enforce_eager": False}, "enforce_eager"),
        ({"ulysses_mode": "advanced_uaa"}, "strict, pure Ulysses"),
        ({"ring_degree": 2}, "strict, pure Ulysses"),
        ({"ulysses_degree": 1}, "strict, pure Ulysses"),
    ],
)
def test_npu_backend_rejects_unsupported_config(monkeypatch, overrides, message):
    values = dict(ulysses_a2a_permute=True, ulysses_degree=2, ring_degree=1, ulysses_mode="strict")
    values.update(overrides)
    enforce_eager = values.pop("enforce_eager", True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(**values),
        enforce_eager=enforce_eager,
    )
    monkeypatch.setattr(factory, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: SimpleNamespace(omni_diffusion_config=config))
    with pytest.raises(ValueError, match=message):
        factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)


def test_cuda_backend_does_not_require_eager(monkeypatch):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(ulysses_a2a_permute=True, ulysses_degree=2), enforce_eager=False
    )
    monkeypatch.setattr(factory, "current_omni_platform", SimpleNamespace(is_npu=lambda: False))
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: SimpleNamespace(omni_diffusion_config=config))
    monkeypatch.setattr(factory, "get_sp_group", lambda: object())
    monkeypatch.setattr(factory, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(factory, "UlyssesParallelAttention", lambda **kwargs: kwargs)
    result = factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)
    assert result["ulysses_a2a_permute"] is True


def test_disabled_optimization_does_not_load_backend(monkeypatch):
    def unexpected():
        pytest.fail("Disabled optimization must not initialize a transport")

    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", unexpected)
    group = SimpleNamespace(ulysses_group=object(), ulysses_world_size=2, ring_world_size=1)
    strategy = ulysses.UlyssesParallelAttention(group, 2, 1, False)
    assert strategy.name == "ulysses"
