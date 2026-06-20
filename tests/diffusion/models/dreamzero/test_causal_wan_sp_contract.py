# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention, append_attention_kv_cache
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
    validate_sp_plan,
)
from vllm_omni.diffusion.models.dreamzero.causal_wan_model import (
    CausalWanModel,
    CausalWanSelfAttention,
    WanI2VCrossAttention,
    WanT2VCrossAttention,
)


pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_causal_wan_model_declares_sp_plan_for_video_only():
    plan = getattr(CausalWanModel, "_sp_plan", None)

    assert plan is not None
    validate_sp_plan(plan)
    assert "sp_prepare" in plan
    assert "head.head" not in plan

    prepare = plan["sp_prepare"]
    assert isinstance(prepare[0], SequenceParallelInput)
    assert prepare[0].split_dim == 1
    assert prepare[0].expected_dims == 3
    assert prepare[0].split_output is True

    assert isinstance(prepare[1], SequenceParallelInput)
    assert prepare[1].split_dim == 1
    assert prepare[1].expected_dims == 3
    assert prepare[1].split_output is True

    assert isinstance(prepare[2], SequenceParallelInput)
    assert prepare[2].split_dim == 0
    assert prepare[2].expected_dims == 3
    assert prepare[2].split_output is True

def test_causal_wan_projects_time_embedding_before_sp_prepare():
    class RecordingSPPrepare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_e_shape = None

        def forward(self, x_video, e_video, freqs):
            self.seen_e_shape = tuple(e_video.shape)
            return x_video[:, :2], e_video[:, :2], freqs[:2]

    class IdentityHead(nn.Module):
        def forward(self, x, e):
            return x

    model = CausalWanModel.__new__(CausalWanModel)
    nn.Module.__init__(model)
    model.freq_dim = 8
    model.dim = 4
    model.time_embedding = nn.Sequential(nn.Linear(model.freq_dim, model.dim), nn.SiLU(), nn.Linear(model.dim, model.dim))
    model.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(model.dim, model.dim * 6))
    model.sp_prepare = RecordingSPPrepare()
    model.text_embedding = nn.Identity()
    model.blocks = nn.ModuleList()
    model.head = IdentityHead()

    x = torch.randn(1, model.dim, 1, 4, 1)
    timestep = torch.zeros(1, 1)
    context = torch.randn(1, 1, model.dim)
    freqs = torch.ones(4, 1, 2, dtype=torch.complex64)

    CausalWanModel._forward_blocks(
        model,
        x=x,
        seq_len=4,
        freqs=freqs,
        timestep=timestep,
        context=context,
        clip_feature=None,
        embodiment_id=None,
        action=None,
        timestep_action=None,
        state=None,
        kv_cache=[],
        crossattn_cache=None,
        current_start_frame=0,
    )

    assert model.sp_prepare.seen_e_shape == (1, 4, model.dim * 6)


def test_causal_wan_gathers_video_tokens_before_head(monkeypatch):
    import vllm_omni.diffusion.models.dreamzero.causal_wan_model as causal_wan_model

    class LocalSPPrepare(nn.Module):
        def forward(self, x_video, e_video, freqs):
            return x_video[:, :2], e_video[:, :2], freqs[:2]

    class RecordingHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_x_shape = None
            self.seen_e_shape = None

        def forward(self, x, e):
            self.seen_x_shape = tuple(x.shape)
            self.seen_e_shape = tuple(e.shape)
            return x

    def fake_sp_gather(tensor, dim, validate=True):
        del validate
        assert dim == 1
        return torch.cat([tensor, tensor + 100], dim=dim)

    monkeypatch.setattr(causal_wan_model, "sp_gather", fake_sp_gather, raising=False)

    model = CausalWanModel.__new__(CausalWanModel)
    nn.Module.__init__(model)
    model.freq_dim = 8
    model.dim = 4
    model.time_embedding = nn.Sequential(nn.Linear(model.freq_dim, model.dim), nn.SiLU(), nn.Linear(model.dim, model.dim))
    model.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(model.dim, model.dim * 6))
    model.sp_prepare = LocalSPPrepare()
    model.text_embedding = nn.Identity()
    model.blocks = nn.ModuleList()
    model.head = RecordingHead()

    x = torch.randn(1, model.dim, 1, 4, 1)
    timestep = torch.zeros(1, 1)
    context = torch.randn(1, 1, model.dim)
    freqs = torch.ones(4, 1, 2, dtype=torch.complex64)

    CausalWanModel._forward_blocks(
        model,
        x=x,
        seq_len=4,
        freqs=freqs,
        timestep=timestep,
        context=context,
        clip_feature=None,
        embodiment_id=None,
        action=None,
        timestep_action=None,
        state=None,
        kv_cache=[],
        crossattn_cache=None,
        current_start_frame=0,
    )

    assert model.head.seen_x_shape == (1, 4, model.dim)
    assert model.head.seen_e_shape == (1, 4, 1, model.dim)


def test_causal_wan_disables_bf16_reduced_precision_reduction_during_blocks():
    if not hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        pytest.skip("bf16 reduced precision reduction flag is not available")

    class FlagRecordingTimeEmbedding(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim
            self.seen_flag = None

        def forward(self, x):
            self.seen_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            return torch.zeros(x.shape[0], self.dim, dtype=x.dtype, device=x.device)

    class RecordingSPPrepare(nn.Module):
        def forward(self, x_video, e_video, freqs):
            return x_video[:, :2], e_video[:, :2], freqs[:2]

    class IdentityHead(nn.Module):
        def forward(self, x, e):
            return x

    old_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    try:
        model = CausalWanModel.__new__(CausalWanModel)
        nn.Module.__init__(model)
        model.freq_dim = 8
        model.dim = 4
        model.time_embedding = FlagRecordingTimeEmbedding(model.dim)
        model.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(model.dim, model.dim * 6)).to(torch.bfloat16)
        model.sp_prepare = RecordingSPPrepare()
        model.text_embedding = nn.Identity()
        model.blocks = nn.ModuleList()
        model.head = IdentityHead()

        x = torch.randn(1, model.dim, 1, 4, 1, dtype=torch.bfloat16)
        timestep = torch.zeros(1, 1)
        context = torch.randn(1, 1, model.dim, dtype=torch.bfloat16)
        freqs = torch.ones(4, 1, 2, dtype=torch.complex64)

        CausalWanModel._forward_blocks(
            model,
            x=x,
            seq_len=4,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=None,
            embodiment_id=None,
            action=None,
            timestep_action=None,
            state=None,
            kv_cache=[],
            crossattn_cache=None,
            current_start_frame=0,
        )

        assert model.time_embedding.seen_flag is False
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_flag


class _FakeParallelLinear(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class _FakeQKVParallelLinear(nn.Module):
    total_num_kv_heads = 4
    total_num_heads = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class _RepeatQKVProjection(nn.Module):
    def forward(self, x):
        return torch.cat([x, x, x], dim=-1), None


class _RecordingCachedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen = {}

    def forward(self, query, key, value, attn_metadata=None):
        raise AssertionError("regular Attention.forward should not receive cached DreamZero KV")

    def forward_with_kv_cache(
        self,
        query,
        key,
        value,
        kv_cache,
        *,
        max_cache_len,
        attn_metadata=None,
    ):
        self.seen["query_shape"] = tuple(query.shape)
        self.seen["key_shape"] = tuple(key.shape)
        self.seen["value_shape"] = tuple(value.shape)
        self.seen["kv_cache"] = kv_cache
        self.seen["max_cache_len"] = max_cache_len
        self.seen["attn_metadata"] = attn_metadata
        if attn_metadata is not None and attn_metadata.joint_query is not None:
            if attn_metadata.joint_strategy == "rear":
                out = torch.cat([query, attn_metadata.joint_query], dim=1)
            else:
                out = torch.cat([attn_metadata.joint_query, query], dim=1)
            return torch.zeros_like(out), torch.stack([key, value], dim=0)
        return torch.zeros_like(query), torch.stack([key, value], dim=0)


def test_self_attention_enables_sequence_parallel_but_cross_attention_does_not(monkeypatch):
    import vllm_omni.diffusion.models.dreamzero.causal_wan_model as causal_wan_model

    monkeypatch.setattr(causal_wan_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(causal_wan_model, "ColumnParallelLinear", _FakeParallelLinear)
    monkeypatch.setattr(causal_wan_model, "RowParallelLinear", _FakeParallelLinear)
    monkeypatch.setattr(causal_wan_model, "QKVParallelLinear", _FakeQKVParallelLinear)

    self_attn = CausalWanSelfAttention(dim=32, num_heads=4, frame_seqlen=4, qk_norm=False)
    assert self_attn.attn.skip_sequence_parallel is False

    t2v_cross = WanT2VCrossAttention(dim=32, num_heads=4, qk_norm=False)
    i2v_cross = WanI2VCrossAttention(dim=32, num_heads=4, qk_norm=False)
    assert t2v_cross.attn.skip_sequence_parallel is True
    assert i2v_cross.attn.skip_sequence_parallel is True


def test_self_attention_uses_generic_cached_attention_for_video_tokens():
    self_attn = CausalWanSelfAttention.__new__(CausalWanSelfAttention)
    nn.Module.__init__(self_attn)
    self_attn.tp_num_heads = 2
    self_attn.head_dim = 4
    self_attn.max_attention_size = 8
    self_attn.num_frame_per_block = 1
    self_attn.num_action_per_block = 2
    self_attn.num_state_per_block = 1
    self_attn.qkv = _RepeatQKVProjection()
    self_attn.o = nn.Identity()
    self_attn.norm_q = nn.Identity()
    self_attn.norm_k = nn.Identity()
    self_attn.attn = _RecordingCachedAttention()

    x = torch.arange(1 * 2 * 8, dtype=torch.float32).reshape(1, 2, 8)
    kv_cache = torch.zeros(2, 1, 1, 2, 4)
    freqs = torch.ones(2, 1, 2, dtype=torch.complex128)
    freqs_action = torch.ones(4, 2, dtype=torch.complex128)
    freqs_state = torch.ones(4, 2, dtype=torch.complex128)

    out, updated_cache = CausalWanSelfAttention.forward(
        self_attn,
        x=x,
        freqs=freqs,
        freqs_action=freqs_action,
        freqs_state=freqs_state,
        action_register_length=None,
        kv_cache=kv_cache,
        current_start_frame=0,
    )

    assert self_attn.attn.seen["query_shape"] == (1, 2, 2, 4)
    assert self_attn.attn.seen["key_shape"] == (1, 2, 2, 4)
    assert self_attn.attn.seen["value_shape"] == (1, 2, 2, 4)
    assert self_attn.attn.seen["kv_cache"] is kv_cache
    assert self_attn.attn.seen["max_cache_len"] == 8
    assert self_attn.attn.seen["attn_metadata"] is None
    assert out.shape == x.shape
    assert updated_cache.shape == (2, 1, 2, 2, 4)


def test_self_attention_passes_action_state_tokens_as_joint_metadata():
    self_attn = CausalWanSelfAttention.__new__(CausalWanSelfAttention)
    nn.Module.__init__(self_attn)
    self_attn.tp_num_heads = 2
    self_attn.head_dim = 4
    self_attn.max_attention_size = 8
    self_attn.num_frame_per_block = 1
    self_attn.num_action_per_block = 2
    self_attn.num_state_per_block = 1
    self_attn.qkv = _RepeatQKVProjection()
    self_attn.o = nn.Identity()
    self_attn.norm_q = nn.Identity()
    self_attn.norm_k = nn.Identity()
    self_attn.attn = _RecordingCachedAttention()

    x = torch.arange(1 * 5 * 8, dtype=torch.float32).reshape(1, 5, 8)
    kv_cache = torch.zeros(2, 1, 1, 2, 4)
    freqs = torch.ones(2, 1, 2, dtype=torch.complex128)
    freqs_action = torch.ones(4, 2, dtype=torch.complex128)
    freqs_state = torch.ones(4, 2, dtype=torch.complex128)

    out, updated_cache = CausalWanSelfAttention.forward(
        self_attn,
        x=x,
        freqs=freqs,
        freqs_action=freqs_action,
        freqs_state=freqs_state,
        action_register_length=3,
        kv_cache=kv_cache,
        current_start_frame=0,
    )

    metadata = self_attn.attn.seen["attn_metadata"]
    assert metadata is not None
    assert metadata.joint_strategy == "rear"
    assert metadata.joint_query.shape == (1, 3, 2, 4)
    assert metadata.joint_key.shape == (1, 3, 2, 4)
    assert metadata.joint_value.shape == (1, 3, 2, 4)
    assert self_attn.attn.seen["query_shape"] == (1, 2, 2, 4)
    assert self_attn.attn.seen["key_shape"] == (1, 2, 2, 4)
    assert self_attn.attn.seen["value_shape"] == (1, 2, 2, 4)
    assert self_attn.attn.seen["kv_cache"] is kv_cache
    assert out.shape == x.shape
    assert updated_cache.shape == (2, 1, 2, 2, 4)


def test_append_attention_kv_cache_appends_fresh_tokens():
    batch = 1
    old_len = 3
    fresh_len = 5
    heads = 2
    head_dim = 4

    old_k = torch.arange(batch * old_len * heads * head_dim, dtype=torch.float32).reshape(
        batch, old_len, heads, head_dim
    )
    old_v = old_k + 1000
    cache = torch.stack([old_k, old_v], dim=0)

    fresh_k = torch.full((batch, fresh_len, heads, head_dim), 7.0)
    fresh_v = torch.full((batch, fresh_len, heads, head_dim), 11.0)

    updated = append_attention_kv_cache(
        kv_cache=cache,
        fresh_key=fresh_k,
        fresh_value=fresh_v,
        max_cache_len=32,
    )

    assert updated.shape == (2, batch, old_len + fresh_len, heads, head_dim)
    torch.testing.assert_close(updated[0, :, :old_len], old_k)
    torch.testing.assert_close(updated[1, :, :old_len], old_v)
    torch.testing.assert_close(updated[0, :, old_len:], fresh_k)
    torch.testing.assert_close(updated[1, :, old_len:], fresh_v)


def test_append_attention_kv_cache_trims_sequence_window_not_heads():
    cache = torch.zeros(2, 1, 6, 3, 4)
    fresh_k = torch.ones(1, 5, 3, 4)
    fresh_v = torch.ones(1, 5, 3, 4) * 2

    updated = append_attention_kv_cache(
        kv_cache=cache,
        fresh_key=fresh_k,
        fresh_value=fresh_v,
        max_cache_len=8,
    )

    assert updated.shape == (2, 1, 8, 3, 4)
    torch.testing.assert_close(updated[0, :, -5:], fresh_k)
    torch.testing.assert_close(updated[1, :, -5:], fresh_v)


def test_attention_forward_with_kv_cache_appends_after_parallel_prepare(monkeypatch):
    attn = Attention(
        num_heads=2,
        head_size=4,
        causal=False,
        softmax_scale=0.5,
        skip_sequence_parallel=False,
    )
    seen_key_lengths = []

    def record_local_attention(query, key, value, attn_metadata):
        seen_key_lengths.append(key.shape[1])
        del key, value, attn_metadata
        return query + len(seen_key_lengths)

    monkeypatch.setattr(attn, "_run_local_attention", record_local_attention)

    query_1 = torch.zeros(1, 2, 2, 4)
    key_1 = torch.ones(1, 2, 2, 4)
    value_1 = torch.ones(1, 2, 2, 4) * 10

    out_1, cache_1 = attn.forward_with_kv_cache(
        query_1,
        key_1,
        value_1,
        kv_cache=None,
        max_cache_len=8,
    )

    assert seen_key_lengths == [2]
    torch.testing.assert_close(out_1, torch.ones_like(query_1))
    torch.testing.assert_close(cache_1[0], key_1)
    torch.testing.assert_close(cache_1[1], value_1)

    query_2 = torch.zeros(1, 3, 2, 4)
    key_2 = torch.ones(1, 3, 2, 4) * 2
    value_2 = torch.ones(1, 3, 2, 4) * 20

    out_2, cache_2 = attn.forward_with_kv_cache(
        query_2,
        key_2,
        value_2,
        kv_cache=cache_1,
        max_cache_len=8,
    )

    assert seen_key_lengths == [2, 5]
    torch.testing.assert_close(out_2, torch.ones_like(query_2) * 2)
    torch.testing.assert_close(cache_2[0, :, :2], key_1)
    torch.testing.assert_close(cache_2[1, :, :2], value_1)
    torch.testing.assert_close(cache_2[0, :, 2:], key_2)
    torch.testing.assert_close(cache_2[1, :, 2:], value_2)


def test_attention_forward_with_kv_cache_does_not_persist_joint_tokens(monkeypatch):
    attn = Attention(
        num_heads=2,
        head_size=4,
        causal=False,
        softmax_scale=0.5,
        skip_sequence_parallel=False,
    )
    seen = {}

    def record_local_attention(query, key, value, attn_metadata):
        del attn_metadata
        seen["query_shape"] = tuple(query.shape)
        seen["key_shape"] = tuple(key.shape)
        seen["value_shape"] = tuple(value.shape)
        return query

    monkeypatch.setattr(attn, "_run_local_attention", record_local_attention)

    query = torch.zeros(1, 2, 2, 4)
    key = torch.ones(1, 2, 2, 4)
    value = torch.ones(1, 2, 2, 4) * 10
    kv_cache = torch.zeros(2, 1, 1, 2, 4)
    joint_query = torch.ones(1, 3, 2, 4) * 2
    joint_key = torch.ones(1, 3, 2, 4) * 3
    joint_value = torch.ones(1, 3, 2, 4) * 30

    out, updated_cache = attn.forward_with_kv_cache(
        query,
        key,
        value,
        kv_cache=kv_cache,
        max_cache_len=8,
        attn_metadata=AttentionMetadata(
            joint_query=joint_query,
            joint_key=joint_key,
            joint_value=joint_value,
            joint_strategy="rear",
        ),
    )

    assert seen["query_shape"] == (1, 5, 2, 4)
    assert seen["key_shape"] == (1, 6, 2, 4)
    assert seen["value_shape"] == (1, 6, 2, 4)
    assert out.shape == (1, 5, 2, 4)
    assert updated_cache.shape == (2, 1, 3, 2, 4)
    torch.testing.assert_close(updated_cache[0, :, :1], kv_cache[0])
    torch.testing.assert_close(updated_cache[1, :, :1], kv_cache[1])
    torch.testing.assert_close(updated_cache[0, :, 1:], key)
    torch.testing.assert_close(updated_cache[1, :, 1:], value)


def test_attention_forward_with_kv_cache_rejects_ring_parallel_cache(monkeypatch):
    attn = Attention(
        num_heads=2,
        head_size=4,
        causal=False,
        softmax_scale=0.5,
        skip_sequence_parallel=False,
    )
    attn.use_ring = True

    query = torch.zeros(1, 2, 2, 4)
    key = torch.ones(1, 2, 2, 4)
    value = torch.ones(1, 2, 2, 4)

    with pytest.raises(NotImplementedError, match="Ring sequence parallel KV cache"):
        attn.forward_with_kv_cache(
            query,
            key,
            value,
            kv_cache=None,
            max_cache_len=8,
        )


def test_causal_wan_model_reports_kv_cache_heads_for_ulysses_degree():
    model = CausalWanModel.__new__(CausalWanModel)
    model.blocks = [SimpleNamespace(self_attn=SimpleNamespace(tp_num_heads=4))]

    assert model.kv_cache_num_heads(ulysses_degree=1) == 4
    assert model.kv_cache_num_heads(ulysses_degree=2) == 2


def test_causal_wan_model_rejects_non_divisible_strict_kv_heads():
    model = CausalWanModel.__new__(CausalWanModel)
    model.blocks = [SimpleNamespace(self_attn=SimpleNamespace(tp_num_heads=3))]

    with pytest.raises(ValueError, match="must be divisible"):
        model.kv_cache_num_heads(ulysses_degree=2)


def test_sp_prepare_is_called_before_action_tokens_are_reconcatenated(monkeypatch):
    model = CausalWanModel(
        model_type="t2v",
        patch_size=(1, 1, 1),
        frame_seqlen=4,
        text_len=8,
        in_dim=2,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=2,
        num_heads=4,
        num_layers=0,
        hidden_size=16,
        action_dim=3,
        max_state_dim=5,
        num_action_per_block=2,
        num_state_per_block=1,
    )

    seen = {}

    def record_prepare(x_video, e_video, freqs):
        seen["x_video_shape"] = tuple(x_video.shape)
        seen["e_video_shape"] = tuple(e_video.shape)
        seen["freqs_shape"] = tuple(freqs.shape)
        return x_video, e_video, freqs

    monkeypatch.setattr(model.sp_prepare, "forward", record_prepare)

    x = torch.randn(1, 2, 1, 2, 2)
    timestep = torch.zeros(1, 1)
    context = torch.randn(1, 8, 16)
    action = torch.randn(1, 2, 3)
    timestep_action = torch.zeros(1, 2)
    state = torch.randn(1, 1, 5)

    model._forward_blocks(
        x=model.patch_embedding(x),
        seq_len=4,
        freqs=torch.randn(4, 1, 4, dtype=torch.complex128),
        timestep=timestep,
        context=context,
        clip_feature=None,
        embodiment_id=None,
        action=action,
        timestep_action=timestep_action,
        state=state,
        kv_cache=[],
        crossattn_cache=None,
        current_start_frame=0,
    )

    assert seen["x_video_shape"][1] == 4
    assert seen["e_video_shape"][1] == 4
    assert seen["freqs_shape"][0] == 4
