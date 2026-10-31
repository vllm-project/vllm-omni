# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BAGEL on AllGather-KV SP: async K/V gather, pre-gathered attention, persistent shard."""

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.bagel.bagel_transformer as bagel_transformer
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.allgather_kv import (
    ALLGATHER_KV_PRE_GATHERED,
    AllGatherKVParallelAttention,
    async_all_gather_sequence,
)
from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    Bagel,
    PackedAttentionMoT,
)
from vllm_omni.diffusion.models.bagel.mot.mot_layernorm import MoTRMSNorm
from vllm_omni.diffusion.models.bagel.mot.mot_qkv_parallel_linear import MoTQKVParallelLinear

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _CompletedWork:
    def wait(self) -> None:
        return None


class _MockSPGroup:
    """Enough of SequenceParallelGroupCoordinator for the strategy under test."""

    def __init__(self, world_size: int, rank: int = 0) -> None:
        self.allgather_group = object()
        self.allgather_world_size = world_size
        self.allgather_rank = rank
        self.gather_calls: list[torch.Tensor] = []

    def all_gather(self, tensor: torch.Tensor, dim: int, group: object) -> torch.Tensor:
        assert group is self.allgather_group
        self.gather_calls.append(tensor.clone())
        return torch.cat([tensor + 10 * r for r in range(self.allgather_world_size)], dim=dim)


@pytest.mark.parametrize(("batch_size", "shares_storage"), [(1, True), (2, False)])
def test_async_all_gather_sequence_concatenates_on_sequence(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    shares_storage: bool,
) -> None:
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def _all_gather_into_tensor(output, tensor, *, group, async_op):
        del group
        assert async_op
        output[: tensor.shape[0]].copy_(tensor)
        output[tensor.shape[0] :].copy_(tensor + 10)
        return _CompletedWork()

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor)
    local = torch.randn(batch_size, 3, 4, 8)

    pending = async_all_gather_sequence(local, object())
    gathered = pending.wait()

    # Same layout as SequenceParallelGroupCoordinator.all_gather(dim=1): rank 0, then rank 1.
    assert gathered.shape == (batch_size, 6, 4, 8)
    assert torch.equal(gathered[:, :3], local)
    assert torch.equal(gathered[:, 3:], local + 10)
    assert gathered.is_contiguous()
    assert (gathered.data_ptr() == pending.output.data_ptr()) is shares_storage


def test_async_all_gather_sequence_rejects_rank_one_tensors() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        async_all_gather_sequence(torch.zeros(4), object())


def test_allgather_kv_skips_gather_when_model_pre_gathered_kv() -> None:
    sp_group = _MockSPGroup(world_size=2, rank=1)
    strategy = AllGatherKVParallelAttention(sp_group=sp_group)
    query = torch.randn(1, 2, 28, 8)
    key_full = torch.randn(1, 4, 4, 8)  # already gathered by the model
    value_full = torch.randn(1, 4, 4, 8)
    joint_query = torch.randn(1, 1, 28, 8)
    joint_key = torch.randn(1, 3, 4, 8)
    joint_value = torch.randn(1, 3, 4, 8)

    out_q, out_k, out_v, out_meta, ctx = strategy.pre_attention(
        query,
        key_full,
        value_full,
        AttentionMetadata(
            joint_query=joint_query,
            joint_key=joint_key,
            joint_value=joint_value,
            joint_strategy="front",
            extra={ALLGATHER_KV_PRE_GATHERED: True},
        ),
    )

    assert sp_group.gather_calls == []
    assert ctx.name == "allgather_kv"
    assert ALLGATHER_KV_PRE_GATHERED not in out_meta.extra
    assert torch.equal(out_q[:, 1:], query)
    assert torch.equal(out_k[:, :3], joint_key)
    assert torch.equal(out_k[:, 3:], key_full)
    assert torch.equal(out_v[:, 3:], value_full)
    # Query ranges still describe this rank's slice of the gathered image span.
    assert out_meta.query_ranges is not None
    joint_range, image_range = out_meta.query_ranges
    assert (joint_range.local_start, joint_range.local_end) == (0, 1)
    assert (image_range.local_start, image_range.local_end) == (1, 3)
    assert image_range.global_start == 3 + 1 * 2  # joint_len + rank * img_seq_local


def test_allgather_kv_still_gathers_without_the_pre_gathered_flag() -> None:
    sp_group = _MockSPGroup(world_size=2, rank=0)
    strategy = AllGatherKVParallelAttention(sp_group=sp_group)
    query = torch.randn(1, 2, 28, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)

    _, out_k, out_v, _, _ = strategy.pre_attention(query, key, value, AttentionMetadata())

    assert len(sp_group.gather_calls) == 2
    assert torch.equal(sp_group.gather_calls[0], key)
    assert torch.equal(sp_group.gather_calls[1], value)
    assert out_k.shape == (1, 4, 4, 8)
    assert out_v.shape == (1, 4, 4, 8)


def test_gen_component_projection_uses_separate_weight_slices() -> None:
    layer = MoTQKVParallelLinear.__new__(MoTQKVParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.gen_exp = torch.nn.Module()
    layer.gen_exp.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    layer.gen_exp.bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    layer.output_partition_sizes = [2, 1, 1]
    inputs = torch.ones(3, 4)

    q = layer.forward_gen_component(inputs, "q")
    k = layer.forward_gen_component(inputs, "k")
    v = layer.forward_gen_component(inputs, "v")

    expected = torch.nn.functional.linear(inputs, layer.gen_exp.weight, layer.gen_exp.bias)
    assert torch.equal(q, expected[:, :2])
    assert torch.equal(k, expected[:, 2:3])
    assert torch.equal(v, expected[:, 3:])
    with pytest.raises(ValueError, match="Unknown QKV component"):
        layer.forward_gen_component(inputs, "x")


def test_gen_component_projection_rejects_packed_quantized_weights() -> None:
    layer = MoTQKVParallelLinear.__new__(MoTQKVParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.gen_exp = torch.nn.Module()
    layer.gen_exp.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.int8), requires_grad=False)
    layer.gen_exp.bias = None
    layer.output_partition_sizes = [2, 1, 1]

    assert not layer.supports_separate_gen_projection
    with pytest.raises(NotImplementedError):
        layer.forward_gen_component(torch.ones(3, 4), "q")


def test_forward_gen_norm_uses_generation_weight() -> None:
    layer = MoTRMSNorm(4, head_norm=True)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.gen_weight.fill_(2.0)
    inputs = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[4.0, 3.0, 2.0, 1.0]],
        ]
    )

    actual = layer.forward_gen(inputs)
    variance = inputs.float().square().mean(dim=-1, keepdim=True)
    expected = inputs * torch.rsqrt(variance + layer.variance_epsilon) * 2.0

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, layer.forward_native(inputs))


def test_forward_gen_allgather_kv_runs_overlapped_path(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    captured_metadata: list[AttentionMetadata] = []

    class _QKVProjection:
        def forward_gen_component(self, inputs: torch.Tensor, component: str) -> torch.Tensor:
            events.append(f"project:{component}")
            width = 4 if component == "q" else 2
            value = {"k": 1.0, "v": 2.0, "q": 3.0}[component]
            return torch.full((inputs.shape[0], width), value, dtype=inputs.dtype)

        def __call__(self, inputs: torch.Tensor):
            events.append("project:text")
            return torch.zeros((inputs.shape[0], 8), dtype=inputs.dtype), None

    class _Norm:
        def forward_gen(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

    class _OutputProjection:
        def __call__(self, inputs: torch.Tensor):
            events.append("project:text_output")
            return inputs, None

        def forward_gen(self, inputs: torch.Tensor):
            events.append("project:vae_output")
            return inputs, None

    def _rotary(inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        del cos, sin
        return inputs

    def _async_gather(inputs: torch.Tensor, group: object):
        assert group is process_group
        name = "k" if torch.all(inputs == 1) else "v"
        events.append(f"gather:{name}")

        class _Gather:
            def wait(self) -> torch.Tensor:
                events.append(f"wait:{name}")
                return inputs

        return _Gather()

    def _attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        events.append("attention")
        captured_metadata.append(metadata)
        assert key.shape == value.shape == (1, 2, 1, 2)
        return torch.cat((metadata.joint_query, query), dim=1)

    process_group = object()
    monkeypatch.setattr(bagel_transformer, "get_sp_group", lambda: SimpleNamespace(allgather_group=process_group))
    monkeypatch.setattr(bagel_transformer, "async_all_gather_sequence", _async_gather)
    layer = SimpleNamespace(
        hidden_size=4,
        head_dim=2,
        num_heads=2,
        num_kv_heads=1,
        q_size=4,
        kv_size=2,
        layer_idx=0,
        qkv_proj=_QKVProjection(),
        q_norm=_Norm(),
        k_norm=_Norm(),
        rotary_op=_rotary,
        attn_noncausal=_attention,
        o_proj=_OutputProjection(),
    )

    output, cache = PackedAttentionMoT._forward_gen_allgather_kv(
        layer,
        packed_query_sequence=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        packed_query_position_embeddings=(torch.ones(3, 2), torch.zeros(3, 2)),
        past_key_values=None,
        packed_vae_token_indexes=torch.tensor([1, 2]),
        packed_text_indexes=torch.tensor([0]),
    )

    assert output.shape == (3, 4)
    assert cache is None
    assert events[:7] == [
        "project:k",
        "gather:k",
        "project:v",
        "gather:v",
        "project:q",
        "project:text",
        "wait:k",
    ]
    assert events[7:9] == ["wait:v", "attention"]
    assert captured_metadata[0].extra == {ALLGATHER_KV_PRE_GATHERED: True}


def test_forward_gen_allgather_kv_drains_inflight_gathers_on_projection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[str] = []

    class _QKVProjection:
        def forward_gen_component(self, inputs: torch.Tensor, component: str) -> torch.Tensor:
            if component == "q":
                raise RuntimeError("Q projection failed")
            width = 2
            value = {"k": 1.0, "v": 2.0}[component]
            return torch.full((inputs.shape[0], width), value, dtype=inputs.dtype)

    class _Norm:
        def forward_gen(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

    def _rotary(inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        del cos, sin
        return inputs

    def _async_gather(inputs: torch.Tensor, group: object):
        assert group is process_group
        name = "k" if torch.all(inputs == 1) else "v"

        class _Gather:
            def wait(self) -> torch.Tensor:
                waits.append(name)
                return inputs

        return _Gather()

    process_group = object()
    monkeypatch.setattr(bagel_transformer, "get_sp_group", lambda: SimpleNamespace(allgather_group=process_group))
    monkeypatch.setattr(bagel_transformer, "async_all_gather_sequence", _async_gather)
    layer = SimpleNamespace(
        head_dim=2,
        num_heads=2,
        num_kv_heads=1,
        qkv_proj=_QKVProjection(),
        q_norm=_Norm(),
        k_norm=_Norm(),
        rotary_op=_rotary,
    )

    with pytest.raises(RuntimeError, match="Q projection failed"):
        PackedAttentionMoT._forward_gen_allgather_kv(
            layer,
            packed_query_sequence=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            packed_query_position_embeddings=(torch.ones(3, 2), torch.zeros(3, 2)),
            past_key_values=None,
            packed_vae_token_indexes=torch.tensor([1, 2]),
            packed_text_indexes=torch.tensor([0]),
        )

    assert waits == ["k", "v"]


def test_forward_gen_uses_fused_path_when_separate_projection_is_unsupported() -> None:
    class _FusedPathReachedError(RuntimeError):
        pass

    class _FusedOnlyQKV(torch.nn.Module):
        supports_separate_gen_projection = False

        def forward(self, *args, **kwargs):
            raise _FusedPathReachedError

    layer = PackedAttentionMoT.__new__(PackedAttentionMoT)
    torch.nn.Module.__init__(layer)
    layer.qkv_proj = _FusedOnlyQKV()
    layer._is_allgather_kv_active = lambda: True

    with pytest.raises(_FusedPathReachedError):
        layer._forward_gen(
            packed_query_sequence=torch.zeros(2, 4),
            query_lens=torch.tensor([2]),
            packed_query_position_embeddings=(torch.ones(2, 2), torch.zeros(2, 2)),
            past_key_values=None,
            packed_vae_token_indexes=torch.tensor([1]),
            packed_text_indexes=torch.tensor([0]),
        )


def test_global_cfg_renorm_reduces_norms_across_the_allgather_group(monkeypatch: pytest.MonkeyPatch) -> None:
    local_v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    local_cfg = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    remote_v = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    remote_cfg = torch.tensor([[2.5, 3.0], [3.5, 4.0]])
    text_scale = 2.0
    remote_guided = remote_cfg + text_scale * (remote_v - remote_cfg)
    process_group = object()

    def _all_reduce(tensor, *, op, group) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert group is process_group
        tensor.add_(torch.stack((remote_v.square().sum(), remote_guided.square().sum())))

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    actual = Bagel._combine_cfg(
        local_v,
        local_cfg,
        None,
        text_scale,
        1.0,
        "global",
        0.0,
        global_norm_group=process_group,
    )
    expected = Bagel._combine_cfg(
        torch.cat((local_v, remote_v)),
        torch.cat((local_cfg, remote_cfg)),
        None,
        text_scale,
        1.0,
        "global",
        0.0,
    )[: local_v.shape[0]]

    torch.testing.assert_close(actual, expected)


def test_persistent_shard_gathers_only_the_final_latent() -> None:
    calls = {"forward": 0, "gather": 0}
    model = SimpleNamespace()

    def _split_vae_for_sp(x_t, vae_position_ids, *args):
        del args
        local_size = x_t.shape[0] // 2
        return (
            x_t[:local_size],
            vae_position_ids[:local_size],
            torch.arange(local_size),
            torch.empty(0, dtype=torch.long),
            torch.tensor([local_size]),
            torch.arange(local_size),
        )

    def _forward_denoise_branch_local(local_x_t, *args):
        del args
        calls["forward"] += 1
        return torch.ones_like(local_x_t)

    def _gather_vae_for_sp(local_x_t):
        calls["gather"] += 1
        return torch.cat((local_x_t, local_x_t))

    model._split_vae_for_sp = _split_vae_for_sp
    model._forward_denoise_branch_local = _forward_denoise_branch_local
    model._gather_vae_for_sp = _gather_vae_for_sp

    result, trajectories, trajectory_timesteps, log_probs = Bagel._generate_image_allgather_kv(
        model,
        x_t=torch.zeros(4, 2),
        timesteps=torch.tensor([1.0, 0.5, 0.0]),
        dts=torch.ones(3),
        packed_text_ids=torch.empty(0, dtype=torch.long),
        packed_text_indexes=torch.empty(0, dtype=torch.long),
        packed_vae_position_ids=torch.arange(4),
        packed_vae_token_indexes=torch.arange(4),
        packed_seqlens=torch.tensor([6]),
        packed_position_ids=torch.arange(4),
        past_key_values=object(),
        cfg_interval=(0.0, 1.0),
        cfg_text_scale=1.0,
        cfg_text_packed_position_ids=None,
        cfg_text_past_key_values=None,
        cfg_img_scale=1.0,
        cfg_img_packed_position_ids=None,
        cfg_img_past_key_values=None,
        cfg_renorm_type="global",
        cfg_renorm_min=0.0,
    )

    # Three denoise steps ran on the local shard; the latent crossed the group once.
    assert calls == {"forward": 3, "gather": 1}
    assert len(result) == 1
    assert torch.equal(result[0], torch.full((4, 2), -3.0))
    assert trajectories is trajectory_timesteps is log_probs is None


def test_skip_sequence_parallel_layer_does_not_build_a_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.diffusion.attention.layer as layer_mod

    def _boom(**kwargs):
        raise AssertionError("factory must not run for a layer that opts out of SP")

    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", _boom)
    attn = Attention(num_heads=4, head_size=8, softmax_scale=0.5, causal=True, skip_sequence_parallel=True)

    assert attn.parallel_strategy is attn._no_parallel_strategy
    with pytest.raises(AssertionError, match="factory must not run"):
        Attention(num_heads=4, head_size=8, softmax_scale=0.5, causal=True)
