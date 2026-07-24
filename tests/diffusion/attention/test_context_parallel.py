from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.context_parallel import ContextParallelAttention
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.layers.mot.mot_qkv_parallel_linear import MoTQKVParallelLinear
from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel


class _CompletedWork:
    def wait(self) -> None:
        return None


def test_context_parallel_gathers_only_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    gathered_inputs: list[torch.Tensor] = []
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def _all_gather_into_tensor(output, tensor, *, group, async_op):
        del group
        assert async_op
        gathered_inputs.append(tensor.clone())
        output[: tensor.shape[0]].copy_(tensor)
        output[tensor.shape[0] :].copy_(tensor + 10)
        return _CompletedWork()

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor)
    strategy = ContextParallelAttention(SimpleNamespace(device_group=object()))
    query = torch.randn(1, 2, 28, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    joint_query = torch.randn(1, 1, 28, 8)
    joint_key = torch.randn(1, 3, 4, 8)
    joint_value = torch.randn(1, 3, 4, 8)

    out_q, out_k, out_v, _, _ = strategy.pre_attention(
        query,
        key,
        value,
        AttentionMetadata(
            joint_query=joint_query,
            joint_key=joint_key,
            joint_value=joint_value,
        ),
    )

    assert len(gathered_inputs) == 2
    assert torch.equal(gathered_inputs[0], key)
    assert torch.equal(gathered_inputs[1], value)
    assert torch.equal(out_q[:, 1:], query)
    assert out_k.shape == (1, 7, 4, 8)
    assert out_v.shape == (1, 7, 4, 8)
    assert torch.equal(out_k[:, :3], joint_key)
    assert torch.equal(out_k[:, 3:5], key)
    assert torch.equal(out_k[:, 5:], key + 10)


def test_context_parallel_config_env_fallback_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_CP_DEGREE", "4")
    from_env = DiffusionParallelConfig()
    explicit = DiffusionParallelConfig(context_parallel_degree=2)

    assert from_env.context_parallel_degree == 4
    assert from_env.sequence_parallel_size == 4
    assert explicit.context_parallel_degree == 2
    assert explicit.sequence_parallel_size == 2


def test_context_parallel_rejects_ulysses_or_ring() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        DiffusionParallelConfig(context_parallel_degree=2, ulysses_degree=2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        DiffusionParallelConfig(context_parallel_degree=2, ring_degree=2)


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


def test_context_parallel_global_cfg_renorm_uses_all_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    local_v = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    local_cfg = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    remote_v = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    remote_cfg = torch.tensor([[2.5, 3.0], [3.5, 4.0]])
    text_scale = 2.0
    remote_guided = remote_cfg + text_scale * (remote_v - remote_cfg)

    def _all_reduce(tensor, *, op, group) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert group is process_group
        tensor.add_(
            torch.stack(
                (
                    remote_v.square().sum(),
                    remote_guided.square().sum(),
                )
            )
        )

    process_group = object()
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


def test_persistent_context_parallel_gathers_only_final_latent() -> None:
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

    def _forward_single_branch_local(local_x_t, *args):
        del args
        calls["forward"] += 1
        return torch.ones_like(local_x_t)

    def _gather_vae_for_sp(local_x_t):
        calls["gather"] += 1
        return torch.cat((local_x_t, local_x_t))

    model._split_vae_for_sp = _split_vae_for_sp
    model._forward_single_branch_local = _forward_single_branch_local
    model._gather_vae_for_sp = _gather_vae_for_sp

    result, trajectories, trajectory_timesteps, log_probs = Bagel._generate_image_context_parallel(
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

    assert calls == {"forward": 3, "gather": 1}
    assert len(result) == 1
    assert torch.equal(result[0], torch.full((4, 2), -3.0))
    assert trajectories is trajectory_timesteps is log_probs is None
