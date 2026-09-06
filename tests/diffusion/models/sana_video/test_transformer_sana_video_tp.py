# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TP unit tests for SANA-Video: distributed RMSNorm, sharded attention and
sharded GLUMB temporal conv.

CPU-only. The TP world size and the collective are mocked so a single process
simulates the shard/reduce logic a real multi-rank run depends on.
"""

import os

import pytest
import torch

from vllm_omni.diffusion.models.sana_video.transformer_sana_video import (
    GLUMBTempConv,
    SanaCrossAttention,
    SanaDistributedRMSNorm,
    SanaLinearAttention,
    SanaRMSNorm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODULE = "vllm_omni.diffusion.models.sana_video.transformer_sana_video"


@pytest.fixture
def tp1_group():
    """Real single-process TP group so the parallel linear layers construct and
    run on CPU at tensor_parallel_size=1."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29501")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture
def force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for the parallel linear layers."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def test_distributed_rms_norm_tp1_matches_sana_rms_norm(mocker) -> None:
    """TP1 distributed norm must be bit-identical to the checkpoint SanaRMSNorm."""
    dim = 24
    torch.manual_seed(0)

    dense = SanaRMSNorm(dim, eps=1e-5, elementwise_affine=True)
    dense.weight.data.normal_()

    dist = SanaDistributedRMSNorm(dim, eps=1e-5)
    dist.weight.data.copy_(dense.weight.data)

    x = torch.randn(2, 5, dim)
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1)
    out = dist(x)
    ref = dense(x)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_distributed_rms_norm_tp1_bf16_weight_matches_sana_rms_norm(mocker) -> None:
    """bf16 weight must follow SanaRMSNorm's cast-to-weight-dtype branch.

    An unconditional cast back to the input dtype would diverge here; this pins
    the fp16/bf16-only cast that SanaRMSNorm performs.
    """
    dim = 24
    torch.manual_seed(0)

    dense = SanaRMSNorm(dim, eps=1e-5, elementwise_affine=True)
    dense.weight.data.normal_()
    dense.weight.data = dense.weight.data.to(torch.bfloat16)

    dist = SanaDistributedRMSNorm(dim, eps=1e-5)
    dist.weight.data = dense.weight.data.clone()

    x = torch.randn(2, 5, dim, dtype=torch.bfloat16)
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1)
    out = dist(x)
    ref = dense(x)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_distributed_rms_norm_tp2_shards_aggregate_to_dense(mocker) -> None:
    """Two TP2 shards must reproduce dense SanaRMSNorm.

    The RMS must use the full (unsharded) hidden size: each rank reduces its
    local sum-of-squares across ranks, not a per-shard mean over its half.
    """
    full_dim, tp_size = 24, 2
    half = full_dim // tp_size
    torch.manual_seed(0)

    dense = SanaRMSNorm(full_dim, eps=1e-5, elementwise_affine=True)
    dense.weight.data.normal_()

    dist0 = SanaDistributedRMSNorm(half, eps=1e-5)
    dist1 = SanaDistributedRMSNorm(half, eps=1e-5)
    dist0.weight.data.copy_(dense.weight.data[:half])
    dist1.weight.data.copy_(dense.weight.data[half:])

    x = torch.randn(2, 5, full_dim)
    x0, x1 = x[..., :half], x[..., half:]

    # A real all-reduce returns the sum of every rank's local sum-of-squares.
    global_sum_sq = x.to(torch.float32).pow(2).sum(-1, keepdim=True)

    def fake_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        return global_sum_sq

    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=tp_size)
    mocker.patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=fake_all_reduce)
    out0 = dist0(x0)
    out1 = dist1(x1)

    out = torch.cat([out0, out1], dim=-1)
    ref = dense(x)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_distributed_rms_norm_weight_loader_shards_dim0(mocker) -> None:
    """Each rank loads its own dim-0 slice of the global affine weight."""
    full_dim, tp_size, rank = 24, 2, 1
    half = full_dim // tp_size

    norm = SanaDistributedRMSNorm(half, eps=1e-5)
    full_weight = torch.randn(full_dim)

    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=tp_size)
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=rank)
    norm.weight.weight_loader(norm.weight, full_weight)

    torch.testing.assert_close(norm.weight.data, full_weight[rank * half : (rank + 1) * half])


def test_distributed_rms_norm_weight_loader_tp1_copies_full() -> None:
    dim = 24
    norm = SanaDistributedRMSNorm(dim, eps=1e-5)
    full_weight = torch.randn(dim)

    norm.weight.weight_loader(norm.weight, full_weight)

    torch.testing.assert_close(norm.weight.data, full_weight)


_SELF_ATTN_GOLDEN = torch.tensor(
    [
        5.0944157,
        -16.9981823,
        55.4457436,
        42.5174408,
        -9.3055859,
        19.3744793,
        -47.4345360,
        -5.8110709,
        -14.0679598,
        -22.3440742,
        -8.3834534,
        8.9109459,
    ]
)


def _fill_by_name(module: torch.nn.Module) -> None:
    """Deterministic per-parameter fill that is stable across TP refactors."""
    torch.manual_seed(0)
    for _, param in sorted(module.named_parameters()):
        param.data.normal_()


def _self_attn_inputs(seq_len: int, dim: int, head_dim: int):
    torch.manual_seed(1)
    x = torch.randn(2, seq_len, dim)
    freqs_cos = torch.randn(1, seq_len, 1, head_dim)
    freqs_sin = torch.randn(1, seq_len, 1, head_dim)
    return x, (freqs_cos, freqs_sin)


def test_self_attention_tp1_matches_golden_and_keeps_param_names(tp1_group, force_default_gemm) -> None:
    """TP1 self-attention stays bit-identical to PR1 and preserves checkpoint keys."""
    dim, num_heads, head_dim, seq_len = 24, 4, 6, 8
    attn = SanaLinearAttention(
        dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=0.0, bias=False, qk_norm="rms_norm_across_heads"
    )
    attn.eval()
    _fill_by_name(attn)
    x, rotary_emb = _self_attn_inputs(seq_len, dim, head_dim)

    with torch.no_grad():
        out = attn(x, rotary_emb=rotary_emb)

    assert {name for name, _ in attn.named_parameters()} == {
        "to_q.weight",
        "to_k.weight",
        "to_v.weight",
        "norm_q.weight",
        "norm_k.weight",
        "to_out.0.weight",
        "to_out.0.bias",
    }
    torch.testing.assert_close(out.flatten()[:12], _SELF_ATTN_GOLDEN, rtol=1e-5, atol=1e-5)


def _mock_tp2(mocker):
    """Mock a TP world size of 2 for module construction (no real group)."""
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=2)
    mocker.patch(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        return_value=2,
    )
    tp_group = mocker.MagicMock()
    tp_group.world_size = 2
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp_group)


def test_self_attention_tp2_shards_projections_and_heads(mocker) -> None:
    """TP2 splits heads column-wise (q/k/v) and the output projection row-wise."""
    dim, num_heads, head_dim = 24, 4, 6
    inner_dim = num_heads * head_dim
    _mock_tp2(mocker)

    attn = SanaLinearAttention(
        dim=dim, num_heads=num_heads, head_dim=head_dim, dropout=0.0, bias=False, qk_norm="rms_norm_across_heads"
    )

    assert attn.heads == num_heads // 2
    assert attn.to_q.weight.shape == (inner_dim // 2, dim)
    assert attn.to_k.weight.shape == (inner_dim // 2, dim)
    assert attn.to_v.weight.shape == (inner_dim // 2, dim)
    assert attn.to_out[0].weight.shape == (dim, inner_dim // 2)
    assert attn.norm_q.weight.shape == (inner_dim // 2,)
    assert attn.norm_k.weight.shape == (inner_dim // 2,)


def test_cross_attention_tp2_shards_projections_and_heads(mocker) -> None:
    """TP2 cross-attention shards q/k/v column-wise, to_out row-wise, heads locally."""
    dim, cross_dim, num_heads, head_dim = 24, 16, 4, 6
    inner_dim = num_heads * head_dim
    _mock_tp2(mocker)

    attn = SanaCrossAttention(
        dim=dim,
        cross_attention_dim=cross_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        dropout=0.0,
        bias=False,
        out_bias=True,
        qk_norm="rms_norm_across_heads",
        prefix="attn2",
    )

    assert attn.heads == num_heads // 2
    assert attn.to_q.weight.shape == (inner_dim // 2, dim)
    assert attn.to_k.weight.shape == (inner_dim // 2, cross_dim)
    assert attn.to_v.weight.shape == (inner_dim // 2, cross_dim)
    assert attn.to_out[0].weight.shape == (dim, inner_dim // 2)
    assert isinstance(attn.norm_q, SanaDistributedRMSNorm)
    assert attn.norm_q.weight.shape == (inner_dim // 2,)


_TINY_TRANSFORMER_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "num_layers": 1,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 12,
    "cross_attention_dim": 24,
    "caption_channels": 8,
    "mlp_ratio": 2.0,
    "patch_size": (1, 2, 2),
    "sample_size": 4,
    "rope_max_seq_len": 32,
}


def test_tp2_load_weights_shards_full_checkpoint_without_missing(mocker) -> None:
    """The full checkpoint loads under TP2 via weight_loaders with no missing/unexpected keys."""
    from diffusers import SanaVideoTransformer3DModel as DiffusersTransformer

    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    _mock_tp2(mocker)
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=0)
    mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0)

    reference = DiffusersTransformer(**_TINY_TRANSFORMER_CONFIG)
    model = SanaVideoTransformer3DModel(**_TINY_TRANSFORMER_CONFIG)

    loaded = model.load_weights(list(reference.state_dict().items()))

    assert loaded == set(model.state_dict())


# ── GLUMB temporal conv TP ──

# hidden_channels = expand_ratio * in_channels = 8, so conv_inverted emits 2h = 16.
_GLUMB_IN, _GLUMB_OUT, _GLUMB_RATIO = 4, 4, 2.0

_GLUMB_GOLDEN = torch.tensor(
    [
        51.4757385,
        -10.9912930,
        -4.1482553,
        -26.1580830,
        -19.7727261,
        17.9901562,
        -1.6007471,
        13.3993073,
        -8.6459389,
        -10.4008112,
        -7.4962354,
        1.5446399,
    ]
)


def _glumb_inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, 2, 3, 3, _GLUMB_IN)  # [B, F, H, W, C]


def test_glumb_tp1_matches_golden_and_keeps_param_names(tp1_group) -> None:
    """TP1 GLUMB stays bit-identical to the PR1 conv output and preserves keys."""
    glumb = GLUMBTempConv(_GLUMB_IN, _GLUMB_OUT, _GLUMB_RATIO, norm_type=None, residual_connection=False)
    glumb.eval()
    _fill_by_name(glumb)
    x = _glumb_inputs()

    with torch.no_grad():
        out = glumb(x)

    assert {name for name, _ in glumb.named_parameters()} == {
        "conv_inverted.weight",
        "conv_inverted.bias",
        "conv_depth.weight",
        "conv_depth.bias",
        "conv_point.weight",
        "conv_temp.weight",
    }
    torch.testing.assert_close(out.flatten()[:12], _GLUMB_GOLDEN, rtol=1e-5, atol=1e-5)


def _random_glumb_checkpoint(hidden_channels: int) -> dict[str, torch.Tensor]:
    """Checkpoint-format (nn.Conv2d) GLUMB weights with per-segment asymmetry.

    conv_inverted / conv_depth carry two h-channel segments (value then gate).
    Filling the gate half from a different distribution than the value half is
    what exposes a naive contiguous dim-0 split: that bug would place the whole
    value segment on rank 0 and the whole gate segment on rank 1, so each rank's
    local chunk(2) would slice value-vs-value instead of value-vs-gate.
    """
    torch.manual_seed(0)
    h = hidden_channels
    value_w = torch.randn(h, _GLUMB_IN, 1, 1)
    gate_w = torch.randn(h, _GLUMB_IN, 1, 1) * 3.0 + 5.0
    value_b = torch.randn(h)
    gate_b = torch.randn(h) * 3.0 + 5.0
    value_dw = torch.randn(h, 1, 3, 3)
    gate_dw = torch.randn(h, 1, 3, 3) * 3.0 + 5.0
    value_db = torch.randn(h)
    gate_db = torch.randn(h) * 3.0 + 5.0
    return {
        "conv_inverted.weight": torch.cat([value_w, gate_w], dim=0),
        "conv_inverted.bias": torch.cat([value_b, gate_b], dim=0),
        "conv_depth.weight": torch.cat([value_dw, gate_dw], dim=0),
        "conv_depth.bias": torch.cat([value_db, gate_db], dim=0),
        "conv_point.weight": torch.randn(_GLUMB_OUT, h, 1, 1),
        "conv_temp.weight": torch.randn(_GLUMB_OUT, _GLUMB_OUT, 3, 1),
    }


def _mock_glumb_tp(mocker, world_size: int, rank: int) -> None:
    """Mock a TP group of ``world_size`` at ``rank`` with an identity all-reduce,
    so a single process can build and run one rank's shard in isolation."""
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=world_size)
    mocker.patch(f"{_MODULE}.get_tensor_model_parallel_rank", return_value=rank)
    # Identity conv_point all-reduce returns each rank's local partial for the
    # test to sum.
    mocker.patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=lambda t: t)


def _load_glumb_checkpoint(glumb: GLUMBTempConv, checkpoint: dict[str, torch.Tensor]) -> None:
    """Load like AutoWeightsLoader: custom weight_loader when present (the sharded
    convs), plain copy for replicated params (conv_temp)."""
    params = dict(glumb.named_parameters())
    for name, tensor in checkpoint.items():
        param = params[name]
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            param.data.copy_(tensor)
        else:
            loader(param, tensor)


def test_glumb_tp2_shards_aggregate_to_dense(mocker) -> None:
    """The merged value/gate two-segment shard must aggregate to the dense conv.

    Everything after conv_point (temporal conv + residual) is linear and
    bias-free, so summing each rank's full forward (with the conv_point
    all-reduce mocked to identity) reproduces the replicated dense output.
    """
    hidden_channels = int(_GLUMB_RATIO * _GLUMB_IN)
    checkpoint = _random_glumb_checkpoint(hidden_channels)
    x = _glumb_inputs()

    _mock_glumb_tp(mocker, world_size=1, rank=0)
    dense = GLUMBTempConv(_GLUMB_IN, _GLUMB_OUT, _GLUMB_RATIO, norm_type=None, residual_connection=False)
    dense.eval()
    _load_glumb_checkpoint(dense, checkpoint)
    with torch.no_grad():
        ref = dense(x)

    partials = []
    for rank in (0, 1):
        _mock_glumb_tp(mocker, world_size=2, rank=rank)
        shard = GLUMBTempConv(_GLUMB_IN, _GLUMB_OUT, _GLUMB_RATIO, norm_type=None, residual_connection=False)
        shard.eval()
        _load_glumb_checkpoint(shard, checkpoint)
        with torch.no_grad():
            partials.append(shard(x))

    aggregated = partials[0] + partials[1]
    torch.testing.assert_close(aggregated, ref, rtol=1e-5, atol=1e-5)
