# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def tp1(monkeypatch):
    from vllm.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))


def test_curve_interpolates_and_clamps_each_stream_timestep():
    from vllm_omni.diffusion.models.minimax_h3.adaln_curve import MiniMaxH3AdalnCurve

    curve = MiniMaxH3AdalnCurve(3, 2)
    curve.table.data.copy_(torch.tensor([[0.0, 4.0], [2.0, 8.0], [10.0, -4.0]]))
    times = torch.tensor([-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1], dtype=torch.float64)
    expected = torch.tensor([[0, 4], [0, 4], [1, 6], [2, 8], [6, 2], [10, -4], [10, -4]], dtype=torch.float32)
    torch.testing.assert_close(curve(times), expected, rtol=0, atol=0)


@pytest.mark.parametrize("grid, width", [(0, 2), (1, 2), (3, 0), (3, -1)])
def test_curve_rejects_invalid_table_dimensions(grid, width):
    from vllm_omni.diffusion.models.minimax_h3.adaln_curve import MiniMaxH3AdalnCurve

    with pytest.raises(ValueError, match="curve"):
        MiniMaxH3AdalnCurve(grid, width)


@pytest.mark.parametrize("expand, modalities", [(6, 3), (2, 1)])
def test_curve_projection_preserves_fp32_and_does_not_apply_silu(expand, modalities):
    arch = h3.MiniMaxH3DiTArchConfig(hidden_size=2, time_embed_dim=2, adaln_curve_grid=3)
    proj = h3.MiniMaxH3AdalnProj(
        arch, expand * modalities * 2, None, expand_ratio=expand, modality_num=modalities, prefix="test"
    )
    assert proj.linear.weight.dtype == torch.float32
    proj.linear.weight.data.fill_(1)
    proj.linear.bias.data.fill_(0.125)
    coords = torch.tensor([[-2.0, 1.001], [0.001, 2.0]], dtype=torch.float32)
    actual = proj(coords)
    expected = torch.tensor([-0.874, 2.126]).repeat_interleave(modalities).unsqueeze(1).expand(-1, 2)
    for part in actual:
        assert part.dtype == torch.float32
        torch.testing.assert_close(part, expected, rtol=0, atol=3e-7)


def test_dense_projection_keeps_original_silu_and_bf16():
    arch = h3.MiniMaxH3DiTArchConfig(hidden_size=2, time_embed_dim=2)
    proj = h3.MiniMaxH3AdalnProj(arch, 4, None, expand_ratio=2, modality_num=1, prefix="test")
    proj.linear.weight.data.fill_(1)
    proj.linear.bias.data.zero_()
    coords = torch.tensor([[-2.0, 1.001]])
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(coords).to(torch.bfloat16), proj.linear.weight, proj.linear.bias
    )
    assert proj.linear.weight.dtype == torch.bfloat16
    torch.testing.assert_close(torch.cat(proj(coords), dim=-1), expected, rtol=0, atol=0)


@pytest.fixture
def curve_model(monkeypatch):
    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import _FakeAttention, _small_od_config
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks

    monkeypatch.setattr(blocks, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    # Upstream probes the accelerator for the SM120 AdaLN-offload default; keep these CPU-only tests off CUDA.
    monkeypatch.setattr(h3.current_omni_platform, "get_device_capability", lambda *args, **kwargs: (8, 9))
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    config = _small_od_config()
    config.tf_model_config.update(time_embed_dim=2, adaln_curve_grid=3)
    return h3.MiniMaxH3DiTModel(config, diffusers_weights=False)


def test_native_curve_model_loads_table_and_uses_it_for_time_embeddings(curve_model):
    table = torch.tensor([[0, 4], [2, 8], [10, -4]], dtype=torch.bfloat16)
    loaded = curve_model.load_weights(
        [
            ("adaln_t_table", table),
            ("adaln_basis", torch.ones(2, 4, dtype=torch.bfloat16)),
            ("adaln_mean", torch.ones(4, dtype=torch.bfloat16)),
        ]
    )
    assert loaded == {"time_embedder.table"}
    assert curve_model.time_embedder.table.dtype == torch.float32
    assert not hasattr(curve_model.time_embedder, "proj_in")
    expected = torch.tensor([[1, 6], [10, -4]], dtype=torch.float32)
    torch.testing.assert_close(curve_model.time_embedder(torch.tensor([0.25, 1.0])), expected, rtol=0, atol=0)
    curve_model.post_load_weights()


def test_native_curve_model_rejects_unknown_weights(curve_model):
    with pytest.raises(ValueError, match="unsupported.*curve"):
        curve_model.load_weights([("adaln_t_table_unknown", torch.zeros(3, 2))])


@pytest.mark.parametrize("parameter", ["time_embedder.table", "blocks.0.adaln_proj.linear.weight"])
def test_curve_model_checks_fp32_after_offload_restore(curve_model, parameter):
    param = dict(curve_model.named_parameters()).get(parameter)
    assert param is not None
    param.data = param.data.to(torch.bfloat16)
    with pytest.raises(ValueError, match="fp32"):
        curve_model.validate_restored_host_weights()


@pytest.mark.parametrize("compressed", [True, False])
def test_hsdp_preserves_compressed_projection_precision(curve_model, monkeypatch, compressed):
    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import _small_od_config
    from vllm_omni.diffusion.distributed import hsdp

    model = curve_model if compressed else h3.MiniMaxH3DiTModel(_small_od_config(), diffusers_weights=False)
    monkeypatch.setattr(hsdp, "get_world_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0))
    monkeypatch.setattr(hsdp, "_create_hsdp_mesh", lambda **kwargs: object())
    context = hsdp.prepare_hsdp_shard_context(
        model,
        hsdp.HSDPInferenceConfig(enabled=True, hsdp_shard_size=2),
        target_device=torch.device("cpu"),
    )
    # The default policy casts block parameters on all-gather. Curve projections
    # must retain FP32, while the official dense model keeps its existing policy.
    assert context.hsdp_kwargs["mp_policy"].param_dtype == (None if compressed else torch.bfloat16)
