# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.adaln_lookup import (
    canonical_lookup_timesteps,
    lookup_timestep_indices,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3AdalnProj, MiniMaxH3DiTArchConfig
from vllm_omni.quantization import svdquant_config
from vllm_omni.quantization.tools.export_minimax_h3_low_memory_checkpoint import export_timesteps, pack_nvfp4

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("values", [[], [float("nan")], [1.01], [-0.01], [0.5, 0.5], [0.7, 0.5], [0.5, 0.5 + 1e-10]])
def test_lookup_metadata_rejects_invalid_or_ambiguous_timesteps(values):
    with pytest.raises(ValueError, match="adaln_lookup_timesteps"):
        canonical_lookup_timesteps(values)


def test_lookup_preserves_modality_rows_and_rejects_interpolation():
    arch = MiniMaxH3DiTArchConfig(hidden_size=4, time_embed_dim=8, adaln_lookup_timesteps=[0.0, 0.5, 1.0])
    module = MiniMaxH3AdalnProj(arch, 72, None, expand_ratio=6, modality_num=3, prefix="blocks.0.adaln_proj")
    rows = torch.arange(3 * 72, dtype=torch.bfloat16).reshape(3, 72)
    module.lookup_values.data.copy_(rows)
    assert not hasattr(module, "linear")
    times = torch.tensor([1.0, 0.0, 0.5, 0.0])
    indices = lookup_timestep_indices(times, torch.tensor(arch.adaln_lookup_timesteps))
    actual = module(indices)
    assert all(part.shape == (12, 4) for part in actual)
    torch.testing.assert_close(torch.cat(actual, dim=-1), rows[[2, 0, 1, 0]].reshape(12, 24), rtol=0, atol=0)
    with pytest.raises(ValueError, match="absent"):
        lookup_timestep_indices(torch.tensor([0.50001]), torch.tensor(arch.adaln_lookup_timesteps))


def test_exported_schedule_covers_actual_video_audio_and_reference_times():
    from vllm_omni.diffusion.models.minimax_h3.time_request import _time_shift_sigmas

    table = torch.tensor(export_timesteps([2, 5, 50], 12.0, 3.0))
    for count in (2, 5, 50):
        for shift, anchor in ((12.0, 0.999), (3.0, 1.0)):
            values: list[float] = []
            for sigma in _time_shift_sigmas(num_steps=count, shift_scale=shift)[:-1]:
                values.extend((1.0 - sigma, max(1.0 - sigma, anchor)))
            lookup_timestep_indices(torch.tensor(values), table)


def test_exported_adaln_rows_match_dense_time_embedder_and_projection(monkeypatch, default_vllm_config):
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3TimeEmbedder
    from vllm_omni.quantization.tools.export_minimax_h3_low_memory_checkpoint import time_embeddings

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))
    arch = MiniMaxH3DiTArchConfig(hidden_size=4, timestep_input_dim=8, time_embed_hidden_size=16, time_embed_dim=8)
    embedder = MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
    dense = MiniMaxH3AdalnProj(arch, 72, None, expand_ratio=6, modality_num=3, prefix="adaln_proj")
    generator = torch.Generator().manual_seed(1101)
    for module in (embedder, dense):
        for parameter in module.parameters():
            parameter.data.copy_(torch.randn(parameter.shape, generator=generator).to(parameter.dtype))
    tensors = {f"time_embedder.{name}": parameter for name, parameter in embedder.named_parameters()}
    times = torch.tensor([0.0, 0.123, 0.999, 1.0])
    exported_embeddings = time_embeddings(SimpleNamespace(get=tensors.__getitem__), times, arch)
    torch.testing.assert_close(exported_embeddings, embedder(times), rtol=0, atol=0)
    exported_rows = torch.nn.functional.linear(
        torch.nn.functional.silu(exported_embeddings).to(torch.bfloat16), dense.linear.weight, dense.linear.bias
    )
    expected = torch.cat(dense(embedder(times)), dim=-1)
    torch.testing.assert_close(exported_rows.reshape(12, 24), expected, rtol=0, atol=0)


def test_packing_matches_signed_e2m1_codes_and_handles_zero_weights():
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2)
    packed, scales, outer = pack_nvfp4(torch.stack((values, -values)))
    assert packed.view(torch.uint8).tolist() == [[16, 50, 84, 118] * 2, [144, 186, 220, 254] * 2]
    assert scales.shape == (1, 2)
    assert scales.dtype == torch.float8_e4m3fn
    assert outer.dtype == torch.bfloat16
    zero, zero_scales, zero_outer = pack_nvfp4(torch.zeros(3, 32))
    assert not bool(zero.any())
    assert bool(torch.isfinite(zero_scales.float()).all())
    assert bool(torch.isfinite(zero_outer).all())


def test_w4a16_keeps_packed_weights_and_matches_dense_reference(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils

    monkeypatch.setattr(svdquant_config, "_assert_supported", lambda: None)
    monkeypatch.setattr(nvfp4_emulation_utils, "current_platform", SimpleNamespace(is_cuda_alike=lambda: False))
    method = svdquant_config.DiffusionSVDQuantLinearMethod(svdquant_config.DiffusionSVDQuantConfig(activation_bits=16))
    layer = torch.nn.Module()
    method.create_weights(layer, 32, [64], 32, 64, torch.bfloat16)
    generator = torch.Generator().manual_seed(6493)
    weight = torch.randn(64, 32, generator=generator)
    packed, scales, outer = pack_nvfp4(weight)
    layer.qweight.data.copy_(packed)
    layer.wscales.data.copy_(scales)
    layer.wtscale.data.copy_(outer)
    layer.proj_down.data.zero_()
    layer.proj_up.data.zero_()
    layer.smooth_factor.data.fill_(1)
    method.process_weights_after_loading(layer)
    assert layer.qweight.dtype == torch.int8
    assert not hasattr(layer, "weight")
    inputs = torch.randn(2, 32, 7, generator=generator).transpose(1, 2).bfloat16()
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    unsigned = packed.view(torch.uint8).long()
    codes = torch.stack((unsigned & 15, unsigned >> 4), dim=-1).reshape(64, 32)
    dense = (grid[codes].reshape(64, 2, 16) * scales.T.float().unsqueeze(-1) * outer.float()).reshape(64, 32).bfloat16()
    actual = method.apply(layer, inputs)
    torch.testing.assert_close(actual, torch.nn.functional.linear(inputs, dense), rtol=0, atol=0)


def test_encoder_uses_only_its_own_w4a16_checkpoint_metadata():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _resolve_minimax_h3_text_encoder_quant_config

    global_config = svdquant_config.DiffusionSVDQuantConfig()
    assert _resolve_minimax_h3_text_encoder_quant_config(global_config) is None
    disk = {"quant_method": "svdquant", "rank": 32, "activation_bits": 16}
    resolved = _resolve_minimax_h3_text_encoder_quant_config(global_config, disk)
    assert resolved.activation_bits == 16
    with pytest.raises(ValueError, match="activation_bits=16"):
        _resolve_minimax_h3_text_encoder_quant_config(global_config, {**disk, "activation_bits": 4})
