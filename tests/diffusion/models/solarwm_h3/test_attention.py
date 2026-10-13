# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical and lifecycle contracts for native SolarWM-H3 attention."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ROOT = Path(__file__).parents[4]
_MODEL = _ROOT / "vllm_omni/diffusion/models/solarwm_h3"


@pytest.fixture
def solar(monkeypatch):
    # Import only dependency-light model arithmetic, independently of GPU
    # platform initialization and the installed vLLM/diffusers versions.
    name = "_solarwm_h3_attention_tests"
    package = ModuleType(name)
    package.__path__ = [str(_MODEL)]
    monkeypatch.setitem(sys.modules, name, package)
    modules = {}
    for leaf in ("layout", "camera", "cache", "attention"):
        spec = importlib.util.spec_from_file_location(f"{name}.{leaf}", _MODEL / f"{leaf}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules[leaf] = module
    package.__dict__.update(**modules, package=name)
    return package


def _rope(positions):
    inv_freq = 10000 ** -(torch.arange(16, dtype=torch.float32) / 16)
    half = (positions[0].float().unsqueeze(-1) * inv_freq).flatten(1)
    return torch.cat((half, half), dim=-1)


def _control(solar, cache, *, index=0, audio_latents=1, commit=False):
    # A Qwen image row has VIDEO_TAG but still belongs to the text condition.
    layout = solar.layout.build_layout(torch.tensor([1, 0]), 5, 2, 4, audio_latents)
    views = torch.eye(4).repeat(1, 51, 1, 1)
    views[0, :, 0, 3] = torch.arange(51) * 0.13
    angle = torch.arange(51) * 0.025
    views[0, :, 0, 0] = angle.cos()
    views[0, :, 0, 1] = -angle.sin()
    views[0, :, 1, 0] = angle.sin()
    views[0, :, 1, 1] = angle.cos()
    K = torch.eye(3).repeat(1, 51, 1, 1)
    context = solar.attention.SolarAttentionContext(layout, views, K, index, cache, commit_cache=commit)
    return solar.attention.prepare_solar_rotaries(context, _rope)


def _dense_reference(solar, q, k, v, control, entries):
    """Independent rectangular-mask oracle for each query's visible document."""
    layout = control.layout
    conditions = layout.condition_indices.numel()
    prefix = conditions + layout.audio_indices.numel()
    rows = layout.rows_per_video_frame
    start = max(0, control.chunk_index - 5)
    local_frames = 5 * (control.chunk_index - start + 1)
    history_k = torch.cat([entry[1] for entry in entries], 1) if entries else k[:, :0]
    history_v = torch.cat([entry[2] for entry in entries], 1) if entries else v[:, :0]
    full_k = torch.cat((k[:, :prefix], history_k, k[:, prefix:]), 1)
    full_v = torch.cat((v[:, :prefix], history_v, v[:, prefix:]), 1)
    views = torch.eye(4).repeat(1, full_k.shape[1], 1, 1)
    intrinsics = torch.eye(3).repeat(1, full_k.shape[1], 1, 1)
    views[:, layout.condition_video_indices] = control.viewmats[:, :1]
    intrinsics[:, layout.condition_video_indices] = control.intrinsics[:, :1]
    views[:, prefix:] = control.viewmats[:, 1 + 5 * start : 1 + 5 * (control.chunk_index + 1)].repeat_interleave(
        rows, 1
    )
    intrinsics[:, prefix:] = control.intrinsics[:, 1 + 5 * start : 1 + 5 * (control.chunk_index + 1)].repeat_interleave(
        rows, 1
    )
    key_positions = torch.cat(
        (
            layout.position_ids[:prefix],
            solar.layout.native_video_position_grid(
                local_frames, layout.latent_height, layout.latent_width, origin=float(layout.text_indices.numel())
            ),
        )
    )
    key_freq = _rope(key_positions.unsqueeze(0))
    key_rotary = (key_freq.cos(), key_freq.sin())
    current_start = prefix + history_k.shape[1]
    query_positions = torch.cat((key_positions[:prefix], key_positions[current_start:]))
    query_freq = _rope(query_positions.unsqueeze(0))
    query_rotary = query_freq.cos(), query_freq.sin()

    def rotate(values, rotary):
        # Separate expression from the model implementation, with a matrix
        # view of H3's two rotary halves.
        pair = values[..., :96].reshape(*values.shape[:-1], 2, 48)
        cos, sin = (part[:, :48].to(values)[None, :, None] for part in rotary)
        first = pair[..., 0, :] * cos - pair[..., 1, :] * sin
        second = pair[..., 1, :] * cos + pair[..., 0, :] * sin
        return torch.cat((first, second, values[..., 96:]), -1)

    q = rotate(q, query_rotary)
    full_k = rotate(full_k, key_rotary)
    query_views = torch.cat((views[:, :prefix], views[:, current_start:]), 1)
    query_K = torch.cat((intrinsics[:, :prefix], intrinsics[:, current_start:]), 1)
    q, full_k, full_v, inverse = solar.camera.prope_qkv_separate(
        q.transpose(1, 2),
        full_k.transpose(1, 2),
        full_v.transpose(1, 2),
        query_viewmats=query_views,
        query_K=query_K,
        key_viewmats=views,
        key_K=intrinsics,
    )
    mask = torch.zeros(q.shape[2], full_k.shape[2], dtype=torch.bool)
    mask[:conditions, :conditions] = True
    mask[conditions:prefix, :prefix] = True
    mask[prefix:] = True
    return inverse(F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=mask)).transpose(1, 2)


@pytest.mark.parametrize("audio_latents", [0, 2])
def test_attention_matches_dense_reference_through_sliding_window(solar, audio_latents):
    generator = torch.Generator().manual_seed(37)
    cache = solar.cache.SolarRawKVCache(num_layers=1)
    with torch.inference_mode():
        for index in range(9):
            control = _control(solar, cache, index=index, audio_latents=audio_latents, commit=True)
            shape = (1, control.layout.sequence_length, 2, 128)
            q, k, v = [torch.randn(shape, generator=generator) for _ in range(3)]
            # Keys arrive after QK normalization; preserve their exact values.
            k = F.rms_norm(k, (128,))
            before = [(i, key.clone(), value.clone()) for i, key, value in cache.history(0, index)]
            expected = _dense_reference(solar, q, k, v, control, before)
            actual = solar.attention.solar_rollout_attention(q, k, v, control=control, layer=0)
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
            assert [entry[0] for entry in cache.history(0, index)] == list(range(max(0, index - 5), index))
            for original, retained in zip(before, cache.history(0, index), strict=True):
                torch.testing.assert_close(original[1], retained[1], atol=0, rtol=0)
                torch.testing.assert_close(original[2], retained[2], atol=0, rtol=0)
            cache.commit_chunk(index)
            prefix = control.layout.condition_indices.numel() + control.layout.audio_indices.numel()
            torch.testing.assert_close(cache.layers[0][-1][1], k[:, prefix:], atol=0, rtol=0)
            torch.testing.assert_close(cache.layers[0][-1][2], v[:, prefix:], atol=0, rtol=0)
            assert len(cache.layers[0]) == min(index + 1, 5)


def test_probes_leave_history_unchanged_and_prefix_cannot_see_video(solar):
    cache = solar.cache.SolarRawKVCache(num_layers=1)
    control = _control(solar, cache)
    shape = (1, control.layout.sequence_length, 1, 128)
    generator = torch.Generator().manual_seed(9)
    q, k, v = [torch.randn(shape, generator=generator) for _ in range(3)]
    prefix = control.layout.condition_indices.numel() + control.layout.audio_indices.numel()
    first = solar.attention.solar_rollout_attention(q, k, v, control=control, layer=0)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, prefix:] *= -20
    changed_v[:, prefix:] += 50
    second = solar.attention.solar_rollout_attention(q, changed_k, changed_v, control=control, layer=0)
    torch.testing.assert_close(first[:, :prefix], second[:, :prefix], atol=0, rtol=0)
    assert not torch.allclose(first[:, prefix:], second[:, prefix:])
    assert cache.layers == {} and cache.pending == {} and cache.chunk_index == 0


def test_attention_rejects_skipped_history_and_grad_enabled_commit(solar):
    cache = solar.cache.SolarRawKVCache(num_layers=1)
    control = _control(solar, cache, commit=True)
    values = torch.randn(1, control.layout.sequence_length, 1, 128)
    with pytest.raises(RuntimeError, match="no-grad"):
        solar.attention.solar_rollout_attention(values, values, values, control=control, layer=0)
    control = _control(solar, cache, index=1)
    with pytest.raises(ValueError, match="chunk index"):
        solar.attention.solar_rollout_attention(values, values, values, control=control, layer=0)


def _transformer_module(solar, monkeypatch):
    native = ModuleType("_solar_native_stub")

    class NativeAttention(nn.Module):
        pass

    class NativeModel(nn.Module):
        def forward(self, **kwargs):
            self.observed = kwargs
            if getattr(self, "fail", False):
                raise RuntimeError("forward failed")
            selected = kwargs["img_pos_for_infer_output_info"]["position_ids"]
            return kwargs["x"][0, selected], kwargs["audio_x"][0, :0]

    native.__dict__.update(MiniMaxH3Attention=NativeAttention, MiniMaxH3DiTModel=NativeModel)
    module_attributes: dict[str, dict[str, object]] = {
        "vllm.distributed": {"get_tensor_model_parallel_world_size": lambda: 1},
        "vllm_omni.diffusion.distributed.comm": {"SeqAllToAll4D": None, "all_to_all_5D": None},
        "vllm_omni.diffusion.distributed.parallel_state": {"get_sp_group": lambda: None},
    }
    for name, attributes in module_attributes.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer", native)
    name = f"{solar.package}.transformer"
    spec = importlib.util.spec_from_file_location(name, _MODEL / "transformer.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_native_prediction_maps_rows_timesteps_and_clears_context(solar, monkeypatch):
    # Packing uses the real configuration contracts. The forward stub isolates
    # boundary mapping from weights; arithmetic tests above need no native stack.
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig

    module = _transformer_module(solar, monkeypatch)
    model = module.SolarWMH3Transformer.__new__(module.SolarWMH3Transformer)
    nn.Module.__init__(model)
    model.arch = MiniMaxH3DiTArchConfig(latents_dim=24, audio_latents_dim=32, text_dim=8)
    model.parallel_config = DiffusionParallelConfig(ulysses_degree=4)
    model.rope = _rope
    cache = solar.cache.SolarRawKVCache(1)
    control = _control(solar, cache, audio_latents=2)
    layout = control.layout
    current = torch.randn(1, 24, 5, 2, 4)
    anchor = torch.randn(1, layout.condition_video_indices.numel(), 96)
    audio = torch.randn(1, layout.audio_indices.numel(), 32)
    prompt = torch.randn(1, layout.text_indices.numel(), 8)
    actual = model.predict_video(prompt, anchor, current, audio, layout, 0.25, 0.6, control)
    torch.testing.assert_close(actual, current, atol=0, rtol=0)
    packed = model.observed
    assert packed["x"].shape[1] % 4 == 0
    times = packed["unique_timesteps"][packed["inverse_indices"]]
    torch.testing.assert_close(times[layout.text_indices], torch.ones(layout.text_indices.numel()))
    torch.testing.assert_close(
        times[layout.condition_video_indices], torch.full_like(times[layout.condition_video_indices], 0.999)
    )
    torch.testing.assert_close(times[layout.audio_indices], torch.full_like(times[layout.audio_indices], 0.6))
    torch.testing.assert_close(
        times[layout.noisy_video_indices], torch.full_like(times[layout.noisy_video_indices], 0.25)
    )
    assert packed["token_tags"][1] == 0  # Qwen image rows remain text inputs.
    assert module._SOLAR_ATTENTION.get() is None
    model.fail = True
    with pytest.raises(RuntimeError, match="forward failed"):
        model.predict_video(prompt, anchor, current, audio, layout, 0.25, 0.6, control)
    assert module._SOLAR_ATTENTION.get() is None


def test_native_attention_stages_postnorm_key_and_untransformed_value(solar, monkeypatch):
    module = _transformer_module(solar, monkeypatch)

    class TupleLinear(nn.Linear):
        def forward(self, value):
            return super().forward(value), None

    class NoParallel(nn.Module):
        def _get_active_parallel_strategy(self):
            return SimpleNamespace(enabled=False)

    native = SimpleNamespace(
        qkv_proj=TupleLinear(8, 384, bias=False),
        out_proj=TupleLinear(128, 8, bias=False),
        q_norm=nn.RMSNorm(128, eps=1e-5),
        k_norm=nn.RMSNorm(128, eps=1e-5),
        rope=nn.Identity(),
        attention=NoParallel(),
        total_num_heads=1,
        num_heads=1,
        num_kv_heads=1,
        head_dim=128,
        rot_dim=96,
        _prefix="blocks.0.attn",
    )
    attention = module.SolarWMH3Attention(native, 0)
    cache = solar.cache.SolarRawKVCache(1)
    control = _control(solar, cache, commit=True)
    x = torch.randn(control.layout.sequence_length, 8)
    prefix = control.layout.condition_indices.numel() + control.layout.audio_indices.numel()
    with torch.inference_mode():
        _, projected_key, projected_value = native.qkv_proj(x)[0].chunk(3, -1)
        expected_key = native.k_norm(projected_key.reshape(1, x.shape[0], 1, 128))[:, prefix:]
        expected_value = projected_value.reshape(1, x.shape[0], 1, 128)[:, prefix:]
        token = module._SOLAR_ATTENTION.set(control)
        try:
            result = attention(x, rope_table=torch.randn(x.shape[0], 96), cu_seqlens=None, max_seqlen=x.shape[0])
        finally:
            module._SOLAR_ATTENTION.reset(token)
        assert result.shape == x.shape
        staged = cache.pending[0]
        torch.testing.assert_close(staged[1], expected_key, atol=0, rtol=0)
        torch.testing.assert_close(staged[2], expected_value, atol=0, rtol=0)
        assert cache.layers == {}
