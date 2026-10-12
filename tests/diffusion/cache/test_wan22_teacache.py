# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU regressions for Wan TeaCache's single-device forward contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from vllm_omni.diffusion.cache.teacache.extractors import extract_wan2_2_context
from vllm_omni.diffusion.models.wan2_2 import wan2_2_transformer as wan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _adaptive_norm(hidden_states, scale, shift):
    # Preserve the native AdaLayerNorm signature and float32 modulation math.
    return F.layer_norm(hidden_states.float(), (hidden_states.shape[-1],)) * (1 + scale) + shift


def _make_wan_model():
    inner_dim = 4
    block = Mock()
    block.scale_shift_table = torch.arange(24, dtype=torch.bfloat16).reshape(1, 6, inner_dim) / 100
    block.norm1 = Mock(side_effect=_adaptive_norm)

    def run_block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, *args):
        table = block.scale_shift_table
        if timestep_proj.ndim == 4:
            shift, scale, *_ = (table.unsqueeze(0) + timestep_proj).unbind(2)
        else:
            shift, scale, *_ = (table + timestep_proj).chunk(6, dim=1)
        modulated = block.norm1(hidden_states, scale, shift).type_as(hidden_states)
        return hidden_states + modulated / 2

    block.side_effect = run_block

    def condition_embedder(timestep, text, image, timestep_seq_len=None):
        batch_size = text.shape[0]
        shape = (batch_size, inner_dim) if timestep_seq_len is None else (batch_size, timestep_seq_len, inner_dim)
        temb = torch.arange(torch.tensor(shape).prod().item(), dtype=text.dtype).reshape(shape) / 100
        projection = torch.cat([temb + index / 10 for index in range(6)], dim=-1)
        return temb, projection, text + 5, None if image is None else image + 7

    def output_scale_shift_prepare(temb):
        return temb + 0.3, temb + 0.7

    # Patch/unpatchify a nontrivial spatial grid while keeping all work on CPU.
    return SimpleNamespace(
        config=SimpleNamespace(patch_size=(1, 2, 2)),
        blocks=[block],
        start_layer=0,
        end_layer=1,
        preserve_vsa_all_blocks=False,
        _cached_rope_resolution=None,
        _cached_rope_emb=None,
        _sp_shard_point=lambda value: value,
        rope=Mock(
            return_value=(
                torch.arange(96, dtype=torch.float32).reshape(1, 12, 1, 8),
                torch.arange(96, dtype=torch.float32).reshape(1, 12, 1, 8) + 100,
            )
        ),
        patch_embedding=Mock(
            side_effect=lambda value: torch.cat([value[:, :, :, ::2, ::2], value[:, :, :, ::2, ::2] + 1], dim=1)
        ),
        condition_embedder=Mock(side_effect=condition_embedder),
        timestep_proj_prepare=Mock(
            side_effect=lambda value, seq_len: value.unflatten(1 if seq_len is None else 2, (6, -1))
        ),
        output_scale_shift_prepare=Mock(side_effect=output_scale_shift_prepare),
        norm_out=Mock(side_effect=_adaptive_norm),
        proj_out=Mock(side_effect=lambda value: torch.cat([value, value + 2], dim=-1)),
    )


@pytest.mark.parametrize("mode", ["t2v", "i2v", "ti2v"])
@pytest.mark.parametrize("return_dict", [True, False])
def test_wan_extractor_matches_native_single_device_forward(monkeypatch, mode, return_dict):
    monkeypatch.setattr(wan, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(wan, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(
        wan,
        "get_forward_context",
        lambda: SimpleNamespace(omni_diffusion_config=SimpleNamespace(parallel_config=None)),
    )
    hidden_states = torch.arange(192, dtype=torch.bfloat16).reshape(2, 2, 2, 4, 6) / 100
    text = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    image = torch.ones(2, 2, 4, dtype=torch.bfloat16) if mode == "i2v" else None
    timestep = torch.arange(24).reshape(2, 12) if mode == "ti2v" else torch.tensor([100, 200])
    inputs = dict(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=text,
        encoder_hidden_states_image=image,
        return_dict=return_dict,
    )
    native_model = _make_wan_model()
    expected = wan.WanTransformer3DModel.forward(native_model, **inputs)
    cached_model = _make_wan_model()
    context = extract_wan2_2_context(cached_model, **inputs)
    context.validate()
    (processed_states,) = context.run_transformer_blocks()
    actual = context.postprocess(processed_states)

    # Norms must receive (hidden_states, scale, shift), including BF16 input.
    for cached_norm, native_norm in (
        (cached_model.blocks[0].norm1, native_model.blocks[0].norm1),
        (cached_model.norm_out, native_model.norm_out),
    ):
        cached_args = cached_norm.call_args_list[0].args
        native_args = native_norm.call_args_list[0].args
        assert len(cached_args) == 3
        assert cached_args[0].dtype == hidden_states.dtype
        for actual_arg, expected_arg in zip(cached_args, native_args):
            torch.testing.assert_close(actual_arg, expected_arg)

    native_args = native_model.blocks[0].call_args.args
    cached_args = cached_model.blocks[0].call_args.args
    for actual_arg, expected_arg in zip(cached_args[:3], native_args[:3]):
        torch.testing.assert_close(actual_arg, expected_arg)
    for actual_rope, expected_rope in zip(cached_args[3], native_args[3]):
        torch.testing.assert_close(actual_rope, expected_rope)
        assert actual_rope.dtype == hidden_states.dtype
    assert context.modulated_input.dtype == hidden_states.dtype
    expected_modulated = _adaptive_norm(*native_model.blocks[0].norm1.call_args.args).type_as(hidden_states)
    torch.testing.assert_close(context.modulated_input, expected_modulated)
    conditioning_args = cached_model.condition_embedder.call_args
    assert conditioning_args.kwargs["timestep_seq_len"] == (12 if mode == "ti2v" else None)
    torch.testing.assert_close(conditioning_args.args[0], timestep.flatten())
    assert cached_args[2].ndim == (4 if mode == "ti2v" else 3)

    if return_dict:
        assert isinstance(actual, Transformer2DModelOutput)
        assert isinstance(expected, Transformer2DModelOutput)
        actual_sample, expected_sample = actual.sample, expected.sample
    else:
        assert isinstance(actual, tuple) and len(actual) == 1
        actual_sample, expected_sample = actual[0], expected[0]
    assert actual_sample.shape == hidden_states.shape
    assert actual_sample.dtype == hidden_states.dtype
    torch.testing.assert_close(actual_sample, expected_sample)
