# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.models.ltx2 import ltx2_transformer
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    LTX2FeedForward,
    LTX2VideoTransformer3DModel,
    LTX2VideoTransformerBlock,
    _make_rms_norm,
    apply_interleaved_rotary_emb,
    apply_keyframes_absolute_embedding,
    apply_split_rotary_emb,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_ltx_interleaved_rope_matches_official_model_dtype_arithmetic():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    cos = torch.randn(2, 3, 8, dtype=x.dtype)
    sin = torch.randn(2, 3, 8, dtype=x.dtype)
    x_real, x_imag = x.unflatten(2, (4, 2)).unbind(-1)
    rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    expected = x * cos + rotated * sin

    actual = apply_interleaved_rotary_emb(x, (cos, sin))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ltx_split_rope_matches_official_model_dtype_arithmetic():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    cos = torch.randn(2, 3, 4, dtype=x.dtype)
    sin = torch.randn(2, 3, 4, dtype=x.dtype)
    split_x = x.reshape(2, 3, 2, 4)
    first_x, second_x = split_x.chunk(2, dim=-2)
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)
    expected = split_x * cos_u
    expected[..., :1, :].addcmul_(-sin_u, second_x)
    expected[..., 1:, :].addcmul_(sin_u, first_x)
    expected = expected.reshape_as(x)

    actual = apply_split_rotary_emb(x, (cos, sin), head_dim=8)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ltx_rms_norm_no_affine_matches_official_torch_module():
    norm = _make_rms_norm(8, eps=1e-6, elementwise_affine=False)

    assert "weight" not in dict(norm.named_parameters())
    assert norm.weight is None
    assert "weight" not in dict(norm.named_buffers())
    assert "weight" not in norm.state_dict()


def test_ltx_rms_norm_affine_weight_remains_parameter():
    norm = _make_rms_norm(8, eps=1e-6, elementwise_affine=True)

    assert isinstance(dict(norm.named_parameters())["weight"], nn.Parameter)
    assert "weight" not in dict(norm.named_buffers())
    assert "weight" in norm.state_dict()


def test_ltx_attention_restores_projection_dtype_after_upcasting_qk_norms():
    calls = []

    class UpcastingNorm(nn.Module):
        def forward(self, tensor):
            return tensor.float()

    class RecordingAttention:
        def __call__(self, query, key, value, metadata):
            calls.append((query, key, value, metadata))
            return query

    attention = SimpleNamespace(
        heads=2,
        head_dim=4,
        to_qkv=None,
        to_q=nn.Identity(),
        to_k=nn.Identity(),
        to_v=nn.Identity(),
        norm_q=UpcastingNorm(),
        norm_k=UpcastingNorm(),
        to_gate_logits=None,
        attn=RecordingAttention(),
        to_out=(nn.Identity(), nn.Identity()),
        rope_type="split",
    )
    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    rotary = (
        torch.ones(2, 2, 3, 2, dtype=torch.float32),
        torch.zeros(2, 2, 3, 2, dtype=torch.float32),
    )

    processor = ltx2_transformer.LTX2AudioVideoAttnProcessor()
    processor._slice_rope_for_tp = lambda rope, _: rope
    output = processor(attention, hidden_states, query_rotary_emb=rotary)

    query, key, value, metadata = calls[0]
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert metadata is None
    torch.testing.assert_close(output, hidden_states)


def test_ltx_transformer_has_fused_cfg_cache_dit_config():
    adapter_config = getattr(LTX2VideoTransformer3DModel, "_cache_dit_adapter_config")

    assert isinstance(adapter_config, CacheDiTAdapterConfig)
    assert not adapter_config.has_separate_cfg


def test_ltx_transformer_exposes_hsdp_shard_conditions_for_blocks():
    model = object.__new__(LTX2VideoTransformer3DModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.norm_out = nn.LayerNorm(4)

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(condition(name, module) for condition in conditions):
            matched.append(name)

    assert matched == ["transformer_blocks.0", "transformer_blocks.1"]


@pytest.mark.parametrize("rope_type", ["interleaved", "split"])
def test_ltx_sp_plan_shards_video_and_audio_timesteps_together(rope_type):
    root_plan = LTX2VideoTransformer3DModel._build_sp_plan(rope_type)[""]

    video_timestep = root_plan["timestep"]
    audio_timestep = root_plan["audio_timestep"]

    assert video_timestep.split_dim == audio_timestep.split_dim == 1
    assert video_timestep.expected_dims == audio_timestep.expected_dims == 2
    assert not video_timestep.split_output
    assert not audio_timestep.split_output


def test_ltx_sp_keeps_cross_attention_key_padding_mask(monkeypatch):
    processor_cls = ltx2_transformer.LTX2AudioVideoAttnProcessor
    monkeypatch.setattr(processor_cls, "_is_sp_enabled", staticmethod(lambda: True))
    processor = processor_cls()
    hidden_states = torch.zeros(1, 3, 4)
    encoder_hidden_states = torch.zeros(1, 4, 4)
    key_padding_mask = torch.tensor([[True, True, True, False]])

    actual = processor._prepare_attention_mask(
        SimpleNamespace(),
        hidden_states,
        encoder_hidden_states,
        key_padding_mask,
        batch_size=1,
        sequence_length=4,
    )

    torch.testing.assert_close(actual, key_padding_mask[:, None, None, :])


def _fake_ltx_parallel_attention():
    calls = []

    class Backend:
        @staticmethod
        def get_name():
            return "FLASH_ATTN"

        @staticmethod
        def supports_attention_mask(attention_spec=None):
            del attention_spec
            return True

    def run(name):
        def forward(query, _key, _value, _metadata):
            calls.append(name)
            return torch.zeros_like(query)

        return forward

    layer = object.__new__(ltx2_transformer._LTX2ParallelAttention)
    nn.Module.__init__(layer)
    layer.attn_backend = Backend
    layer.attention = SimpleNamespace(forward=run("native"))
    layer.sdpa_fallback = SimpleNamespace(forward=run("sdpa"))
    layer.backend_pref = "FLASH_ATTN"
    return layer, calls


@pytest.mark.parametrize(("query_length", "key_length"), [(4, 4), (3, 4)])
def test_ltx_masked_attention_uses_sdpa(query_length, key_length):
    layer, calls = _fake_ltx_parallel_attention()
    query = torch.zeros(1, query_length, 2, 4, dtype=torch.bfloat16)
    key = value = torch.zeros(1, key_length, 2, 4, dtype=torch.bfloat16)
    key_padding_mask = torch.tensor([[True] * (key_length - 1) + [False]])
    metadata = ltx2_transformer.AttentionMetadata(attn_mask=key_padding_mask[:, None, None, :])

    layer._run_local_attention(query, key, value, metadata)

    assert calls == ["sdpa"]


def test_ltx_unmasked_attention_keeps_native_backend():
    layer, calls = _fake_ltx_parallel_attention()
    query = key = value = torch.zeros(1, 4, 2, 4, dtype=torch.bfloat16)

    layer._run_local_attention(query, key, value, None)

    assert calls == ["native"]


def test_ltx25_feed_forward_bias_flag_controls_both_linears(monkeypatch):
    class FakeParallelLinear(nn.Linear):
        def __init__(self, dim_in, dim_out, *, bias=True, **kwargs):
            super().__init__(dim_in, dim_out, bias=bias)

    monkeypatch.setattr(ltx2_transformer, "ColumnParallelLinear", FakeParallelLinear)
    monkeypatch.setattr(ltx2_transformer, "RowParallelLinear", FakeParallelLinear)

    without_bias = LTX2FeedForward(8, inner_dim=16, bias=False)
    with_bias = LTX2FeedForward(8, inner_dim=16, bias=True)

    assert without_bias.net[0].proj.bias is None
    assert without_bias.net[2].bias is None
    assert with_bias.net[0].proj.bias is not None
    assert with_bias.net[2].bias is not None


def _build_tiny_ltx_transformer(monkeypatch, **overrides):
    class FakeBlock(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.init_kwargs = kwargs

        def forward(self, hidden_states, audio_hidden_states, *_args, **_kwargs):
            return hidden_states, audio_hidden_states

    monkeypatch.setattr(ltx2_transformer, "LTX2VideoTransformerBlock", FakeBlock)
    kwargs = {
        "in_channels": 4,
        "out_channels": 4,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "cross_attention_dim": 8,
        "vae_scale_factors": (1, 1, 1),
        "pos_embed_max_pos": 4,
        "base_height": 8,
        "base_width": 8,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 4,
        "audio_cross_attention_dim": 8,
        "audio_scale_factor": 1,
        "audio_pos_embed_max_pos": 4,
        "audio_sampling_rate": 16,
        "audio_hop_length": 4,
        "num_layers": 1,
        "caption_channels": 8,
        "use_prompt_embeddings": False,
        "cross_attn_mod": True,
        "audio_cross_attn_mod": True,
    }
    kwargs.update(overrides)
    return LTX2VideoTransformer3DModel(**kwargs)


def test_ltx25_keyframe_absolute_embedding_matches_official():
    hidden_states = torch.arange(32, dtype=torch.bfloat16).view(1, 4, 8)
    keyframes_mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    embedding = torch.linspace(-0.5, 0.5, 8, dtype=torch.bfloat16).view(1, 8)

    actual = apply_keyframes_absolute_embedding(hidden_states, keyframes_mask, embedding)
    expected = hidden_states + (keyframes_mask > 0).to(torch.bfloat16) * embedding

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert apply_keyframes_absolute_embedding(hidden_states, None, embedding) is hidden_states


def test_ltx25_constructor_flags_and_native_weight_loader(monkeypatch):
    legacy_model = _build_tiny_ltx_transformer(monkeypatch)

    assert legacy_model.config.ff_bias is True
    assert legacy_model.config.audio_ff_bias is True
    assert legacy_model.config.use_prompt_adaln_single is True
    assert legacy_model.config.use_keyframes_abs_pos_embedding is False
    assert hasattr(legacy_model, "prompt_adaln")
    assert hasattr(legacy_model, "audio_prompt_adaln")
    assert not hasattr(legacy_model, "keyframes_abs_pos_embedding")
    assert legacy_model.transformer_blocks[0].init_kwargs["ff_bias"] is True
    assert legacy_model.transformer_blocks[0].init_kwargs["audio_ff_bias"] is True

    ltx25_model = _build_tiny_ltx_transformer(
        monkeypatch,
        ff_bias=False,
        audio_ff_bias=True,
        use_prompt_adaln_single=False,
        use_keyframes_abs_pos_embedding=True,
    )

    assert ltx25_model.keyframes_abs_pos_embedding.shape == (1, 8)
    torch.testing.assert_close(ltx25_model.keyframes_abs_pos_embedding, torch.zeros(1, 8))
    assert not hasattr(ltx25_model, "prompt_adaln")
    assert not hasattr(ltx25_model, "audio_prompt_adaln")
    assert ltx25_model.transformer_blocks[0].init_kwargs["ff_bias"] is False
    assert ltx25_model.transformer_blocks[0].init_kwargs["audio_ff_bias"] is True

    monkeypatch.setattr(ltx2_transformer, "get_tensor_model_parallel_world_size", lambda: 1)
    expected = torch.arange(8, dtype=torch.float32).view(1, 8)
    loaded = ltx25_model.load_weights([("keyframes_abs_pos_embedding", expected)])

    assert loaded == {"keyframes_abs_pos_embedding"}
    torch.testing.assert_close(ltx25_model.keyframes_abs_pos_embedding, expected)


def test_ltx_cross_attention_modulations_use_official_timesteps(monkeypatch):
    class CaptureAda(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, timestep, *, batch_size, hidden_dtype):
            self.calls.append(timestep.detach().clone())
            embedding = torch.zeros(timestep.numel(), 8, dtype=hidden_dtype)
            return embedding, embedding

    class CaptureRope(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_dtypes = []

        def forward(self, _coords, *, device, out_dtype=None):
            self.output_dtypes.append(out_dtype)
            empty = torch.empty(0, device=device, dtype=out_dtype)
            return empty, empty

    model = _build_tiny_ltx_transformer(monkeypatch)
    rope = CaptureRope()
    model.rope = model.audio_rope = model.cross_attn_rope = model.cross_attn_audio_rope = rope
    model.prompt_modulation = False
    video_scale_shift = CaptureAda()
    video_gate = CaptureAda()
    audio_scale_shift = CaptureAda()
    audio_gate = CaptureAda()
    model.av_cross_attn_video_scale_shift = video_scale_shift
    model.av_cross_attn_video_a2v_gate = video_gate
    model.av_cross_attn_audio_scale_shift = audio_scale_shift
    model.av_cross_attn_audio_v2a_gate = audio_gate

    model(
        hidden_states=torch.zeros(1, 2, 4),
        audio_hidden_states=torch.zeros(1, 1, 4),
        encoder_hidden_states=torch.zeros(1, 1, 8),
        audio_encoder_hidden_states=torch.zeros(1, 1, 8),
        timestep=torch.tensor([[0.0, 2.0]]),
        audio_timestep=torch.tensor([[5.0]]),
        sigma=torch.tensor([2.0]),
        audio_sigma=torch.tensor([5.0]),
        video_coords=torch.zeros(1, 3, 2, 2),
        audio_coords=torch.zeros(1, 1, 1, 2),
    )

    torch.testing.assert_close(video_scale_shift.calls[0], torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(video_gate.calls[0], torch.tensor([5.0]))
    torch.testing.assert_close(audio_scale_shift.calls[0], torch.tensor([5.0]))
    torch.testing.assert_close(audio_gate.calls[0], torch.tensor([2.0]))
    assert len(rope.output_dtypes) == 4
    assert all(dtype is torch.float32 for dtype in rope.output_dtypes)


def test_ltx25_static_prompt_table_is_used_without_prompt_adaln():
    class CaptureAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_hidden_states = None

        def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
            if encoder_hidden_states is not None:
                self.encoder_hidden_states = encoder_hidden_states.detach().clone()
            return torch.zeros_like(hidden_states)

    class ZeroOutput(nn.Module):
        def forward(self, hidden_states):
            return torch.zeros_like(hidden_states)

    block = object.__new__(LTX2VideoTransformerBlock)
    nn.Module.__init__(block)
    block.video_cross_attn_adaln = True
    block.audio_cross_attn_adaln = True
    block.cross_attn_adaln = True
    block.scale_shift_table = nn.Parameter(torch.zeros(9, 2))
    block.audio_scale_shift_table = nn.Parameter(torch.zeros(9, 2))
    block.prompt_scale_shift_table = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    block.audio_prompt_scale_shift_table = nn.Parameter(torch.tensor([[0.5, 1.0], [1.5, 2.0]]))
    block.norm1 = nn.Identity()
    block.audio_norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.audio_norm2 = nn.Identity()
    block.norm3 = nn.Identity()
    block.audio_norm3 = nn.Identity()
    block.attn1 = CaptureAttention()
    block.audio_attn1 = CaptureAttention()
    block.attn2 = CaptureAttention()
    block.audio_attn2 = CaptureAttention()
    block.ff = ZeroOutput()
    block.audio_ff = ZeroOutput()

    hidden_states = torch.ones(1, 1, 2)
    audio_hidden_states = torch.ones(1, 1, 2)
    encoder_hidden_states = torch.ones(1, 1, 2)
    audio_encoder_hidden_states = torch.ones(1, 1, 2)
    block(
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        temb=torch.zeros(1, 1, 18),
        temb_audio=torch.zeros(1, 1, 18),
        temb_ca_scale_shift=torch.empty(0),
        temb_ca_audio_scale_shift=torch.empty(0),
        temb_ca_gate=torch.empty(0),
        temb_ca_audio_gate=torch.empty(0),
        temb_prompt=None,
        temb_prompt_audio=None,
        use_a2v_cross_attention=False,
        use_v2a_cross_attention=False,
    )

    torch.testing.assert_close(block.attn2.encoder_hidden_states, torch.tensor([[[5.0, 7.0]]]))
    torch.testing.assert_close(block.audio_attn2.encoder_hidden_states, torch.tensor([[[3.0, 4.0]]]))


def test_ltx_gated_attention_uses_tp_sharded_projection(monkeypatch):
    column_calls = []

    class FakeColumnParallelLinear(nn.Linear):
        def __init__(self, input_size, output_size, *, bias=True, **kwargs):
            column_calls.append((output_size, kwargs))
            super().__init__(input_size, output_size, bias=bias)

    class FakeRowParallelLinear(nn.Linear):
        def __init__(self, input_size, output_size, *, bias=True, **_kwargs):
            super().__init__(input_size, output_size, bias=bias)

    class FakeParallelAttention(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()

    monkeypatch.setattr(ltx2_transformer, "ColumnParallelLinear", FakeColumnParallelLinear)
    monkeypatch.setattr(ltx2_transformer, "RowParallelLinear", FakeRowParallelLinear)
    monkeypatch.setattr(ltx2_transformer, "_LTX2ParallelAttention", FakeParallelAttention)
    monkeypatch.setattr(ltx2_transformer, "get_tensor_model_parallel_world_size", lambda: 2)

    attention = ltx2_transformer.LTX2Attention(
        query_dim=8,
        heads=4,
        kv_heads=4,
        dim_head=2,
        qk_norm="rms_norm",
        apply_gated_attention=True,
        pack_qkv=False,
    )

    gate_output_size, gate_kwargs = column_calls[-1]
    assert gate_output_size == 4
    assert gate_kwargs["gather_output"] is False
    assert gate_kwargs["return_bias"] is False
    assert attention.query_num_heads == 2
