# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _tiny_cosmos3_config(**overrides):
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 32,
        "latent_patch_size": 1,
        "latent_channel": 2,
        "rope_scaling": {"mrope_section": [1, 1, 0]},
    }
    config.update(overrides)
    return config


def test_mrope_position_ids_cover_text_and_video() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        compute_mrope_position_ids_text,
        compute_mrope_position_ids_vision,
    )

    text_ids, text_offset = compute_mrope_position_ids_text(num_tokens=3, temporal_offset=5)
    assert text_ids.tolist() == [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
    assert text_offset == 8

    vision_ids, vision_offset = compute_mrope_position_ids_vision(2, 2, 3, temporal_offset=10, fps=None)
    assert vision_ids.shape == (3, 12)
    assert vision_ids[0].tolist() == [10] * 6 + [11] * 6
    assert vision_offset == 12

    modulated_ids, modulated_offset = compute_mrope_position_ids_vision(
        2,
        1,
        1,
        temporal_offset=10,
        fps=12.0,
        base_fps=24.0,
        temporal_compression_factor=4,
    )
    torch.testing.assert_close(modulated_ids[0], torch.tensor([10.0, 12.0]))
    assert modulated_offset == 13


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("qk_norm_for_diffusion", False),
        ("qk_norm_for_text", False),
        ("position_embedding_type", "rotary"),
        ("unified_3d_mrope_reset_spatial_ids", False),
        ("joint_attn_implementation", "one_way"),
    ],
)
def test_validate_supported_config_rejects_unsupported_flags(key: str, value) -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    with pytest.raises(ValueError, match=f"{key}="):
        Cosmos3VFMTransformer._validate_supported_config({key: value})
    Cosmos3VFMTransformer._validate_supported_config({})
    Cosmos3VFMTransformer._validate_supported_config(None)


def test_transformer_sharding_offload_and_patch_round_trip_contracts() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = nn.Module()
    model.language_model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
    model.gen_layers = nn.ModuleList([nn.Linear(2, 2)])
    model.norm_moe_gen = nn.LayerNorm(2)

    matched = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in model._hsdp_shard_conditions)
    ]
    assert matched == ["language_model.layers.0", "language_model.layers.1", "gen_layers.0"]
    assert Cosmos3VFMTransformer._layerwise_offload_blocks_attrs == ["gen_layers"]
    assert Cosmos3VFMTransformer._repeated_blocks == ["Cosmos3GenDecoderLayer"]

    model.latent_patch_size = 2
    model.latent_channel_size = 3
    latents = torch.arange(1 * 3 * 1 * 3 * 5, dtype=torch.float32).reshape(1, 3, 1, 3, 5)
    torch.testing.assert_close(model.unpatchify(model.patchify(latents, t=1, h=3, w=5), t=1, h=3, w=5), latents)


def test_forward_returns_video_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    monkeypatch.setattr(transformer_cosmos3, "_get_ulysses_state", lambda: (1, 0, None))

    output = transformer_cosmos3.Cosmos3VFMTransformer(
        SimpleNamespace(tf_model_config=_tiny_cosmos3_config(), dtype=torch.float32)
    )(
        hidden_states=torch.zeros(1, 2, 1, 2, 2),
        timestep=torch.tensor([1.0]),
        text_ids=torch.tensor([[1, 2]], dtype=torch.long),
        text_mask=torch.ones(1, 2, dtype=torch.long),
        video_shape=(1, 2, 2),
        fps=24.0,
    )

    assert tuple(output.shape) == (1, 2, 1, 2, 2)


def test_compute_rope_freqs_places_text_and_video_positions() -> None:
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer

    class FakeRotary:
        def __init__(self) -> None:
            self.position_ids: list[torch.Tensor] = []

        def __call__(self, x, position_ids):
            del x
            self.position_ids.append(position_ids.detach().cpu())
            batch, seq = position_ids.shape[1], position_ids.shape[2]
            return torch.zeros(batch, seq, 4), torch.ones(batch, seq, 4)

    rotary = FakeRotary()
    model = object.__new__(Cosmos3VFMTransformer)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(rotary_emb=rotary)
    model.temporal_modality_margin = 100
    model.base_fps = 24.0
    model.temporal_compression_factor = 4
    model.enable_fps_modulation = False

    freqs_und, freqs_gen = model._compute_rope_freqs(
        text_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
        t=2,
        hp=1,
        wp=1,
        fps=24.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    text_pos, vision_pos = rotary.position_ids
    assert text_pos[:, 0, :].tolist() == [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert vision_pos[0, 0].tolist() == [102, 103]
    assert freqs_und[0].shape == (2, 3, 1, 4)
    assert freqs_gen[0].shape == (2, 2, 1, 4)
