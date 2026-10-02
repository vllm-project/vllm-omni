# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Parity of the unpadded, grid-grouped MiniCPM-o 4.5 vision path with the padded batch."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_llm as minicpmo_llm
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    Resampler,
    SiglipAttention,
    SiglipEncoderLayer,
    SiglipSdpaAttention,
    SiglipVisionConfig,
    SiglipVisionTransformer,
    _plan_packed_vision_chunks,
    _resolve_vision_attention_implementation,
)

_PATCH = 2
# Mixed grids: equal sequence length with different shapes, and repeats.
_RAGGED_LAYOUTS = [(4, 4), (2, 8), (3, 5), (4, 4), (5, 3), (1, 16), (4, 4)]


@dataclass
class _Config:
    vision_batch_size: int


def _vision_config(attn_implementation: str, *, hidden_size: int = 32, heads: int = 4) -> SiglipVisionConfig:
    config = SiglipVisionConfig(
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=2,
        num_attention_heads=heads,
        image_size=28,
        patch_size=_PATCH,
        attention_dropout=0.0,
    )
    config._attn_implementation = attn_implementation
    return config


def _build_towers(
    attn_implementation: str, *, seed: int = 0, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> tuple[SiglipVisionTransformer, Resampler]:
    torch.manual_seed(seed)
    vpm = SiglipVisionTransformer(_vision_config(attn_implementation)).eval()
    resampler = Resampler(num_queries=4, embed_dim=64, num_heads=4, kv_dim=32, adaptive=True, max_size=(8, 8)).eval()
    for param in resampler.parameters():
        # Resampler parameters default to zeros/constant: randomize for a meaningful parity check.
        torch.nn.init.normal_(param, std=0.2)
    resampler.attn_implementation = attn_implementation
    return vpm.to(device=device, dtype=dtype), resampler.to(device=device, dtype=dtype)


def _copy_towers(
    source: tuple[SiglipVisionTransformer, Resampler], attn_implementation: str
) -> tuple[SiglipVisionTransformer, Resampler]:
    weight = source[0].embeddings.patch_embedding.weight
    vpm, resampler = _build_towers(attn_implementation, dtype=weight.dtype, device=str(weight.device))
    vpm.load_state_dict(source[0].state_dict())
    resampler.load_state_dict(source[1].state_dict())
    return vpm, resampler


def _model(towers: tuple[SiglipVisionTransformer, Resampler], *, packed: bool, vision_batch_size: int = 16):
    return SimpleNamespace(
        vpm=towers[0],
        resampler=towers[1],
        config=_Config(vision_batch_size=vision_batch_size),
        vision_packed_encode=packed,
    )


def _pixels(layouts, *, seed: int = 1, dtype: torch.dtype = torch.float32, device: str = "cpu") -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn((3, _PATCH, h * w * _PATCH), generator=generator).to(device=device, dtype=dtype) for h, w in layouts
    ]


def _encode(model, pixel_values, layouts) -> torch.Tensor:
    tgt_sizes = torch.tensor(layouts, dtype=torch.int32)
    with torch.inference_mode():
        return MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(
            model,
            {"pixel_values": pixel_values, "tgt_sizes": tgt_sizes},
        )


def _per_item(model, pixel_values, layouts) -> torch.Tensor:
    return torch.cat([_encode(model, [pixels], [layout]) for pixels, layout in zip(pixel_values, layouts)])


pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_vision_attention_defaults_to_sdpa() -> None:
    assert _resolve_vision_attention_implementation(SimpleNamespace(), SimpleNamespace()) == "sdpa"
    unset = SimpleNamespace(_attn_implementation=None)
    assert _resolve_vision_attention_implementation(unset, SimpleNamespace()) == "sdpa"


@pytest.mark.cpu
def test_explicit_vision_attention_override_is_preserved(monkeypatch) -> None:
    vision_config = SimpleNamespace(_attn_implementation="eager")
    assert _resolve_vision_attention_implementation(vision_config, SimpleNamespace()) == "eager"

    flash_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    monkeypatch.setattr(minicpmo_llm, "is_flash_attn_2_available", lambda: True)
    assert _resolve_vision_attention_implementation(flash_config, SimpleNamespace()) == "flash_attention_2"
    monkeypatch.setattr(minicpmo_llm, "is_flash_attn_2_available", lambda: False)
    assert _resolve_vision_attention_implementation(flash_config, SimpleNamespace()) == "sdpa"


@pytest.mark.cpu
def test_legacy_engine_attention_override_wins() -> None:
    vision_config = SimpleNamespace(_attn_implementation="sdpa")
    model_config = SimpleNamespace(_attn_implementation="eager")
    assert _resolve_vision_attention_implementation(vision_config, model_config) == "eager"


@pytest.mark.cpu
def test_sdpa_encoder_builds_and_selects_sdpa_attention() -> None:
    vpm = SiglipVisionTransformer(_vision_config("sdpa"))
    assert all(type(layer.self_attn) is SiglipSdpaAttention for layer in vpm.encoder.layers)
    eager = SiglipVisionTransformer(_vision_config("eager"))
    assert all(type(layer.self_attn) is SiglipAttention for layer in eager.encoder.layers)


@pytest.mark.cpu
def test_sdpa_layer_matches_eager_layer_with_padding_mask() -> None:
    torch.manual_seed(7)
    eager = SiglipEncoderLayer(_vision_config("eager")).eval()
    sdpa = SiglipEncoderLayer(_vision_config("sdpa")).eval()
    sdpa.load_state_dict(eager.state_dict())
    hidden_states = torch.randn(2, 5, 32)
    attention_mask = torch.zeros(2, 1, 5, 5)
    attention_mask[1, :, :, -2:] = torch.finfo(torch.float32).min

    with torch.inference_mode():
        eager_output = eager(hidden_states, attention_mask)[0]
        sdpa_output = sdpa(hidden_states, attention_mask)[0]

    torch.testing.assert_close(sdpa_output, eager_output, rtol=1e-5, atol=1e-6)


@pytest.mark.cpu
def test_plan_groups_equal_grids_and_bounds_chunks() -> None:
    chunks = _plan_packed_vision_chunks(_RAGGED_LAYOUTS, max_items=16)
    assert len(chunks) == 1
    indices, runs = chunks[0]
    assert sorted(indices) == list(range(len(_RAGGED_LAYOUTS)))
    # (5, 3) and (3, 5) share L=15 and are adjacent; the three (4, 4) grids form one run.
    assert runs == [(3, 5, 1), (5, 3, 1), (1, 16, 1), (2, 8, 1), (4, 4, 3)]

    chunks = _plan_packed_vision_chunks(_RAGGED_LAYOUTS, max_items=3)
    assert [len(indices) for indices, _ in chunks] == [3, 3, 1]
    assert sorted(i for indices, _ in chunks for i in indices) == list(range(len(_RAGGED_LAYOUTS)))
    for indices, runs in chunks:
        assert sum(count for _, _, count in runs) == len(indices)


@pytest.mark.cpu
@pytest.mark.parametrize("vision_batch_size", [1, 3, 16])
@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_packed_matches_padded_and_per_item(vision_batch_size: int, attn_implementation: str) -> None:
    reference = _build_towers("eager")
    towers = _copy_towers(reference, attn_implementation)
    pixel_values = _pixels(_RAGGED_LAYOUTS)

    padded = _encode(
        _model(reference, packed=False, vision_batch_size=vision_batch_size), pixel_values, _RAGGED_LAYOUTS
    )
    per_item = _per_item(_model(reference, packed=False), pixel_values, _RAGGED_LAYOUTS)
    packed = _encode(_model(towers, packed=True, vision_batch_size=vision_batch_size), pixel_values, _RAGGED_LAYOUTS)

    assert packed.shape == (len(_RAGGED_LAYOUTS), 4, 64)
    torch.testing.assert_close(packed, per_item, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(packed, padded, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
def test_packed_output_does_not_depend_on_batch_composition() -> None:
    towers = _build_towers("sdpa")
    model = _model(towers, packed=True)
    pixel_values = _pixels(_RAGGED_LAYOUTS)

    alone = _encode(model, pixel_values[:1], _RAGGED_LAYOUTS[:1])
    mixed = _encode(model, pixel_values, _RAGGED_LAYOUTS)

    torch.testing.assert_close(mixed[:1], alone, rtol=1e-6, atol=1e-6)


@pytest.mark.cpu
def test_packed_grows_resampler_position_cache_like_padded_path() -> None:
    reference = _build_towers("eager")
    towers = _copy_towers(reference, "sdpa")
    layouts = [(3, 10), (2, 2)]  # w=10 exceeds the resampler's 8x8 sin-cos cache
    pixel_values = _pixels(layouts)

    packed = _encode(_model(towers, packed=True), pixel_values, layouts)
    padded = _encode(_model(reference, packed=False), pixel_values, layouts)

    assert list(towers[1].max_size) == [8, 10]
    torch.testing.assert_close(packed, padded, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
def test_packed_falls_back_to_padded_path_on_inconsistent_grids(mocker) -> None:
    towers = _build_towers("sdpa")
    pixel_values = _pixels([(2, 2)])
    packed_spy = mocker.spy(minicpmo_llm, "_encode_vision_packed")

    # The pixels hold 4 patches, the grid claims 6: the packed path defers to
    # the padded path, which keeps its own (failing) behavior for such input.
    with torch.inference_mode(), pytest.raises(RuntimeError):
        MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(
            _model(towers, packed=True),
            {"pixel_values": pixel_values, "tgt_sizes": torch.tensor([[2, 3]])},
        )

    packed_spy.assert_not_called()


@pytest.mark.cpu
def test_packed_position_ids_match_padded_embeddings() -> None:
    vpm, _ = _build_towers("sdpa")
    embeddings = vpm.embeddings
    for height, width in _RAGGED_LAYOUTS:
        mask = torch.ones((1, height, width), dtype=torch.bool)
        expected = embeddings._create_position_ids(mask, torch.tensor([[height, width]]), device=torch.device("cpu"))
        actual = embeddings.layout_position_ids(height, width, torch.device("cpu"))
        assert torch.equal(actual, expected[0])
        assert embeddings.layout_position_ids(height, width, torch.device("cpu")) is actual


def _cuda_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@hardware_test(res={"cuda": "L4"})
def test_packed_bf16_error_is_no_worse_than_padded_eager_bf16() -> None:
    """bf16 SDPA packed path vs the shipped eager padded path, both measured against fp32."""
    _cuda_or_skip()
    reference = _build_towers("eager", device="cuda")
    layouts = [(28, 37), (32, 32), (28, 37), (37, 28), (16, 64)]
    pixel_values = _pixels(layouts, device="cuda")
    fp32 = _encode(_model(reference, packed=False), pixel_values, layouts)

    eager_bf16 = _copy_towers(reference, "eager")
    eager_bf16 = (eager_bf16[0].to(torch.bfloat16), eager_bf16[1].to(torch.bfloat16))
    sdpa_bf16 = _copy_towers(reference, "sdpa")
    sdpa_bf16 = (sdpa_bf16[0].to(torch.bfloat16), sdpa_bf16[1].to(torch.bfloat16))
    pixel_values_bf16 = [pixels.to(torch.bfloat16) for pixels in pixel_values]

    baseline = _encode(_model(eager_bf16, packed=False), pixel_values_bf16, layouts).float()
    packed = _encode(_model(sdpa_bf16, packed=True), pixel_values_bf16, layouts).float()

    baseline_error = (baseline - fp32).abs().max().item()
    packed_error = (packed - fp32).abs().max().item()
    assert packed_error <= 1.5 * baseline_error + 1e-3, (packed_error, baseline_error)
