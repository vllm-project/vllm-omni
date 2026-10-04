# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.audex.speech_decoder.modeling_audex_causal_speech_decoder import (
    CausalCodecDecoderCache,
    CausalVocosBackbone,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts, pytest.mark.cache]


def _kv(values: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1)
    return key, -key


def test_growable_cache_reuses_storage_and_preserves_values() -> None:
    cache = CausalCodecDecoderCache(initial_capacity=2)

    key, value = cache.update(0, *_kv([1.0, 2.0]))
    first_storage = key.untyped_storage().data_ptr()
    torch.testing.assert_close(key.flatten(), torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(value.flatten(), torch.tensor([-1.0, -2.0]))
    cache.advance(2)

    key, value = cache.update(0, *_kv([3.0]))
    grown_storage = key.untyped_storage().data_ptr()
    assert grown_storage != first_storage
    torch.testing.assert_close(key.flatten(), torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(value.flatten(), torch.tensor([-1.0, -2.0, -3.0]))
    cache.advance(1)

    key, value = cache.update(0, *_kv([4.0]))
    assert key.untyped_storage().data_ptr() == grown_storage
    torch.testing.assert_close(key.flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(value.flatten(), torch.tensor([-1.0, -2.0, -3.0, -4.0]))
    cache.advance(1)

    assert cache.position == 4
    assert cache.key_values[0][0].shape == (1, 1, 4, 1)


def test_cache_validates_layer_progress_and_reset() -> None:
    cache = CausalCodecDecoderCache(initial_capacity=2)
    cache.update(0, *_kv([1.0]))
    cache.update(1, *_kv([2.0]))
    cache.advance(1)

    cache.update(0, *_kv([3.0]))
    with pytest.raises(RuntimeError, match="layers were not updated"):
        cache.advance(1)

    cache.reset()
    assert cache.position == 0
    assert cache.key_values == {}


@pytest.mark.parametrize("chunk_sizes", [(1,) * 11, (3, 1, 4, 3), (5, 6)])
def test_cached_decode_matches_full_causal_decode(chunk_sizes: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    model = CausalVocosBackbone(hidden_dim=32, depth=2, heads=4, pos_meb_dim=8).eval()
    hidden_states = torch.randn(2, sum(chunk_sizes), 32)

    with torch.inference_mode():
        expected = model(hidden_states)
        cache = CausalCodecDecoderCache(initial_capacity=2)
        actual = torch.cat(
            [model(chunk, cache=cache) for chunk in hidden_states.split(chunk_sizes, dim=1)],
            dim=1,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert cache.position == hidden_states.size(1)


def test_cached_decode_does_not_read_positions_back_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    model = CausalVocosBackbone(hidden_dim=16, depth=2, heads=2, pos_meb_dim=8).eval()
    cache = CausalCodecDecoderCache()
    hidden_states = torch.randn(1, 2, 16)

    def fail_item(_tensor: torch.Tensor) -> None:
        raise AssertionError("streaming decode must not call Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    with torch.inference_mode():
        model(hidden_states, cache=cache)


def test_cached_decode_builds_attention_mask_once_per_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    model = CausalVocosBackbone(hidden_dim=16, depth=3, heads=2, pos_meb_dim=8).eval()
    cache = CausalCodecDecoderCache()
    hidden_states = torch.randn(1, 2, 16)
    original_arange = torch.arange
    arange_calls = 0

    def count_arange(*args, **kwargs):
        nonlocal arange_calls
        arange_calls += 1
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", count_arange)
    with torch.inference_mode():
        model(hidden_states, cache=cache)

    # One range creates input positions and one creates the shared causal mask.
    assert arange_calls == 2
