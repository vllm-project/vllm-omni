# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Subset token coordinates, native fallbacks and bounded model ownership."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from vllm_omni.model_executor.models.lychee_fd.configuration_lychee import LycheeFDConfig
from vllm_omni.model_executor.models.lychee_fd.side_head import LycheeSideHead

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _config():
    return SimpleNamespace(
        stoken_token_ids_min=10,
        stoken_token_ids_max=20,
        control_token_ids_min=25,
        control_token_ids_max=29,
        stoken_delay_token_id=33,
        stoken_pad_token_id=32,
        start_speaking_token_id=25,
        start_listening_token_id=26,
        keep_speaking_token_id=28,
        keep_listening_token_id=27,
        sleep_token_id=30,
        detect_token_id=29,
        start_bc_token_id=31,
    )


def _native():
    return SimpleNamespace(
        head_dtype=None,
        soft_cap=None,
        scale=1.0,
        logits_as_input=False,
    )


def _head():
    return SimpleNamespace(
        tp_size=1,
        quant_method=UnquantizedEmbeddingMethod(),
        weight=torch.arange(160, dtype=torch.float32).view(40, 4) / 4,
    )


def test_released_subsets_preserve_exact_old_projection_shapes():
    side = LycheeSideHead(LycheeFDConfig(), 158363)
    assert side.token_ids["speech"].numel() == 6666
    assert side.token_ids["control"].tolist() == [158352, 158353, 158354, 158355, 158356, 158357, 158362]
    assert 151694 in side.token_ids["speech"]
    assert 158256 in side.token_ids["speech"]
    assert 151693 not in side.token_ids["speech"]  # Forced start never samples this logit.


@pytest.mark.parametrize("branch", ["speech", "control"])
@pytest.mark.parametrize("batch", [1, 4])
def test_projection_has_native_coordinates_and_skips_unused_rows(branch, batch, monkeypatch):
    side = LycheeSideHead(_config(), 40)
    head = _head()
    hidden = torch.arange(batch * 4, dtype=torch.float32).view(batch, 4)
    processor = _native()
    expected = torch.nn.functional.linear(hidden, head.weight)
    linear = torch.nn.functional.linear
    projected_rows = []

    def record(inputs, weight):
        projected_rows.append(weight.shape[0])
        return linear(inputs, weight)

    monkeypatch.setattr(torch.nn.functional, "linear", record)
    value = side.project(branch, hidden, head, processor)
    ids = side.token_ids[branch]
    assert projected_rows == [ids.numel()]
    torch.testing.assert_close(value[:, ids], expected[:, ids], rtol=0, atol=0)
    other = torch.ones(40, dtype=torch.bool)
    other[ids] = False
    assert torch.isneginf(value[:, other]).all()
    assert head.weight.shape == (40, 4)


def test_cached_subsets_remain_bounded_and_explicit_reload_clears_them():
    side = LycheeSideHead(_config(), 40)
    head = _head()
    hidden = torch.ones(1, 4)
    side.project("speech", hidden, head, _native())
    cached = side._cache["speech"].weight
    for _ in range(5):
        side.project("speech", hidden, head, _native())
        side.project("control", hidden, head, _native())
    assert side._cache["speech"].weight is cached and len(side._cache) == 2
    head.weight.add_(2)
    side.clear()
    assert not side._cache
    value = side.project("speech", hidden, head, _native())
    assert side._cache["speech"].weight is not cached
    ids = side.token_ids["speech"]
    torch.testing.assert_close(value[:, ids], torch.nn.functional.linear(hidden, head.weight[ids]))
    side.clear()
    side.clear()


def test_replaced_weight_storage_invalidates_cached_subset():
    side = LycheeSideHead(_config(), 40)
    head = _head()
    hidden = torch.ones(1, 4)
    old = side.project("control", hidden, head, _native())
    head.weight = head.weight + 2
    new = side.project("control", hidden, head, _native())
    ids = side.token_ids["control"]
    torch.testing.assert_close(new[:, ids], old[:, ids] + 8)


@pytest.mark.parametrize("fallback", ["quant", "tp", "head_dtype", "logits_input"])
def test_unsupported_side_projection_uses_native_processor(fallback):
    side = LycheeSideHead(_config(), 40)
    head = _head()
    hidden = torch.ones(1, 4)
    processor = MagicMock(**vars(_native()))
    if fallback == "quant":
        head.quant_method = object()
    elif fallback == "tp":
        head.tp_size = 2
    elif fallback == "head_dtype":
        processor.head_dtype = torch.float64
    else:
        processor.logits_as_input = True
    sentinel = object()
    processor.return_value = sentinel
    assert side.project("speech", hidden, head, processor) is sentinel
    processor.assert_called_once_with(head, hidden)
    assert not side._cache


def test_native_scale_and_soft_cap_are_preserved_with_unselected_rows_masked():
    side = LycheeSideHead(_config(), 40)
    head = _head()
    hidden = torch.ones(1, 4)
    processor = _native()
    processor.soft_cap = 30
    processor.scale = -2
    value = side.project("control", hidden, head, processor)
    ids = side.token_ids["control"]
    raw = torch.nn.functional.linear(hidden, head.weight[ids])
    torch.testing.assert_close(value[:, ids], torch.tanh(raw / 30) * 30 * -2)
    assert torch.isneginf(value[:, 0]).all()


def test_unknown_branch_fails_before_cache_allocation():
    side = LycheeSideHead(_config(), 40)
    with pytest.raises(ValueError, match="Unknown"):
        side.project("text", torch.ones(1, 4), _head(), _native())
    assert not side._cache


def test_inference_weight_without_version_counter_can_be_projected():
    side = LycheeSideHead(_config(), 40)
    with torch.inference_mode():
        head = _head()
        hidden = torch.ones(1, 4)
        result = side.project("speech", hidden, head, _native())
        assert result.shape == (1, 40)


def test_forced_start_does_not_read_excluded_logit_or_advance_rng():
    from vllm_omni.model_executor.models.lychee_fd.sampling import LycheeControlMode, sample_speech_tokens

    config = LycheeFDConfig()
    side = LycheeSideHead(config, 158363)
    head = SimpleNamespace(tp_size=1, quant_method=UnquantizedEmbeddingMethod(), weight=torch.zeros(158363, 1))
    logits = side.project("speech", torch.ones(1, 1), head, _native())
    assert torch.isneginf(logits[0, config.tts_start_token_id])
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state().clone()
    token = sample_speech_tokens(
        logits,
        modes=torch.tensor([int(LycheeControlMode.SPEAKING)]),
        speaking_steps=torch.tensor([config.stoken_delay_num]),
        config=config,
        generator=generator,
    )
    assert token.tolist() == [config.tts_start_token_id]
    torch.testing.assert_close(generator.get_state(), before, rtol=0, atol=0)
