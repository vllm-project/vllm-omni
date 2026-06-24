# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_minimal_omni() -> Qwen3OmniMoeForConditionalGeneration:
    model = Qwen3OmniMoeForConditionalGeneration.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.talker = SimpleNamespace(text_projection=lambda x: x + 10)
    model.tts_pad_embed = torch.full((2,), -1.0)
    model.tts_eos_embed = torch.full((2,), -2.0)
    return model


def test_async_chunk_decode_consumes_cached_handoff_decode() -> None:
    model = _make_minimal_omni()
    payload = {
        "embed": {
            "cached_decode": torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        },
        "meta": {
            "num_processed_tokens": 1,
            "prefill_consumed_text_tokens": 1,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, torch.tensor([11.0, 12.0]))
    assert update["_advance_num_processed_tokens"] is True


def test_async_chunk_decode_appends_current_decode_after_cached_prefix() -> None:
    model = _make_minimal_omni()
    payload = {
        "embed": {
            "cached_decode": torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            "decode": torch.tensor([[5.0, 6.0]]),
        },
        "meta": {
            "num_processed_tokens": 3,
            "prefill_consumed_text_tokens": 1,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, torch.tensor([15.0, 16.0]))
    assert update["_advance_num_processed_tokens"] is True


def test_async_chunk_decode_uses_accumulated_decode_when_cache_is_prefix() -> None:
    model = _make_minimal_omni()
    payload = {
        "embed": {
            "cached_decode": torch.tensor([[1.0, 2.0]]),
            "decode": torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
        },
        "meta": {
            "num_processed_tokens": 2,
            "prefill_consumed_text_tokens": 1,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, torch.tensor([13.0, 14.0]))
    assert update["_advance_num_processed_tokens"] is True


def test_async_chunk_decode_clears_prior_eos_when_new_decode_arrives() -> None:
    model = _make_minimal_omni()
    payload = {
        "embed": {"decode": torch.tensor([[7.0, 8.0]])},
        "meta": {
            "num_processed_tokens": 1,
            "prefill_consumed_text_tokens": 1,
            "eos_emitted": True,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, torch.tensor([17.0, 18.0]))
    assert update["_advance_num_processed_tokens"] is True
    assert update["meta"]["eos_emitted"] is False
