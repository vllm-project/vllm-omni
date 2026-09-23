# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.realtime_prompt import build_realtime_prompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "<|im_start|>assistant\n":
            return [20, 21]
        return list(range(100, 100 + len(text.split())))


class _Processor:
    audio_channel_pad = 1024
    audio_bos_token = 1025

    def __init__(self) -> None:
        self.reference_codes = None

    def make_ensemble(self, prompt_audio_tokens):
        self.reference_codes = prompt_audio_tokens
        return np.full((3, 17), self.audio_channel_pad, dtype=np.int64)


def test_build_realtime_prompt_aligns_reference_and_prefill_text() -> None:
    processor = _Processor()
    reference_codes = torch.arange(80, dtype=torch.int64).reshape(5, 16)

    params = build_realtime_prompt(
        _Tokenizer(),
        processor,
        "one two three",
        reference_codes,
    )

    assert np.array_equal(processor.reference_codes, reference_codes.numpy())
    assert params["prompt_token_ids"] == [1024, 1024, 1024, 20, 21, 100, 101, 102]
    audio_grid = params["codes"]["ref"]
    assert audio_grid.shape == (len(params["prompt_token_ids"]), 16)
    assert audio_grid[-1, 0].item() == processor.audio_bos_token
    assert "ids" not in params


def test_build_realtime_prompt_streams_text_after_first_twelve_tokens() -> None:
    params = build_realtime_prompt(
        _Tokenizer(),
        _Processor(),
        " ".join(f"word{i}" for i in range(15)),
        torch.zeros((2, 16), dtype=torch.int64),
    )

    assert params["prompt_token_ids"][-12:] == list(range(100, 112))
    assert params["ids"] == {"all": [112, 113, 114]}


def test_build_realtime_prompt_rejects_wrong_processor_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _Processor()
    monkeypatch.setattr(
        processor,
        "make_ensemble",
        lambda prompt_audio_tokens: np.zeros((3, 16), dtype=np.int64),
    )

    with pytest.raises(ValueError, match=r"expected \(sequence_length, 17\)"):
        build_realtime_prompt(
            _Tokenizer(),
            processor,
            "one",
            torch.zeros((2, 16), dtype=torch.int64),
        )
