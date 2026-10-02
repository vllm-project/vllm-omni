# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Code2Wav decode buckets: ``code2wav_bucket_stats`` and ``cfm_cross_turn_buckets`` (connector extras)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav as code2wav_module
from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import _forward, _info, _model
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
    _BucketStats,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _two_turn_rows() -> list[dict]:
    # Two fresh streams of the same voice that are on different turns.
    return [_info("a", 0, [1, 2], cache_epoch=0), _info("b", 0, [3, 4], cache_epoch=1)]


def test_switches_default_off_and_keep_the_historical_key() -> None:
    model, _ = _model()
    assert model._cross_turn_buckets is False
    assert model._bucket_stats is None
    item = SimpleNamespace(previous=None, prompt_cache_id="p", prompt_wav="w", cache_epoch=3)
    assert MiniCPMO45Code2Wav._bucket_key(item) == ("p", "w", ("uninitialized",), 3)
    assert MiniCPMO45Code2Wav._bucket_key(item, cross_turn=True) == ("p", "w", ("uninitialized",))
    key = ("p", "w", ("sig",), 3)
    assert MiniCPMO45Code2Wav._onset_group_key(key) == ("p", "w", 3)
    assert MiniCPMO45Code2Wav._onset_group_key(key[:3], cross_turn=True) == ("p", "w")


def test_rows_of_different_turns_split_by_default_and_merge_when_enabled() -> None:
    split_model, split_t2w = _model()
    split = _forward(split_model, _two_turn_rows())
    assert split_t2w.hift.calls == [1, 1]

    merged_model, merged_t2w = _model()
    merged_model._cross_turn_buckets = True
    merged = _forward(merged_model, _two_turn_rows())
    assert merged_t2w.hift.calls == [2]
    for got, want in zip(
        merged.multimodal_outputs["model_outputs"], split.multimodal_outputs["model_outputs"], strict=True
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-6)
    # Each stream keeps its own turn's state.
    assert merged_model._states["a"].cache_epoch == 0
    assert merged_model._states["b"].cache_epoch == 1


def test_bucket_stats_count_rows_buckets_decodes_and_split_reasons(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(code2wav_module, "logger", logger)
    model, _ = _model()
    model._bucket_stats = _BucketStats(every=2)
    _forward(model, _two_turn_rows())
    assert logger.info.call_count == 0
    _forward(model, [_info("c", 0, [5, 6])])
    assert logger.info.call_count == 1
    message, forwards, rows, buckets, multi_buckets, multi_forwards, decodes, rows_per_decode, reasons = (
        logger.info.call_args.args
    )
    assert "Code2Wav buckets" in message
    assert forwards == 2
    assert rows == {1: 1, 2: 1}
    assert buckets == pytest.approx(1.5)
    assert (multi_buckets, multi_forwards) == (pytest.approx(2.0), 1)
    assert decodes == pytest.approx(1.5)
    assert rows_per_decode == pytest.approx(1.0)
    assert reasons == {"cache_epoch": 1}
    # The window restarts after each line.
    assert model._bucket_stats.forwards == 0


def test_split_reason_names_prompt_onset_resident_and_tensor_signatures() -> None:
    stats = _BucketStats()

    def row(prompt: str, att_cache: object | None, epoch: int = 0) -> SimpleNamespace:
        previous = (
            None
            if att_cache is None
            else SimpleNamespace(token2wav=SimpleNamespace(flow_cache={"estimator_att_cache": att_cache}))
        )
        return SimpleNamespace(previous=previous, prompt_cache_id=prompt, prompt_wav="w", cache_epoch=epoch)

    tensor = torch.zeros(1)
    assert stats.split_reason(row("p", tensor), row("q", tensor), cross_turn=False) == "prompt"
    # A fresh stream next to a continuation is the onset split, not a cache layout one.
    assert stats.split_reason(row("p", None), row("p", object()), cross_turn=True) == "onset"
    assert stats.split_reason(row("p", tensor), row("p", None), cross_turn=True) == "onset"
    assert stats.split_reason(row("p", object()), row("p", tensor), cross_turn=True) == "signature_resident"
    assert stats.split_reason(row("p", tensor), row("p", torch.zeros(2)), cross_turn=True) == "signature_tensor"
