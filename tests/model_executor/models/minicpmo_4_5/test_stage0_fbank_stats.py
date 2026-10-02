# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``duplex_fbank_stats``: timing the Stage-0 streaming mel front end without touching its output."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 as stage0_module
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime, _FbankStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Processor:
    def __init__(self) -> None:
        self._streaming_mel_processor = SimpleNamespace(buffer=np.zeros(32000, dtype=np.float32), sample_rate=16000)
        self.calls = 0

    def process_audio_streaming(self, audio_chunk, *, reset: bool, return_batch_feature: bool):
        self.calls += 1
        return {"audio_features": audio_chunk * 2, "calls": self.calls}


def _runtime(stats: _FbankStats | None) -> MiniCPMO45Stage0DuplexRuntime:
    runtime = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.processor = _Processor()
    runtime._fbank_stats = stats
    return runtime


def test_fbank_stats_off_returns_the_processor_output_untimed(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(stage0_module, "logger", logger)
    runtime = _runtime(None)
    chunk = np.ones(16000, dtype=np.float32)
    result = runtime._process_streaming_audio(chunk, 3)
    np.testing.assert_array_equal(result["audio_features"], chunk * 2)
    logger.info.assert_not_called()


def test_fbank_stats_on_logs_every_n_calls_and_keeps_the_output(monkeypatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(stage0_module, "logger", logger)
    stats = _FbankStats(every=3)
    runtime = _runtime(stats)
    chunk = np.ones(16000, dtype=np.float32)
    for index in range(5):
        result = runtime._process_streaming_audio(chunk, index)
        np.testing.assert_array_equal(result["audio_features"], chunk * 2)
        assert result["calls"] == index + 1
    assert stats.calls == 5
    assert logger.info.call_count == 1
    message, count, *_rest, buffer_p50, buffer_max = logger.info.call_args.args
    assert "fbank" in message
    assert count == 3
    assert buffer_p50 == pytest.approx(2.0) and buffer_max == pytest.approx(2.0)


@pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False), ("true", False), (None, False)])
def test_runtime_reads_the_switch_from_the_hf_config(value: object, expected: bool) -> None:
    # Same read as duplex_audio_encoder_pinned_h2d: an explicit True only.
    stage_model = SimpleNamespace(config=SimpleNamespace(duplex_fbank_stats=value), processor=_Processor())
    runtime = MiniCPMO45Stage0DuplexRuntime(stage_model, device="cpu")
    assert (runtime._fbank_stats is not None) is expected
