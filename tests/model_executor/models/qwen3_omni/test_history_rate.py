# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.session.history_calibration import HeardTextSnapshot
from vllm_omni.model_executor.models.qwen3_omni.duplex.history_rate import QwenTokenRateCalibration
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class OffsetTokenizer:
    is_fast = True

    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_offsets_mapping": True}
        return {"offset_mapping": self.offsets}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "offsets", "played_ms", "expected"),
    [
        ("Hello world again", [(0, 5), (5, 8), (8, 11), (11, 17)], 660, "Hello"),
        ("Hello world again", [(0, 5), (5, 8), (8, 11), (11, 17)], 910, "Hello world"),
        ("Hello world again", [(0, 5), (5, 11), (11, 17)], 10000, "Hello world"),
        ("你好吗", [(0, 1), (0, 1), (1, 2), (2, 3)], 410, ""),
        ("你好吗", [(0, 1), (0, 1), (1, 2), (2, 3)], 660, "你"),
        ("今天很好", [(0, 2), (2, 3), (3, 4)], 409, ""),
        ("今天很好", [(0, 2), (2, 3), (3, 4)], 410, "今天"),
        ("今天很好", [(0, 2), (2, 3), (3, 4)], 0, ""),
        ("只有一个词", [(0, 5)], 10000, "只有一个"),
        ("word", [(0, 4)], 10000, ""),
        ("", [], 10000, ""),
    ],
)
async def test_rate_preserves_original_offsets_and_whole_units(text, offsets, played_ms, expected):
    calibrate = QwenTokenRateCalibration(OffsetTokenizer(offsets), 250)
    end = await calibrate(HeardTextSnapshot("r", text, b"", 0, played_ms))
    assert text[:end] == expected


@pytest.mark.asyncio
async def test_rate_bounds_tokenization_work():
    def fail(*args, **kwargs):
        pytest.fail("Oversized text must be refused before tokenization")

    class Tokenizer:
        is_fast = True
        __call__ = fail

    calibrate = QwenTokenRateCalibration(Tokenizer(), 250)
    for text in ("a" * 8193, "天" * 2049):
        assert await calibrate(HeardTextSnapshot("r", text, b"", 0, 1000)) is None


@pytest.mark.asyncio
async def test_rate_accepts_unit_limit_and_withholds_final_unit():
    text = "天" * 2048
    calibrate = QwenTokenRateCalibration(OffsetTokenizer([(i, i + 1) for i in range(len(text))]), 383)
    assert await calibrate(HeardTextSnapshot("r", text, b"", 0, 1_000_000)) == 2047


@pytest.mark.asyncio
@pytest.mark.parametrize(("played_ms", "expected"), [(542, ""), (543, "今天"), (925, "今天"), (926, "今天很")])
async def test_configured_383_rate_rounds_down_after_audio_margin(played_ms, expected):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = SimpleNamespace(tokenizer=OffsetTokenizer([(0, 2), (2, 3), (3, 4)]))
    policy = plugin.history_calibrator(DuplexSessionRuntimeConfig(history_ms_per_token=383))
    text = "今天很好"
    end = await policy.calibrate(HeardTextSnapshot("r", text, b"", 0, played_ms))
    assert text[:end] == expected


def test_plugin_selects_rate_without_pcm_retention_or_asr():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = SimpleNamespace(tokenizer=OffsetTokenizer([(0, 5)]))
    config = DuplexSessionRuntimeConfig(history_ms_per_token=260)
    policy = plugin.history_calibrator(config)
    assert not policy.requires_audio
    assert not plugin.data_plane.retain_history_audio
    assert isinstance(policy.calibrate, QwenTokenRateCalibration)
    assert plugin.history_calibrator(config) is policy


def test_plugin_default_and_asr_policy():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    assert plugin.history_calibrator(DuplexSessionRuntimeConfig()) is None
    policy = plugin.history_calibrator(
        DuplexSessionRuntimeConfig(history_asr_url="http://localhost/transcribe", history_asr_model="asr")
    )
    assert policy.requires_audio
    assert plugin.data_plane.retain_history_audio


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_rate_refuses_invalid_priors(rate):
    with pytest.raises(ValueError):
        QwenTokenRateCalibration(OffsetTokenizer([]), rate)
