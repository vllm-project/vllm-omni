# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""When the Qwen3-Omni Code2Wav stage captures its CUDA graphs: from the
worker's ``capture_auxiliary_graphs`` hook after the first load, at once on a
weight reload, and while loading on NPU."""

from unittest.mock import MagicMock

import pytest
import torch

import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni as qwen3_omni
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model(*, code2wav: bool = True) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.thinker = None
    model.talker = None
    model.code2wav = MagicMock(load_weights=MagicMock(return_value=set())) if code2wav else None
    model.model_stage = "code2wav" if code2wav else "thinker"
    return model


@pytest.fixture
def platform(monkeypatch):
    npu = MagicMock(return_value=False)
    monkeypatch.setattr(qwen3_omni.current_omni_platform, "is_npu", npu)
    return npu


@pytest.fixture
def enable(monkeypatch, platform):
    calls = MagicMock()
    monkeypatch.setattr(Qwen3OmniMoeForConditionalGeneration, "_maybe_enable_code2wav_cudagraph", calls)
    return calls


def test_load_defers_capture_to_the_warmup_hook_which_captures_once(enable):
    model = _model()

    model.load_weights(iter(()))
    model.code2wav.precompute_snake_caches.assert_called_once()
    enable.assert_not_called()

    model.capture_auxiliary_graphs()
    model.capture_auxiliary_graphs()
    enable.assert_called_once()


@pytest.mark.parametrize("code2wav", [True, False])
def test_warmup_hook_captures_nothing_without_loaded_code2wav_weights(enable, code2wav):
    _model(code2wav=code2wav).capture_auxiliary_graphs()

    enable.assert_not_called()


def test_weight_reload_after_warmup_recaptures_at_once(enable):
    model = _model()
    model.load_weights(iter(()))
    model.capture_auxiliary_graphs()

    model.load_weights(iter(()))

    assert enable.call_count == 2
    assert model.code2wav.precompute_snake_caches.call_count == 2


def test_npu_captures_while_loading(enable, platform):
    platform.return_value = True
    model = _model()

    model.load_weights(iter(()))
    enable.assert_called_once()

    model.capture_auxiliary_graphs()
    enable.assert_called_once()


def test_npu_weight_reload_recaptures_while_loading(enable, platform):
    platform.return_value = True
    model = _model()
    model.load_weights(iter(()))
    model.capture_auxiliary_graphs()

    model.load_weights(iter(()))

    assert enable.call_count == 2


def test_failed_capture_falls_back_to_eager_with_a_warning(enable, monkeypatch):
    enable.side_effect = RuntimeError("capture failed")
    warning = MagicMock()
    monkeypatch.setattr(qwen3_omni.logger, "warning", warning)
    model = _model()
    model.load_weights(iter(()))

    model.capture_auxiliary_graphs()

    assert "falling back to eager" in warning.call_args.args[0]


def test_failed_capture_raises_on_npu(enable, platform):
    enable.side_effect = RuntimeError("capture failed")
    platform.return_value = True

    with pytest.raises(RuntimeError, match="capture failed"):
        _model().load_weights(iter(()))


def test_failed_snake_precompute_skips_graph_capture(enable, monkeypatch):
    warning = MagicMock()
    monkeypatch.setattr(qwen3_omni.logger, "warning", warning)
    model = _model()
    model.code2wav.precompute_snake_caches.side_effect = RuntimeError("precompute failed")

    model.load_weights(iter(()))
    model.capture_auxiliary_graphs()

    enable.assert_not_called()
    assert "not capturing" in warning.call_args.args[0]
