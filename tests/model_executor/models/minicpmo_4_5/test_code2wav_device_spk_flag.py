# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the ``token2wav_device_spk`` backend switch.

The switch selects between the on-device CAM++ speaker embedding and the CPU
onnxruntime fallback used for accuracy triage. ``bool("false")`` is ``True``,
so a quoted YAML/JSON stage override would silently keep the device path on and
make the fallback unusable. These tests drive the real backend-selection branch
in ``_build_backend`` (only the model assets are faked) and assert which
embedding path gets enabled.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5 import (
    minicpmo_4_5_code2wav as code2wav_module,
)
from vllm_omni.model_executor.models.minicpmo_4_5 import (
    minicpmo_4_5_token2wav as token2wav_module,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeToken2Wav:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.device_spk_calls = 0
        self.trt_spk_calls = 0

    def enable_device_spk_embedding(self):
        self.device_spk_calls += 1

    def enable_trt_spk_embedding(self):
        self.trt_spk_calls += 1


class _FakeBackend:
    def __init__(self, token2wav, **kwargs):
        self.token2wav = token2wav
        self.kwargs = kwargs


@pytest.fixture
def npu_platform(monkeypatch):
    """Route ``_build_backend`` down the NPU branch on a CPU-only machine."""
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)


def _build_backend(tmp_path, monkeypatch, extra):
    """Run ``_build_backend`` with fake assets, returning (model, token2wav)."""
    assets = tmp_path / "assets"
    (assets / "token2wav").mkdir(parents=True)
    (assets / "HT_ref_audio.wav").write_bytes(b"")
    created = {}

    def _fake_token2wav(path, **kwargs):
        created["token2wav"] = _FakeToken2Wav(path, **kwargs)
        return created["token2wav"]

    monkeypatch.setattr(token2wav_module, "MiniCPMO45Token2wav", _fake_token2wav)
    monkeypatch.setattr(code2wav_module, "BatchedToken2Wav", _FakeBackend)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(tmp_path),
            stage_connector_config={"extra": extra},
        )
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    model._build_backend()
    # Reaching the end proves the switch branch ran, not just the parser.
    assert isinstance(model.backend, _FakeBackend)
    return model, created["token2wav"]


@pytest.mark.parametrize("value", [False, "false", "  FALSE  ", 0, "0"])
def test_explicit_false_keeps_the_cpu_onnxruntime_fallback(tmp_path, monkeypatch, npu_platform, value):
    _, token2wav = _build_backend(tmp_path, monkeypatch, {"token2wav_device_spk": value})
    assert token2wav.device_spk_calls == 0
    assert token2wav.trt_spk_calls == 0


@pytest.mark.parametrize("value", [None, True, "true", 1, "1"])
def test_unset_or_true_enables_the_device_embedding(tmp_path, monkeypatch, npu_platform, value):
    extra = {} if value is None else {"token2wav_device_spk": value}
    _, token2wav = _build_backend(tmp_path, monkeypatch, extra)
    assert token2wav.device_spk_calls == 1


@pytest.mark.parametrize("value", ["maybe", "yes", 2, 1.5, []])
def test_invalid_value_is_rejected(tmp_path, monkeypatch, npu_platform, value):
    with pytest.raises(ValueError, match="token2wav_device_spk"):
        _build_backend(tmp_path, monkeypatch, {"token2wav_device_spk": value})
