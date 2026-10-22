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
        self.device_spk_use_trt = None
        self.speaker_graph_config = None

    def enable_device_spk_embedding(self, *, use_trt=True):
        self.device_spk_calls += 1
        self.device_spk_use_trt = use_trt

    def enable_speaker_cuda_graph(self, config):
        self.speaker_graph_config = config

    def enable_trt_spk_embedding(self):
        self.trt_spk_calls += 1


class _FakeBackend:
    def __init__(self, token2wav, **kwargs):
        self.token2wav = token2wav
        self.kwargs = kwargs


@pytest.fixture(params=["cuda", "npu"])
def device_platform(monkeypatch, request):
    """Exercise each device speaker backend without loading model assets."""
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: request.param == "npu")
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: request.param == "cuda")


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
def test_explicit_false_keeps_the_cpu_onnxruntime_fallback(tmp_path, monkeypatch, device_platform, value):
    _, token2wav = _build_backend(tmp_path, monkeypatch, {"token2wav_device_spk": value})
    assert token2wav.device_spk_calls == 0
    assert token2wav.trt_spk_calls == 0


@pytest.mark.parametrize("value", [None, True, "true", 1, "1"])
def test_unset_or_true_enables_the_device_embedding(tmp_path, monkeypatch, device_platform, value):
    extra = {} if value is None else {"token2wav_device_spk": value}
    _, token2wav = _build_backend(tmp_path, monkeypatch, extra)
    assert token2wav.device_spk_calls == 1
    assert token2wav.device_spk_use_trt is False
    assert token2wav.trt_spk_calls == 0


@pytest.mark.parametrize("value", ["maybe", "yes", 2, 1.5, []])
def test_invalid_value_is_rejected(tmp_path, monkeypatch, device_platform, value):
    with pytest.raises(ValueError, match="token2wav_device_spk"):
        _build_backend(tmp_path, monkeypatch, {"token2wav_device_spk": value})


def test_invalid_value_is_rejected_off_device_too(tmp_path, monkeypatch):
    """The flag is parsed before the platform branch, so a typo fails anywhere."""
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)

    with pytest.raises(ValueError, match="token2wav_device_spk"):
        _build_backend(tmp_path, monkeypatch, {"token2wav_device_spk": "maybe"})


@pytest.mark.parametrize("use_trt", [False, True])
def test_cuda_speaker_backend_preserves_lazy_model_loading(tmp_path, monkeypatch, use_trt):
    """The facade can select torch on CUDA while other callers retain TRT."""
    from vllm_omni.model_executor.models.cosyvoice3 import speaker_embedding_torch

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    core = token2wav_module.StepAudio2Token2WavCore(str(tmp_path), device="cuda")
    calls = []
    speaker = object()

    def torch_backend(path, *, device):
        assert path == f"{tmp_path}/campplus.onnx"
        assert device == core.device
        calls.append("torch")
        return speaker

    monkeypatch.setattr(speaker_embedding_torch, "get_campplus_torch", torch_backend)
    monkeypatch.setattr(core, "enable_trt_spk_embedding", lambda: calls.append("trt"))
    facade = token2wav_module.MiniCPMO45Token2wav.__new__(token2wav_module.MiniCPMO45Token2wav)
    facade._core = core
    if use_trt:
        facade.enable_device_spk_embedding()
        assert calls == ["trt"]
    else:
        facade.enable_device_spk_embedding(use_trt=False)
        assert calls == ["torch"]
        assert core._spk_model is speaker
    assert core._models_loaded is False
    assert core._flow is None
    assert core._hift is None


_OTHER_DEFAULTS = {
    "token2wav_spk_cuda_graph",
    "token2wav_float16",
    "token2wav_trt",
    "code2wav_bfloat16_attention_cache",
    "cfm_graph_precapture",
    "token2wav_ref_prefetch",
    "token2wav_ref_prefetch_setup",
    "enable_hift_graph",
    "enable_cfm_graph",
    "cfm_onset_merge",
    "enable_code2wav_encoder_graph",
    "code2wav_encoder_capture_after_precapture",
    "cfm_fused_body",
    "cfm_slot_pool",
    "enable_whole_euler",
    "cfm_row_offset_merge",
    "cfm_encoder_cuda_graph",
    "cfm_cross_turn_buckets",
    "cfm_precapture_empty_cache",
    "cfm_drop_upstream_att_buffer",
}


@pytest.mark.parametrize("key", sorted(_OTHER_DEFAULTS))
def test_other_stage_extra_flags_are_strict(tmp_path, monkeypatch, device_platform, key):
    with pytest.raises(ValueError, match=key):
        _build_backend(tmp_path, monkeypatch, {key: "maybe"})


def test_quoted_false_disables_other_stage_flags(tmp_path, monkeypatch, device_platform):
    model, _ = _build_backend(
        tmp_path,
        monkeypatch,
        {
            "cfm_graph_precapture": "false",
            "enable_hift_graph": "false",
            "token2wav_ref_prefetch": "false",
        },
    )
    assert model._precapture_pending is False
    assert model._hift_graph_config["enabled"] is False
    assert model._ref_prefetch_enabled is False


def test_quoted_false_disables_default_on_flags(tmp_path, monkeypatch, device_platform):
    """`cfm_fused_body` and `enable_whole_euler` default on: a quoted "false" must still turn them off."""
    model, _ = _build_backend(
        tmp_path,
        monkeypatch,
        {"cfm_fused_body": "false", "enable_whole_euler": "false"},
    )
    assert model._cfm_graph_config["fused_body"] is False
    assert model._cfm_graph_config["enable_whole_euler"] is False


def test_quoted_true_enables_flags_previously_compared_with_is_true(tmp_path, monkeypatch, device_platform):
    model, _ = _build_backend(
        tmp_path,
        monkeypatch,
        {
            "cfm_row_offset_merge": "true",
            "cfm_slot_pool": "true",
            "cfm_encoder_cuda_graph": "true",
            "enable_code2wav_encoder_graph": "true",
            "cfm_cross_turn_buckets": "true",
            "cfm_precapture_empty_cache": "true",
        },
    )
    assert model._cfm_graph_config["row_offset_merge"] is True
    assert model._cfm_graph_config["slot_pool"] is True
    assert model._encoder_graph_config["enabled"] is True
    assert model._chunk_encoder_graph_config["enabled"] is True
    assert model._cross_turn_buckets is True
    assert model._precapture_empty_cache is True


def test_explicit_null_falls_back_to_the_default(tmp_path, monkeypatch, device_platform):
    """`key: ~` in a YAML override means "unset", not "false"."""
    model, token2wav = _build_backend(
        tmp_path,
        monkeypatch,
        {
            "token2wav_device_spk": None,
            "cfm_fused_body": None,
            "cfm_graph_precapture": None,
            "enable_whole_euler": None,
        },
    )
    assert token2wav.device_spk_calls == 1
    assert token2wav.device_spk_use_trt is False
    assert token2wav.trt_spk_calls == 0
    assert model._cfm_graph_config["fused_body"] is True
    assert model._cfm_graph_config["enable_whole_euler"] is True
    assert model._precapture_pending is True


@pytest.mark.parametrize("key", ["token2wav_allow_tf32", "code2wav_allow_tf32"])
def test_invalid_tf32_value_fails_at_startup_even_off_cuda(tmp_path, monkeypatch, device_platform, key):
    """The TF32 mode is resolved in `__init__`, so a typo cannot wait for the first forward."""
    with pytest.raises(ValueError, match="allow_tf32"):
        _build_backend(tmp_path, monkeypatch, {key: "maybe"})


def test_invalid_tf32_env_fails_at_startup_even_off_cuda(tmp_path, monkeypatch, device_platform):
    monkeypatch.setenv("MINICPMO_CODE2WAV_TF32", "maybe")
    with pytest.raises(ValueError, match="allow_tf32"):
        _build_backend(tmp_path, monkeypatch, {})


def test_tf32_mode_is_resolved_once_and_reused(tmp_path, monkeypatch, device_platform):
    model, _ = _build_backend(tmp_path, monkeypatch, {"token2wav_allow_tf32": "off"})
    assert model._tf32_mode == "off"
    # Mutating the stage config afterwards must not re-read it per forward.
    model.vllm_config.model_config.stage_connector_config["extra"]["token2wav_allow_tf32"] = True
    assert model._tf32_mode == "off"


@pytest.mark.parametrize("value,enabled", [(True, True), ("true", True), (False, False), ("false", False)])
def test_speaker_graph_flag_reaches_selected_backend(tmp_path, monkeypatch, value, enabled):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    model, speaker = _build_backend(tmp_path, monkeypatch, {"token2wav_spk_cuda_graph": value})
    assert speaker.device_spk_calls == 1
    assert speaker.speaker_graph_config is (model.vllm_config if enabled else None)
