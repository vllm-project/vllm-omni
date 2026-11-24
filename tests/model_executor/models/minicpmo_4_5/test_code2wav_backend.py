# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Code2Wav asset loading, backend selection, and startup capture contracts.

``token2wav_device_spk`` selects between the on-device CAM++ embedding and the CPU
onnxruntime fallback used for accuracy triage. ``bool("false")`` is ``True``,
so a quoted YAML/JSON stage override would silently keep the device path on and
make the fallback unusable. These tests drive the real backend-selection branch
in ``_build_backend`` (only the model assets are faked) and assert which
embedding path gets enabled.
"""

from __future__ import annotations

import contextlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mock import patch_hf_snapshot_download
from vllm_omni.model_executor.models.minicpmo_4_5 import (
    minicpmo_4_5_code2wav as code2wav_module,
)
from vllm_omni.model_executor.models.minicpmo_4_5.code2wav import token2wav as token2wav_module
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
    _resolve_model_dir,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeToken2Wav:
    def __init__(self, path, **kwargs):
        self.device = (
            "cuda"
            if current_omni_platform.is_cuda()
            else "npu"
            if current_omni_platform.is_npu()
            else current_omni_platform.device_type or "cpu"
        )
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
        return True

    def enable_trt_spk_embedding(self, *, cpu_reference=False):
        self.trt_spk_calls += 1
        self.trt_cpu_reference = cpu_reference

    @property
    def flow(self):
        estimator = SimpleNamespace()
        decoder = SimpleNamespace(estimator=estimator)
        device = torch.device("privateuseone" if self.device == "npu" else self.device)
        return SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=device)]),
            decoder=decoder,
        )


class _FakeBackend:
    def __init__(self, token2wav, **kwargs):
        self.token2wav = token2wav
        self.kwargs = kwargs
        self._whole_euler_graph_wrapper = None


@pytest.fixture(autouse=True)
def tensorrt_importable(monkeypatch):
    """Backend selection probes ``import tensorrt``; CPU CI images do not ship it."""
    monkeypatch.setitem(sys.modules, "tensorrt", ModuleType("tensorrt"))


@pytest.fixture(params=["cuda", "npu"])
def device_platform(monkeypatch, request):
    """Exercise each device speaker backend without loading model assets."""
    monkeypatch.setattr(current_omni_platform, "device_type", request.param)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: request.param == "npu")
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: request.param == "cuda")
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: False)


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


@pytest.mark.parametrize(
    "flow,reason",
    [
        pytest.param(torch.nn.Identity(), "flow has no parameters", id="parameterless"),
        pytest.param(torch.nn.Linear(1, 1, device="cpu"), "flow is on CPU", id="cpu"),
    ],
)
def test_device_speaker_fallback_without_accelerator(tmp_path, monkeypatch, mocker, flow, reason):
    monkeypatch.setattr(_FakeToken2Wav, "flow", property(lambda self: flow))
    warning = mocker.patch.object(code2wav_module.logger, "warning")

    _, token2wav = _build_backend(tmp_path, monkeypatch, {"token2wav_device_spk": True})

    assert token2wav.device_spk_calls == 0
    assert token2wav.trt_spk_calls == 0
    warning.assert_called_once_with(
        "MiniCPM-o Code2Wav: token2wav_device_spk=true but %s; using CPU onnxruntime", reason
    )


@pytest.mark.parametrize("value", ["maybe", "yes", 2, 1.5, []])
def test_invalid_value_is_rejected(tmp_path, monkeypatch, device_platform, value):
    with pytest.raises(ValueError, match="token2wav_device_spk"):
        _build_backend(tmp_path, monkeypatch, {"token2wav_device_spk": value})


@pytest.mark.parametrize("value", [None, False, True])
def test_other_torch_accelerator_honors_device_speaker_flag(tmp_path, monkeypatch, value):
    monkeypatch.setattr(current_omni_platform, "device_type", "mps")
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    extra = {} if value is None else {"token2wav_device_spk": value}
    _, token2wav = _build_backend(tmp_path, monkeypatch, extra)
    assert token2wav.device_spk_calls == int(value is True)
    assert token2wav.trt_spk_calls == 0


def test_device_speaker_cache_reuses_unindexed_mps(tmp_path, monkeypatch):
    """An unindexed MPS device and mps:0 share one loaded speaker."""
    from vllm_omni.model_executor.models.cosyvoice3 import speaker_embedding_torch

    device = torch.device("mps:0")
    build = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(speaker_embedding_torch, "_CAMPPLUS_TORCH_CACHE", {})
    monkeypatch.setattr(speaker_embedding_torch, "_build_campplus_fp32", build)
    monkeypatch.setattr(torch, "empty", Mock(return_value=Mock(spec=torch.Tensor, device=device)))
    path = str(tmp_path / "campplus.onnx")

    speaker = speaker_embedding_torch.get_campplus_torch(path, "mps")

    assert speaker.device == device
    assert speaker_embedding_torch.get_campplus_torch(path, torch.device("mps")) is speaker
    assert speaker_embedding_torch.get_campplus_torch(path, "mps:0") is speaker
    build.assert_called_once_with(path)


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
    from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import StepAudio2Token2WavCore

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    core = StepAudio2Token2WavCore(str(tmp_path), device="cuda")
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
    "token2wav_s3_cuda_graph",
    "token2wav_spk_cuda_graph",
    "token2wav_spk_cpu_reference",
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


@pytest.mark.parametrize("platform", ["cuda", "rocm", "npu", "cpu"])
@pytest.mark.parametrize("precision_flag", ["token2wav_float16", "cfm_float16"])
@pytest.mark.parametrize("value", [True, "true", False, "false"])
@pytest.mark.parametrize("tf32_source", ["extra", "env"])
def test_fused_body_default_with_tf32_disabled(tmp_path, monkeypatch, platform, precision_flag, value, tf32_source):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: platform == "cuda")
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: platform == "rocm")
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: platform == "npu")
    monkeypatch.delenv("MINICPMO_CODE2WAV_TF32", raising=False)
    extra = {precision_flag: value}
    if tf32_source == "env":
        monkeypatch.setenv("MINICPMO_CODE2WAV_TF32", "off")
    else:
        extra["token2wav_allow_tf32"] = False

    model, _ = _build_backend(tmp_path, monkeypatch, extra)

    assert model._tf32_mode == "off"
    expected = platform == "npu" or (platform in ("cuda", "rocm") and value in (True, "true"))
    assert model.backend.kwargs["cfm_graph_config"]["fused_body"] is expected
    assert model.backend.kwargs["cfm_tf32"] is False


@pytest.mark.parametrize("precision_flag", ["token2wav_float16", "cfm_float16"])
@pytest.mark.parametrize("value,expected", [(True, True), ("true", True), (False, False), ("false", False)])
def test_fp16_fused_body_explicit_override(tmp_path, monkeypatch, precision_flag, value, expected):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setenv("MINICPMO_CODE2WAV_TF32", "off")
    model, _ = _build_backend(
        tmp_path,
        monkeypatch,
        {precision_flag: True, "cfm_fused_body": value},
    )
    assert model.backend.kwargs["cfm_graph_config"]["fused_body"] is expected
    assert model.backend.kwargs["cfm_tf32"] is False


@pytest.mark.parametrize("value,enabled", [(True, True), ("true", True), (False, False), ("false", False)])
def test_speaker_graph_flag_reaches_selected_backend(tmp_path, monkeypatch, value, enabled):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    model, speaker = _build_backend(tmp_path, monkeypatch, {"token2wav_spk_cuda_graph": value})
    assert speaker.device_spk_calls == 1
    assert speaker.speaker_graph_config is (model.vllm_config if enabled else None)


@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("selection", ["torch", "cpu", "trt-env", "trt-config", "trt-speaker"])
def test_reference_graphs_respect_selected_backend(tmp_path, monkeypatch, mocker, platform, selection):
    from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import CampplusTorch
    from vllm_omni.model_executor.models.minicpmo_4_5.code2wav import reference_audio_graph
    from vllm_omni.model_executor.models.step_audio2 import step_audio2_dit_trt

    _force_platform(monkeypatch, platform)
    monkeypatch.setenv("MINICPMO_TOKEN2WAV_TRT", "1" if selection == "trt-env" else "0")
    mocker.patch.object(step_audio2_dit_trt, "build_dit_trt_stepper", return_value=object())
    speaker_wrapper = mocker.patch.object(reference_audio_graph, "SpeakerCudaGraph")
    s3_wrapper = mocker.patch.object(reference_audio_graph, "S3CudaGraph")
    original_init = _FakeToken2Wav.__init__
    device = torch.device("cuda" if platform == "cuda" else "privateuseone")

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tokenizer = mocker.Mock()
        tokenizer.parameters = lambda: iter([mocker.Mock(device=device)])
        self._core = mocker.Mock(device=device, _spk_model=None, audio_tokenizer=tokenizer)

    def device_speaker(self, *, use_trt):
        speaker = CampplusTorch.__new__(CampplusTorch)
        speaker.device = device
        self._core._spk_model = speaker

    def trt_speaker(self, **kwargs):
        self._core._spk_model = object()

    monkeypatch.setattr(_FakeToken2Wav, "__init__", init)
    monkeypatch.setattr(_FakeToken2Wav, "enable_device_spk_embedding", device_speaker)
    monkeypatch.setattr(_FakeToken2Wav, "enable_trt_spk_embedding", trt_speaker)
    for name in ("enable_speaker_cuda_graph", "enable_s3_cuda_graph"):
        monkeypatch.setattr(_FakeToken2Wav, name, getattr(token2wav_module.MiniCPMO45Token2wav, name), raising=False)

    _build_backend(
        tmp_path,
        monkeypatch,
        {
            "token2wav_spk_cuda_graph": True,
            "token2wav_s3_cuda_graph": True,
            "token2wav_device_spk": selection != "cpu",
            # The legacy env var applies only while the key is unset (#6232).
            "token2wav_trt": None if selection == "trt-env" else selection == "trt-config",
            "token2wav_spk_cpu_reference": selection == "trt-speaker",
        },
    )
    assert speaker_wrapper.call_count == int(platform == "cuda" and selection == "torch")
    assert s3_wrapper.call_count == int(platform == "cuda")


def test_cpu_reference_speaker_accepts_the_default_six_second_window(tmp_path, monkeypatch):
    """598 fbank frames fit the CAM++ profile, so the opt-in still installs."""
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    _, token2wav = _build_backend(
        tmp_path,
        monkeypatch,
        {"token2wav_spk_cpu_reference": True, "ref_audio_max_seconds": 6},
    )
    assert token2wav.trt_spk_calls == 1
    assert token2wav.trt_cpu_reference is True


def test_cpu_reference_speaker_rejects_a_31_second_window(tmp_path, monkeypatch):
    """31 s is 3098 fbank frames, past the profile max of 3000."""
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    with pytest.raises(ValueError, match="3098"):
        _build_backend(
            tmp_path,
            monkeypatch,
            {"token2wav_spk_cpu_reference": True, "ref_audio_max_seconds": 31},
        )


def test_trt_speaker_rejects_frames_outside_the_profile():
    import torch

    from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import CampplusTRT

    speaker = CampplusTRT.__new__(CampplusTRT)
    with pytest.raises(ValueError, match=r"4\.\.3000"):
        speaker(torch.zeros(3098, 80))


@pytest.mark.parametrize("value", [True, "true", 1])
def test_cpu_reference_speaker_is_explicit_cuda_opt_in(tmp_path, monkeypatch, value):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    _, token2wav = _build_backend(tmp_path, monkeypatch, {"token2wav_spk_cpu_reference": value})
    assert token2wav.trt_spk_calls == 1
    assert token2wav.trt_cpu_reference is True
    assert token2wav.device_spk_calls == 0


def test_device_spk_false_overrides_cpu_reference_speaker(tmp_path, monkeypatch):
    """The CPU onnxruntime triage switch still wins after the reference flag is set."""
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    _, token2wav = _build_backend(
        tmp_path,
        monkeypatch,
        {"token2wav_spk_cpu_reference": True, "token2wav_device_spk": False},
    )
    assert token2wav.trt_spk_calls == 0
    assert token2wav.device_spk_calls == 0


def _record_dit_trt_builds(monkeypatch) -> list[dict]:
    """Replace the TensorRT engine builder with a recorder of its keyword arguments."""
    from vllm_omni.model_executor.models.step_audio2 import step_audio2_dit_trt

    builds: list[dict] = []

    def _fake_build(*args, **kwargs):
        builds.append(kwargs)
        return object()

    monkeypatch.setattr(step_audio2_dit_trt, "build_dit_trt_stepper", _fake_build)
    return builds


@pytest.fixture
def cuda_platform_without_trt_env(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    for name in ("MINICPMO_TOKEN2WAV_TRT", "MINICPMO_TOKEN2WAV_TRT_DTYPE", "MINICPMO_TOKEN2WAV_TRT_CACHE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "extra,env,expect_trt",
    [
        pytest.param({"token2wav_trt": True}, None, True, id="key-true"),
        pytest.param({}, "1", True, id="env-only"),
        pytest.param({"token2wav_trt": False}, "1", False, id="key-false-beats-env"),
    ],
)
def test_full_dit_trt_overrides_cpu_reference_speaker(
    tmp_path, monkeypatch, cuda_platform_without_trt_env, extra, env, expect_trt
):
    """token2wav_trt selects full DiT TensorRT; the legacy env var counts only while it is unset (#6232)."""
    if env is not None:
        monkeypatch.setenv("MINICPMO_TOKEN2WAV_TRT", env)
    builds = _record_dit_trt_builds(monkeypatch)
    _, token2wav = _build_backend(tmp_path, monkeypatch, {"token2wav_spk_cpu_reference": True, **extra})
    assert len(builds) == int(expect_trt)
    # Without full DiT TensorRT the default CPU-reference speaker engine is used.
    assert token2wav.trt_spk_calls == 1
    assert token2wav.trt_cpu_reference is (not expect_trt)


@pytest.mark.parametrize(
    "extra,expected_dtype,expected_cache",
    [
        pytest.param({}, torch.float16, None, id="defaults"),
        pytest.param(
            {"token2wav_trt_dtype": "FP32", "token2wav_trt_cache_dir": "/keys/cache"},
            torch.float32,
            "/keys/cache",
            id="keys",
        ),
    ],
)
def test_full_dit_trt_dtype_and_cache_dir_reach_engine_builder(
    tmp_path, monkeypatch, cuda_platform_without_trt_env, extra, expected_dtype, expected_cache
):
    builds = _record_dit_trt_builds(monkeypatch)
    _build_backend(tmp_path, monkeypatch, {"token2wav_trt": True, **extra})
    assert len(builds) == 1
    assert builds[0]["dtype"] is expected_dtype
    assert builds[0]["cache_dir"] == expected_cache


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_invalid_token2wav_trt_dtype_fails_at_startup(tmp_path, monkeypatch, cuda_platform_without_trt_env, platform):
    """A dtype typo in a stage key fails on every platform instead of silently selecting fp16."""
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: platform == "cuda")
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: platform == "npu")
    builds = _record_dit_trt_builds(monkeypatch)
    with pytest.raises(ValueError, match="token2wav_trt_dtype must be one of"):
        _build_backend(tmp_path, monkeypatch, {"token2wav_trt": True, "token2wav_trt_dtype": "bf16"})
    assert builds == []


def test_cpu_reference_speaker_falls_back_without_tensorrt(tmp_path, monkeypatch):
    """The default profile stays loadable in images that do not ship TensorRT."""
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    _, token2wav = _build_backend(tmp_path, monkeypatch, {"token2wav_spk_cpu_reference": True})
    assert token2wav.trt_spk_calls == 0
    assert token2wav.device_spk_calls == 1
    assert token2wav.device_spk_use_trt is False


def test_cpu_reference_speaker_is_ignored_off_cuda(tmp_path, monkeypatch):
    """A shared config may set the flag; NPU must keep its device backend."""
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    _, token2wav = _build_backend(tmp_path, monkeypatch, {"token2wav_spk_cpu_reference": True})
    assert token2wav.trt_spk_calls == 0
    assert token2wav.device_spk_calls == 1


@pytest.mark.parametrize("cpu_reference", [False, True])
def test_facade_forwards_speaker_reference_policy(mocker, cpu_reference):
    from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import StepAudio2Token2WavCore

    core = mocker.Mock(spec=StepAudio2Token2WavCore)
    facade = token2wav_module.MiniCPMO45Token2wav.__new__(token2wav_module.MiniCPMO45Token2wav)
    facade._core = core
    facade.enable_trt_spk_embedding(cpu_reference=cpu_reference)
    core.enable_trt_spk_embedding.assert_called_once_with(cpu_reference=cpu_reference)


def _force_platform(monkeypatch, platform: str) -> None:
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: platform == "npu")
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: platform == "cuda")


@pytest.mark.parametrize(
    ("extra", "capture", "pending"),
    [
        pytest.param({}, True, True, id="default"),
        pytest.param({"cfm_graph_precapture_at_init": "true"}, True, True, id="explicit-on"),
        pytest.param({"cfm_graph_precapture_at_init": "false"}, False, True, id="lazy"),
        pytest.param(
            {"cfm_graph_precapture": "false", "cfm_graph_precapture_at_init": "true"},
            False,
            False,
            id="disabled",
        ),
    ],
)
def test_startup_sweep_lifecycle(tmp_path, monkeypatch, device_platform, extra, capture, pending):
    """Weight loading leaves capture to warmup, once, under the forward context."""
    model, _ = _build_backend(tmp_path, monkeypatch, extra)
    contexts, policies = [], []

    def sweep() -> None:
        contexts.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        model._precapture_pending = False

    @contextlib.contextmanager
    def policy(mode):
        policies.append(mode)
        yield

    monkeypatch.setattr(model, "_precapture_default_prompt", sweep)
    monkeypatch.setattr(code2wav_module, "_stage_matmul_policy", policy)
    model.load_weights([])
    assert contexts == policies == []
    assert model._precapture_pending is pending

    model.capture_auxiliary_graphs()
    assert contexts == ([(True, False)] if capture else [])
    assert policies == ([model._tf32_mode] if capture else [])
    assert model._precapture_pending is (pending and not capture)
    model.capture_auxiliary_graphs()
    assert len(contexts) == len(policies) == int(capture)


@pytest.mark.parametrize("release_fails", [False, True], ids=["release-ok", "release-fails"])
def test_startup_sweep_preserves_failure_and_releases_cache(tmp_path, monkeypatch, device_platform, release_fails):
    model, _ = _build_backend(
        tmp_path,
        monkeypatch,
        {"cfm_graph_precapture_at_init": "true", "cfm_precapture_empty_cache": "true"},
    )
    released = []

    def sweep() -> None:
        raise RuntimeError("poisoned wrapper")

    def release() -> None:
        released.append(1)
        if release_fails:
            raise RuntimeError("empty_cache failed")

    monkeypatch.setattr(model, "_precapture_default_prompt", sweep)
    monkeypatch.setattr(model, "_release_precapture_cache", release)
    with pytest.raises(RuntimeError, match="poisoned wrapper"):
        model.capture_auxiliary_graphs()
    assert released == [1]


@pytest.mark.parametrize("missing_device_type", [False, True], ids=["npu-api", "unregistered-device"])
def test_release_precapture_cache_uses_npu_allocator(monkeypatch, missing_device_type):
    """Release uses the NPU API even without CUDA or a registered NPU device type."""
    calls = []
    npu = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        current_device=lambda: 0,
        memory_allocated=lambda device=None: 1024,
        memory_reserved=lambda device=None: 2048,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    _force_platform(monkeypatch, "npu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("torch.accelerator allocator APIs must not run on NPU")

    monkeypatch.setattr(torch.accelerator, "empty_cache", forbidden)
    monkeypatch.setattr(torch.accelerator, "synchronize", forbidden)
    device = torch.device

    def npu_device(*args, **kwargs):
        if missing_device_type:
            raise RuntimeError("NPU device type is not registered")
        # Model the registered device without requiring torch_npu on CPU CI.
        return device("privateuseone", 0)

    monkeypatch.setattr(torch, "device", npu_device)
    MiniCPMO45Code2Wav.__new__(MiniCPMO45Code2Wav)._release_precapture_cache()
    assert calls == ["synchronize", "empty_cache"]


def test_npu_worker_warmup_hook_reaches_the_model(monkeypatch):
    """The NPU worker must forward its warmup hook to the model's capture hook.

    ``_capture_auxiliary_graphs`` resolves ``capture_auxiliary_graphs`` through
    ``getattr(self.model_runner.model, ...)``, which depends on any graph wrapper
    forwarding attribute access. If a future wrapper stops forwarding, the hook
    would silently become a no-op and the sweep would fall back to the first
    request again — the exact failure this PR removes. Pin the forwarding.
    """

    class _FakeModel:
        calls = 0

        def capture_auxiliary_graphs(self) -> None:
            _FakeModel.calls += 1

    worker_module = pytest.importorskip("vllm_omni.platforms.npu.worker.base")
    worker = object.__new__(worker_module.OmniNPUWorkerBase)

    worker.model_runner = SimpleNamespace(model=_FakeModel())
    worker._capture_auxiliary_graphs()
    assert _FakeModel.calls == 1

    # Models without the hook stay a no-op instead of failing startup.
    worker.model_runner = SimpleNamespace(model=SimpleNamespace())
    worker._capture_auxiliary_graphs()


def _no_hub(monkeypatch, tmp_path):
    def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("snapshot_download must not be called here")

    patch_hf_snapshot_download(monkeypatch, _fail, hf_home=tmp_path)


def test_local_directory_is_returned_unchanged(tmp_path, monkeypatch):
    _no_hub(monkeypatch, tmp_path)
    assert _resolve_model_dir(str(tmp_path)) == str(tmp_path)


def test_repo_id_resolves_via_snapshot_download(tmp_path, monkeypatch):
    calls = {}

    def _fake_snapshot_download(model_ref, revision=None, allow_patterns=None):
        calls["model_ref"] = model_ref
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        return str(tmp_path / "snapshot")

    patch_hf_snapshot_download(monkeypatch, _fake_snapshot_download, hf_home=tmp_path)
    resolved = _resolve_model_dir("openbmb/MiniCPM-o-4_5", revision="abc123")
    assert resolved == str(tmp_path / "snapshot")
    assert calls["model_ref"] == "openbmb/MiniCPM-o-4_5"
    assert calls["revision"] == "abc123"
    assert calls["allow_patterns"] == ["assets/*"]


def test_snapshot_download_failure_propagates(monkeypatch, tmp_path):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("offline and not cached")

    patch_hf_snapshot_download(monkeypatch, _raise, hf_home=tmp_path)
    with pytest.raises(FileNotFoundError):
        _resolve_model_dir("openbmb/MiniCPM-o-4_5")


def test_init_with_fake_path_does_not_resolve(monkeypatch, tmp_path):
    """Mirrors the CPU-test construction: fake model path, no ``revision``."""
    _no_hub(monkeypatch, tmp_path)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config=None,
        )
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    assert model.model_path == "/fake/model"
    assert model._default_prompt_wav == "/fake/model/assets/HT_ref_audio.wav"


def test_default_prompt_wav_follows_resolved_model_path(monkeypatch, tmp_path):
    _no_hub(monkeypatch, tmp_path)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="openbmb/MiniCPM-o-4_5",
            stage_connector_config=None,
        )
    )
    model = MiniCPMO45Code2Wav(vllm_config=config)
    model.model_path = "/resolved/snapshot"
    assert model._default_prompt_wav == "/resolved/snapshot/assets/HT_ref_audio.wav"
