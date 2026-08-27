# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for A5 (Ascend 950 PR) patch wiring.

The A5 logic lives in the shared NPU platform files (``npu/__init__.py`` for
device detection, ``npu/models/qwen3_tts_code2wav.py`` for the Qwen3-TTS
patch). The tests load those modules from source with fake Qwen3-TTS
dependencies, so they validate the patch contract without loading real model
or NPU kernels. A5 device detection is exercised with a fake
``vllm_ascend.utils`` module.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _repo_root() -> Path:
    marker = Path("vllm_omni") / "platforms" / "npu" / "models"
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"could not locate repo root containing {marker}")


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_npu_init(monkeypatch: pytest.MonkeyPatch):
    _install_fake_module(monkeypatch, "vllm")
    _install_fake_module(
        monkeypatch,
        "vllm.logger",
        init_logger=lambda _name: SimpleNamespace(debug=lambda *_: None),
    )
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "__init__.py"
    return _load_source_module("vllm_omni_test_npu_init", path)


def _load_qwen3_tts_code2wav_patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    a5: bool,
):
    class FakePromptEmbedsBuilder:
        pass

    class FakeCode2Wav:
        def __init__(self, *, vllm_config, prefix=""):
            pass

        def load_weights(self, weights):
            pass

    fake_prompt_builder = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder",
        Qwen3TTSPromptEmbedsBuilder=FakePromptEmbedsBuilder,
        mel_spectrogram=lambda *_args, **_kwargs: torch.empty(0),
    )
    fake_talker = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker",
        Qwen3TTSPromptEmbedsBuilder=FakePromptEmbedsBuilder,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav",
        Qwen3TTSCode2Wav=FakeCode2Wav,
    )
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    pkg_qwen3_tts = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts",
        prompt_embeds_builder=fake_prompt_builder,
        qwen3_tts_talker=fake_talker,
    )
    # Mark the package fakes as real packages so the patch's lazy submodule
    # imports (e.g. ``qwen3_tts_code2wav``) resolve inside the harness.
    pkg_qwen3_tts.__path__ = []
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms",
        current_omni_platform=SimpleNamespace(is_npu=lambda: False),
    )
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu", is_a5=lambda: a5)
    _install_fake_module(monkeypatch, "vllm")
    _install_fake_module(monkeypatch, "vllm.config", VllmConfig=object)
    _install_fake_module(
        monkeypatch,
        "vllm.logger",
        init_logger=lambda _name: SimpleNamespace(info=lambda *_: None, debug=lambda *_: None),
    )
    _install_fake_module(monkeypatch, "vllm_ascend")
    _install_fake_module(monkeypatch, "vllm_ascend.utils", maybe_trans_nz=lambda weight: weight)
    _install_fake_module(monkeypatch, "torch_npu", npu_format_cast=lambda weight, _fmt: weight)

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "models" / "qwen3_tts_code2wav.py"
    module = _load_source_module("vllm_omni_test_qwen3_tts_code2wav_patch", path)
    return module, fake_prompt_builder, fake_talker


def test_is_a5_detects_a5_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_npu_init(monkeypatch)
    _install_fake_module(monkeypatch, "vllm_ascend")
    _install_fake_module(monkeypatch, "vllm_ascend.utils", is_950=lambda: True)

    assert module.is_a5() is True
    assert module.is_a5(torch.device("npu")) is True


def test_is_a5_falls_back_false_without_is_950(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_npu_init(monkeypatch)
    # Older vllm-ascend releases have no `is_950`; detection must fall back.
    _install_fake_module(monkeypatch, "vllm_ascend")
    _install_fake_module(monkeypatch, "vllm_ascend.utils")

    assert module.is_a5(torch.device("cuda")) is False
    assert module.is_a5() is False


def test_qwen3_tts_patch_swaps_prompt_builder_on_a5(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fake_prompt_builder, fake_talker = _load_qwen3_tts_code2wav_patch(
        monkeypatch,
        a5=True,
    )
    original = fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder

    module.apply_qwen3_tts_patches()

    assert fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder is not original
    assert fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder is fake_talker.Qwen3TTSPromptEmbedsBuilder

    module.apply_qwen3_tts_patches()
    assert fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder is fake_talker.Qwen3TTSPromptEmbedsBuilder


def test_qwen3_tts_patch_keeps_prompt_builder_off_a5(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fake_prompt_builder, fake_talker = _load_qwen3_tts_code2wav_patch(
        monkeypatch,
        a5=False,
    )
    original = fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder

    module.apply_qwen3_tts_patches()

    assert fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder is original
    assert fake_talker.Qwen3TTSPromptEmbedsBuilder is original


def test_a5_prompt_patch_runs_stft_frontend_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module, fake_prompt_builder, _ = _load_qwen3_tts_code2wav_patch(monkeypatch, a5=True)
    captured = {}

    def fake_mel_spectrogram(wav_tensor, **kwargs):
        captured["wav_device"] = wav_tensor.device
        captured["wav_dtype"] = wav_tensor.dtype
        captured["kwargs"] = kwargs
        return torch.ones(1, 128, 3, dtype=torch.float32)

    class FakeSpeakerEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))

        def forward(self, mels):
            captured["speaker_input_dtype"] = mels.dtype
            captured["speaker_input_device"] = mels.device
            return (torch.ones(4, dtype=mels.dtype),)

    module.apply_qwen3_tts_patches()
    monkeypatch.setattr(fake_prompt_builder, "mel_spectrogram", fake_mel_spectrogram)
    builder = object.__new__(fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder)
    builder._device = lambda: torch.device("cpu")
    builder._embedding_dtype = torch.float16
    builder._speaker_encoder = FakeSpeakerEncoder()
    builder._config = SimpleNamespace(speaker_encoder_config=SimpleNamespace(sample_rate=24000))

    speaker = builder.extract_speaker_embedding(np.zeros(16, dtype=np.float32), 24000)

    assert captured["wav_device"] == torch.device("cpu")
    assert captured["wav_dtype"] is torch.float32
    assert captured["kwargs"]["sampling_rate"] == 24000
    assert captured["speaker_input_dtype"] is torch.float16
    assert captured["speaker_input_device"] == torch.device("cpu")
    assert speaker.dtype is torch.float16


def test_qwen3_tts_code2wav_patch_skips_fractal_z_conv_cast_on_a5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _, _ = _load_qwen3_tts_code2wav_patch(monkeypatch, a5=True)
    linear_weights = []
    conv_weights = []

    monkeypatch.setattr(
        module,
        "maybe_trans_nz",
        lambda weight: linear_weights.append(weight) or weight,
    )
    monkeypatch.setattr(
        module.torch_npu,
        "npu_format_cast",
        lambda weight, fmt: conv_weights.append((weight, fmt)) or weight,
    )

    class FakeDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.conv = torch.nn.Conv1d(4, 4, 3)
            self.deconv = torch.nn.ConvTranspose1d(4, 4, 4)
            self.grouped_conv = torch.nn.Conv1d(4, 4, 3, groups=2)
            self.cache_precompute_calls = 0

        def precompute_snake_caches(self):
            self.cache_precompute_calls += 1

    class FakeCode2Wav:
        def __init__(self, *, vllm_config, prefix=""):
            self.vllm_config = vllm_config
            self.prefix = prefix
            self.decoder = FakeDecoder()

        def _npu_decoder_runtime_dtype(self, _device):
            return torch.float16

        def load_weights(self, weights):
            assert list(weights) == []
            return {"loaded"}

    target = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav",
        Qwen3TTSCode2Wav=FakeCode2Wav,
    )
    module.apply_qwen3_tts_code2wav_patch()

    model = target.Qwen3TTSCode2Wav(
        vllm_config=SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu"))),
        prefix="stage1",
    )
    assert model.load_weights(iter(())) == {"loaded"}

    assert model.prefix == "stage1"
    assert model.decoder.linear.weight.dtype is torch.float16
    assert [weight.data_ptr() for weight in linear_weights] == [model.decoder.linear.weight.data_ptr()]
    # On A5 convs keep the contiguous ND layout: no FRACTAL_Z cast is issued.
    assert conv_weights == []
    assert model.decoder.cache_precompute_calls == 1
