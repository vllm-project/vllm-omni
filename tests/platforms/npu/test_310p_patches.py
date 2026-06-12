# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for 310P patch wiring.

The tests load patch modules from source with fake Qwen3-TTS dependencies, so
they validate the patch contract without loading real model or NPU kernels.
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
    marker = Path("vllm_omni") / "platforms" / "npu" / "_310p" / "patch"
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


def _install_qwen3_tts_patch_fakes(monkeypatch: pytest.MonkeyPatch):
    class FakeAudioResampler:
        def __init__(self, *, target_sr: int):
            self.target_sr = target_sr

        def resample(self, wav, *, orig_sr: int):
            del orig_sr
            return wav

    class FakeCodePredictorAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
            self.register_buffer("_fusion_causal_mask", torch.ones(1), persistent=False)

    class FakeCodePredictorDecoderLayer(torch.nn.Module):
        pass

    class FakeCodePredictorBaseModel(torch.nn.Module):
        pass

    class FakeMimiEuclideanCodebook(torch.nn.Module):
        @property
        def embed(self):
            return self._embed

    class FakeEncoder:
        def __init__(self):
            self.to_calls: list[dict] = []
            self.last_input_dtype = None

        def to(self, **kwargs):
            self.to_calls.append(kwargs)
            return self

        def encode(self, *, input_values, return_dict: bool):
            assert return_dict
            self.last_input_dtype = input_values.dtype
            return SimpleNamespace(audio_codes=torch.arange(12, dtype=torch.long).reshape(1, 3, 4))

    class FakeFeatureBatch(dict):
        def to(self, target):
            if isinstance(target, torch.dtype):
                for key, value in list(self.items()):
                    if torch.is_floating_point(value):
                        self[key] = value.to(dtype=target)
                self.dtype = target
            else:
                self.device = torch.device(target)
                for key, value in list(self.items()):
                    self[key] = value.to(device=self.device)
            return self

    class FakeFeatureExtractor:
        sampling_rate = 24000

        def __call__(self, *, raw_audio, sampling_rate: int, return_tensors: str):
            assert len(raw_audio) == 1
            assert sampling_rate == self.sampling_rate
            assert return_tensors == "pt"
            return FakeFeatureBatch(
                input_values=torch.ones(1, 1, 8, dtype=torch.float32),
                padding_mask=torch.ones(1, 1, 8, dtype=torch.long),
            )

    class FakeTalkerBase(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix: str = ""):
            del vllm_config, prefix
            super().__init__()
            self._embedding_dtype = torch.bfloat16
            self._prompt_builder = SimpleNamespace(_embedding_dtype=torch.bfloat16)
            self.encoder = FakeEncoder()
            self._encoder_feature_extractor = FakeFeatureExtractor()
            self._encoder_valid_num_quantizers = 2
            self._encoder_downsample_rate = 2

        def load_weights(self, weights):
            del weights
            self.encoder.to(dtype=torch.bfloat16)
            return {"loaded"}

    class FakePromptEmbedsBuilder:
        pass

    fake_qwen3_code_predictor = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.common.qwen3_code_predictor",
        CodePredictorAttention=FakeCodePredictorAttention,
        CodePredictorDecoderLayer=FakeCodePredictorDecoderLayer,
        CodePredictorBaseModel=FakeCodePredictorBaseModel,
        _rotate_half=lambda x: x,
    )
    fake_prompt_builder = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder",
        Qwen3TTSPromptEmbedsBuilder=FakePromptEmbedsBuilder,
        mel_spectrogram=lambda *_args, **_kwargs: torch.empty(0),
    )
    fake_talker = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker",
        Qwen3TTSTalkerForConditionalGeneration=FakeTalkerBase,
        Qwen3TTSPromptEmbedsBuilder=FakePromptEmbedsBuilder,
    )
    fake_modeling_mimi = _install_fake_module(
        monkeypatch,
        "transformers.models.mimi.modeling_mimi",
        MimiEuclideanCodebook=FakeMimiEuclideanCodebook,
    )
    fake_mimi = _install_fake_module(
        monkeypatch,
        "transformers.models.mimi",
        modeling_mimi=fake_modeling_mimi,
    )

    _install_fake_module(monkeypatch, "vllm")
    _install_fake_module(monkeypatch, "vllm.multimodal")
    _install_fake_module(monkeypatch, "vllm.multimodal.audio", AudioResampler=FakeAudioResampler)
    _install_fake_module(monkeypatch, "transformers")
    _install_fake_module(monkeypatch, "transformers.models", mimi=fake_mimi)
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.common",
        qwen3_code_predictor=fake_qwen3_code_predictor,
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts",
        prompt_embeds_builder=fake_prompt_builder,
        qwen3_tts_talker=fake_talker,
    )
    return fake_qwen3_code_predictor, fake_prompt_builder, fake_talker


def _load_qwen3_tts_patch(monkeypatch: pytest.MonkeyPatch):
    fakes = _install_qwen3_tts_patch_fakes(monkeypatch)
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "qwen3_tts.py"
    module = _load_source_module("vllm_omni_test_310p_qwen3_tts_patch", path)
    return module, fakes


def test_registry_applies_worker_once_and_model_patch_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    registry = _load_source_module("vllm_omni_test_310p_patch_registry", registry_path)
    calls = {"worker": 0, "talker": 0}

    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.worker",
        apply_patch=lambda: calls.__setitem__("worker", calls["worker"] + 1),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=lambda: calls.__setitem__("talker", calls["talker"] + 1),
    )

    registry.apply_patches()
    registry.apply_patches()
    registry.apply_model_patches(SimpleNamespace(model_arch="OtherModel"))
    registry.apply_model_patches(SimpleNamespace(model_arch="Qwen3TTSTalkerForConditionalGeneration"))

    assert calls == {"worker": 1, "talker": 1}


def test_worker_patch_replaces_base_and_runs_disable_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeOmniNPUWorkerBase:
        def _init_device(self):
            calls.append("parent")
            return "npu:0"

    fake_worker_base = _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu.worker.base",
        OmniNPUWorkerBase=FakeOmniNPUWorkerBase,
    )
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms")
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu")
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p",
        disable_jit_compile=lambda: calls.append("disable_jit"),
    )
    _install_fake_module(monkeypatch, "vllm_omni.platforms.npu.worker", base=fake_worker_base)

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "worker.py"
    module = _load_source_module("vllm_omni_test_310p_worker_patch", path)
    module.apply_patch()

    assert fake_worker_base.OmniNPUWorkerBase is module._OmniNPUWorkerBase310P
    assert fake_worker_base.OmniNPUWorkerBase()._init_device() == "npu:0"
    assert calls == ["parent", "disable_jit"]


def test_qwen3_tts_patch_replaces_target_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    module, (fake_code_predictor, fake_prompt_builder, fake_talker) = _load_qwen3_tts_patch(monkeypatch)

    module.apply_talker_patches()

    assert fake_talker.Qwen3TTSTalkerForConditionalGeneration is module._Qwen3TTSTalker310P
    assert fake_talker.Qwen3TTSPromptEmbedsBuilder is module._Qwen3TTSPromptEmbedsBuilder310P
    assert fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder is module._Qwen3TTSPromptEmbedsBuilder310P
    assert fake_code_predictor.CodePredictorAttention is module._Qwen3CodePredictorAttention310P
    assert fake_code_predictor.CodePredictorDecoderLayer is module._Qwen3CodePredictorDecoderLayer310P
    assert fake_code_predictor.CodePredictorBaseModel is module._Qwen3CodePredictorBaseModel310P


def test_qwen3_tts_talker_patch_uses_fp16_runtime_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
    talker = module._Qwen3TTSTalker310P(vllm_config=object())

    assert talker._embedding_dtype is torch.float16
    assert talker._prompt_builder._embedding_dtype is torch.float16
    assert talker.load_weights([]) == {"loaded"}
    assert talker.encoder.to_calls[-1] == {"dtype": torch.float16}

    codes = talker._encode_ref_audio_batch([np.zeros(8, dtype=np.float32)], 24000, device=torch.device("cpu"))

    assert talker.encoder.last_input_dtype is torch.float16
    assert len(codes) == 1
    assert codes[0].dtype is torch.long
    assert codes[0].shape == (4, 2)


def test_qwen3_tts_prompt_patch_runs_stft_frontend_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
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
            return (torch.ones(4, dtype=mels.dtype),)

    monkeypatch.setattr(module.prompt_embeds_builder, "mel_spectrogram", fake_mel_spectrogram)
    builder = object.__new__(module._Qwen3TTSPromptEmbedsBuilder310P)
    builder._device = lambda: torch.device("cpu")
    builder._embedding_dtype = torch.float16
    builder._speaker_encoder = FakeSpeakerEncoder()
    builder._config = SimpleNamespace(speaker_encoder_config=SimpleNamespace(sample_rate=24000))

    speaker = builder.extract_speaker_embedding(np.zeros(16, dtype=np.float32), 24000)

    assert captured["wav_device"] == torch.device("cpu")
    assert captured["wav_dtype"] is torch.float32
    assert captured["kwargs"]["sampling_rate"] == 24000
    assert captured["speaker_input_dtype"] is torch.float16
    assert speaker.dtype is torch.float16


def test_qwen3_tts_mimi_codebook_quantize_uses_cpu_fp32_cdist(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
    real_cdist = torch.cdist
    captured = {}

    def fake_cdist(x1, x2, p=2):
        captured["x1_device"] = x1.device
        captured["x2_device"] = x2.device
        captured["x1_dtype"] = x1.dtype
        captured["x2_dtype"] = x2.dtype
        return real_cdist(x1, x2, p=p)

    monkeypatch.setattr(torch, "cdist", fake_cdist)
    codebook = object.__new__(module._MimiEuclideanCodebook310P)
    codebook._embed = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float16)

    indices = codebook.quantize(torch.tensor([[0.1, 0.0], [1.8, 0.0]], dtype=torch.float16))

    assert captured == {
        "x1_device": torch.device("cpu"),
        "x2_device": torch.device("cpu"),
        "x1_dtype": torch.float32,
        "x2_dtype": torch.float32,
    }
    assert indices.tolist() == [0, 1]
