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

    class FakeCodePredictorWrapper(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
            self.model = torch.nn.Module()
            self.model.codec_embedding = torch.nn.ModuleList([torch.nn.Embedding(8, 4), torch.nn.Embedding(8, 4)])
            self.model.linear = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.ModuleList([torch.nn.Linear(4, 8, bias=False), torch.nn.Linear(4, 8, bias=False)])
            self.small_to_mtp_projection = torch.nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.small_to_mtp_projection.weight.copy_(torch.eye(4))
            self._wrapper_config = SimpleNamespace(use_parallel_embedding=False)
            self._static_310p_ready = False
            self._projected_codec_embed_weight = None

        def load_weights(self, weights):
            del weights
            return {"loaded"}

        def forward(self, *args, **kwargs):
            self.forward_args = args
            self.forward_kwargs = kwargs
            return "fallback"

    class FakeCode2WavBase(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()

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
        CodePredictorWrapper=FakeCodePredictorWrapper,
        _rotate_half=lambda x: x,
    )
    fake_qwen3_tts_code_predictor_vllm = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm",
        Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM=FakeCodePredictorWrapper,
        Qwen3TTSTalkerCodePredictorModelVLLM=FakeCodePredictorBaseModel,
        CodePredictorWrapper=FakeCodePredictorWrapper,
    )
    fake_qwen3_tts_code2wav = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav",
        Qwen3TTSCode2Wav=FakeCode2WavBase,
    )
    fake_tokenizer_12hz = _install_fake_module(monkeypatch, "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz")
    fake_tokenizer_v2 = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
        Qwen3TTSTokenizerV2DecoderRMSNorm=torch.nn.Module,
        apply_rotary_pos_emb=lambda q, k, cos, sin, position_ids=None, unsqueeze_dim=1: (q, k),
    )
    fake_tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 = fake_tokenizer_v2
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

    _install_fake_module(monkeypatch, "vllm")
    _install_fake_module(monkeypatch, "vllm.multimodal")
    _install_fake_module(monkeypatch, "vllm.multimodal.audio", AudioResampler=FakeAudioResampler)
    _install_fake_module(monkeypatch, "torch_npu", npu_format_cast=lambda weight, _fmt: weight)
    _install_fake_module(monkeypatch, "vllm_ascend")
    _install_fake_module(monkeypatch, "vllm_ascend._310p")
    _install_fake_module(monkeypatch, "vllm_ascend._310p.attention")
    _install_fake_module(
        monkeypatch,
        "vllm_ascend._310p.attention.attention_mask",
        AttentionMaskBuilder310=SimpleNamespace(
            gen_causal_additive_mask=lambda max_seq, device: torch.zeros(
                max_seq,
                max_seq,
                device=device,
            )
        ),
    )
    _install_fake_module(monkeypatch, "vllm_ascend.sample")
    _install_fake_module(
        monkeypatch,
        "vllm_ascend.sample.sampler",
        apply_top_k_top_p=lambda logits, **_kwargs: logits,
        random_sample=lambda probs, _generators: probs.argmax(dim=-1, keepdim=True),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_ascend.utils",
        ACL_FORMAT_FRACTAL_NZ=29,
        aligned_16=lambda tensor: tensor,
        maybe_trans_nz=lambda weight: weight,
        nd_to_nz_2d=lambda tensor: tensor,
    )
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
        qwen3_tts_code2wav=fake_qwen3_tts_code2wav,
        qwen3_tts_code_predictor_vllm=fake_qwen3_tts_code_predictor_vllm,
        qwen3_tts_talker=fake_talker,
        tokenizer_12hz=fake_tokenizer_12hz,
    )
    return (
        fake_qwen3_code_predictor,
        fake_qwen3_tts_code_predictor_vllm,
        fake_qwen3_tts_code2wav,
        fake_tokenizer_v2,
        fake_prompt_builder,
        fake_talker,
    )


def _load_qwen3_tts_patch(monkeypatch: pytest.MonkeyPatch):
    fakes = _install_qwen3_tts_patch_fakes(monkeypatch)
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "qwen3_tts.py"
    module = _load_source_module("vllm_omni_test_310p_qwen3_tts_patch", path)
    return module, fakes


def test_registry_applies_worker_once_and_model_patch_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    registry_path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "__init__.py"
    registry = _load_source_module("vllm_omni_test_310p_patch_registry", registry_path)
    calls = {"worker": 0, "talker": 0, "code2wav": 0}

    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.worker",
        apply_patch=lambda: calls.__setitem__("worker", calls["worker"] + 1),
    )
    _install_fake_module(
        monkeypatch,
        "vllm_omni.platforms.npu._310p.patch.qwen3_tts",
        apply_talker_patches=lambda: calls.__setitem__("talker", calls["talker"] + 1),
        apply_code2wav_patches=lambda: calls.__setitem__("code2wav", calls["code2wav"] + 1),
    )

    registry.apply_patches()
    registry.apply_patches()
    registry.apply_model_patches(SimpleNamespace(model_arch="OtherModel"))
    registry.apply_model_patches(SimpleNamespace(model_arch="Qwen3TTSTalkerForConditionalGeneration"))
    registry.apply_model_patches(SimpleNamespace(model_arch="Qwen3TTSCode2Wav"))

    assert calls == {"worker": 1, "talker": 1, "code2wav": 1}


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
    (
        module,
        (
            fake_code_predictor,
            fake_code_predictor_vllm,
            _fake_code2wav,
            _fake_tokenizer_v2,
            fake_prompt_builder,
            fake_talker,
        ),
    ) = _load_qwen3_tts_patch(monkeypatch)

    original_common_wrapper = fake_code_predictor.CodePredictorWrapper
    original_vllm_wrapper = fake_code_predictor_vllm.CodePredictorWrapper

    module.apply_talker_patches()

    assert fake_talker.Qwen3TTSTalkerForConditionalGeneration is module._Qwen3TTSTalker310P
    assert fake_talker.Qwen3TTSPromptEmbedsBuilder is module._Qwen3TTSPromptEmbedsBuilder310P
    assert fake_prompt_builder.Qwen3TTSPromptEmbedsBuilder is module._Qwen3TTSPromptEmbedsBuilder310P
    assert fake_code_predictor.CodePredictorAttention is module._Qwen3CodePredictorAttention310P
    assert fake_code_predictor.CodePredictorDecoderLayer is module._Qwen3CodePredictorDecoderLayer310P
    assert fake_code_predictor.CodePredictorBaseModel is module._Qwen3CodePredictorBaseModel310P
    assert (
        fake_code_predictor_vllm.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM
        is module._Qwen3TTSTalkerCodePredictor310P
    )
    assert fake_code_predictor.CodePredictorWrapper is original_common_wrapper
    assert fake_code_predictor_vllm.CodePredictorWrapper is original_vllm_wrapper


def test_qwen3_tts_code2wav_patch_only_selects_310p_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    module, (_, _, fake_code2wav, fake_tokenizer_v2, _, _) = _load_qwen3_tts_patch(monkeypatch)
    original_rms_norm = fake_tokenizer_v2.Qwen3TTSTokenizerV2DecoderRMSNorm
    original_apply_rotary = fake_tokenizer_v2.apply_rotary_pos_emb

    module.apply_code2wav_patches()
    module.apply_code2wav_patches()

    assert fake_code2wav.Qwen3TTSCode2Wav is module._Qwen3TTSCode2Wav310P
    assert fake_tokenizer_v2.Qwen3TTSTokenizerV2DecoderRMSNorm is original_rms_norm
    assert fake_tokenizer_v2.apply_rotary_pos_emb is original_apply_rotary

    code2wav = module._Qwen3TTSCode2Wav310P(
        vllm_config=SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    )

    assert code2wav._decoder_runtime_dtype(torch.device("cpu")) is torch.float16


def test_qwen3_tts_code_predictor_patch_prepares_static_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
    predictor = module._Qwen3TTSTalkerCodePredictor310P(
        vllm_config=object(),
        config=object(),
        talker_config=object(),
    )
    codec_weight_before = predictor.model.codec_embedding[0].weight.detach().clone()

    loaded = predictor.load_weights(iter(()))
    predictor._prepare_static_weights_310p()

    assert loaded == {"loaded"}
    assert predictor._static_310p_ready is True
    assert predictor._lm_heads_list == list(predictor.lm_head)
    assert predictor._codec_embeds_list == list(predictor.model.codec_embedding)
    assert predictor._projected_codec_embed_weight.shape == (2, 8, 4)
    torch.testing.assert_close(predictor._projected_codec_embed_weight[0], codec_weight_before)


def test_qwen3_tts_code_predictor_cpu_fallback_preserves_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
    predictor = module._Qwen3TTSTalkerCodePredictor310P(
        vllm_config=object(),
        config=object(),
        talker_config=object(),
    )
    generators = [None]

    result = predictor.forward(
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        generator=None,
        generators=generators,
    )

    assert result == "fallback"
    assert predictor.forward_kwargs["generator"] is None
    assert predictor.forward_kwargs["generators"] is generators


def test_qwen3_tts_talker_patch_uses_fp16_runtime_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
    talker = module._Qwen3TTSTalker310P(vllm_config=object())

    assert talker._embedding_dtype is torch.float16
    assert talker._prompt_builder._embedding_dtype is torch.float16
    assert talker.load_weights([]) == {"loaded"}
    assert talker.encoder.to_calls[-1] == {"device": torch.device("cpu"), "dtype": torch.float32}

    codes = talker._encode_ref_audio_batch([np.zeros(8, dtype=np.float32)], 24000, device=torch.device("cpu"))

    assert talker.encoder.last_input_dtype is torch.float32
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


def test_qwen3_tts_common_npu_optimizations_live_outside_310p_patch() -> None:
    root = _repo_root()
    patch_source = (root / "vllm_omni" / "platforms" / "npu" / "_310p" / "patch" / "qwen3_tts.py").read_text()
    code_predictor_source = (
        root / "vllm_omni" / "model_executor" / "models" / "common" / "qwen3_code_predictor.py"
    ).read_text()
    code2wav_source = (
        root / "vllm_omni" / "model_executor" / "models" / "qwen3_tts" / "qwen3_tts_code2wav.py"
    ).read_text()
    tokenizer_source = (
        root
        / "vllm_omni"
        / "model_executor"
        / "models"
        / "qwen3_tts"
        / "tokenizer_12hz"
        / "modeling_qwen3_tts_tokenizer_v2.py"
    ).read_text()

    assert "_MimiEuclideanCodebook310P" not in patch_source
    assert "_Qwen3TTSTokenizerV2DecoderRMSNorm310P" not in patch_source
    assert "_code2wav_apply_rotary_pos_emb_310p" not in patch_source
    assert "maybe_trans_nz" not in patch_source

    assert "torch_npu.npu_rms_norm" in code_predictor_source
    assert "torch_npu.npu_add_rms_norm" in code_predictor_source
    assert "torch_npu.npu_rotary_mul" in code_predictor_source
    assert "def _prepare_npu_weights" in code_predictor_source
    assert "maybe_trans_nz" in code_predictor_source

    assert "def _prepare_npu_decoder_weights" in code2wav_source
    assert "nn.Conv1d" in code2wav_source
    assert "nn.ConvTranspose1d" in code2wav_source
    assert "_ACL_FORMAT_FRACTAL_Z" in code2wav_source
    assert "maybe_trans_nz" in code2wav_source

    assert "torch_npu.npu_rms_norm" in tokenizer_source
    assert "torch_npu.npu_rotary_mul" in tokenizer_source
