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

    class FakeProjection(torch.nn.Linear):
        def __init__(self):
            super().__init__(4, 4, bias=False)
            self.call_shapes: list[tuple[int, ...]] = []

        def forward(self, hidden_states):
            self.call_shapes.append(tuple(hidden_states.shape))
            return super().forward(hidden_states)

    class FakeCodePredictorWrapper(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
            self.model = torch.nn.Module()
            self.model.codec_embedding = torch.nn.ModuleList([torch.nn.Embedding(8, 4), torch.nn.Embedding(8, 4)])
            self.model.linear = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.ModuleList([torch.nn.Linear(4, 8, bias=False), torch.nn.Linear(4, 8, bias=False)])
            self.small_to_mtp_projection = FakeProjection()
            with torch.no_grad():
                self.small_to_mtp_projection.weight.copy_(torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])))
            self._wrapper_config = SimpleNamespace(use_parallel_embedding=False, sampling_mode="per_call")
            self._projected_codec_embed_weight = None

        def load_weights(self, weights):
            del weights
            return {"loaded"}

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
    )
    return (
        fake_qwen3_code_predictor,
        fake_qwen3_tts_code_predictor_vllm,
        fake_qwen3_tts_code2wav,
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


def test_qwen3_tts_patch_registers_target_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    (
        module,
        (
            fake_code_predictor,
            fake_code_predictor_vllm,
            fake_code2wav,
            fake_prompt_builder,
            fake_talker,
        ),
    ) = _load_qwen3_tts_patch(monkeypatch)

    original_common_wrapper = fake_code_predictor.CodePredictorWrapper
    original_vllm_wrapper = fake_code_predictor_vllm.CodePredictorWrapper

    module.apply_talker_patches()
    module.apply_code2wav_patches()

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
    assert fake_code2wav.Qwen3TTSCode2Wav is module._Qwen3TTSCode2Wav310P

    code2wav = module._Qwen3TTSCode2Wav310P(
        vllm_config=SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu")))
    )

    assert code2wav._npu_decoder_runtime_dtype(torch.device("cpu")) is torch.float16


def test_qwen3_tts_code_predictor_forward_uses_projected_embedding_and_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
    predictor = module._Qwen3TTSTalkerCodePredictor310P(
        vllm_config=object(),
        config=object(),
        talker_config=object(),
    )
    codec_weight = predictor.model.codec_embedding[0].weight.detach().clone()
    projection_weight = predictor.small_to_mtp_projection.weight.detach().clone()

    assert predictor.load_weights(iter(())) == {"loaded"}
    torch.testing.assert_close(
        predictor._projected_codec_embed_weight[0],
        torch.nn.functional.linear(codec_weight, projection_weight),
    )
    assert predictor.small_to_mtp_projection.call_shapes == [(8, 4), (8, 4)]
    predictor.small_to_mtp_projection.call_shapes.clear()
    predictor._num_groups = 3
    predictor._model_dtype = torch.float32
    predictor._prefix_graphs_enabled = False
    predictor._bucket_pos_ids = {}
    predictor._device_graphs = {}
    predictor._lm_heads_list = [
        lambda hidden: torch.nn.functional.one_hot(
            torch.full((hidden.shape[0],), 2, dtype=torch.long),
            num_classes=8,
        ).to(torch.float32),
        lambda hidden: torch.nn.functional.one_hot(
            torch.full((hidden.shape[0],), 3, dtype=torch.long),
            num_classes=8,
        ).to(torch.float32),
    ]
    predictor._setup_compile = lambda: None
    predictor._padded_bsz = lambda bsz: bsz

    def ensure_buffers(device, dtype, padded_bsz):
        predictor._proj_buf = torch.zeros(padded_bsz, 4, 4, device=device, dtype=dtype)

    predictor._ensure_buffers = ensure_buffers
    predictor._compiled_model_fwd = lambda embeds, _positions: torch.zeros_like(embeds)

    codes = predictor.forward(
        torch.tensor([1]),
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        do_sample=False,
    )

    assert codes.tolist() == [[1, 2, 3]]
    assert predictor.small_to_mtp_projection.call_shapes == [(1, 2, 4)]
    torch.testing.assert_close(
        predictor._proj_buf[0, 2],
        predictor._projected_codec_embed_weight[0, 2],
    )

    filter_calls = []
    sample_calls = []

    def apply_top_k_top_p(logits, *, p, k, top_k):
        filter_calls.append((p, k, top_k))
        return logits

    def random_sample(_probs, generators):
        sample_calls.append(generators)
        return torch.tensor([[4]])

    monkeypatch.setattr(module, "apply_top_k_top_p", apply_top_k_top_p)
    monkeypatch.setattr(module, "random_sample", random_sample)
    predictor._num_groups = 2
    predictor._wrapper_config.sampling_mode = "stored"
    predictor._top_k = 2
    predictor._top_p = 0.8
    predictor._lm_heads_list = predictor._lm_heads_list[:1]
    generator = torch.Generator().manual_seed(1234)

    sampled_codes = predictor.forward(
        torch.tensor([1]),
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        generator=generator,
    )

    assert sampled_codes.tolist() == [[1, 4]]
    assert len(filter_calls) == 1
    top_p_tensor, top_k_tensor, top_k_hint = filter_calls[0]
    torch.testing.assert_close(top_p_tensor, torch.tensor([0.8]))
    assert top_k_tensor.tolist() == [2]
    assert top_k_hint == 2
    assert sample_calls == [{0: generator}]


def test_qwen3_tts_talker_patch_uses_fp16_runtime_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_qwen3_tts_patch(monkeypatch)
    talker = module._Qwen3TTSTalker310P(vllm_config=object())

    assert talker._embedding_dtype is torch.float16
    assert talker._prompt_builder._embedding_dtype is torch.float16
    assert talker.talker_mtp_graph_safe is False
    assert talker.talker_mtp_accepts_per_row_generators is True
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


def test_qwen3_tts_tokenizer_npu_patch_dispatches_fused_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    rotary_calls = []
    rms_calls = []

    def rotary_mul(hidden_states, cos, sin):
        rotary_calls.append((hidden_states, cos, sin))
        return hidden_states + 1

    def rms_norm(hidden_states, weight, *, epsilon):
        rms_calls.append((hidden_states, weight, epsilon))
        return hidden_states * weight, None

    _install_fake_module(
        monkeypatch,
        "torch_npu",
        npu_rotary_mul=rotary_mul,
        npu_rms_norm=rms_norm,
    )
    _install_fake_module(monkeypatch, "vllm")
    _install_fake_module(monkeypatch, "vllm.logger", init_logger=lambda _name: SimpleNamespace(debug=lambda *_: None))

    class FakeRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
            self.variance_epsilon = 1e-5

        def forward(self, hidden_states):
            return hidden_states

    def original_rope(q, k, cos, sin):
        del cos, sin
        return q, k

    tokenizer = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
        Qwen3TTSTokenizerV2DecoderRMSNorm=FakeRMSNorm,
        apply_rotary_pos_emb=original_rope,
    )
    tokenizer_package = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz",
        modeling_qwen3_tts_tokenizer_v2=tokenizer,
    )
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models.qwen3_tts")
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz",
        tokenizer_package,
    )

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "models" / "qwen3_tts_tokenizer_v2.py"
    module = _load_source_module("vllm_omni_test_qwen3_tts_tokenizer_npu_patch", path)
    module.apply_qwen3_tts_tokenizer_v2_patch()

    q = torch.zeros(1, 2, 3, 2)
    k = torch.ones_like(q)
    cos = torch.zeros(1, 3, 2)
    sin = torch.ones_like(cos)
    q_out, k_out = tokenizer.apply_rotary_pos_emb(q, k, cos, sin)
    norm = FakeRMSNorm()
    norm_out = norm(torch.ones(1, 2))

    assert len(rotary_calls) == 2
    assert rotary_calls[0][1].shape == (1, 1, 3, 2)
    torch.testing.assert_close(q_out, q + 1)
    torch.testing.assert_close(k_out, k + 1)
    assert len(rms_calls) == 1
    assert rms_calls[0][2] == pytest.approx(1e-5)
    torch.testing.assert_close(norm_out, torch.tensor([[2.0, 3.0]]))


def test_qwen3_tts_code2wav_npu_patch_prepares_loaded_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    linear_weights = []
    conv_weights = []

    def maybe_trans_nz(weight):
        linear_weights.append(weight)
        return weight

    def format_cast(weight, fmt):
        conv_weights.append((weight, fmt))
        return weight

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

    logger = SimpleNamespace(info=lambda *_: None, debug=lambda *_: None)
    current_platform = SimpleNamespace(is_npu=lambda: False)
    target = _install_fake_module(
        monkeypatch,
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav",
        Qwen3TTSCode2Wav=FakeCode2Wav,
    )
    _install_fake_module(monkeypatch, "torch_npu", npu_format_cast=format_cast)
    _install_fake_module(monkeypatch, "vllm")
    _install_fake_module(monkeypatch, "vllm.config", VllmConfig=object)
    _install_fake_module(monkeypatch, "vllm.logger", init_logger=lambda _name: logger)
    _install_fake_module(monkeypatch, "vllm_ascend")
    _install_fake_module(monkeypatch, "vllm_ascend.utils", maybe_trans_nz=maybe_trans_nz)
    _install_fake_module(monkeypatch, "vllm_omni")
    _install_fake_module(monkeypatch, "vllm_omni.platforms", current_omni_platform=current_platform)
    _install_fake_module(monkeypatch, "vllm_omni.model_executor")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models")
    _install_fake_module(monkeypatch, "vllm_omni.model_executor.models.qwen3_tts")

    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "models" / "qwen3_tts_code2wav.py"
    module = _load_source_module("vllm_omni_test_qwen3_tts_code2wav_npu_patch", path)
    module.apply_qwen3_tts_code2wav_patch()

    model = target.Qwen3TTSCode2Wav(
        vllm_config=SimpleNamespace(device_config=SimpleNamespace(device=torch.device("cpu"))),
        prefix="stage1",
    )
    assert model.load_weights(iter(())) == {"loaded"}

    assert model.prefix == "stage1"
    assert model.decoder.linear.weight.dtype is torch.float16
    assert [weight.data_ptr() for weight in linear_weights] == [model.decoder.linear.weight.data_ptr()]
    assert {weight.data_ptr() for weight, _ in conv_weights} == {
        model.decoder.conv.weight.data_ptr(),
        model.decoder.deconv.weight.data_ptr(),
    }
    assert all(fmt == module._ACL_FORMAT_FRACTAL_Z for _, fmt in conv_weights)
    assert model.decoder.cache_precompute_calls == 1
