# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def processor(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as module
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    calls = []
    config = SimpleNamespace(allowed_special=[], sample_rate=24000)
    ctx = SimpleNamespace(get_hf_config=lambda: config, model_config=SimpleNamespace(model="model-a"))
    obj = SimpleNamespace(
        info=SimpleNamespace(ctx=ctx),
        tokenizer=None,
        feat_extractor=None,
        campplus_trt=None,
        campplus_session=None,
        _speaker_cache=SpeakerEmbeddingCache(max_bytes=1024),
        _ensure_cached_runtime_components=lambda *args: None,
    )

    def speech(audio, device):
        calls.append(audio)
        return torch.tensor([[1, 2]], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)

    obj._extract_speech_token_via_s3 = speech
    monkeypatch.setattr(
        module, "extract_text_token", lambda text, *args: (torch.tensor([[ord(c) for c in text]]), len(text))
    )
    monkeypatch.setattr(module, "extract_speech_feat", lambda *args: (torch.ones(1, 4, 2), torch.tensor([4])))
    monkeypatch.setattr(module, "extract_spk_embedding", lambda *args: torch.ones(1, 2))

    def run(audio, text="target", reference="reference"):
        return module.CosyVoice3MultiModalProcessor._call_hf_processor(
            obj, text, {"audio": audio}, {"prompt_text": reference}, {}
        )

    return obj, run, calls


def test_reference_cache_reuses_audio_but_not_text_and_owns_tensors(processor):
    obj, run, calls = processor
    audio = (np.arange(8, dtype=np.float32), 16000)
    first = run(audio)
    first["speech_feat"].zero_()
    first["speech_token"].zero_()
    second = run((audio[0].copy(), 16000), text="different", reference="changed")
    assert len(calls) == 1
    assert torch.all(second["speech_feat"] == 1)
    assert second["speech_token"].tolist() == [[1, 2]]
    assert not torch.equal(first["input_ids"], second["input_ids"])
    second["embedding"].zero_()
    assert torch.all(run(audio)["embedding"] == 1)
    assert obj._speaker_cache.stats()["hits"] == 2


@pytest.mark.parametrize("change", ["samples", "rate", "dtype", "model"])
def test_reference_cache_distinguishes_inputs(processor, change):
    obj, run, calls = processor
    wave = np.arange(8, dtype=np.float32)
    run((wave, 16000))
    rate = 16000
    if change == "samples":
        wave[0] = 20
    elif change == "rate":
        rate = 24000
    elif change == "dtype":
        wave = wave.astype(np.float64)
    else:
        obj.info.ctx.model_config.model = "model-b"
    run((wave, rate))
    assert len(calls) == 2


def test_precomputed_conditioning_bypasses_encoders_and_still_tokenizes_text(processor, monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as module
    from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import (
        REFERENCE_CONDITIONING_KEY,
        CosyReferenceArtifacts,
    )

    obj, run, calls = processor
    audio = (np.arange(8, dtype=np.float32), 16000)
    local = run(audio)
    artifacts = CosyReferenceArtifacts(
        *(
            local[name] if name != "speech_token_len" else local[name][0]
            for name in ("speech_feat", "speech_token", "speech_token_len", "embedding")
        )
    )
    config = obj.info.ctx.get_hf_config()
    config.qwen_pretrain_path = "tokenizer"
    config.skip_special_tokens = True
    config.version = "cosyvoice3"

    def fail(*args):
        pytest.fail("a shared reference client tried to initialize or run reference encoders")

    obj._ensure_cached_runtime_components = fail
    monkeypatch.setattr(module, "_audio_conditioning", fail)
    monkeypatch.setattr(module, "_text_only_runtime_tokenizer", lambda *args: "text-only-tokenizer")

    def shared(text, reference):
        return module.CosyVoice3MultiModalProcessor._call_hf_processor(
            obj,
            text,
            {"audio": audio},
            {"prompt_text": reference, REFERENCE_CONDITIONING_KEY: artifacts.processor_kwargs()},
            {},
        )

    first = shared("target", "reference")
    assert torch.equal(first["input_ids"], local["input_ids"])
    for name in ("speech_feat", "speech_token", "embedding"):
        torch.testing.assert_close(first[name], local[name], rtol=0, atol=0)
    changed = shared("new target", "new reference")
    assert not torch.equal(first["input_ids"], changed["input_ids"])
    first["embedding"].zero_()
    assert torch.all(artifacts.embedding == 1)
    assert len(calls) == 1


def test_apply_retains_precomputed_artifacts_without_initializing_encoders(processor, monkeypatch, mocker):
    from vllm.multimodal.parse import MultiModalDataItems
    from vllm.multimodal.processing import ProcessorInputs

    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as module
    from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import REFERENCE_CONDITIONING_KEY

    obj, _, _ = processor
    config = obj.info.ctx.get_hf_config()
    config.qwen_pretrain_path, config.skip_special_tokens, config.version = "tokenizer", True, "cosyvoice3"
    tokenizer = mocker.Mock()
    tokenizer.decode.return_value = "target"
    obj.info.get_tokenizer = lambda: tokenizer
    obj._OMNI_PROMPT_TEXT_KEY = module.OmniMultiModalProcessor._OMNI_PROMPT_TEXT_KEY
    obj._ensure_cached_runtime_components = mocker.Mock(side_effect=AssertionError("encoders loaded"))
    monkeypatch.setattr(module, "_text_only_runtime_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(module.OmniMultiModalProcessor, "apply", lambda self, inputs, timing_ctx: inputs)
    artifacts = {"sentinel": "must pass through unchanged"}
    inputs = ProcessorInputs(
        prompt=[1, 2],
        mm_data_items=MultiModalDataItems(),
        hf_processor_mm_kwargs={REFERENCE_CONDITIONING_KEY: artifacts},
    )
    instance = object.__new__(module.CosyVoice3MultiModalProcessor)
    vars(instance).update(vars(obj))
    result = instance.apply(inputs, None)
    assert result.hf_processor_mm_kwargs[REFERENCE_CONDITIONING_KEY] is artifacts
    assert result.prompt == [ord(c) for c in "target"]
    obj._ensure_cached_runtime_components.assert_not_called()


def test_shared_worker_preserves_resolved_config_overrides(monkeypatch):
    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as module
    from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import CosyVoice3ReferenceEncoder
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    resolved = CosyVoice3Config()
    resolved.qwen_pretrain_path = "custom-text-tokenizer"
    resolved.target_sr = 16000
    observed = []
    monkeypatch.setattr(
        module.CosyVoice3MultiModalProcessor,
        "_ensure_cached_runtime_components",
        lambda self, path, config: observed.append((path, config.qwen_pretrain_path, config.target_sr)),
    )
    monkeypatch.setattr(module.CosyVoice3MultiModalProcessor, "_ensure_s3_model", lambda self: None)
    encoder = CosyVoice3ReferenceEncoder("local-model", resolved.to_dict(), 1024)
    try:
        assert observed == [("local-model", "custom-text-tokenizer", 16000)]
        assert encoder.resampler.target_sr == 16000
    finally:
        encoder._executor.shutdown()


@pytest.mark.parametrize("audio_count", [0, 1, 8])
def test_encoder_budget_does_not_initialize_dummy_reference_encoders(audio_count):
    from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as module

    info = object.__new__(module.CosyVoice3MultiModalProcessingInfo)
    info.ctx = SimpleNamespace(
        get_hf_config=lambda *args: SimpleNamespace(token_frame_rate=25),
        model_config=SimpleNamespace(max_model_len=32768),
    )

    def fail(*args, **kwargs):
        pytest.fail("budget discovery must not encode dummy audio")

    result = get_mm_max_toks_per_item(
        SimpleNamespace(info=info, get_dummy_mm_inputs=fail), {"audio": audio_count}, cache=None
    )
    assert result == ({"audio": 752} if audio_count else {})


@pytest.mark.parametrize("seconds", [2, 30])
def test_renderer_dummy_inputs_supply_synthetic_conditioning(monkeypatch, seconds):
    from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
    from vllm.multimodal.processing import ProcessorInputs

    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as module
    from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import (
        REFERENCE_CONDITIONING_KEY,
        CosyReferenceArtifacts,
    )
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    config = CosyVoice3Config()
    inputs = ProcessorInputs(
        prompt=[1, 2],
        mm_data_items=MultiModalDataItems({"audio": AudioProcessorItems([np.zeros(seconds * config.target_sr)])}),
    )
    monkeypatch.setattr(module.OmniDummyInputsBuilder, "get_dummy_processor_inputs", lambda *args: inputs)
    builder = module.CosyVoice3DummyInputsBuilder(SimpleNamespace(get_hf_config=lambda: config))
    result = builder.get_dummy_processor_inputs(32768, {"audio": 1})
    artifact = CosyReferenceArtifacts.from_mapping(result.hf_processor_mm_kwargs[REFERENCE_CONDITIONING_KEY])
    assert artifact.speech_token.shape == (1, seconds * 25)
    assert artifact.speech_feat.shape == (1, seconds * 50, 80)
    assert artifact.embedding.shape == (1, 192)
    assert int(artifact.speech_token_len[0]) == seconds * 25
    assert all(tensor.device.type == "cpu" for tensor in artifact.tensors())
