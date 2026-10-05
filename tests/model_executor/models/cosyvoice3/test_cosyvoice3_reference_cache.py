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
