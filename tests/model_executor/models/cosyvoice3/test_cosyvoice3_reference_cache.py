# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

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
    audio: tuple[NDArray[np.float32], int] = (np.arange(8, dtype=np.float32), 16000)
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
    wave: NDArray[np.float32 | np.float64] = np.arange(8, dtype=np.float32)
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


def _condition_reference(obj, waveform):
    from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3 as module

    return module._audio_conditioning(
        obj,
        (waveform, 16000),
        obj.info.ctx.model_config.model,
        obj.info.ctx.get_hf_config(),
        {},
    )


def _cold_reference_burst(processor, monkeypatch, *, failure=None):
    obj, _, _ = processor
    callers = 8
    waveform: NDArray[np.float32] = np.arange(8, dtype=np.float32)
    original_get = obj._speaker_cache.get
    original_speech = obj._extract_speech_token_via_s3
    owner_started = threading.Event()
    release_owner = threading.Event()
    all_missed = threading.Event()
    lock = threading.Lock()
    misses = 0
    encoder_calls = 0

    def get(key):
        nonlocal misses
        result = original_get(key)
        if result is None:
            with lock:
                misses += 1
                if misses == callers:
                    all_missed.set()
        return result

    def speech(audio, device):
        nonlocal encoder_calls
        with lock:
            encoder_calls += 1
        owner_started.set()
        assert release_owner.wait(timeout=5)
        if failure is not None:
            raise RuntimeError(failure)
        return original_speech(audio, device)

    with monkeypatch.context() as patch:
        patch.setattr(obj._speaker_cache, "get", get)
        patch.setattr(obj, "_extract_speech_token_via_s3", speech)
        with ThreadPoolExecutor(max_workers=callers) as executor:
            futures = []
            try:
                futures.append(executor.submit(_condition_reference, SimpleNamespace(**vars(obj)), waveform.copy()))
                assert owner_started.wait(timeout=5)
                futures.extend(
                    executor.submit(_condition_reference, SimpleNamespace(**vars(obj)), waveform.copy())
                    for _ in range(callers - 1)
                )
                assert all_missed.wait(timeout=5)
            finally:
                release_owner.set()
            results = []
            for future in futures:
                error = future.exception(timeout=5)
                results.append(error if error is not None else future.result(timeout=5))
    return results, encoder_calls


def test_reference_cache_merges_cold_calls_and_returns_independent_tensors(processor, monkeypatch):
    obj, _, calls = processor
    results, encoder_calls = _cold_reference_burst(processor, monkeypatch)
    assert encoder_calls == 1
    assert len(calls) == 1
    assert len(results) == 8
    expected = {key: value.clone() for key, value in results[0].items()}
    assert expected["speech_token"].tolist() == [[1, 2]]
    for result in results:
        assert all(torch.equal(result[key], value) for key, value in expected.items())
    for value in results[1].values():
        value.zero_()
    for result in (results[0], results[2], _condition_reference(obj, np.arange(8, dtype=np.float32))):
        assert all(torch.equal(result[key], value) for key, value in expected.items())
    assert len(calls) == 1


def test_reference_cache_fans_out_failure_and_retries(processor, monkeypatch):
    obj, _, calls = processor
    results, encoder_calls = _cold_reference_burst(processor, monkeypatch, failure="reference encoding failed")
    assert encoder_calls == 1
    assert len(results) == 8
    assert all(isinstance(error, RuntimeError) and str(error) == "reference encoding failed" for error in results)
    assert obj._speaker_cache.stats()["entries"] == 0
    result = _condition_reference(obj, np.arange(8, dtype=np.float32))
    assert result["speech_token"].tolist() == [[1, 2]]
    assert len(calls) == 1


def test_reference_cache_keeps_distinct_keys_parallel(processor, monkeypatch):
    obj, _, calls = processor
    original_speech = obj._extract_speech_token_via_s3
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    def speech(audio, device):
        if audio[0][0] == 0:
            first_started.set()
            assert release_first.wait(timeout=5)
        else:
            second_started.set()
        return original_speech(audio, device)

    monkeypatch.setattr(obj, "_extract_speech_token_via_s3", speech)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_condition_reference, obj, np.arange(8, dtype=np.float32))
        try:
            assert first_started.wait(timeout=5)
            second = executor.submit(_condition_reference, obj, np.arange(8, dtype=np.float32) + 20)
            assert second_started.wait(timeout=5)
        finally:
            release_first.set()
        assert first.result(timeout=5)["speech_token"].tolist() == [[1, 2]]
        assert second.result(timeout=5)["speech_token"].tolist() == [[1, 2]]
    assert len(calls) == 2
