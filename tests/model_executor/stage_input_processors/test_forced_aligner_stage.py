from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.forced_aligner import (
    ALIGNER_WORDS_KEY,
    _extract_text,
    _extract_waveform,
    code2wav2aligner,
    decode_pooling_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeCompletion:
    def __init__(self, multimodal_output):
        self.multimodal_output = multimodal_output


class _FakeOutput:
    def __init__(self, multimodal_output, finished=True):
        self.finished = finished
        self.outputs = [_FakeCompletion(multimodal_output)]


def _audio_mm(n=2400, sr=24000):
    return {"audio": torch.zeros(n, dtype=torch.float32), "sr": torch.tensor(sr, dtype=torch.int32)}


def test_extract_waveform_from_audio_key():
    wav, sr = _extract_waveform(_audio_mm(n=100, sr=16000))
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert wav.shape == (100,)
    assert sr == 16000


def test_extract_waveform_falls_back_to_model_outputs_and_default_sr():
    wav, sr = _extract_waveform({"model_outputs": [torch.ones(50)]})
    assert wav.shape == (50,)
    assert sr == 24000  # default when sr missing


def test_extract_text_unwraps_additional_information_list():
    prompt = {"additional_information": {"text": ["Hello world."]}}
    assert _extract_text(prompt) == "Hello world."


def test_code2wav2aligner_builds_text_prompt_and_audio():
    prompt = {"additional_information": {"text": ["U.S.A test"], "language": ["English"]}}
    out = code2wav2aligner([_FakeOutput(_audio_mm())], prompt)

    assert len(out) == 1
    item = out[0]
    # text prompt: tokenization + audio features deferred to the stage preprocessor
    assert "prompt_token_ids" not in item
    # prompt built from official segmentation: "U.S.A" -> "USA"
    assert "USA<timestamp><timestamp>test" in item["prompt"]
    # audio carried as (np.float32, sr) tuple
    audio, sr = item["multi_modal_data"]["audio"]
    assert isinstance(audio, np.ndarray) and sr == 24000
    assert item["additional_information"]["aligner_words"] == [["USA", "test"]]
    assert item["additional_information"]["language"] == ["English"]


def test_code2wav2aligner_skips_unfinished_and_empty():
    # unfinished -> skipped
    assert code2wav2aligner([_FakeOutput(_audio_mm(), finished=False)], {}) == []
    # finished but empty waveform -> skipped
    empty = _FakeOutput({"audio": torch.zeros(0), "sr": torch.tensor(24000)})
    assert code2wav2aligner([empty], {"additional_information": {"text": ["hi"]}}) == []


def test_decode_pooling_output_produces_word_timestamps_ms():
    # 2 words -> 4 <timestamp> markers (id=99). classify_num=5, seg=200ms.
    hf_config = SimpleNamespace(
        timestamp_token_id=99,
        timestamp_segment_time=200.0,
        thinker_config=SimpleNamespace(classify_num=5),
    )
    # prompt: marker positions at indices 2,3,5,6
    prompt_token_ids = [1, 1, 99, 99, 1, 99, 99]
    request = SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        additional_information={ALIGNER_WORDS_KEY: [["hello", "world"]]},
    )
    logits = torch.zeros((7, 5), dtype=torch.float32)
    logits[2, 0] = 1.0  # word0 start -> bin 0
    logits[3, 2] = 1.0  # word0 end   -> bin 2
    logits[5, 2] = 1.0  # word1 start -> bin 2
    logits[6, 4] = 1.0  # word1 end   -> bin 4

    ms = decode_pooling_output(logits, request, hf_config)

    # decoder returns a bare [n_words, 2] int32 tensor (not a dict)
    assert ms.tolist() == [[0, 400], [400, 800]]  # bins x 200ms


def test_decode_pooling_output_handles_hf_config_without_thinker():
    # hf_config without thinker_config falls back to itself for classify_num.
    hf_config = SimpleNamespace(
        timestamp_token_id=99,
        timestamp_segment_time=100.0,
        classify_num=4,
    )
    request = SimpleNamespace(
        prompt_token_ids=[99, 99],
        additional_information={ALIGNER_WORDS_KEY: [["hi"]]},
    )
    logits = torch.zeros((2, 4), dtype=torch.float32)
    logits[0, 1] = 1.0
    logits[1, 3] = 1.0

    ms = decode_pooling_output(logits, request, hf_config)
    assert ms.tolist() == [[100, 300]]
