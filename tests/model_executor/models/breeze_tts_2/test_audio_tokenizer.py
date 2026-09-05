import numpy as np
import torch

from vllm_omni.model_executor.models.breeze_tts_2.audio_tokenizer import (
    BreezeReferenceAudioTokenizer,
)


class _Tokenizer:
    def encode(self, audio, sr=None):
        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        return {"audio_codes": [torch.arange(32).reshape(16, 2)]}


def test_reference_audio_tokenizer_normalizes_codebook_major_output():
    adapter = BreezeReferenceAudioTokenizer(_Tokenizer())

    codes = adapter.encode(np.zeros(32, dtype=np.float32), 16000)

    assert tuple(codes.shape) == (2, 16)
    assert codes.dtype == torch.int16
    assert codes.device.type == "cpu"


def test_reference_audio_tokenizer_rejects_waveform_without_sample_rate():
    adapter = BreezeReferenceAudioTokenizer(_Tokenizer())

    try:
        adapter.encode(np.zeros(32, dtype=np.float32))
    except ValueError as exc:
        assert "sample_rate is required" in str(exc)
    else:
        raise AssertionError("missing sample rate should be rejected")
