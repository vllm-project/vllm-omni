# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiniCPM-o 4.5 Talker / Token2Wav split."""

from __future__ import annotations

import io
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bare_model(stage: str = "talker", model_path: str = "/tmp/minicpmo") -> MiniCPMO45OmniTTSForConditionalGeneration:
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = stage
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model=model_path))
    model.audio_tokenizer = None
    model._hidden_size = 4
    return model


def _bare_wrapper(stage: str = "talker") -> MiniCPMO45OmniForConditionalGeneration:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = stage
    model.config = SimpleNamespace(hidden_size=4)
    return model


class _FakeMiniCPMTTS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_eos_token_id=9,
            num_audio_tokens=20,
            hidden_size=4,
            normalize_projected_hidden=False,
        )
        self.audio_bos_token_id = 8
        self.emb_text = nn.Embedding(32, 4)
        self.projector_semantic = nn.Linear(4, 4, bias=False)
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return SimpleNamespace(new_ids=torch.tensor([[[5], [6], [7]]], dtype=torch.long))


def _wav_bytes(samples: np.ndarray | None = None, sample_rate: int = 24_000) -> bytes:
    if samples is None:
        samples = np.zeros(240, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def test_prepare_tts_inputs_and_generate_audio_tokens() -> None:
    model = _bare_model("talker")
    fake_tts = _FakeMiniCPMTTS().to(dtype=torch.bfloat16)
    model.tts_obj = fake_tts
    model._lazy_init_talker = lambda: None

    tts_token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    tts_hidden_states = torch.ones(3, 4)

    inputs_embeds, eos_token, max_new_token, num_text = model.prepare_tts_inputs(tts_token_ids, tts_hidden_states)
    assert inputs_embeds.shape == (1, 5, 4)
    assert eos_token.tolist() == [19]
    assert max_new_token == 2048
    assert num_text == 3

    audio_tokens = model.generate_audio_tokens(tts_token_ids, tts_hidden_states)
    assert audio_tokens.tolist() == [5, 6, 7]
    assert fake_tts.last_generate_kwargs["max_new_token"] == 2048


def test_decode_audio_tokens_uses_token2wav(monkeypatch, tmp_path) -> None:
    model = _bare_model("token2wav", str(tmp_path))
    model._lazy_init_token2wav = lambda: None
    monkeypatch.setitem(sys.modules, "torchaudio", types.SimpleNamespace(save=lambda *args, **kwargs: None))

    calls: list[tuple[list[int], str | None]] = []

    def fake_tokenizer(tokens, prompt_wav_path):
        calls.append((list(tokens), prompt_wav_path))
        return _wav_bytes(np.linspace(-0.1, 0.1, 120, dtype=np.float32))

    model.audio_tokenizer = fake_tokenizer
    waveform = model.decode_audio_tokens(torch.tensor([1, 2, 3]))

    assert calls == [([1, 2, 3], None)]
    assert waveform.dtype == np.float32
    assert waveform.shape == (120,)
    assert np.isfinite(waveform).all()


def test_decode_audio_tokens_fails_fast_without_tokenizer() -> None:
    model = _bare_model("token2wav")
    model._lazy_init_token2wav = lambda: None
    model.audio_tokenizer = None

    with pytest.raises(RuntimeError, match="Token2Wav is not initialized"):
        model.decode_audio_tokens(torch.tensor([1], dtype=torch.long))


def test_token2wav_forward_empty_finished_payload_returns_empty_waveform() -> None:
    model = _bare_model("token2wav")
    output = model.forward(
        input_ids=torch.tensor([0], dtype=torch.long),
        additional_information={
            "codes": {"audio": torch.empty(0, dtype=torch.long)},
            "meta": {"code_flat_numel": 0},
        },
    )

    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32
    assert output.numel() == 0


def test_token2wav_forward_missing_payload_ignores_placeholder_tokens() -> None:
    model = _bare_model("token2wav")
    model.decode_audio_tokens = lambda _codes: pytest.fail("placeholder input_ids must not be decoded")

    output = model.forward(input_ids=torch.tensor([1, 2, 3], dtype=torch.long))

    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32
    assert output.numel() == 0


def test_wrapper_talker_missing_runtime_payload_returns_dummy_hidden() -> None:
    model = _bare_wrapper("talker")
    output = model.forward(
        input_ids=torch.tensor([1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1], dtype=torch.long),
        runtime_additional_information=[{}],
    )

    assert output.text_hidden_states.shape == (2, 4)
    assert output.multimodal_outputs is None


def test_wrapper_token2wav_load_weights_does_not_scan_checkpoint() -> None:
    model = _bare_wrapper("token2wav")

    class FakeToken2Wav:
        def load_weights(self, weights):
            assert weights == []
            return {"assets"}

    class NoIterCheckpoint:
        def __iter__(self):
            raise AssertionError("token2wav stage must not iterate HF checkpoint weights")

    model.token2wav = FakeToken2Wav()

    assert model.load_weights(NoIterCheckpoint()) == {"token2wav.assets"}


def test_token2wav_forward_non_empty_decode_cannot_silently_return_empty() -> None:
    model = _bare_model("token2wav")
    model.decode_audio_tokens = lambda _codes: np.asarray([], dtype=np.float32)

    with pytest.raises(RuntimeError, match="decoded empty audio"):
        model.forward(
            input_ids=torch.tensor([0, 0], dtype=torch.long),
            additional_information={"codes": {"audio": torch.tensor([1, 2], dtype=torch.long)}},
        )


def test_package_waveform() -> None:
    waveform = MiniCPMO45OmniTTSForConditionalGeneration.package_waveform(np.asarray([0.25, -0.25]))
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert waveform.tolist() == [0.25, -0.25]
