# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for per-request Qwen2.5-Omni speaker selection.

The talker used to cache the first ``voice_type`` it saw on the model and
Code2Wav always used the ``forward`` default, so every request was spoken by
Chelsie whatever ``speaker`` it asked for.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniToken2WavConfig

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import (
    DEFAULT_SPEAKER,
    Qwen2_5OmniForConditionalGeneration,
    resolve_speaker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bare_model(stage: str) -> Qwen2_5OmniForConditionalGeneration:
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = stage
    return model


class _FakeToken2Wav(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.token2wav = SimpleNamespace(factor=2, code2wav_dit_model=SimpleNamespace(mel_dim=4))
        self.conditionings: list[torch.Tensor] = []

    def process_chunk(self, *, conditioning, **kwargs):
        self.conditionings.append(conditioning)
        return None, torch.zeros(8)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, "Chelsie"),
        ("", "Chelsie"),
        ([], "Chelsie"),
        ("ethan", "Ethan"),
        (" ETHAN ", "Ethan"),
        (["ethan"], "Ethan"),
        ("chelsie", "Chelsie"),
        ("not-a-voice", "Chelsie"),
    ],
)
def test_resolve_speaker(requested, expected) -> None:
    assert DEFAULT_SPEAKER == "Chelsie"
    assert resolve_speaker(requested) == expected


def test_talker_prefill_uses_each_request_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_model("talker")
    model.model = nn.Linear(2, 2)
    speakers: list[str] = []

    def fake_prefill(*, speaker, **kwargs):
        speakers.append(speaker)
        return torch.zeros(1, dtype=torch.long), torch.zeros(1, 2)

    monkeypatch.setattr(model, "_thinker_to_talker_prefill", fake_prefill)
    payload = {
        "embed": {"prefill": torch.zeros(3, 2)},
        "hidden_states": {"output": torch.zeros(2, 2)},
        "ids": {"prompt": [1, 2, 3], "output": [4, 5]},
    }
    # A later request without a speaker must not inherit the earlier one.
    for speaker in (["ethan"], None, "CHELSIE"):
        request_payload = {**payload, "speaker": speaker} if speaker is not None else payload
        model.thinker_to_talker_process(torch.zeros(5, dtype=torch.long), torch.zeros(5, 2), request_payload)

    assert speakers == ["Ethan", "Chelsie", "Chelsie"]


def test_code2wav_uses_request_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_model("code2wav")
    speakers: list[str] = []

    def fake_generate_audio(code, speaker):
        speakers.append(speaker)
        return torch.zeros(4)

    monkeypatch.setattr(model, "generate_audio", fake_generate_audio)
    codes, positions = torch.tensor([5, 6, 7]), torch.arange(3)

    model.forward(codes, positions, model_intermediate_buffer=[{"speaker": ["ethan"]}])
    model.forward(codes, positions, model_intermediate_buffer=[{}])
    model.forward(codes, positions)

    assert speakers == ["Ethan", "Chelsie", "Chelsie"]


# spk_dict.pt keeps the voice's casing; legacy npy inputs use lowercase keys.
@pytest.mark.parametrize("keys", [("Chelsie", "Ethan"), ("chelsie", "ethan")])
def test_codec_to_audio_selects_requested_voice(keys) -> None:
    model = _bare_model("code2wav")
    model.token2wav = _FakeToken2Wav()
    model.token2wav_config = Qwen2_5OmniToken2WavConfig(dit_config={"mel_dim": 4, "enc_emb_dim": 3})
    model._token2wav_conds = {key: torch.full((1, 3), float(i)) for i, key in enumerate(keys)}
    model._token2wav_ref_mels = {key: torch.zeros(1, 300, 4) for key in keys}

    model._codec_to_audio(torch.tensor([5, 6, 7]), "Ethan")
    model._codec_to_audio(torch.tensor([5, 6, 7]), "Chelsie")

    assert [cond[0, 0].item() for cond in model.token2wav.conditionings] == [1.0, 0.0]
