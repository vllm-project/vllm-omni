# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from vllm_omni.model_executor.models.breeze_tts_2.prompt_builder import (
    BreezeTTS2PromptBuilder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "offline_inference"
    / "text_to_speech"
    / "breeze_tts_2"
    / "end2end.py"
)


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {
            "<|AUDIO|>": 262144,
            "<|audio_eos|>": 262145,
        }.get(token, self.unk_token_id)

    def __call__(self, texts, *, add_special_tokens=True, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[1000 + ord(char) % 1000 for char in text] for text in texts]
        for row_index, text in enumerate(texts):
            if "<|AUDIO|>" in text or "<|audio_eos|>" in text:
                cursor = 0
                row = []
                while cursor < len(text):
                    if text.startswith("<|AUDIO|>", cursor):
                        row.append(262144)
                        cursor += len("<|AUDIO|>")
                    elif text.startswith("<|audio_eos|>", cursor):
                        row.append(262145)
                        cursor += len("<|audio_eos|>")
                    else:
                        row.append(1000 + ord(text[cursor]) % 1000)
                        cursor += 1
                ids[row_index] = row
        if add_special_tokens:
            ids = [[2, *row, 1] for row in ids]
        return {"input_ids": ids}

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        rendered = []
        for token_id in ids:
            if token_id == 262144:
                rendered.append("<|AUDIO|>")
            elif token_id == 262145:
                rendered.append("<|audio_eos|>")
            elif token_id in (1, 2):
                continue
            else:
                rendered.append(chr(int(token_id) - 1000))
        return "".join(rendered)


class _ReferenceEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int | None]] = []

    def encode(self, audio, sample_rate=None):
        self.calls.append((audio, sample_rate))
        return torch.arange(32, dtype=torch.int16).reshape(2, 16)


def _builder() -> tuple[BreezeTTS2PromptBuilder, _ReferenceEncoder]:
    encoder = _ReferenceEncoder()
    builder = BreezeTTS2PromptBuilder(
        _Tokenizer(),
        SimpleNamespace(
            audio_token_id=262144,
            audio_eos_token_id=262145,
            pad_token_id=0,
            backbone_config=SimpleNamespace(vocab_size=100000),
            num_codebooks=16,
            codec_config={"codebook_size": 2048},
        ),
        encoder,
    )
    return builder, encoder


def test_reference_voice_uses_exact_transcript_and_audio_codes():
    builder, encoder = _builder()

    prompt = builder.build(
        {
            "template": "ref_clone_tata",
            "speaker": "S1",
            "ref_text": "exact reference transcript",
            "ref_audio": "reference.wav",
            "ref_audio_sample_rate": 24_000,
            "text": "new target text",
        }
    )

    info = prompt["additional_information"]
    assert tuple(info["input_values"].shape) == (2, 16)
    assert info["ref_code_len"] == 2
    assert info["prompt_ids"].tolist().count(262144) == 2
    assert info["text_ids_len"].numel() == 2
    assert int(info["text_ids_mask"].sum()) == int(info["text_ids_len"].sum())
    assert encoder.calls == [("reference.wav", 24_000)]


def test_reference_plus_instruction_composes_voice_direction_prompt():
    builder, _ = _builder()

    prompt = builder.build(
        {
            "template": "ref_edit_tata",
            "speaker": "S0",
            "ref_text": "exact reference transcript",
            "ref_audio": "reference.wav",
            "instruction": "speak calmly and warmly",
            "text": "new target text",
        }
    )

    info = prompt["additional_information"]
    assert info["template"] == "ref_edit_tata"
    assert tuple(info["input_values"].shape) == (2, 16)
    assert info["prompt_ids"].tolist().count(262144) == 2
    assert info["text_ids_len"].numel() == 2


def test_offline_example_reference_payload_feeds_prompt_builder(tmp_path):
    """Smoke: the offline example's payload keys flow into the real builder.

    Loads ``end2end.py`` (voxtral-style importlib), reads a real stereo wav
    through its ``reference_payload`` helper, and feeds the result into the
    real ``BreezeTTS2PromptBuilder`` — guarding the example↔builder field
    contract that previously broke via an unread ``ref_audio_path`` key.
    """
    spec = importlib.util.spec_from_file_location("breeze_tts_2_end2end", _EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = np.sin(2 * np.pi * 220.0 * t)
    ref_path = tmp_path / "reference.wav"
    sf.write(str(ref_path), np.stack([tone, tone], axis=1), sample_rate)

    payload = module.reference_payload(str(ref_path))

    # The helper emits exactly the keys the prompt builder consumes.
    assert set(payload) == {"ref_audio", "ref_audio_sample_rate"}
    assert payload["ref_audio_sample_rate"] == sample_rate
    assert payload["ref_audio"].ndim == 1  # stereo down-mixed to one clip

    builder, encoder = _builder()
    prompt = builder.build(
        {
            "template": "ref_clone_tata",
            "speaker": "S0",
            "ref_text": "exact reference transcript",
            "text": "new target text",
            **payload,
        }
    )

    info = prompt["additional_information"]
    assert tuple(info["input_values"].shape) == (2, 16)
    assert info["ref_code_len"] == 2
    assert info["prompt_ids"].tolist().count(262144) == 2
    # The builder received the example's waveform and sample rate verbatim.
    assert encoder.calls == [(payload["ref_audio"], sample_rate)]
