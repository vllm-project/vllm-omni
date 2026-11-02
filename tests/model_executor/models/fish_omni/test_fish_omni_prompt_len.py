# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The serving layer sizes the Drama 3 placeholder prompt before the reference
audio is encoded, and stage 0 rejects a size mismatch. This checks that the
estimate equals the rows built from the real DAC encoder output.

The frame count does not depend on the codec weights, so a randomly
initialized codec runs on CPU without the checkpoint.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("torchaudio")

from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniForConditionalGeneration
from vllm_omni.model_executor.models.fish_omni.prompt import Reference, build_prompt_frames
from vllm_omni.model_executor.models.fish_speech import dac_encoder
from vllm_omni.model_executor.models.fish_speech.dac_utils import build_dac_codec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEMANTIC_BEGIN = 248099
NUM_CODEBOOKS = 10


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]


@pytest.fixture(scope="module")
def random_codec():
    torch.manual_seed(0)
    codec = build_dac_codec()
    codec.decoder = None
    return codec.eval()


@pytest.mark.parametrize(
    ("sample_rate", "num_samples"),
    [
        (44100, 2047),
        (44100, 2048),
        (44100, 2049),
        (44100, 44100),
        (16000, 743),
        (16000, 744),
        (16000, 16000),
        (24000, 36001),
        (22050, 11026),
        (48000, 48000),
    ],
)
def test_estimate_matches_encoded_frames(monkeypatch, random_codec, sample_rate, num_samples):
    monkeypatch.setattr(dac_encoder, "_load_dac_codec", lambda *args, **kwargs: random_codec)
    wav = np.random.default_rng(0).uniform(-0.5, 0.5, num_samples).astype(np.float32)
    with torch.no_grad():
        codes = dac_encoder.encode_reference_audio_codes("unused", wav, sample_rate, device="cpu")
    assert codes.shape[1] == NUM_CODEBOOKS

    info = {"text": "Hello there.", "ref_text": "Reference words.", "ref_audio_wav": wav, "ref_audio_sr": sample_rate}
    estimate = FishOmniForConditionalGeneration.estimate_prompt_len_from_additional_information(
        info, tokenizer=CharTokenizer()
    )
    frames = build_prompt_frames(
        CharTokenizer(),
        "Hello there.",
        [Reference("Reference words.", codes)],
        semantic_begin_id=SEMANTIC_BEGIN,
        num_codebooks=NUM_CODEBOOKS,
    )
    assert estimate == frames.shape[0]


def test_estimate_uses_given_codes():
    codes = torch.zeros((12, NUM_CODEBOOKS), dtype=torch.long)
    info = {"text": "Hi.", "ref_text": "Ref.", "ref_codes": codes}
    estimate = FishOmniForConditionalGeneration.estimate_prompt_len_from_additional_information(
        info, tokenizer=CharTokenizer()
    )
    frames = build_prompt_frames(
        CharTokenizer(),
        "Hi.",
        [Reference("Ref.", codes)],
        semantic_begin_id=SEMANTIC_BEGIN,
        num_codebooks=NUM_CODEBOOKS,
    )
    assert estimate == frames.shape[0]


@pytest.mark.parametrize(
    ("sample_rate", "shape"),
    [
        (44100, (2, 44100)),
        (44100, (44100, 2)),
        (16000, (2, 16000)),
        (24000, (36001, 2)),
    ],
)
def test_estimate_matches_encoded_frames_stereo(monkeypatch, random_codec, sample_rate, shape):
    monkeypatch.setattr(dac_encoder, "_load_dac_codec", lambda *args, **kwargs: random_codec)
    wav = np.random.default_rng(0).uniform(-0.5, 0.5, shape).astype(np.float32)
    with torch.no_grad():
        codes = dac_encoder.encode_reference_audio_codes("unused", wav, sample_rate, device="cpu")

    info = {"text": "Hello there.", "ref_text": "Reference words.", "ref_audio_wav": wav, "ref_audio_sr": sample_rate}
    estimate = FishOmniForConditionalGeneration.estimate_prompt_len_from_additional_information(
        info, tokenizer=CharTokenizer()
    )
    frames = build_prompt_frames(
        CharTokenizer(),
        "Hello there.",
        [Reference("Reference words.", codes)],
        semantic_begin_id=SEMANTIC_BEGIN,
        num_codebooks=NUM_CODEBOOKS,
    )
    assert estimate == frames.shape[0]
