# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import _build_hift

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

flashcosyvoice_hifigan = pytest.importorskip("flashcosyvoice.modules.hifigan")
flashcosyvoice_audio = pytest.importorskip("flashcosyvoice.utils.audio")


def test_vendored_hift_matches_flashcosyvoice() -> None:
    reference = flashcosyvoice_hifigan.HiFTGenerator().eval()
    vendored = _build_hift().eval()

    # Strict loading verifies that the vendored architecture remains compatible
    # with the Step-Audio2 checkpoint produced for flashcosyvoice.
    vendored.load_state_dict(reference.state_dict(), strict=True)

    speech_feat = torch.randn(1, 80, 4)
    cache_source = torch.zeros(1, 1, 0)
    with torch.inference_mode():
        torch.manual_seed(0)
        expected_speech, expected_source = reference(speech_feat, cache_source)
        torch.manual_seed(0)
        actual_speech, actual_source = vendored.inference(speech_feat, cache_source)

    torch.testing.assert_close(actual_source, expected_source)
    torch.testing.assert_close(actual_speech, expected_speech)


def test_vendored_mel_spectrogram_matches_flashcosyvoice() -> None:
    reference = flashcosyvoice_audio.mel_spectrogram
    vendored = mel_spectrogram

    audio = torch.randn(1, 48000)
    kwargs = {
        "n_fft": 1920,
        "num_mels": 80,
        "sampling_rate": 24000,
        "hop_size": 480,
        "win_size": 1920,
        "fmin": 0,
        "fmax": 8000,
    }
    with torch.inference_mode():
        expected_spec = reference(audio, **kwargs)
        actual_spec = vendored(audio, **kwargs)

    torch.testing.assert_close(actual_spec, expected_spec)
