# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline inference tests: Qwen3-TTS.
See examples/offline_inference/qwen3_tts/README.md

Picks classic test examples covering all three query types (CustomVoice,
VoiceDesign, Base) and the streaming execution path.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_OMNI_USE_V2_RUNNER"] = "1"

import pytest

from tests.examples.conftest import EXAMPLES, run_cmd
from tests.utils import hardware_test

pytestmark = [pytest.mark.advanced_model, pytest.mark.example]

END2END = str(EXAMPLES / "offline_inference" / "qwen3_tts" / "end2end.py")

CUSTOM_VOICE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


MIN_DURATION_S = 0.5
EXPECTED_SAMPLE_RATE = 24000


def _assert_wav_output(output_dir: str) -> None:
    """Assert that at least one valid .wav file was produced.

    Checks: file exists, sample rate, minimum duration, and not silence.
    """
    wav_files = list(Path(output_dir).glob("*.wav"))
    assert len(wav_files) > 0, f"No .wav files found in {output_dir}"
    for wav in wav_files:
        data, sr = sf.read(wav, dtype="float32")
        assert sr == EXPECTED_SAMPLE_RATE, f"{wav.name}: sample_rate={sr}, expected {EXPECTED_SAMPLE_RATE}"
        duration = len(data) / sr
        assert duration >= MIN_DURATION_S, f"{wav.name}: duration={duration:.3f}s, expected >= {MIN_DURATION_S}s"
        assert np.max(np.abs(data)) > 0.01, f"{wav.name}: audio appears to be silence"


@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_custom_voice():
    """CustomVoice single prompt — the most common TTS use case."""
    with tempfile.TemporaryDirectory() as output_dir:
        command = [
            "python",
            END2END,
            "--model",
            CUSTOM_VOICE_MODEL,
            "--query-type",
            "CustomVoice",
            "--output-dir",
            output_dir,
        ]
        run_cmd(command)
        _assert_wav_output(output_dir)


@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_voice_design():
    """VoiceDesign single prompt — generates speech from a voice description."""
    with tempfile.TemporaryDirectory() as output_dir:
        command = [
            "python",
            END2END,
            "--model",
            VOICE_DESIGN_MODEL,
            "--query-type",
            "VoiceDesign",
            "--output-dir",
            output_dir,
        ]
        run_cmd(command)
        _assert_wav_output(output_dir)


@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_base_icl():
    """Base ICL mode — voice cloning with reference audio and transcript."""
    with tempfile.TemporaryDirectory() as output_dir:
        command = [
            "python",
            END2END,
            "--model",
            BASE_MODEL,
            "--query-type",
            "Base",
            "--mode-tag",
            "icl",
            "--output-dir",
            output_dir,
        ]
        run_cmd(command)
        _assert_wav_output(output_dir)


@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_custom_voice_streaming():
    """CustomVoice streaming — exercises the AsyncOmni streaming path."""
    with tempfile.TemporaryDirectory() as output_dir:
        command = [
            "python",
            END2END,
            "--model",
            CUSTOM_VOICE_MODEL,
            "--query-type",
            "CustomVoice",
            "--streaming",
            "--output-dir",
            output_dir,
        ]
        run_cmd(command)
        _assert_wav_output(output_dir)
