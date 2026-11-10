# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
import json

import pytest

from tests.examples import qwen3_omni_audio_artifacts as artifacts
from tests.helpers.media import WHISPER_TEMPERATURE_FALLBACK, WHISPER_TEMPERATURE_FALLBACK_SEED

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("asr_error", [None, RuntimeError("ASR failed")], ids=["success", "asr-failure"])
def test_qwen3_audio_evidence_preserves_exact_wav_and_metadata(monkeypatch, tmp_path, asr_error):
    source = tmp_path / "generated.wav"
    source_bytes = b"exact generated audio bytes"
    source.write_bytes(source_bytes)
    checkout = tmp_path / "checkout"
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")
    monkeypatch.setenv("BUILDKITE_BUILD_CHECKOUT_PATH", str(checkout))
    calls = []

    def transcribe(output_path, model_size, *, temperature_fallback):
        artifact = checkout / "qwen3-omni-doc-artifacts" / "test_audio_case_param_" / source.name
        assert artifact.read_bytes() == source_bytes
        calls.append((output_path, model_size, temperature_fallback))
        if asr_error is not None:
            raise asr_error
        return "spoken content"

    monkeypatch.setattr(artifacts, "convert_audio_file_to_text", transcribe)

    if asr_error is None:
        assert (
            artifacts.transcribe_qwen3_omni_audio(
                str(source),
                "test_audio_case[param]",
                "raw client output",
            )
            == "spoken content"
        )
    else:
        with pytest.raises(RuntimeError, match="ASR failed"):
            artifacts.transcribe_qwen3_omni_audio(
                str(source),
                "test_audio_case[param]",
                "raw client output",
            )

    artifact_dir = checkout / "qwen3-omni-doc-artifacts" / "test_audio_case_param_"
    retained_audio = artifact_dir / source.name
    assert calls == [(str(retained_audio), "small", True)]
    assert retained_audio.read_bytes() == source_bytes

    evidence = json.loads((artifact_dir / "transcription.json").read_text(encoding="utf-8"))
    assert evidence == {
        "asr_error": "RuntimeError: ASR failed" if asr_error is not None else None,
        "asr_model": "small",
        "asr_seed": WHISPER_TEMPERATURE_FALLBACK_SEED,
        "asr_temperature_schedule": list(WHISPER_TEMPERATURE_FALLBACK),
        "audio_file": source.name,
        "audio_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "client_output": "raw client output",
        "test_name": "test_audio_case[param]",
        "transcript": None if asr_error is not None else "spoken content",
    }


def test_qwen3_audio_evidence_requires_checkout_path_in_buildkite(monkeypatch, tmp_path):
    source = tmp_path / "generated.wav"
    source.write_bytes(b"audio")
    monkeypatch.setenv("BUILDKITE_BUILD_ID", "build-id")
    monkeypatch.delenv("BUILDKITE_BUILD_CHECKOUT_PATH", raising=False)

    with pytest.raises(RuntimeError, match="BUILDKITE_BUILD_CHECKOUT_PATH is required"):
        artifacts.transcribe_qwen3_omni_audio(str(source), "test_name", "client output")
