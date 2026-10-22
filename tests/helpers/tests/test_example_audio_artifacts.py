# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest

from tests.examples import audio_artifacts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("fail", [False, True])
def test_example_audio_opts_into_fallback_and_preserves_original_evidence(monkeypatch, tmp_path, fail):
    source = tmp_path / "clip.wav"
    source.write_bytes(b"original audio")
    checkout = tmp_path / "checkout"
    monkeypatch.setenv("BUILDKITE_BUILD_CHECKOUT_PATH", str(checkout))
    calls = []

    def transcribe(output_path, *, temperature_fallback):
        calls.append((output_path, temperature_fallback))
        if fail:
            raise RuntimeError("ASR failed")
        return "spoken content"

    monkeypatch.setattr(audio_artifacts, "convert_audio_file_to_text", transcribe)
    if fail:
        with pytest.raises(RuntimeError, match="ASR failed"):
            audio_artifacts.transcribe_example_audio(str(source), "test_modality", "client output")
    else:
        assert (
            audio_artifacts.transcribe_example_audio(str(source), "test_modality", "client output") == "spoken content"
        )

    assert calls == [(str(source), True)]
    folder = checkout / "qwen3-omni-doc-artifacts/test_modality"
    assert (folder / "clip.wav").read_bytes() == source.read_bytes()
    evidence = json.loads((folder / "transcription.json").read_text())
    assert evidence["client_output"] == "client output"
    assert evidence["asr_temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert evidence["asr_fallback_seed"] == 0
    assert evidence.get("asr_error") == ("RuntimeError: ASR failed" if fail else None)
