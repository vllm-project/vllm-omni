# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve Qwen3-Omni documentation audio and its ASR evidence."""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from tests.helpers.media import (
    WHISPER_TEMPERATURE_FALLBACK,
    WHISPER_TEMPERATURE_FALLBACK_SEED,
    convert_audio_file_to_text,
)

_ARTIFACT_ROOT = "qwen3-omni-doc-artifacts"
_ASR_MODEL = "small"


def _artifact_checkout() -> Path:
    checkout = os.environ.get("BUILDKITE_BUILD_CHECKOUT_PATH")
    if os.environ.get("BUILDKITE_BUILD_ID") and not checkout:
        raise RuntimeError("BUILDKITE_BUILD_CHECKOUT_PATH is required in Buildkite")
    return Path(checkout) if checkout else Path.cwd()


def _safe_test_name(test_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", test_name) or "unnamed-test"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcribe_qwen3_omni_audio(output_path: str, test_name: str, client_output: str) -> str:
    """Retain the exact WAV and ASR metadata used by a Qwen3-Omni check."""
    source = Path(output_path)
    artifact_dir = _artifact_checkout() / _ARTIFACT_ROOT / _safe_test_name(test_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Copy before ASR so a transcription failure still leaves its exact input.
    audio_artifact = artifact_dir / source.name
    shutil.copyfile(source, audio_artifact)
    evidence = {
        "test_name": test_name,
        "client_output": client_output,
        "audio_file": audio_artifact.name,
        "audio_sha256": _sha256(audio_artifact),
        "asr_model": _ASR_MODEL,
        "asr_temperature_schedule": list(WHISPER_TEMPERATURE_FALLBACK),
        "asr_seed": WHISPER_TEMPERATURE_FALLBACK_SEED,
        "transcript": None,
        "asr_error": None,
    }

    try:
        transcript = convert_audio_file_to_text(
            str(audio_artifact),
            model_size=_ASR_MODEL,
            temperature_fallback=True,
        )
        evidence["transcript"] = transcript
        return transcript
    except BaseException as exc:
        evidence["asr_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        evidence_path = artifact_dir / "transcription.json"
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Qwen3-Omni audio evidence: {artifact_dir}")


__all__ = ["transcribe_qwen3_omni_audio"]
