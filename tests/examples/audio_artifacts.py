# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve the exact generated audio and transcript used by an example check."""

import json
import os
import re
import shutil
from pathlib import Path

from tests.helpers.media import convert_audio_file_to_text


def transcribe_example_audio(output_path: str, test_name: str, client_output: str) -> str:
    source = Path(output_path)
    checkout = Path(os.environ.get("BUILDKITE_BUILD_CHECKOUT_PATH") or Path.cwd())
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", test_name)
    artifact_dir = checkout / "qwen3-omni-doc-artifacts" / safe_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Copy before ASR so a transcriber failure still leaves its input available.
    audio_artifact = artifact_dir / source.name
    shutil.copyfile(source, audio_artifact)
    evidence = {
        "test": test_name,
        "audio_file": source.name,
        "client_output": client_output,
        "asr_model": "small",
        "asr_temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "asr_fallback_seed": 0,
    }
    try:
        transcript = convert_audio_file_to_text(output_path, temperature_fallback=True)
        evidence["transcript"] = transcript
        return transcript
    except BaseException as exc:
        evidence["asr_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (artifact_dir / "transcription.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"Qwen3-Omni audio evidence: {artifact_dir}")
