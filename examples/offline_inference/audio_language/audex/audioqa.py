# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex (Nemotron-Labs-Audex-2B) audio understanding example.

Speech (or general audio) + text instruction → text, through the
single-stage ``nemotron_labs_audex_thinker_only`` pipeline: NV-Whisper
encoder + relu2 projector + 2B dense LM on ``checkpoint_folder_full``.

The prompt format mirrors the official audioqa script: a ChatML user turn
holding the ``<so_embedding>`` placeholder (expanded by the processor to
750 embedding positions per 30 s clip) plus the instruction, with a closed
``<think></think>`` priming.

Examples:

    # Transcribe WAVs (ASR):
    python examples/offline_inference/audio_language/audex/audioqa.py \\
        --audio-files a.wav b.wav

    # Free-form audio QA:
    python examples/offline_inference/audio_language/audex/audioqa.py \\
        --audio-files a.wav --question "What language is being spoken?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from vllm_omni import Omni

ASR_QUESTION = "Transcribe the input speech."
TARGET_SR = 16_000
# The model root's default deploy yaml is the TTS pipeline; audio
# understanding needs the single-stage thinker-only pipeline, so default to it.
_DEFAULT_DEPLOY_CONFIG = str(
    Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy" / "nemotron_labs_audex_thinker_only.yaml"
)


def build_prompt(question: str) -> str:
    return f"<|im_start|>user\n<so_embedding>\n{question}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Audex audio understanding")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument("--audio-files", type=str, nargs="+", required=True)
    parser.add_argument("--question", type=str, default=ASR_QUESTION)
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=_DEFAULT_DEPLOY_CONFIG,
        help="Deploy yaml (defaults to the nemotron_labs_audex_thinker_only pipeline).",
    )
    return parser.parse_args()


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # The processor resamples via its data parser; pass through as-is.
        pass
    return audio, sr


def main():
    args = parse_args()
    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    prompt_text = build_prompt(args.question)
    prompts = []
    for path in args.audio_files:
        audio, sr = _load_audio(path)
        prompts.append(
            {
                "prompt": prompt_text,
                "multi_modal_data": {"audio": (audio, sr)},
            }
        )

    outputs = engine.generate(prompts)

    # Outputs may return out of submission order; the numeric request-id
    # prefix follows submission order (lexicographic sort would misplace
    # "10_..." before "2_...").
    def _req_index(req_output) -> int:
        import re as _re

        match = _re.search(r"(\d+)", str(req_output.request_id))
        return int(match.group(1)) if match else 0

    ordered = sorted(outputs, key=_req_index)
    for path, req_output in zip(args.audio_files, ordered):
        text = req_output.outputs[0].text if req_output.outputs else ""
        print(f"{Path(path).name}: {text.strip()}")


if __name__ == "__main__":
    main()
