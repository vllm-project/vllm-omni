# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex (Nemotron-Labs-Audex-2B) audio understanding example.

Speech (or general audio) + text instruction → text, through the
single-stage ``audex_thinker_only`` pipeline: NV-Whisper
encoder + relu2 projector + 2B dense LM on ``checkpoint_folder_full``.

The prompt format mirrors the official audioqa script: a ChatML user turn
holding the ``<so_embedding>`` placeholder (expanded by the processor to
750 embedding positions per 30 s clip) plus the instruction, with a closed
``<think></think>`` priming.

Examples (without --audio-files, vLLM's public ``mary_had_lamb`` asset is
transcribed):

    # Transcribe WAVs (ASR):
    python examples/offline_inference/audex/audio_qa.py \\
        --audio-files a.wav b.wav

    # Free-form audio QA:
    python examples/offline_inference/audex/audio_qa.py \\
        --audio-files a.wav --question "What language is being spoken?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import default_deploy_config, load_audio_file, request_index
from vllm.assets.audio import AudioAsset

from vllm_omni import Omni

ASR_QUESTION = "Transcribe the input speech."


def build_prompt(question: str) -> str:
    return f"<|im_start|>user\n<so_embedding>\n{question}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Audex audio understanding")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument(
        "--audio-files",
        type=str,
        nargs="+",
        default=None,
        help="Input clips. Defaults to vLLM's mary_had_lamb audio asset.",
    )
    parser.add_argument("--question", type=str, default=ASR_QUESTION)
    parser.add_argument(
        "--deploy-config",
        type=str,
        # The model root's default deploy yaml is the TTS pipeline; audio
        # understanding needs the single-stage thinker-only pipeline.
        default=default_deploy_config("audex_thinker_only.yaml"),
        help="Deploy yaml (defaults to the audex_thinker_only pipeline).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    prompt_text = build_prompt(args.question)
    if args.audio_files:
        clips = [load_audio_file(path) for path in args.audio_files]
    else:
        clips = [AudioAsset("mary_had_lamb").audio_and_sample_rate]
    prompts = []
    for audio, sr in clips:
        prompts.append(
            {
                "prompt": prompt_text,
                "multi_modal_data": {"audio": (audio, sr)},
            }
        )

    outputs = engine.generate(prompts)

    ordered = sorted(outputs, key=request_index)
    names = [Path(p).name for p in args.audio_files] if args.audio_files else ["mary_had_lamb"]
    for name, req_output in zip(names, ordered):
        text = req_output.outputs[0].text if req_output.outputs else ""
        print(f"{name}: {text.strip()}")


if __name__ == "__main__":
    main()
