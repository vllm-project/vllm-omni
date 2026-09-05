# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Breeze-TTS-2 inference example (two-stage talker + codec pipeline).

Uses the async-chunk deploy config (vllm_omni/deploy/breeze_tts_2.yaml).
Reference checkpoint: BreezeBlue/Breeze-TTS-2.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoConfig

from vllm_omni import Omni
from vllm_omni.model_executor.models.breeze_tts_2.prompt_builder import (
    BreezeTTS2PromptBuilder,
)
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_RATE = 24_000


def parse_args():
    parser = TrackingArgumentParser(description="Offline Breeze-TTS-2 inference")
    parser.add_argument(
        "--model",
        type=str,
        default="BreezeBlue/Breeze-TTS-2",
        help="Breeze-TTS-2 model path or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is Breeze TTS 2 running on vLLM Omni.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="S0",
        help="Speaker tag such as S0..S9.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional style instruction (voice design / voice direction).",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Optional reference wav path for cloning (requires --ref-text).",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Exact transcript of the reference audio.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Cap on generated codec frames (one frame is 80 ms of audio).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Directory for output WAV files.",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override the deploy config path. If unset, auto-loads "
        "vllm_omni/deploy/breeze_tts_2.yaml based on the HF model_type.",
    )
    return parser.parse_args()


def resolve_template(args) -> str:
    if args.ref_audio is not None:
        return "ref_edit_tata" if args.instruction else "ref_clone_tata"
    return "tts_instruction" if args.instruction else "tts_plain"


def build_prompt(args) -> dict:
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    reference_encoder = None
    if args.ref_audio is not None:
        from vllm_omni.model_executor.models.breeze_tts_2.audio_tokenizer import (
            BreezeReferenceAudioTokenizer,
        )

        reference_encoder = BreezeReferenceAudioTokenizer.from_pretrained(
            args.model,
            num_codebooks=int(getattr(config, "num_codebooks", 16)),
            device_map="cpu",
        )
    builder = BreezeTTS2PromptBuilder.from_pretrained(
        args.model,
        config,
        reference_audio_encoder=reference_encoder,
    )
    payload = {
        "text": args.text,
        "speaker": args.voice,
        "ref_audio_path": args.ref_audio,
    }
    if args.instruction:
        payload["instruction"] = args.instruction
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    prompt = builder.build(payload, template=resolve_template(args))
    if args.max_new_tokens is not None:
        prompt["additional_information"]["breeze_max_new_frames"] = int(args.max_new_tokens)
    return prompt


def extract_audio(multimodal_output: dict) -> torch.Tensor:
    """Extract the final complete audio tensor from multimodal output.

    The output processor concatenates per-step delta tensors under
    ``model_outputs``.  Falls back to ``audio`` for backwards compat.
    """
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"No audio key in multimodal_output: {list(multimodal_output.keys())}")

    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("Audio list is empty or all elements are None.")
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]

    return torch.as_tensor(audio).float().cpu().reshape(-1)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
    )

    prompt = build_prompt(args)

    print(f"Model       : {args.model}")
    print(f"Text        : {args.text}")
    print(f"Template    : {resolve_template(args)}")
    print(f"Output dir  : {output_dir}")
    print(f"Prompt len  : {len(prompt['prompt_token_ids'])} tokens")

    t_start = time.perf_counter()
    outputs = engine.generate([prompt])
    elapsed = time.perf_counter() - t_start

    request_output = outputs[0]
    mm = request_output.outputs[0].multimodal_output
    audio = extract_audio(mm)

    duration = audio.numel() / SAMPLE_RATE
    rtf = elapsed / duration if duration > 0 else float("inf")

    output_path = output_dir / "output.wav"
    sf.write(str(output_path), audio.numpy(), SAMPLE_RATE, format="WAV")

    print(f"Saved       : {output_path}")
    print(f"Duration    : {duration:.2f}s")
    print(f"Inference   : {elapsed:.2f}s")
    print(f"RTF         : {rtf:.3f}")


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
