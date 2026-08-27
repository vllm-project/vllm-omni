"""Offline dots.tts inference example (single-stage native AR pipeline).

Uses the single-stage native AR config (vllm_omni/deploy/dots_tts.yaml).
Reference checkpoint: rednote-hilab/dots.tts-soar.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_RATE = 48_000


def parse_args():
    parser = TrackingArgumentParser(description="Offline dots.tts native AR inference")
    parser.add_argument(
        "--model",
        type=str,
        default="rednote-hilab/dots.tts-soar",
        help="dots.tts model path or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of dots TTS running on vLLM Omni.",
        help="Text to synthesize.",
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
        "vllm_omni/deploy/dots_tts.yaml based on the HF model_type.",
    )
    return parser.parse_args()


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

    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.dots_tts.dots_tts_prompt import (
        build_dots_tts_prompt,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt = build_dots_tts_prompt(
        tokenizer=tokenizer,
        text=args.text,
    )

    print(f"Model       : {args.model}")
    print(f"Text        : {args.text}")
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
