# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline higgs-audio v3 inference example.

Runs Stage 0 (Qwen3 talker) + Stage 1 (HiggsAudio codec) end-to-end
through the vLLM-Omni engine without going through the HTTP server, and
saves a 24 kHz mono WAV per prompt.

Example:

    python examples/offline_inference/text_to_speech/higgs_audio_v3/end2end.py \
        --texts "Hello world." \
                "The quick brown fox jumps over the lazy dog." \
        --output-dir results/higgs_v3_wavs
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni import Omni

SAMPLE_RATE = 24_000
DEFAULT_TEXTS = (
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Today is a beautiful day for a walk in the park.",
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Offline higgs-audio v3 inference")
    parser.add_argument(
        "--model",
        type=str,
        default="bosonai/higgs-audio-v3-tts-4b",
        help="Stage-0 talker model id or path.",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=list(DEFAULT_TEXTS),
        help="One or more plain-text prompts to synthesize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/higgs_v3_wavs",
        help="Directory to write per-prompt WAV files.",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override the deploy config path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Cap on Stage-0 codec frames per request.",
    )
    return parser.parse_args()


def _slugify(text: str) -> str:
    slug = re.sub(r"\s+", "_", text.strip().lower())
    slug = re.sub(r"[^a-z0-9_]+", "", slug)
    return slug[:48] or "prompt"


def _extract_pcm(multimodal_output: dict) -> torch.Tensor:
    """Pull the final concatenated PCM tensor out of a request's multimodal_output."""
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"no audio key in multimodal_output: {list(multimodal_output.keys())}")
    if isinstance(audio, list):
        valid = [torch.as_tensor(a).float().cpu().reshape(-1) for a in audio if a is not None]
        if not valid:
            raise ValueError("audio list is empty")
        return torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]
    return torch.as_tensor(audio).float().cpu().reshape(-1)


def _pcm_to_int16(pcm: torch.Tensor) -> np.ndarray:
    arr = pcm.numpy()
    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    else:
        arr = arr.astype(np.int16)
    return arr


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    # Build prompts using the v3 tokenizer adapter
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
        HiggsAudioV3TokenizerAdapter,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    adapter = HiggsAudioV3TokenizerAdapter(tokenizer)

    print(f"Model       : {args.model}")
    print(f"Prompts     : {len(args.texts)}")
    print(f"Output dir  : {output_dir}")

    total_elapsed = 0.0
    total_dur = 0.0
    for text in args.texts:
        prompt_ids = adapter.build_prompt(text)
        prompt = {"prompt_token_ids": prompt_ids}

        t_start = time.perf_counter()
        outputs = engine.generate([prompt])
        elapsed = time.perf_counter() - t_start
        total_elapsed += elapsed

        mm = outputs[0].outputs[0].multimodal_output
        pcm = _extract_pcm(mm)
        slug = _slugify(text)
        out_path = output_dir / f"{slug}.wav"
        sf.write(str(out_path), _pcm_to_int16(pcm), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        dur = pcm.numel() / SAMPLE_RATE
        total_dur += dur
        print(f"  {slug:<50} dur={dur:5.2f}s  -> {out_path}")

    rtf = total_elapsed / total_dur if total_dur > 0 else float("inf")
    print(f"Total infer : {total_elapsed:.2f}s  total audio: {total_dur:.2f}s  RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
