# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline higgs-audio v2 inference example.

Runs Stage 0 (DualFFN talker) + Stage 1 (HiggsAudio codec) end-to-end
through the vLLM-Omni engine without going through the HTTP server, and
saves a 24 kHz mono WAV per prompt.

Example:

    python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \\
        --texts "Hello world." \\
                "The quick brown fox jumps over the lazy dog." \\
        --output-dir results/wavs
"""

from __future__ import annotations

import os

# DeepGEMM FP8 kernels require an optional backend that may not be installed.
# Disable the warmup before importing vLLM so engine startup falls back to the
# regular gemm path. Users with deep_gemm installed can override these.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

SAMPLE_RATE = 24_000
DEFAULT_TEXTS = (
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
)


def parse_args():
    parser = TrackingArgumentParser(description="Offline higgs-audio v2 inference")
    parser.add_argument(
        "--model",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
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
        default="results/wavs",
        help="Directory to write per-prompt WAV files.",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override the deploy config path. Auto-loads "
        "vllm_omni/deploy/higgs_audio_v2.yaml from the HF model_type by default.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Cap on Stage-0 codec frames per request.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference clip for voice clone (path to a WAV file). Paired with --ref-text.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the reference clip. Required when --ref-audio is set.",
    )
    return parser.parse_args()


def _slugify(text: str) -> str:
    import re

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
    if (args.ref_audio is None) != (args.ref_text is None):
        raise SystemExit("--ref-audio and --ref-text must be supplied together")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = Omni(model=args.model, deploy_config=args.deploy_config)

    # Build prompt_token_ids using the same path serving_speech.py takes online.
    from transformers import AutoProcessor

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        build_plain_text_prompt,
        build_voice_clone_prompt,
        input_ids_to_python_list,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Voice-clone path: load the reference clip once. The HF processor will
    # encode it via the bundled HiggsAudioV2TokenizerModel each time we build
    # a prompt below. This is cheap on CPU for a few-second clip.
    ref_wav: np.ndarray | None = None
    ref_sr: int | None = None
    if args.ref_audio is not None:
        ref_wav, ref_sr = sf.read(args.ref_audio, always_2d=False)
        if ref_wav.ndim == 2:
            ref_wav = ref_wav.mean(axis=1)

    print(f"Model       : {args.model}")
    print(f"Prompts     : {len(args.texts)}")
    print(f"Output dir  : {output_dir}")
    print(f"Voice clone : {'yes' if ref_wav is not None else 'no'}")

    # Run one prompt at a time. The Stage-0 talker's per-slot audio state is
    # request-scoped; submitting multiple prompts in the same engine.generate()
    # call would batch them in the AR runner and exercise a code path that is
    # not validated for this model yet.
    total_elapsed = 0.0
    total_dur = 0.0
    for text in args.texts:
        if ref_wav is not None:
            out = build_voice_clone_prompt(processor, text, ref_wav, int(ref_sr), args.ref_text)
            prompt = {
                "prompt_token_ids": out["prompt_token_ids"],
                # Bare tensors (NOT list-wrapped): the msgspec serializer in
                # vllm_omni.data_entry_keys routes torch.Tensor → tensor_data
                # and list[Tensor] → list_data (which silently strips tensors).
                "additional_information": {
                    "audio_input_ids": out["audio_input_ids"],
                    "audio_input_ids_mask": out["audio_input_ids_mask"],
                },
            }
        else:
            inputs = build_plain_text_prompt(processor, text)
            prompt = {"prompt_token_ids": input_ids_to_python_list(inputs)}
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
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
