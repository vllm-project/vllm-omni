# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline end-to-end inference example for NeuTTS-Air."""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from vllm import SamplingParams

from vllm_omni import Omni

OUTPUT_SAMPLE_RATE = 24_000
SPEECH_GENERATION_END_TOKEN_ID = 151670


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuTTS-Air E2E TTS inference")
    parser.add_argument("--model", default="neuphonic/neutts-air")
    parser.add_argument("--text", required=True, help="Target text to synthesize.")
    parser.add_argument("--ref-audio", required=True, help="Reference voice audio path.")
    parser.add_argument(
        "--ref-text",
        required=True,
        help="Transcript of the reference audio.",
    )
    parser.add_argument("--output", default="outputs/neutts_air.wav")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def load_reference_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_audio, reference_sample_rate = load_reference_audio(args.ref_audio)
    request = {
        "prompt": args.text,
        "multi_modal_data": {
            "audio": (reference_audio, reference_sample_rate),
        },
        "mm_processor_kwargs": {
            "ref_text": args.ref_text,
        },
    }

    stage0_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        min_tokens=50,
        seed=args.seed,
        detokenize=False,
        stop_token_ids=[SPEECH_GENERATION_END_TOKEN_ID],
        ignore_eos=True,
    )
    stage1_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        seed=args.seed,
        detokenize=False,
    )

    omni = None
    try:
        omni = Omni(model=args.model, trust_remote_code=True, log_stats=False)
        started_at = time.perf_counter()
        outputs = omni.generate([request], [stage0_params, stage1_params])
        elapsed = time.perf_counter() - started_at

        audio_array = None
        for stage_output in outputs:
            print("final output type:", stage_output.final_output_type)
            if stage_output.final_output_type != "audio":
                continue
            multimodal_output = stage_output.request_output.outputs[0].multimodal_output
            audio = multimodal_output.get("audio")
            if audio is None:
                raise RuntimeError("NeuTTS-Air returned an empty audio output.")
            if isinstance(audio, torch.Tensor):
                audio = audio.float().detach().cpu().numpy()
            audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio_array is None or audio_array.size == 0:
            raise RuntimeError("NeuTTS-Air produced no final audio samples.")
        if not np.isfinite(audio_array).all():
            raise RuntimeError("NeuTTS-Air produced NaN or infinite audio samples.")

        sf.write(output_path, audio_array, OUTPUT_SAMPLE_RATE, format="WAV")

        peak = float(np.max(np.abs(audio_array)))
        rms = float(np.sqrt(np.mean(audio_array**2)))
        clipped_ratio = float(np.mean(np.abs(audio_array) >= 0.999))
        print("sample rate:", OUTPUT_SAMPLE_RATE)
        print("duration:", audio_array.size / OUTPUT_SAMPLE_RATE)
        print("latency:", elapsed)
        print("max abs:", peak)
        print("rms:", rms)
        print("clipped ratio:", clipped_ratio)
        print("saved:", output_path)
    finally:
        if omni is not None:
            omni.close()


if __name__ == "__main__":
    main()
