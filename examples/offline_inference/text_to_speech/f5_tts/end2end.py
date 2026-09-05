# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline inference example for F5-TTS via vLLM-Omni.

F5-TTS is a flow-matching DiT that generates mel spectrograms from text
and decodes them with a vocoder. The entire pipeline runs in a single
diffusion stage.

Usage:
  # Basic (uses built-in F5-TTS test reference audio)
  python end2end.py --text "Hello, this is a test."

  # Voice cloning with local reference audio
  python end2end.py --text "Hello world." \
    --ref-audio /path/to/ref.wav --ref-text "Transcript of ref."

  # With cache acceleration (Cache-DiT recommended; TeaCache is not
  # supported for F5-TTS — step-level caching is incompatible with
  # flow-matching + sway sampling)
  python end2end.py --text "Hello world." --cache-backend cache_dit

  # With profiling
  python end2end.py --text "Hello world." --enable-profiler
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni.diffusion.data import DiffusionParallelConfig  # noqa: E402
from vllm_omni.entrypoints.omni import Omni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402

MODEL = "SWivid/F5-TTS/F5TTS_v1_Base"
REF_AUDIO_URL = (
    "https://raw.githubusercontent.com/SWivid/F5-TTS/main/"
    "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
)
REF_TEXT = "Some call me nature, others call me mother nature."


def save_audio(audio_data: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    try:
        import soundfile as sf
        sf.write(path, audio_data, sample_rate)
    except ImportError:
        import scipy.io.wavfile as wav
        if audio_data.dtype in (np.float32, np.float64):
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
        wav.write(path, sample_rate, audio_data)


def load_ref_audio(ref_audio_path: str | None) -> bytes | str:
    if ref_audio_path is None:
        return REF_AUDIO_URL
    p = Path(ref_audio_path)
    if p.exists():
        return p.read_bytes()
    return ref_audio_path


def main():
    args = parse_args()

    ref_audio = load_ref_audio(args.ref_audio)
    ref_text = args.ref_text or REF_TEXT

    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            "Fn_compute_blocks": args.cache_dit_fn,
            "Bn_compute_blocks": args.cache_dit_bn,
            "max_warmup_steps": args.cache_dit_warmup,
        }

    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        cfg_parallel_size=args.cfg_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
    )

    print(f"\n{'=' * 60}")
    print("F5-TTS - Text-to-Speech Generation (Offline)")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Text: {args.text}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  CFG: {args.guidance_scale}")
    print(f"  Cache: {args.cache_backend or 'None'}")
    print(f"{'=' * 60}\n")

    omni = Omni(
        model=args.model,
        parallel_config=parallel_config,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_diffusion_pipeline_profiler=args.enable_profiler,
        enable_layerwise_offload=args.enable_layerwise_offload,
    )

    prompts = {
        "prompt": args.text,
        "additional_information": {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "lang": args.lang,
        },
    }

    sampling_params = OmniDiffusionSamplingParams(
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args={
            "sway_sampling_coef": args.sway_sampling_coef,
            "speed": args.speed,
        },
    )
    # Without this flag the pipeline ignores guidance_scale and falls back
    # to the 2.0 default (same convention as the hunyuan_image3 example).
    sampling_params.guidance_scale_provided = True

    t0 = time.perf_counter()
    outputs = omni.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    print(f"Generation time: {elapsed:.2f}s")

    if not outputs:
        raise ValueError("No output generated.")

    output = outputs[0]
    mm = output.request_output.multimodal_output if hasattr(output, "request_output") else None
    if mm is None:
        raise ValueError("No multimodal output.")

    audio = mm.get("audio")
    if audio is None:
        raise ValueError("No audio in output.")

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()

    sr = mm.get("sr", 24000)
    if hasattr(sr, "item"):
        sr = int(sr.item())
    elif isinstance(sr, list):
        sr = int(sr[-1].item() if hasattr(sr[-1], "item") else sr[-1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "f5_tts_output.wav")
    save_audio(audio, out_path, sr)
    print(f"Saved: {out_path} ({len(audio) / sr:.2f}s @ {sr}Hz)")


def parse_args():
    parser = FlexibleArgumentParser(description="F5-TTS offline inference")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--text", default=(
        "I don't really care what you call me. "
        "I've been a silent spectator, watching species evolve, "
        "empires rise and fall. But always, I am here."
    ))
    parser.add_argument("--ref-audio", default=None, help="Path/URL to reference audio.")
    parser.add_argument("--ref-text", default=None, help="Transcript of reference audio.")
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="f5_tts_output")
    parser.add_argument("--cache-backend", default=None, choices=["cache_dit"])
    parser.add_argument("--cache-dit-fn", type=int, default=1)
    parser.add_argument("--cache-dit-bn", type=int, default=0)
    parser.add_argument("--cache-dit-warmup", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--use-hsdp", action="store_true")
    parser.add_argument("--hsdp-shard-size", type=int, default=1)
    parser.add_argument("--enable-profiler", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
