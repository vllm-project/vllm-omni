# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end Audio-Omni offline example: TTS and voice cloning.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import soundfile
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.processors.audio_omni import postprocess_tts_output

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audio-Omni offline TTS / voice cloning.")
    p.add_argument("--model", default="HKUSTAudio/Audio-Omni", help="Local Audio-Omni bundle dir (or HF id).")
    p.add_argument("--mode", default="tts", choices=("tts", "voice_clone"))
    p.add_argument("--prompt", required=True, help="Transcript to synthesize.")
    p.add_argument("--voice-prompt", default="", help="Reference voice WAV (voice_clone).")
    p.add_argument("--voice-ref-text", default="", help="Transcript of the reference WAV.")
    p.add_argument("--output-dir", default=str(ROOT / "audio_omni_outputs"))
    p.add_argument("--num-inference-steps", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-postprocess", action="store_true", help="Skip gradio-style head-cut + silence trim.")
    return p.parse_args()


def ensure_engine_config(model: str) -> None:
    if not os.path.isdir(model):
        return
    config_json = os.path.join(model, "config.json")
    bundle_json = os.path.join(model, "Audio-Omni.json")
    if os.path.exists(config_json) or not os.path.exists(bundle_json):
        return
    with open(bundle_json, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("architectures", ["AudioOmniPipeline"])
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def save_wav(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), audio.clamp(-1.0, 1.0).cpu().T.numpy(), sample_rate, subtype="PCM_16")


def main() -> None:
    args = parse_args()
    if args.mode == "voice_clone" and not args.voice_prompt:
        raise SystemExit("--mode voice_clone requires --voice-prompt (and --voice-ref-text).")

    ensure_engine_config(args.model)
    omni = Omni(model=args.model, model_class_name="AudioOmniPipeline")

    extra: dict = {"task": "tts"}
    if args.voice_prompt:
        extra["voice_prompt_path"] = args.voice_prompt
        extra["voice_ref_text"] = args.voice_ref_text

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    t0 = time.perf_counter()
    outputs = omni.generate(
        args.prompt,
        OmniDiffusionSamplingParams(
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            extra_args=extra,
        ),
    )
    mm_output = outputs[0].request_output.multimodal_output
    audio = mm_output.get("audio")
    if audio is None:
        raise RuntimeError("No audio produced.")
    audio = torch.as_tensor(audio).detach().cpu().float()
    if audio.ndim == 3:
        audio = audio[0]
    sample_rate = int(mm_output.get("audio_sample_rate") or 44100)
    elapsed = time.perf_counter() - t0

    audio = audio / audio.abs().max().clamp(min=1e-8)

    out_dir = Path(args.output_dir)
    save_wav(audio, out_dir / f"{args.mode}_raw.wav", sample_rate)

    if not args.no_postprocess:
        ref_duration = 0.0
        if args.voice_prompt:
            info = soundfile.info(args.voice_prompt)
            ref_duration = min(info.frames / info.samplerate, 6.0)
        final = postprocess_tts_output(audio, sample_rate, voice_ref_duration=ref_duration)
    else:
        final = audio
    save_wav(final, out_dir / f"{args.mode}.wav", sample_rate)
    print(
        f"[{args.mode}] saved {out_dir / (args.mode + '.wav')} "
        f"({audio.shape[-1] / sample_rate:.2f}s raw -> {final.shape[-1] / sample_rate:.2f}s, {elapsed:.2f}s)"
    )

    omni.close()


if __name__ == "__main__":
    main()
