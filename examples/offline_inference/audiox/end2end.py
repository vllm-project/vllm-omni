# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end AudioX offline example covering the 6 t2*/v2*/tv2* tasks.

Provide a directory with the **vLLM-Omni AudioX safetensors bundle** (e.g. from
``zhangj1an/AudioX`` on Hugging Face)::

    huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
    python end2end.py --model ./audiox_weights
    python end2end.py --model ./audiox_weights --tasks t2a tv2a
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile
import torch
import torchaudio.functional as TF

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

ROOT = Path(__file__).resolve().parent

SAMPLE_PROMPTS: dict[str, str] = {
    "t2a": "Fireworks burst twice, followed by a period of silence before a clock begins ticking.",
    "t2m": "Uplifting ukulele tune for a travel vlog",
    "v2a": "",
    "v2m": "",
    "tv2a": "drum beating sound and human talking",
    "tv2m": "uplifting music matching the scene",
}

ALL_TASKS = ("t2a", "t2m", "v2a", "v2m", "tv2a", "tv2m")
VIDEO_TASKS = frozenset({"v2a", "v2m", "tv2a", "tv2m"})
TEXT_TASKS = frozenset({"t2a", "t2m", "tv2a", "tv2m"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AudioX offline end-to-end (6 t2*/v2*/tv2* tasks).")
    p.add_argument("--model", default="zhangj1an/AudioX", help="HF id or local AudioX bundle path.")
    p.add_argument("--tasks", nargs="+", default=list(ALL_TASKS), choices=ALL_TASKS)
    p.add_argument("--video", default="", help="Video path / URL (required for v2*/tv2*).")
    p.add_argument("--reference-audio", default="", help="Optional audio prompt for audio-conditioned generation.")
    p.add_argument("--output-dir", default=str(ROOT / "audiox_task_outputs"))
    p.add_argument("--num-inference-steps", type=int, default=250)
    p.add_argument("--seconds-total", type=float, default=10.0)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rate", type=int, default=48000, help="Output WAV rate (resampled if != model rate).")
    return p.parse_args()


def save_wav(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    """Write 16-bit PCM WAV. ``audio`` is ``[channels, samples]`` float in [-1, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), audio.clamp(-1.0, 1.0).cpu().T.numpy(), sample_rate, subtype="PCM_16")


def main() -> None:
    args = parse_args()

    omni = Omni(model=args.model, model_class_name="AudioXPipeline")

    for task in args.tasks:
        if task in VIDEO_TASKS and not args.video:
            raise SystemExit(f"task={task!r} requires --video")
        prompt = SAMPLE_PROMPTS[task] if task in TEXT_TASKS else ""
        extra: dict = {"audiox_task": task, "seconds_start": 0.0, "seconds_total": float(args.seconds_total)}
        if task in VIDEO_TASKS:
            extra["video_path"] = args.video
        if args.reference_audio:
            extra["audio_path"] = args.reference_audio

        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
        t0 = time.perf_counter()
        outputs = omni.generate(
            prompt,
            OmniDiffusionSamplingParams(
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                extra_args=extra,
            ),
        )
        audio = outputs[0].request_output.multimodal_output.get("audio")
        if audio is None:
            raise RuntimeError(f"No audio produced for task {task!r}")
        audio = torch.as_tensor(audio).detach().cpu().float()
        if audio.ndim == 3:
            audio = audio[0]

        model_sr = int(outputs[0].request_output.multimodal_output.get("audio_sample_rate") or 44100)
        if model_sr != args.sample_rate:
            audio = TF.resample(audio, model_sr, args.sample_rate)

        out_path = Path(args.output_dir) / f"{task}.wav"
        save_wav(audio, out_path, args.sample_rate)
        print(f"[{task}] saved {out_path} ({time.perf_counter() - t0:.2f}s)")

    omni.close()


if __name__ == "__main__":
    main()
