# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for Miso TTS via vLLM-Omni.

Two-stage pipeline: talker (8B + 300M frame AR) → Mimi decode at 24 kHz.

Prerequisites::

    pip install moshi safetensors

If the Hugging Face repo has no ``config.json``, the deploy YAML supplies
``hf_overrides`` (see ``vllm_omni/deploy/miso_tts.yaml``).

Usage::

  python end2end.py --text "Hello from Miso TTS." --speaker 0 --output out.wav
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("NO_TORCH_COMPILE", "1")

from vllm_omni import Omni  # noqa: E402

MODEL = "MisoLabs/MisoTTS"


def build_request(
    text: str,
    speaker: int = 0,
    max_generation_frames: int = 125,
    temperature: float = 0.9,
    topk: int = 50,
    seed: int | None = None,
) -> dict:
    additional: dict = {
        "text": [text],
        "speaker": [speaker],
        "max_generation_frames": [max_generation_frames],
        "temperature": [temperature],
        "topk": [topk],
    }
    if seed is not None:
        additional["seed"] = [seed]
    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": additional,
    }


def main() -> None:
    parser = FlexibleArgumentParser(description="Miso TTS offline inference")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--output", type=str, default="miso_tts_out.wav")
    parser.add_argument("--max-generation-frames", type=int, default=125)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Path to deploy YAML; unset auto-loads vllm_omni/deploy/miso_tts.yaml.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    args = parser.parse_args()

    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=4096,
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )

    req = build_request(
        text=args.text,
        speaker=args.speaker,
        max_generation_frames=args.max_generation_frames,
        temperature=args.temperature,
        topk=args.top_k,
        seed=args.seed,
    )

    outputs = omni.generate([req], sampling_params=sampling)
    out = outputs[0]
    mm = out.outputs[0].multimodal_output
    audio = mm.get("model_outputs")
    if audio is None:
        audio = mm.get("audio")
    if audio is None:
        raise RuntimeError("No audio in multimodal_output")
    if isinstance(audio, list):
        tensors = [t.reshape(-1) for t in audio if t is not None and t.numel() > 0]
        if not tensors:
            raise RuntimeError("Empty audio list")
        waveform = torch.cat(tensors, dim=0)
    else:
        waveform = audio.reshape(-1)

    sr_raw = mm.get("sr")
    if sr_raw is None:
        sample_rate = 24000
    else:
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sample_rate = int(sr_val.item()) if hasattr(sr_val, "item") else int(sr_val)

    wav_np = waveform.detach().cpu().float().numpy()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav_np, sample_rate)
    print(f"Wrote {out_path} ({len(wav_np) / sample_rate:.2f}s @ {sample_rate} Hz)")


if __name__ == "__main__":
    main()
