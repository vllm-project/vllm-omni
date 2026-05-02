# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for MOSS-TTS-Nano via vLLM-Omni.

Single-stage pipeline: the 0.1B AR LM and MOSS-Audio-Tokenizer-Nano codec
both run inside one generation stage.  Output is 48 kHz stereo WAV.

Supports:
  - Built-in voice presets (--voice)
  - Custom voice cloning via reference audio (--prompt-audio)
  - Batch synthesis of multiple texts
  - Reproducible output via --seed

Usage examples:
  # Built-in voice
  python end2end.py --text "Hello from MOSS-TTS-Nano."

  # Voice clone with reference audio
  python end2end.py --text "Hello!" --prompt-audio /path/to/ref.wav

  # Batch with different voices
  python end2end.py --batch --output-dir /tmp/moss_output
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Prevent multiprocessing from re-importing CUDA in the wrong context.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402

MODEL = "OpenMOSS-Team/MOSS-TTS-Nano"

BATCH_SAMPLES = [
    {"text": "Hello, this is a test of MOSS-TTS-Nano.", "voice": "Ava", "label": "en_hello"},
    {"text": "你好，这是 MOSS-TTS-Nano 的语音合成测试。", "voice": "Junhao", "label": "zh_hello"},
    {"text": "Bonjour, ceci est un test de synthèse vocale.", "voice": "Bella", "label": "fr_hello"},
]


def build_request(
    text: str,
    voice: str = "Junhao",
    mode: str = "voice_clone",
    prompt_audio_path: str | None = None,
    prompt_text: str | None = None,
    max_new_frames: int = 375,
    seed: int | None = None,
    audio_temperature: float = 0.8,
    audio_top_k: int = 25,
    audio_top_p: float = 0.95,
    text_temperature: float = 1.0,
) -> dict:
    """Build an Omni request payload for MOSS-TTS-Nano."""
    additional: dict = {
        "text": [text],
        "voice": [voice],
        "mode": [mode],
        "max_new_frames": [max_new_frames],
        "audio_temperature": [audio_temperature],
        "audio_top_k": [audio_top_k],
        "audio_top_p": [audio_top_p],
        "text_temperature": [text_temperature],
    }
    if prompt_audio_path:
        additional["prompt_audio_path"] = [str(prompt_audio_path)]
    if prompt_text:
        additional["prompt_text"] = [prompt_text]
    if seed is not None:
        additional["seed"] = [seed]

    return {
        "prompt": "<|im_start|>assistant\n",  # minimal placeholder prompt
        "additional_information": additional,
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 48000) -> None:
    audio_np = waveform.float().numpy()
    # Reshape stereo: inference_stream yields interleaved [samples*2]; reshape to [samples, 2]
    if audio_np.ndim == 1 and audio_np.shape[0] % 2 == 0:
        audio_np = audio_np.reshape(-1, 2)
    sf.write(path, audio_np, sample_rate)
    print(f"  Saved {path} ({audio_np.shape}, {sample_rate} Hz)")


def main(args) -> None:
    omni = Omni(
        model=MODEL,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        print(f"Running batch synthesis ({len(BATCH_SAMPLES)} samples)...")
        inputs = [build_request(s["text"], voice=s["voice"], seed=args.seed) for s in BATCH_SAMPLES]
        params_list = [sampling_params] * len(inputs)
    else:
        print(f"Synthesizing: {args.text!r}")
        inputs = build_request(
            text=args.text,
            voice=args.voice,
            mode=args.mode,
            prompt_audio_path=args.prompt_audio,
            prompt_text=args.prompt_text,
            max_new_frames=args.max_new_frames,
            seed=args.seed,
            audio_temperature=args.audio_temperature,
            audio_top_k=args.audio_top_k,
            audio_top_p=args.audio_top_p,
            text_temperature=args.text_temperature,
        )
        params_list = sampling_params

    for stage_outputs in omni.generate(inputs, params_list):
        for i, req_output in enumerate(stage_outputs.request_output):
            for j, out in enumerate(req_output.outputs):
                mm = out.multimodal_output
                if mm is None:
                    print(f"  [req {i}] No audio output.")
                    continue
                audio = mm.get("audio")
                sr_tensor = mm.get("sr")
                if audio is None:
                    print(f"  [req {i}] No waveform in multimodal_output.")
                    continue
                sr = int(sr_tensor.item()) if sr_tensor is not None else 48000
                label = BATCH_SAMPLES[i]["label"] if args.batch else f"output_{i}_{j}"
                out_path = str(output_dir / f"{label}.wav")
                save_audio(audio.cpu(), out_path, sr)

    print("Done.")


def parse_args():
    parser = FlexibleArgumentParser(description="MOSS-TTS-Nano offline inference")
    parser.add_argument("--text", default="Hello, this is MOSS-TTS-Nano speaking.", help="Text to synthesize.")
    parser.add_argument("--voice", default="Junhao", help="Built-in voice preset name.")
    parser.add_argument("--mode", default="voice_clone", choices=["voice_clone", "continuation"])
    parser.add_argument("--prompt-audio", default=None, help="Path to reference audio for voice cloning.")
    parser.add_argument("--prompt-text", default=None, help="Reference transcript (continuation mode).")
    parser.add_argument("--max-new-frames", type=int, default=375, help="Max AR frames (~14s at default).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--batch", action="store_true", help="Run built-in batch of diverse samples.")
    parser.add_argument("--output-dir", default="/tmp/moss_tts_nano_output", help="Directory for WAV outputs.")
    parser.add_argument(
        "--deploy-config",
        default=None,
        help="Path to a deploy YAML; leave unset to auto-load vllm_omni/deploy/moss_tts_nano.yaml.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
