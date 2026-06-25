# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for IndexTTS2 via vLLM-Omni.

Two-stage pipeline: GPT AR (Stage 0) → S2Mel + BigVGAN (Stage 1).
Output is 22050 Hz mono WAV.

Usage:
  python end2end.py \
    --model /path/to/IndexTeam/IndexTTS-2 \
    --text "你好，这是一个语音合成测试。" \
    --ref-audio /path/to/ref.wav

  # With emotion audio:
  python end2end.py \
    --model /path/to/IndexTeam/IndexTTS-2 \
    --text "今天天气真好！" \
    --ref-audio /path/to/ref.wav \
    --emo-audio /path/to/happy.wav
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402
from vllm_omni.model_executor.models.indextts2.prompt_utils import (  # noqa: E402
    build_indextts2_prefill_prompt_ids,
)


def build_request(
    model: str,
    text: str,
    ref_audio_path: str | None = None,
    emo_audio_path: str | None = None,
    emo_text: str | None = None,
    emo_vector: list[float] | None = None,
    emo_alpha: float | None = None,
    use_emo_text: bool = False,
    use_random: bool = False,
) -> dict:
    additional: dict = {"text": [text]}
    if ref_audio_path:
        additional["voice"] = [str(ref_audio_path)]
    if emo_audio_path:
        additional["emo_audio"] = [str(emo_audio_path)]
    if emo_text:
        additional["emo_text"] = [emo_text]
    if emo_vector is not None:
        additional["emo_vector"] = [emo_vector]
    if emo_alpha is not None:
        additional["emo_alpha"] = [emo_alpha]
    if use_emo_text:
        additional["use_emo_text"] = [True]
    if use_random:
        additional["use_random"] = [True]
    return {
        "prompt_token_ids": build_indextts2_prefill_prompt_ids(model, text),
        "additional_information": additional,
    }


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 22050) -> None:
    audio_np = waveform.float().numpy()
    sf.write(path, audio_np, sample_rate)
    print(f"  Saved {path} ({audio_np.shape}, {sample_rate} Hz)")


def extract_audio(mm: dict) -> tuple[torch.Tensor | None, int]:
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        chunks = [chunk.reshape(-1) for chunk in audio if isinstance(chunk, torch.Tensor) and chunk.numel() > 0]
        audio = torch.cat(chunks, dim=0) if chunks else None

    sr_val = mm.get("sr")
    if isinstance(sr_val, list):
        sr_val = sr_val[-1] if sr_val else None
    if hasattr(sr_val, "item"):
        sample_rate = int(sr_val.item())
    else:
        sample_rate = int(sr_val) if sr_val is not None else 22050

    return audio if isinstance(audio, torch.Tensor) else None, sample_rate


def main(args) -> None:
    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    # Stage 0: GPT AR (text → mel codes)
    gpt_sampling = SamplingParams(
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        max_tokens=1500,
        repetition_penalty=10.0,
        stop_token_ids=[8193],
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )
    # Stage 1: S2Mel + BigVGAN (non-AR, params mostly ignored)
    s2mel_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=65536,
        detokenize=True,
    )
    sampling_params = [gpt_sampling, s2mel_sampling]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing: {args.text!r}")
    if args.ref_audio:
        print(f"  ref_audio: {args.ref_audio}")
    inputs = build_request(
        model=args.model,
        text=args.text,
        ref_audio_path=args.ref_audio,
        emo_audio_path=args.emo_audio,
        emo_text=args.emo_text,
        emo_vector=args.emo_vector,
        emo_alpha=args.emo_alpha,
        use_emo_text=args.use_emo_text,
        use_random=args.use_random,
    )

    for i, omni_out in enumerate(omni.generate(inputs, sampling_params_list=sampling_params)):
        mm = omni_out.multimodal_output
        if not mm:
            print(f"  [req {i}] No multimodal output.")
            continue
        audio, sr = extract_audio(mm)
        if audio is None:
            print(f"  [req {i}] No waveform in multimodal_output.")
            continue
        out_path = str(output_dir / f"output_{i}.wav")
        save_audio(audio.cpu(), out_path, sr)

    print("Done.")


def parse_args():
    parser = FlexibleArgumentParser(description="IndexTTS2 offline inference")
    parser.add_argument("--model", required=True, help="HF model path for IndexTTS2.")
    parser.add_argument("--text", default="你好，这是IndexTTS2语音合成测试。")
    parser.add_argument("--ref-audio", required=True, help="Reference audio for voice cloning.")
    parser.add_argument("--emo-audio", default=None, help="Emotion reference audio.")
    parser.add_argument("--emo-text", default=None, help="Emotion description text.")
    parser.add_argument(
        "--emo-vector",
        type=float,
        nargs=8,
        default=None,
        help="8-dim emotion vector: happy angry sad afraid disgusted melancholic surprised calm.",
    )
    parser.add_argument("--emo-alpha", type=float, default=None, help="Emotion weight in [0, 1].")
    parser.add_argument("--use-emo-text", action="store_true", help="Infer emotion vector from emo-text or text.")
    parser.add_argument("--use-random", action="store_true", help="Use random emotion prototypes.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "indextts2_output",
        ),
    )
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
