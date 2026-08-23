# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for IndexTTS 2.0 and 2.5 via vLLM-Omni.

Two-stage pipeline: GPT AR (Stage 0) → semantic codec + S2Mel + BigVGAN
(Stage 1).
Output is 22050 Hz mono WAV.

Usage:
  python end2end.py \
    --model /path/to/indextts-2.5 \
    --model-version 2.5 \
    --lang zh \
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
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str | None = None,
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
    if model_type == "indextts2_5":
        additional["lang"] = [lang]
        additional["text_normalization"] = [text_normalization]
    prompt_kwargs = {
        "model_type": model_type,
        "lang": lang,
        "text_normalization": text_normalization,
    }
    if tokenizer_file is not None:
        prompt_kwargs["tokenizer_file"] = tokenizer_file
    return {
        "prompt_token_ids": build_indextts2_prefill_prompt_ids(
            model,
            text,
            **prompt_kwargs,
        ),
        "additional_information": additional,
    }


def _get_stage0_tokenizer_file(omni: Omni) -> str | None:
    """Read the effective tokenizer override from the resolved Stage 0 config."""
    stage_configs = omni.stage_configs
    if not stage_configs:
        return None
    engine_args = getattr(stage_configs[0], "engine_args", None)
    if engine_args is None:
        return None
    hf_overrides = getattr(engine_args, "hf_overrides", None)
    if hf_overrides is None and isinstance(engine_args, dict):
        hf_overrides = engine_args.get("hf_overrides")
    if hf_overrides is None:
        return None
    if hasattr(hf_overrides, "get"):
        return hf_overrides.get("tokenizer_file")
    return getattr(hf_overrides, "tokenizer_file", None)


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
    model_type = "indextts2_5" if args.model_version == "2.5" else "indextts2"
    deploy_config = args.deploy_config
    if deploy_config is None and args.model_version == "2.5":
        repo_root = Path(__file__).resolve().parents[4]
        deploy_config = str(repo_root / "vllm_omni" / "deploy" / "indextts2_5.yaml")
    omni = Omni(
        model=args.model,
        deploy_config=deploy_config,
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
    request_kwargs = dict(
        model=args.model,
        text=args.text,
        ref_audio_path=args.ref_audio,
        emo_audio_path=args.emo_audio,
        emo_text=args.emo_text,
        emo_vector=args.emo_vector,
        emo_alpha=args.emo_alpha,
        use_emo_text=args.use_emo_text,
        use_random=args.use_random,
        model_type=model_type,
        lang=args.lang,
        text_normalization=args.text_normalization,
    )
    if model_type == "indextts2_5":
        tokenizer_file = _get_stage0_tokenizer_file(omni)
        if tokenizer_file is not None:
            request_kwargs["tokenizer_file"] = tokenizer_file
    inputs = build_request(**request_kwargs)
    if model_type == "indextts2_5":
        inputs["additional_information"]["duration_factor"] = [1.0 / getattr(args, "speed", 1.0)]

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
    parser = FlexibleArgumentParser(description="IndexTTS 2.0/2.5 offline inference")
    parser.add_argument(
        "--model",
        required=True,
        help="HF model path or native bundle containing checkpoints/.",
    )
    parser.add_argument(
        "--model-version",
        choices=("2.0", "2.5"),
        default="2.0",
    )
    parser.add_argument(
        "--lang",
        default="zh",
        help=(
            "IndexTTS 2.5 language code, for example zh/en/zhen/ja/yue; "
            "zhen is mixed Chinese/English and Mandarin is a vLLM-Omni alias for zh."
        ),
    )
    parser.add_argument(
        "--no-text-normalization",
        action="store_false",
        dest="text_normalization",
        help="Disable IndexTTS 2.5 text normalization.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="IndexTTS 2.5 native synthesis speed in [0.5, 2.0]; 2.0 is faster.",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed Stage 0 AR sampling and per-request CFM noise; changing the "
            "concurrent batch composition may still change the final waveform."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "indextts2_output",
        ),
    )
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.model_version == "2.5" and not 0.5 <= args.speed <= 2.0:
        parser.error("IndexTTS 2.5 --speed must be between 0.5 and 2.0")
    return args


if __name__ == "__main__":
    main(parse_args())
