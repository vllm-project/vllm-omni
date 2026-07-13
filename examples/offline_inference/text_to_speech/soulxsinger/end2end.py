"""Offline SoulX-Singer SVS / SVC: single-stage DiT (preprocess inline).

Pass raw prompt/target audio; vLLM-Omni runs integrated preprocess then inference.

Usage:
    # SVS (default)
    python end2end.py --model /path/to/SoulX-Singer \\
        --prompt-audio path/to/prompt/audio.mp3 \\
        --target-audio path/to/target/audio.mp3 \\
        --preprocess-weights-dir /path/to/SoulX-Singer-Preprocess \\
        -o output.wav

    # SVC (config.json must declare SoulXSingerSVCPipeline; --model points at SVC view)
    python end2end.py --model /path/to/SoulX-Singer-svc \\
        --prompt-audio path/to/prompt/audio.mp3 \\
        --target-audio path/to/target/audio.mp3 \\
        --preprocess-weights-dir /path/to/SoulX-Singer-Preprocess \\
        -o output_svc.wav
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from vllm_omni.diffusion.models.soulx_singer.utils import resolve_soulx_kind, validate_soulx_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

REPO_ROOT = Path(__file__).resolve().parents[4]
SVS_DEPLOY_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "soulxsinger_svs.yaml"
SVC_DEPLOY_CONFIG = REPO_ROOT / "vllm_omni" / "deploy" / "soulxsinger_svc.yaml"
DEFAULT_PROMPT_AUDIO = REPO_ROOT / "tests" / "assets" / "soulxsinger" / "zh_prompt.mp3"
DEFAULT_TARGET_AUDIO = REPO_ROOT / "tests" / "assets" / "soulxsinger" / "music.mp3"
_SAMPLE_RATE = 24000


def _require_paths(paths: dict[str, str | None]) -> None:
    missing = [name for name, p in paths.items() if not p or not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Missing file(s): " + ", ".join(f"{k}={paths[k]!r}" for k in missing))


def resolve_preprocess_weights_dir(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "rmvpe" / "rmvpe.pt").is_file():
            raise FileNotFoundError(f"--preprocess-weights-dir must contain rmvpe/rmvpe.pt: {path}")
        return str(path)
    env = os.environ.get("SOULX_PREPROCESS_WEIGHTS_DIR")
    if env:
        path = Path(env).expanduser().resolve()
        if (path / "rmvpe" / "rmvpe.pt").is_file():
            return str(path)
    raise FileNotFoundError(
        "SoulX preprocess weights not found. Pass --preprocess-weights-dir or set "
        "SOULX_PREPROCESS_WEIGHTS_DIR to a directory containing rmvpe/rmvpe.pt "
        "(huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess)."
    )


def build_sampling(
    args: argparse.Namespace,
    *,
    preprocess_weights_dir: str,
    kind: str,
) -> OmniDiffusionSamplingParams:
    extra: dict = {
        "prompt_audio": os.path.abspath(args.prompt_audio),
        "target_audio": os.path.abspath(args.target_audio),
        "preprocess_weights_dir": preprocess_weights_dir,
        "vocal_sep": args.vocal_sep,
        "auto_shift": args.auto_shift,
        "pitch_shift": args.pitch_shift,
    }
    if kind == "svs":
        extra["language"] = args.language
        extra["control"] = args.control
        if args.prompt_metadata_path:
            extra["prompt_metadata_path"] = os.path.abspath(args.prompt_metadata_path)
        if args.target_metadata_path:
            extra["target_metadata_path"] = os.path.abspath(args.target_metadata_path)
        if args.prompt_metadata_path or args.target_metadata_path:
            extra["audio_path"] = os.path.abspath(args.prompt_wav_path or args.prompt_audio)

    extra = validate_soulx_extra_args(kind, extra)

    return OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        extra_args=extra,
    )


def extract_audio(outputs) -> tuple[np.ndarray, int]:
    out = outputs[0]
    mm = getattr(out, "multimodal_output", None)
    if mm is None:
        ro = getattr(out, "request_output", None)
        if ro is not None:
            mm = getattr(ro, "multimodal_output", None)
            if mm is None and ro.outputs:
                mm = getattr(ro.outputs[0], "multimodal_output", None)
    if not mm or "audio" not in mm:
        raise RuntimeError("No audio in multimodal_output")

    audio_val = mm["audio"]
    if isinstance(audio_val, list):
        chunks = [
            np.asarray(chunk.detach().cpu().float().numpy() if hasattr(chunk, "detach") else chunk).reshape(-1)
            for chunk in audio_val
            if chunk is not None
        ]
        audio_np = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float32)
    elif hasattr(audio_val, "cpu"):
        audio_np = audio_val.detach().cpu().numpy().squeeze()
    else:
        audio_np = np.asarray(audio_val, dtype=np.float32).squeeze()

    sr_val = mm.get("sr", mm.get("audio_sample_rate", _SAMPLE_RATE))
    if isinstance(sr_val, list) and sr_val:
        sr_val = sr_val[-1]
    if hasattr(sr_val, "item"):
        sr_val = sr_val.item()
    return audio_np.astype(np.float32, copy=False), int(sr_val)


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--svc",
        action="store_true",
        help="Assert SVC mode (must match config.json architectures)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="SoulX-Singer weight directory ($BASE for SVS, $SVC_DIR for SVC)",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override deploy YAML (default: soulxsinger_svs.yaml or soulxsinger_svc.yaml)",
    )
    parser.add_argument(
        "--prompt-audio",
        type=str,
        default=str(DEFAULT_PROMPT_AUDIO),
        help="Prompt / reference audio (mp3 or wav)",
    )
    parser.add_argument(
        "--target-audio",
        type=str,
        default=str(DEFAULT_TARGET_AUDIO),
        help="Target content audio (mp3 or wav)",
    )
    parser.add_argument(
        "--prompt-metadata-path",
        type=str,
        default=None,
        help="SVS precomputed prompt metadata.json (skips online ASR/ROSVOT for prompt)",
    )
    parser.add_argument(
        "--target-metadata-path",
        type=str,
        default=None,
        help="SVS precomputed target metadata.json (skips online ASR/ROSVOT for target)",
    )
    parser.add_argument(
        "--prompt-wav-path",
        type=str,
        default=None,
        help="SVS prompt vocal wav for metadata processor when using --prompt-metadata-path (default: --prompt-audio)",
    )
    parser.add_argument(
        "--preprocess-weights-dir",
        type=str,
        default=None,
        help="SoulX-Singer-Preprocess weights (or set SOULX_PREPROCESS_WEIGHTS_DIR)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Mandarin",
        help="SVS lyrics language (Mandarin, Cantonese, English)",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="melody",
        choices=["score", "melody"],
        help="SVS control mode",
    )
    parser.add_argument(
        "--vocal-sep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run vocal separation during preprocess",
    )
    parser.add_argument(
        "--auto-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto pitch shift to align target pitch with prompt",
    )
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional CFM RNG seed. Omit for non-deterministic sampling.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoulX-Singer offline SVS / SVC (preprocess + DiT)")
    add_inference_args(parser)
    parser.add_argument("--output", "-o", type=str, default="output.wav")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    kind = resolve_soulx_kind(args.model)

    if not (args.svc ^ (kind == "svs")):
        raise ValueError(f"mode mismatch: {args.model}/config.json (declares {kind!r}), --svc={args.svc}")

    required = {
        "prompt_audio": args.prompt_audio,
        "target_audio": args.target_audio,
    }
    if kind == "svs" and bool(args.prompt_metadata_path) ^ bool(args.target_metadata_path):
        raise ValueError(
            "SVS precomputed metadata requires both --prompt-metadata-path and "
            "--target-metadata-path (or neither, for online preprocess)."
        )
    if args.prompt_metadata_path:
        required["prompt_metadata_path"] = args.prompt_metadata_path
        required["target_metadata_path"] = args.target_metadata_path
    if args.prompt_wav_path:
        required["prompt_wav_path"] = args.prompt_wav_path
    _require_paths(required)
    preprocess_dir = resolve_preprocess_weights_dir(args.preprocess_weights_dir)
    deploy_config = args.deploy_config or str(SVC_DEPLOY_CONFIG if kind == "svc" else SVS_DEPLOY_CONFIG)
    mode = kind.upper()

    print(f"Loading SoulX-Singer {mode} from {args.model}")
    print(f"  deploy: {deploy_config}")
    print(f"  prompt: {args.prompt_audio}")
    print(f"  target: {args.target_audio}")

    omni = Omni(model=args.model, deploy_config=deploy_config, async_chunk=False)  # SoulX-Singer is batch-only
    sampling = build_sampling(
        args,
        preprocess_weights_dir=preprocess_dir,
        kind=kind,
    )
    prompt = {"prompt_token_ids": [0], "multi_modal_data": {}}

    print("Running generate (inline preprocess + DiT)...")
    outputs = list(omni.generate([prompt], sampling))
    if not outputs:
        raise RuntimeError("No output from omni.generate")

    err = getattr(outputs[0], "error", None)
    if err is not None:
        raise RuntimeError(f"Pipeline error: {err}")

    audio_np, sr = extract_audio(outputs)
    sf.write(args.output, audio_np, sr)
    print(f"Saved {args.output} ({sr} Hz, {len(audio_np) / sr:.2f}s)")
    omni.close()


if __name__ == "__main__":
    main()
