#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline inference demo for SongGeneration v2 via vLLM Omni.

Runs the two-stage runtime pipeline:

    lyrics + style -> LeLM AR (Stage 0) -> Flow1dVAE code2wav (Stage 1) -> 48 kHz stereo wav

Prerequisites:
    Clone the official SongGeneration repo (source code required for Stage 1):
        git clone https://github.com/tencent-ailab/SongGeneration.git

    Model weights and runtime assets are downloaded automatically on first run.

Usage:
    python end2end.py --model /path/to/SongGeneration --query-type mixed

    Or set the environment variable:
    export SONGGENERATION_REPO=/path/to/SongGeneration
    python end2end.py --query-type mixed
"""

from __future__ import annotations

import logging
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / asset resolution
# ---------------------------------------------------------------------------

_REPO_ENV = "SONGGENERATION_REPO"

_COMMON_LOCAL_PATHS = [
    "/root/SongGeneration",
    "/workspace/SongGeneration",
    os.path.expanduser("~/SongGeneration"),
]

_HF_MODEL_REPO = "lglg666/SongGeneration-v2-large"
_HF_RUNTIME_REPO = "lglg666/SongGeneration-Runtime"

DEFAULT_DEPLOY_CONFIG = str(Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy" / "songgeneration_v2.yaml")


def _has_source_code(path: str | os.PathLike[str]) -> bool:
    """Check if the repo has upstream source code (required for Stage 1 wrapper)."""
    root = Path(path).expanduser()
    return (root / "codeclm").is_dir()


def _has_model_weights(path: str | os.PathLike[str]) -> bool:
    root = Path(path).expanduser()
    return (root / "songgeneration_v2_large" / "model.pt").is_file() or (root / "model.pt").is_file()


def _has_runtime_assets(path: str | os.PathLike[str]) -> bool:
    root = Path(path).expanduser()
    return (root / "ckpt").is_dir() and (root / "third_party").is_dir()


def _resolve_repo_path(model_arg: str | None) -> str:
    """Resolve the SongGeneration repo path.

    Priority: --model > SONGGENERATION_REPO env > common local paths.
    """
    if model_arg:
        if not os.path.isdir(model_arg):
            raise FileNotFoundError(
                f"--model path does not exist: {model_arg}\n"
                "Please clone the official repo:\n"
                "  git clone https://github.com/tencent-ailab/SongGeneration.git"
            )
        return model_arg

    env_path = os.environ.get(_REPO_ENV)
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(
                f"{_REPO_ENV}={env_path} does not exist.\n"
                "Please clone the official repo:\n"
                "  git clone https://github.com/tencent-ailab/SongGeneration.git"
            )
        logger.info("Using %s=%s", _REPO_ENV, env_path)
        return env_path

    for p in _COMMON_LOCAL_PATHS:
        if os.path.isdir(p):
            logger.info("Auto-detected SongGeneration repo at %s", p)
            return p

    raise FileNotFoundError(
        "SongGeneration repo not found.\n\n"
        "Stage 1 wraps upstream Flow1dVAE/Tango source code, so a local\n"
        "clone of the official repo is required:\n\n"
        "  git clone https://github.com/tencent-ailab/SongGeneration.git\n\n"
        "Then run:\n"
        "  python end2end.py --model /path/to/SongGeneration --query-type mixed\n\n"
        "Or set:\n"
        f"  export {_REPO_ENV}=/path/to/SongGeneration"
    )


def _ensure_assets(repo_path: str, no_download: bool = False) -> None:
    """Validate source code exists; auto-download missing weights/assets."""
    if not _has_source_code(repo_path):
        raise FileNotFoundError(
            f"Source code not found in {repo_path} (missing codeclm/ directory).\n\n"
            "Stage 1 requires upstream Flow1dVAE source code.\n"
            "Please ensure you cloned the full repo:\n"
            "  git clone https://github.com/tencent-ailab/SongGeneration.git"
        )

    if not _has_model_weights(repo_path):
        if no_download:
            raise FileNotFoundError(
                f"Model weights not found in {repo_path}.\n"
                "Expected: songgeneration_v2_large/model.pt\n"
                "Download with: huggingface-cli download lglg666/SongGeneration-v2-large "
                f"--local-dir {repo_path}/songgeneration_v2_large"
            )
        logger.info("Model weights not found; downloading from HuggingFace...")
        _download_model_weights(repo_path)

    if not _has_runtime_assets(repo_path):
        if no_download:
            raise FileNotFoundError(
                f"Runtime assets not found in {repo_path}.\n"
                "Expected: ckpt/ and third_party/ directories.\n"
                "Download with: huggingface-cli download lglg666/SongGeneration-Runtime "
                f"--local-dir {repo_path}"
            )
        logger.info("Runtime assets (ckpt/, third_party/) not found; downloading from HuggingFace...")
        _download_runtime_assets(repo_path)


def _download_model_weights(repo_path: str) -> None:
    """Download LeLM weights into repo_path/songgeneration_v2_large/."""
    from huggingface_hub import snapshot_download

    target = os.path.join(repo_path, "songgeneration_v2_large")
    logger.info("Downloading %s -> %s (first run only)...", _HF_MODEL_REPO, target)
    snapshot_download(_HF_MODEL_REPO, local_dir=target)
    logger.info("Model weights downloaded to %s", target)


def _download_runtime_assets(repo_path: str) -> None:
    """Download ckpt/ and third_party/ into repo_path."""
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s -> %s (first run only)...", _HF_RUNTIME_REPO, repo_path)
    snapshot_download(_HF_RUNTIME_REPO, local_dir=repo_path)
    logger.info("Runtime assets downloaded to %s", repo_path)


# ---------------------------------------------------------------------------
# Sample queries
# ---------------------------------------------------------------------------

SAMPLE_QUERIES: dict[str, dict[str, Any]] = {
    "mixed": {
        "lyric": (
            "[intro-long] ; "
            "[verse] 花朵绽放如诗篇.随风轻舞动."
            "湖面波光映倒影.心随波浪荡漾中."
            "感受着这美好时光.仿佛梦境般飘渺 ; "
            "[chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅."
            "阳光洒满湖面.一切都如此美好."
            "仿佛置身人间天堂.让人心醉神迷 ; "
            "[inst-long] ; "
            "[verse] 柳枝轻拂水面.带来淡淡的花香."
            "渔夫划桨而过.留下一道道涟漪."
            "在这宁静的午后.一切都显得如此和谐 ; "
            "[chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮."
            "雨滴轻敲窗棂.夜风中烛火摇曳."
            "在这人间天堂.让我们紧紧相拥 ; "
            "[outro-long]"
        ),
        "descriptions": "female, dark, hip hop, sad, piano and drums",
        "gen_type": "mixed",
    },
    "vocal": {
        "lyric": (
            "[intro-long] ; "
            "[verse] 花朵绽放如诗篇.随风轻舞动."
            "湖面波光映倒影.心随波浪荡漾中."
            "感受着这美好时光.仿佛梦境般飘渺 ; "
            "[chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅."
            "阳光洒满湖面.一切都如此美好."
            "仿佛置身人间天堂.让人心醉神迷 ; "
            "[inst-long] ; "
            "[verse] 柳枝轻拂水面.带来淡淡的花香."
            "渔夫划桨而过.留下一道道涟漪."
            "在这宁静的午后.一切都显得如此和谐 ; "
            "[chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮."
            "雨滴轻敲窗棂.夜风中烛火摇曳."
            "在这人间天堂.让我们紧紧相拥 ; "
            "[outro-long]"
        ),
        "descriptions": "female, dark, hip hop, sad, piano and drums",
        "gen_type": "vocal",
    },
    "bgm": {
        "lyric": (
            "[intro-long] ; "
            "[verse] 花朵绽放如诗篇.随风轻舞动."
            "湖面波光映倒影.心随波浪荡漾中."
            "感受着这美好时光.仿佛梦境般飘渺 ; "
            "[chorus] 唱啊唱.爱意围绕在身边.唱啊唱.心情如此欢畅."
            "阳光洒满湖面.一切都如此美好."
            "仿佛置身人间天堂.让人心醉神迷 ; "
            "[inst-long] ; "
            "[verse] 柳枝轻拂水面.带来淡淡的花香."
            "渔夫划桨而过.留下一道道涟漪."
            "在这宁静的午后.一切都显得如此和谐 ; "
            "[chorus] 唱啊唱.情人在我身旁.唱啊唱.心情如此明亮."
            "雨滴轻敲窗棂.夜风中烛火摇曳."
            "在这人间天堂.让我们紧紧相拥 ; "
            "[outro-long]"
        ),
        "descriptions": "female, dark, hip hop, sad, piano and drums",
        "gen_type": "bgm",
    },
}

SAMPLE_RATE = 48000
FRAME_RATE_HZ = 25
DELAY_FRAMES = 250
PLACEHOLDER_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _estimate_prompt_len(
    *,
    model_path: str,
    lyric: str,
    descriptions: str | None,
    gen_type: str,
    prompt_len_override: int | None = None,
) -> int:
    """Compute the Stage-0 placeholder length before constructing the request."""
    if prompt_len_override is not None and prompt_len_override > 0:
        return int(prompt_len_override)

    env_len = os.environ.get("SONGGEN_PROMPT_LEN")
    if env_len:
        return int(env_len)

    import torch

    from vllm_omni.model_executor.models.songgeneration_v2.condition_fuser import ConditionFuser
    from vllm_omni.model_executor.models.songgeneration_v2.conditioners import (
        build_conditioner_provider,
    )
    from vllm_omni.model_executor.models.songgeneration_v2.configuration_songgeneration_v2 import (
        SongGenerationV2LeLMConfig,
    )
    from vllm_omni.model_executor.models.songgeneration_v2.prompt_builder import (
        build_condition_tensors,
    )

    cfg = SongGenerationV2LeLMConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    provider = build_conditioner_provider(cfg, upstream_repo_path=model_path).to(device)

    ckpt_candidates = [
        Path(model_path) / "songgeneration_v2_large" / "model.pt",
        Path(model_path) / "model.pt",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.is_file()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"model.pt not found under {model_path}. Expected songgeneration_v2_large/model.pt or model.pt."
        )
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    provider_state = {
        name[len("audiolm.condition_provider.") :]: tensor
        for name, tensor in raw.items()
        if name.startswith("audiolm.condition_provider.")
    }
    provider.load_state_dict(provider_state, strict=False)
    del raw, provider_state

    with torch.no_grad():
        cond_tensors, batch_size = build_condition_tensors(
            provider,
            lyrics=[lyric],
            descriptions=[descriptions] if descriptions else None,
            gen_type=gen_type,
            cfg_pair=True,
            device=device,
        )
        cond_half = {
            name: (emb1[:batch_size], emb2[:batch_size], mask[:batch_size])
            for name, (emb1, emb2, mask) in cond_tensors.items()
        }
        fuser_cfg = cfg.fuser or {"sum": [], "prepend": []}
        fuser = ConditionFuser(
            fuse2cond={
                "sum": list(fuser_cfg.get("sum", []) or []),
                "prepend": list(fuser_cfg.get("prepend", []) or []),
            }
        )
        dummy = torch.zeros(1, 1, cfg.dim, device=device)
        fused, _ = fuser(dummy, dummy, cond_half, first_step=True)
        prompt_len = int(fused.shape[1])

    del provider, cond_tensors, cond_half, fuser, dummy, fused
    if torch.cuda.is_available():
        torch.accelerator.empty_cache()
    return prompt_len


def _normalize_gen_type(value: Any) -> str:
    gen_type = str(value or "mixed").strip().lower()
    aliases = {
        "separate": "mixed",
        "all": "mixed",
        "full": "mixed",
        "instrumental": "bgm",
        "accompaniment": "bgm",
    }
    gen_type = aliases.get(gen_type, gen_type)
    if gen_type not in {"mixed", "vocal", "bgm"}:
        raise ValueError(f"Unsupported gen_type: {value!r}")
    return gen_type


def _duration_to_max_gen_len(duration_sec: float) -> int:
    """Convert final audio duration to raw delayed-pattern AR steps."""
    return int(round((float(duration_sec) * FRAME_RATE_HZ) + DELAY_FRAMES))


def _build_prompt(args: Any, query: dict[str, Any]) -> dict[str, Any]:
    lyric = args.lyric or query["lyric"]
    descriptions = args.descriptions if args.descriptions is not None else query.get("descriptions")
    gen_type = _normalize_gen_type(query["gen_type"])
    max_gen_len = int(query.get("max_gen_len", args.max_gen_len))

    prompt_len = _estimate_prompt_len(
        model_path=args.resolved_model_path,
        lyric=lyric,
        descriptions=descriptions,
        gen_type=gen_type,
        prompt_len_override=args.prompt_len,
    )
    logger.info("prompt_len=%d, gen_type=%s, max_gen_len=%d", prompt_len, gen_type, max_gen_len)
    return {
        "prompt_token_ids": [PLACEHOLDER_TOKEN_ID] * prompt_len,
        "additional_information": {
            "lyric": lyric,
            "descriptions": descriptions,
            "gen_type": gen_type,
            "seed": args.seed,
            "max_gen_len": max_gen_len,
        },
    }


# ---------------------------------------------------------------------------
# Audio extraction and saving
# ---------------------------------------------------------------------------


def _extract_audio(mm: dict[str, Any]) -> tuple[Any | None, int | None]:
    import torch

    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    sr = mm.get("sr")

    if isinstance(audio, list):
        tensors = [x for x in audio if isinstance(x, torch.Tensor) and x.numel() > 0]
        audio = torch.cat(tensors, dim=-1) if tensors else None
    if isinstance(sr, list) and sr:
        sr = sr[-1]
    if hasattr(sr, "item"):
        sr = int(sr.item())
    elif sr is not None:
        sr = int(sr)
    return audio, sr if isinstance(sr, int) else None


def _audio_for_soundfile(audio: Any):
    arr = audio.detach().float().cpu().numpy()
    if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    return arr


def _save_wav(output_dir: str, request_id: str, audio: Any, sr: int) -> str:
    import soundfile as sf

    arr = _audio_for_soundfile(audio)
    out_path = Path(output_dir) / f"output_{request_id}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), arr, samplerate=sr, format="WAV")
    duration = arr.shape[0] / sr
    peak = float(abs(arr).max()) if arr.size else 0.0
    logger.info(
        "Saved %s  shape=%s sr=%d duration=%.2fs peak=%.4f",
        out_path,
        arr.shape,
        sr,
        duration,
        peak,
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: Any) -> None:
    from vllm_omni import Omni

    args.resolved_model_path = _resolve_repo_path(args.model)
    _ensure_assets(args.resolved_model_path, no_download=args.no_auto_download)

    os.environ[_REPO_ENV] = args.resolved_model_path

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    query = dict(SAMPLE_QUERIES[args.query_type])
    if args.duration_sec is not None:
        query["max_gen_len"] = _duration_to_max_gen_len(args.duration_sec)
    prompts = [_build_prompt(args, query)]

    logger.info("Constructing Omni(model=%s, deploy=%s)", args.resolved_model_path, args.deploy_config)
    t0 = time.perf_counter()
    omni = Omni(
        model=args.resolved_model_path,
        deploy_config=args.deploy_config,
        trust_remote_code=True,
        stage_init_timeout=args.stage_init_timeout,
        enforce_eager=True,
    )
    logger.info("Omni ready in %.1fs, stages=%d", time.perf_counter() - t0, omni.num_stages)

    t0 = time.perf_counter()
    saved_paths: list[str] = []
    for prompt in prompts:
        for stage_outputs in omni.generate([prompt], py_generator=True):
            request_output = getattr(stage_outputs, "request_output", stage_outputs)
            if request_output is None or not getattr(request_output, "outputs", None):
                continue
            mm = request_output.outputs[0].multimodal_output or {}
            audio, sr = _extract_audio(mm)
            if audio is None or sr is None or audio.numel() == 0:
                continue
            if sr != SAMPLE_RATE:
                raise RuntimeError(f"Expected sample rate {SAMPLE_RATE}, got {sr}")
            path = _save_wav(args.output_dir, str(request_output.request_id), audio, sr)
            saved_paths.append(path)

    elapsed = time.perf_counter() - t0
    if not saved_paths:
        raise RuntimeError("No audio output was produced.")
    logger.info("Inference complete in %.1fs. Output: %s", elapsed, saved_paths[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> Any:
    parser = ArgumentParser(
        description="SongGeneration v2 offline inference with vLLM Omni",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to local SongGeneration repo (must contain codeclm/). "
            "Auto-detects from SONGGENERATION_REPO env or common paths if omitted."
        ),
    )
    parser.add_argument("--deploy-config", type=str, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument(
        "--query-type",
        choices=["mixed", "vocal", "bgm"],
        default="mixed",
        help="Generation mode with built-in sample lyrics.",
    )
    parser.add_argument(
        "--lyric",
        type=str,
        default=None,
        help="Override the built-in sample lyrics.",
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default=None,
        help="Override the built-in style description (comma-separated tags).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gen-len", type=int, default=750)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Target audio duration in seconds (overrides --max-gen-len).",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help="Override Stage-0 prompt length (or set SONGGEN_PROMPT_LEN env).",
    )
    parser.add_argument("--output-dir", type=str, default="output_songgeneration_v2")
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic download of missing weights/assets from HuggingFace.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    run(parse_args())
