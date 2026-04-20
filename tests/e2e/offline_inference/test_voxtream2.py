"""Offline benchmark for Voxtream2 through the vLLM-Omni engine."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.voxtream2.voxtream2_utils import (
    VOXTREAM2_DEFAULT_SAMPLE_RATE,
    build_voxtream2_prompt,
    flatten_voxtream2_audio_output,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "herimor/voxtream2"
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxtream2_1stage.yaml"
DEFAULT_META = REPO_ROOT / "voxtream" / "assets" / "benchmark" / "meta.csv"
DEFAULT_REF_AUDIO = REPO_ROOT / "voxtream" / "assets" / "audio" / "english_male.wav"

DEFAULT_PROMPTS = [
    "Hello, welcome to the Voxtream2 benchmark test.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Please remember to bring your identification documents tomorrow morning.",
    "Learning a new language takes patience and consistent daily practice.",
    "Could you please turn down the music? I am trying to focus on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock.",
]


@dataclass
class RequestResult:
    success: bool = False
    prompt: str = ""
    ref_audio_path: str = ""
    e2e_s: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    num_samples: int = 0
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = "voxtream2_omni"
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    p99_rtf: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    per_request: list[dict[str, Any]] = field(default_factory=list)


def _build_prompt(text: str, ref_audio: Path) -> dict[str, Any]:
    return build_voxtream2_prompt(text, str(ref_audio))


def _extract_output_audio(outputs: list[Any]) -> tuple[torch.Tensor, int]:
    for output in outputs:
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, dict) or not mm:
            continue
        audio, sample_rate = flatten_voxtream2_audio_output(
            mm,
            default_sample_rate=VOXTREAM2_DEFAULT_SAMPLE_RATE,
            require_audio=False,
        )
        if audio.numel() > 0:
            return audio, sample_rate

    raise ValueError("No audio output returned by Voxtream2 Omni benchmark request.")


def _resolve_audio_path(raw_path: str, meta_path: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path

    candidates = [REPO_ROOT / raw_path, REPO_ROOT / "voxtream" / raw_path]
    if meta_path is not None:
        candidates.extend([meta_path.parent / raw_path, meta_path.parents[2] / raw_path])

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Reference audio not found: {raw_path}")


def _read_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt_file:
        prompt_file = Path(args.prompt_file)
        prompts.extend(line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if args.text:
        prompts.extend(text.strip() for text in args.text if text.strip())
    if not prompts:
        prompts = DEFAULT_PROMPTS.copy()
    return [prompts[i % len(prompts)] for i in range(args.num_prompts)]


def _read_cases(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.meta:
        meta_path = Path(args.meta)
        if not meta_path.is_file():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        cases: list[tuple[str, Path]] = []
        with meta_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("text") or "").strip()
                prompt_audio = (row.get("prompt_audio") or "").strip()
                if text and prompt_audio:
                    cases.append((text, _resolve_audio_path(prompt_audio, meta_path)))
        if not cases:
            raise ValueError(f"No valid cases found in {meta_path}")
        return cases[: args.num_prompts]

    ref_audio = Path(args.ref_audio)
    if not ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    return [(prompt, ref_audio) for prompt in _read_prompts(args)]


def _run_single_request(
    engine: Omni,
    prompt: str,
    ref_audio_path: Path,
    save_wav_path: Path | None,
) -> RequestResult:
    result = RequestResult(prompt=prompt, ref_audio_path=str(ref_audio_path))
    start = time.perf_counter()
    try:
        outputs = engine.generate([_build_prompt(prompt, ref_audio_path)], use_tqdm=False)
        e2e_s = time.perf_counter() - start
        audio, sample_rate = _extract_output_audio(outputs)

        if save_wav_path is not None:
            sf.write(str(save_wav_path), audio.numpy(), sample_rate, format="WAV")

        audio_duration_s = float(audio.numel()) / float(sample_rate) if sample_rate > 0 else 0.0
        result.success = True
        result.e2e_s = e2e_s
        result.audio_duration_s = audio_duration_s
        result.rtf = e2e_s / audio_duration_s if audio_duration_s > 0 else 0.0
        result.num_samples = int(audio.numel())
    except Exception as exc:
        result.e2e_s = time.perf_counter() - start
        result.error = str(exc)
    return result


def _print_summary(bench: BenchmarkResult) -> None:
    width = 56
    print("")
    print("=" * width)
    print(f"{'Voxtream2 Omni Benchmark Result':^{width}}")
    print("=" * width)
    print(f"{'Successful requests:':<40}{bench.completed:<12}")
    print(f"{'Failed requests:':<40}{bench.failed:<12}")
    print(f"{'Benchmark duration (s):':<40}{bench.duration_s:<12.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<12.2f}")
    print("-" * width)
    print(f"{'End-to-end Latency':^{width}}")
    print("-" * width)
    print(f"{'Mean E2E (ms):':<40}{bench.mean_e2e_ms:<12.2f}")
    print(f"{'Median E2E (ms):':<40}{bench.median_e2e_ms:<12.2f}")
    print(f"{'P99 E2E (ms):':<40}{bench.p99_e2e_ms:<12.2f}")
    print("-" * width)
    print(f"{'Real Time Factor':^{width}}")
    print("-" * width)
    print(f"{'Mean RTF:':<40}{bench.mean_rtf:<12.3f}")
    print(f"{'Median RTF:':<40}{bench.median_rtf:<12.3f}")
    print(f"{'P99 RTF:':<40}{bench.p99_rtf:<12.3f}")
    print(f"{'Audio throughput (audio s / wall s):':<40}{bench.audio_throughput:<12.2f}")
    print("=" * width)
    print("")


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    if args.voxtream_root:
        os.environ["VOXTREAM2_ROOT"] = args.voxtream_root
        os.environ.setdefault("VOXTREAM_ROOT", args.voxtream_root)

    cases = _read_cases(args)
    audio_dir = None
    if args.save_audio:
        audio_dir = Path(args.result_dir) / "audio_voxtream2"
        audio_dir.mkdir(parents=True, exist_ok=True)

    engine = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    try:
        for i in range(args.num_warmups):
            warmup_prompt, warmup_ref_audio = cases[i % len(cases)]
            _ = _run_single_request(engine, warmup_prompt, warmup_ref_audio, None)

        per_request: list[RequestResult] = []
        bench_start = time.perf_counter()
        for idx, (prompt, ref_audio_path) in enumerate(cases):
            wav_path = audio_dir / f"output_{idx:04d}.wav" if audio_dir is not None else None
            req = _run_single_request(engine, prompt, ref_audio_path, wav_path)
            per_request.append(req)
            if req.success:
                print(
                    f"  [{idx + 1}/{len(cases)}] e2e={req.e2e_s * 1000:.0f}ms "
                    f"rtf={req.rtf:.3f} audio={req.audio_duration_s:.2f}s"
                )
            else:
                print(f"  [{idx + 1}/{len(cases)}] FAILED: {req.error}")
        duration_s = time.perf_counter() - bench_start
    finally:
        engine.close()

    successful = [item for item in per_request if item.success]
    failed = [item for item in per_request if not item.success]

    bench = BenchmarkResult(
        config_name=args.config_name,
        num_prompts=len(cases),
        completed=len(successful),
        failed=len(failed),
        duration_s=duration_s,
        per_request=[asdict(item) for item in per_request],
    )

    if successful:
        e2e_ms = [item.e2e_s * 1000.0 for item in successful]
        rtfs = [item.rtf for item in successful]
        audio_durs = [item.audio_duration_s for item in successful]
        bench.mean_e2e_ms = float(np.mean(e2e_ms))
        bench.median_e2e_ms = float(np.median(e2e_ms))
        bench.p90_e2e_ms = float(np.percentile(e2e_ms, 90))
        bench.p95_e2e_ms = float(np.percentile(e2e_ms, 95))
        bench.p99_e2e_ms = float(np.percentile(e2e_ms, 99))
        bench.mean_rtf = float(np.mean(rtfs))
        bench.median_rtf = float(np.median(rtfs))
        bench.p99_rtf = float(np.percentile(rtfs, 99))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        bench.audio_throughput = bench.total_audio_duration_s / bench.duration_s if bench.duration_s > 0 else 0.0
        bench.request_throughput = bench.completed / bench.duration_s if bench.duration_s > 0 else 0.0

    return bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voxtream2 Omni offline benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--stage-configs-path", type=str, default=str(DEFAULT_STAGE_CONFIG))
    parser.add_argument("--voxtream-root", type=str, default=str(REPO_ROOT / "voxtream"))
    parser.add_argument("--meta", type=str, default=str(DEFAULT_META) if DEFAULT_META.is_file() else None)
    parser.add_argument("--ref-audio", type=str, default=str(DEFAULT_REF_AUDIO))
    parser.add_argument("--text", type=str, action="append", default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument("--config-name", type=str, default="voxtream2_omni")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench = run_benchmark(args)
    _print_summary(bench)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = result_dir / f"bench_{args.config_name}_{ts}.json"
    out_file.write_text(json.dumps([asdict(bench)], indent=2), encoding="utf-8")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
