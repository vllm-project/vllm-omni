#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E benchmark: MOSS-TTS offline inference single-stream latency and RTF.

Loads MOSS-VoiceGenerator via Omni offline inference and measures end-to-end
latency and Real-Time Factor (RTF = wall_clock / audio_duration, lower is
better) across a configurable number of sequential requests.

Requires the model to be cached locally (or network access to HuggingFace).

Args:
    --num-requests          Number of timed requests (default: 8)
    --warmup                Number of warm-up requests, excluded from timing (default: 2)
    --max-tokens            Stage-0 max_tokens, controls audio length (default: 256)
    --gpu-memory-utilization
                            Stage-0 GPU memory fraction (default: 0.70)
    --codec-cuda-graph      Enable CUDA Graph for Stage-1 codec decoder
                            (sets enforce_eager=False on Stage 1)

Usage::

    # default: 2 warmup + 8 timed requests, eager codec
    python benchmarks/tts/bench_moss_tts.py

    # with CUDA Graph codec (Stage-1)
    python benchmarks/tts/bench_moss_tts.py --codec-cuda-graph

    # custom
    python benchmarks/tts/bench_moss_tts.py \\
        --num-requests 10 \\
        --max-tokens 512 \\
        --warmup 2 \\
        --gpu-memory-utilization 0.70 \\
        --codec-cuda-graph

Output is printed as a Markdown table suitable for pasting into a PR description.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch
from vllm import SamplingParams

from vllm_omni import Omni

_MODEL = "OpenMOSS-Team/MOSS-VoiceGenerator"
_SAMPLE_RATE = 24_000
_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy"

_PROMPTS = [
    ("Hello, this is a MOSS voice design benchmark.", "a warm female voice with an American accent"),
    ("今天天气真不错，适合出去走走。", "清晰温暖的女声"),
    ("The quick brown fox jumps over the lazy dog.", "a young male voice with a British accent"),
    ("人工智能正在改变我们的生活方式。", "沉稳男声"),
    ("Benchmarking neural text-to-speech synthesis.", "a neutral professional voice"),
    ("语音合成技术在近年来取得了显著进步。", "明亮活泼的女声"),
    ("This benchmark measures end-to-end generation latency.", "a deep calm male voice"),
    ("开始测试批量语音合成的性能指标。", "标准普通话女声"),
    ("Real-time factor measures how fast we generate audio.", "a warm friendly voice"),
    ("批量推理能够显著提升系统吞吐量。", "年轻男声"),
]


def _build_request(text: str, instruction: str) -> dict:
    from transformers import AutoProcessor

    try:
        proc = AutoProcessor.from_pretrained(_MODEL, trust_remote_code=True)
    except Exception as exc:
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            raise SystemExit(f"Cannot load AutoProcessor: {exc}") from exc
        raise

    user_msg = proc.build_user_message(text=text, instruction=instruction)
    batch = proc(conversations=[[user_msg]], mode="generation")
    unified = batch["input_ids"][0]
    text_ids = unified[:, 0].tolist()
    audio_codes = unified[:, 1:].contiguous().to(torch.int64)
    del proc
    gc.collect()

    return {
        "prompt_token_ids": text_ids,
        "additional_information": {"codes": {"ref": audio_codes}},
    }


def _run_one(omni: Omni, request: dict, sampling: list[SamplingParams]) -> dict:
    """Run one request; return wall-clock time and audio sample count."""
    audio_samples = 0
    t_start = time.perf_counter()

    for out in omni.generate(request, sampling):
        mm = out.multimodal_output
        if mm:
            audio = mm.get("audio")
            if audio is None:
                audio = mm.get("model_outputs")
            if isinstance(audio, list):
                audio = torch.cat(
                    [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                    dim=0,
                )
            if isinstance(audio, torch.Tensor):
                audio_samples += int(audio.numel())

    return {
        "total_s": time.perf_counter() - t_start,
        "audio_samples": audio_samples,
    }


def _build_config(gpu_memory_utilization: float, codec_cuda_graph: bool = False) -> str:
    """Build a benchmark-friendly deploy config from moss_voice_generator.yaml."""
    import tempfile

    import yaml

    yaml_path = _DEPLOY_DIR / "moss_voice_generator.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for stage in cfg.get("stages", []):
        sid = stage.get("stage_id")
        if sid == 0:
            stage["gpu_memory_utilization"] = gpu_memory_utilization
            stage["max_num_seqs"] = 1
        elif sid == 1 and codec_cuda_graph:
            stage["enforce_eager"] = False

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS offline inference: single-stream latency and RTF")
    parser.add_argument("--num-requests", type=int, default=8, help="Number of timed requests")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up requests (untimed)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Stage 0 max_tokens")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.70,
        help="Stage 0 gpu_memory_utilization",
    )
    parser.add_argument(
        "--codec-cuda-graph",
        action="store_true",
        help="Enable CUDA Graph for Stage-1 codec (sets enforce_eager=False)",
    )
    args = parser.parse_args()

    sampling = [
        SamplingParams(
            temperature=1.7,
            top_p=0.8,
            top_k=25,
            max_tokens=args.max_tokens,
            seed=42,
            detokenize=False,
        ),
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=65536,
            seed=42,
            detokenize=False,
        ),
    ]

    print(f"Building requests (model={_MODEL}) …")
    n_prompts = args.warmup + args.num_requests
    requests = []
    for i in range(n_prompts):
        text, instr = _PROMPTS[i % len(_PROMPTS)]
        requests.append(_build_request(text, instr))

    config_path = _build_config(args.gpu_memory_utilization, codec_cuda_graph=args.codec_cuda_graph)
    codec_mode = "cuda-graph" if args.codec_cuda_graph else "eager"
    print(f"Loading Omni (model={_MODEL}, config={config_path}, codec={codec_mode}) …")
    try:
        omni = Omni(model=_MODEL, stage_configs_path=config_path, stage_init_timeout=300)
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(device)}\n")

        print(f"Warming up ({args.warmup} requests) …")
        for i in range(args.warmup):
            _run_one(omni, requests[i], sampling)

        print(f"Timing {args.num_requests} requests (max_tokens={args.max_tokens}) …\n")
        results = []
        for i in range(args.num_requests):
            r = _run_one(omni, requests[args.warmup + i], sampling)
            results.append(r)
            audio_s = r["audio_samples"] / _SAMPLE_RATE
            rtf = r["total_s"] / audio_s if audio_s > 0 else float("inf")
            print(f"  req {i + 1:2d}: total={r['total_s'] * 1000:.0f}ms  audio={audio_s:.1f}s  RTF={rtf:.3f}")

        omni.close()
    finally:
        Path(config_path).unlink(missing_ok=True)

    total_s_list = [r["total_s"] for r in results]
    audio_s_list = [r["audio_samples"] / _SAMPLE_RATE for r in results]
    # RTF = wall_clock / audio_duration; lower is better (consistent with Fish Speech benchmarks)
    rtf_list = [t / a for t, a in zip(total_s_list, audio_s_list) if a > 0]

    print("\n### MOSS-TTS — E2E Benchmark\n")
    print(
        f"GPU: {torch.cuda.get_device_name(device)}  "
        f"model: {_MODEL}  "
        f"max_tokens: {args.max_tokens}  "
        f"n_requests: {args.num_requests}  "
        f"codec: {codec_mode}\n"
    )
    n = len(total_s_list)
    p99_note = "" if n >= 100 else f" (n={n}; P99 ≈ max for small runs)"
    print(f"| Metric | Mean | Median | P99{p99_note} |")
    print("|--------|------|--------|-----|")

    def _row(label: str, values: list[float], fmt: str = ".1f") -> str:
        if not values:
            return f"| {label} | n/a | n/a | n/a |"
        mean = statistics.mean(values)
        med = statistics.median(values)
        p99 = sorted(values)[int(len(values) * 0.99)]
        return f"| {label} | {mean:{fmt}} | {med:{fmt}} | {p99:{fmt}} |"

    print(_row("Total latency (ms)", [v * 1000 for v in total_s_list]))
    print(_row("Audio duration (s)", audio_s_list))
    print(_row("RTF (wall-clock/audio, lower is better)", rtf_list, ".3f"))


if __name__ == "__main__":
    main()
