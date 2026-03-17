"""Benchmark Qwen3-TTS using HuggingFace transformers (qwen_tts library).

Measures E2E latency, RTF, and audio duration for offline (non-serving) inference.
Results are saved in the same JSON format as bench_tts_serve.py for unified plotting.

Usage:
    python bench_tts_hf.py \
        --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
        --num-prompts 50 \
        --num-warmups 3 \
        --gpu-device 0 \
        --result-dir results/
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.",
    "She said she would be here by noon, but nobody showed up.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I can't believe how beautiful the sunset looks from up here on the mountain.",
    "Please remember to bring your identification documents to the appointment tomorrow morning.",
    "Have you ever wondered what it would be like to travel through time and visit ancient civilizations?",
    "The restaurant on the corner serves the best pasta I have ever tasted in my entire life.",
    "After the meeting, we should discuss the quarterly results and plan for the next phase.",
    "Learning a new language takes patience, practice, and a genuine curiosity about other cultures.",
    "The train leaves at half past seven, so we need to arrive at the station before then.",
    "Could you please turn down the music a little bit, I'm trying to concentrate on my work.",
    "It was a dark and stormy night when the old lighthouse keeper heard a knock at the door.",
]


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 1  # always 1 for offline
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFP stats - not applicable for HF offline, set to E2E for compatibility
    mean_ttfp_ms: float = 0.0
    median_ttfp_ms: float = 0.0
    std_ttfp_ms: float = 0.0
    p90_ttfp_ms: float = 0.0
    p95_ttfp_ms: float = 0.0
    p99_ttfp_ms: float = 0.0
    # E2E stats (ms)
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    # RTF stats
    mean_rtf: float = 0.0
    median_rtf: float = 0.0
    std_rtf: float = 0.0
    p99_rtf: float = 0.0
    # Audio stats
    mean_audio_duration_s: float = 0.0
    total_audio_duration_s: float = 0.0
    audio_throughput: float = 0.0
    request_throughput: float = 0.0
    # Per-request details
    per_request: list = field(default_factory=list)


def run_benchmark(args):
    from qwen_tts import Qwen3TTSModel

    device = f"cuda:{args.gpu_device}"
    print(f"Loading model: {args.model} on {device}")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.bfloat16,
    )
    print("Model loaded.")

    # Build prompt list
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]

    # Warmup
    if args.num_warmups > 0:
        print(f"Warming up with {args.num_warmups} requests...")
        for i in range(args.num_warmups):
            p = PROMPTS[i % len(PROMPTS)]
            wavs, sr = model.generate_custom_voice(
                text=p,
                language=args.language,
                speaker=args.voice,
            )
        # Sync GPU
        torch.cuda.synchronize(device)
        print("Warmup done.")

    # Benchmark
    print(f"Running {args.num_prompts} requests sequentially...")
    e2e_times = []
    rtfs = []
    audio_durations = []
    per_request = []
    failed = 0

    audio_dir = None
    if args.save_audio:
        audio_dir = Path(args.result_dir) / "audio_hf"
        audio_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        try:
            torch.cuda.synchronize(device)
            st = time.perf_counter()

            wavs, sr = model.generate_custom_voice(
                text=prompt,
                language=args.language,
                speaker=args.voice,
            )

            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - st

            # Compute audio duration
            audio_samples = wavs[0]
            if isinstance(audio_samples, torch.Tensor):
                audio_samples = audio_samples.cpu().numpy()
            audio_dur = len(audio_samples) / sr

            rtf = elapsed / audio_dur if audio_dur > 0 else 0.0

            e2e_times.append(elapsed)
            rtfs.append(rtf)
            audio_durations.append(audio_dur)
            per_request.append(
                {
                    "e2e_ms": elapsed * 1000,
                    "ttfp_ms": elapsed * 1000,  # no streaming, TTFP = E2E
                    "rtf": rtf,
                    "audio_duration_s": audio_dur,
                    "prompt": prompt,
                }
            )

            if audio_dir:
                sf.write(str(audio_dir / f"output_{i:04d}.wav"), audio_samples, sr)

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i + 1}/{args.num_prompts}] e2e={elapsed * 1000:.0f}ms  rtf={rtf:.3f}  audio={audio_dur:.2f}s"
                )

        except Exception as e:
            print(f"  [{i + 1}/{args.num_prompts}] FAILED: {e}")
            failed += 1

    total_duration = time.perf_counter() - total_start
    completed = len(e2e_times)

    # Compute stats
    result = BenchmarkResult(
        config_name=args.config_name,
        concurrency=1,
        num_prompts=args.num_prompts,
        completed=completed,
        failed=failed,
        duration_s=total_duration,
    )

    if e2e_times:
        e2e_ms = [t * 1000 for t in e2e_times]

        result.mean_e2e_ms = float(np.mean(e2e_ms))
        result.median_e2e_ms = float(np.median(e2e_ms))
        result.std_e2e_ms = float(np.std(e2e_ms))
        result.p90_e2e_ms = float(np.percentile(e2e_ms, 90))
        result.p95_e2e_ms = float(np.percentile(e2e_ms, 95))
        result.p99_e2e_ms = float(np.percentile(e2e_ms, 99))

        # For HF offline, TTFP = E2E (no streaming)
        result.mean_ttfp_ms = result.mean_e2e_ms
        result.median_ttfp_ms = result.median_e2e_ms
        result.std_ttfp_ms = result.std_e2e_ms
        result.p90_ttfp_ms = result.p90_e2e_ms
        result.p95_ttfp_ms = result.p95_e2e_ms
        result.p99_ttfp_ms = result.p99_e2e_ms

        result.mean_rtf = float(np.mean(rtfs))
        result.median_rtf = float(np.median(rtfs))
        result.std_rtf = float(np.std(rtfs))
        result.p99_rtf = float(np.percentile(rtfs, 99))

        result.mean_audio_duration_s = float(np.mean(audio_durations))
        result.total_audio_duration_s = float(np.sum(audio_durations))
        result.audio_throughput = result.total_audio_duration_s / total_duration
        result.request_throughput = completed / total_duration
        result.per_request = per_request

    # Print summary in standardized performance template
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{completed:<10}")
    print(f"{'Failed requests:':<40}{failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{1:<10}")
    print(f"{'Benchmark duration (s):':<40}{total_duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{result.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{result.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{result.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{result.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{result.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{result.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{result.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{result.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{result.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{result.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{result.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{result.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")

    # Save results (as a list with single concurrency=1 entry, matching serve format)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump([asdict(result)], f, indent=2)
    print(f"Results saved to {result_file}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS HuggingFace Benchmark")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", help="HuggingFace model name or path"
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--voice", type=str, default="Vivian")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument(
        "--config-name", type=str, default="hf_transformers", help="Label for this config (used in filenames)"
    )
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--save-audio", action="store_true", help="Save generated audio files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
