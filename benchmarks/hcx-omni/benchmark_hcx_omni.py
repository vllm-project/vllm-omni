# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for HyperCLOVAX-SEED-Omni-8B online serving.

Measures end-to-end latency and throughput for:
  - Speech-to-Speech (S2S): audio input → text + audio output
  - Text-to-Vision  (T2V): text prompt → text + image output
  - Text-to-Text    (T2T): text prompt → text only (thinker stage only)

Metrics reported per mode:
  - Latency   : mean / p50 / p90 / p99 (seconds, wall-clock)
  - Throughput: requests / second
  - Success rate

Usage:
    # Start the server first (see run_server.sh), then:

    # All modes (10 requests each)
    python benchmark_hcx_omni.py --base-url http://localhost:8000/v1 --num-prompts 10

    # S2S only, 50 requests, concurrency 4
    python benchmark_hcx_omni.py --mode s2s --num-prompts 50 --concurrency 4

    # T2V only
    python benchmark_hcx_omni.py --mode t2v --num-prompts 20

    # With a real audio file for S2S
    python benchmark_hcx_omni.py --mode s2s --audio-file /path/to/speech.wav
"""

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B"

# System prompt required for audio/image generation to activate
SYSTEM_PROMPT = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": (
                "당신은 CLOVA X입니다. 네이버가 만든 AI 어시스턴트로서 "
                "오디오와 이미지를 인식하고 텍스트, 음성, 이미지를 생성할 수 있습니다."
            ),
        }
    ],
}

T2V_PROMPTS = [
    "귀여운 고양이 한 마리가 소파에 앉아 있는 그림을 그려줘.",
    "밤하늘에 별이 빛나는 산 풍경 이미지를 만들어줘.",
    "노란 해바라기가 가득한 들판을 그려줘.",
    "현대적인 카페 인테리어 이미지를 생성해줘.",
    "귀여운 강아지가 공원에서 뛰노는 그림을 그려줘.",
    "파란 바다와 흰 모래 해변의 풍경을 그려줘.",
    "봄날의 벚꽃이 흩날리는 공원 이미지를 만들어줘.",
    "아늑한 서재에서 책을 읽는 사람의 그림을 그려줘.",
    "빨간 지붕의 유럽풍 작은 마을 풍경을 그려줘.",
    "우주 공간에서 지구를 바라보는 우주비행사 그림을 그려줘.",
]

S2S_PROMPTS = [
    "이 오디오에서 무슨 내용이 들리나요?",
    "방금 들은 내용을 한국어로 요약해줘.",
    "이 소리가 무엇인지 설명해줘.",
]

T2T_PROMPTS = [
    "대한민국의 수도는 어디인가요?",
    "하늘은 왜 파란가요?",
    "인공지능이란 무엇인가요?",
    "건강한 식습관을 위한 조언을 해줘.",
    "파이썬 프로그래밍 언어의 특징은 무엇인가요?",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    mode: str
    latency: float
    success: bool
    has_audio: bool = False
    has_image: bool = False
    text: str = ""
    error: str = ""


@dataclass
class BenchmarkStats:
    mode: str
    total: int
    success: int
    latencies: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 0.0

    @property
    def throughput(self) -> float:
        return self.success / sum(self.latencies) * self.success if self.latencies else 0.0

    def summary(self) -> dict:
        if not self.latencies:
            return {"mode": self.mode, "total": self.total, "success": 0}
        s = sorted(self.latencies)
        n = len(s)
        return {
            "mode": self.mode,
            "total": self.total,
            "success": self.success,
            "success_rate": f"{self.success_rate:.1%}",
            "latency_mean": f"{statistics.mean(s):.2f}s",
            "latency_p50": f"{s[int(n * 0.50)]:.2f}s",
            "latency_p90": f"{s[int(n * 0.90)]:.2f}s",
            "latency_p99": f"{s[min(int(n * 0.99), n - 1)]:.2f}s",
            "latency_min": f"{s[0]:.2f}s",
            "latency_max": f"{s[-1]:.2f}s",
        }


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def make_sine_wav_b64(duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    """Generate a simple 440 Hz sine wave and return as base64 WAV."""
    try:
        import numpy as np
        import scipy.io.wavfile as wav

        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        buf = io.BytesIO()
        wav.write(buf, sample_rate, (audio * 32767).astype(np.int16))
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Minimal WAV header without numpy (44-byte header + silence)
        sample_rate = 16000
        n_samples = int(sample_rate * duration_sec)
        data = b"\x00\x00" * n_samples  # 16-bit silence
        data_size = len(data)
        header = (
            b"RIFF"
            + (data_size + 36).to_bytes(4, "little")
            + b"WAVEfmt "
            + (16).to_bytes(4, "little")
            + (1).to_bytes(2, "little")  # PCM
            + (1).to_bytes(2, "little")  # mono
            + sample_rate.to_bytes(4, "little")
            + (sample_rate * 2).to_bytes(4, "little")
            + (2).to_bytes(2, "little")
            + (16).to_bytes(2, "little")
            + b"data"
            + data_size.to_bytes(4, "little")
        )
        return base64.b64encode(header + data).decode()


# ---------------------------------------------------------------------------
# Async request functions
# ---------------------------------------------------------------------------
async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    payload: dict,
) -> tuple[float, dict]:
    url = f"{base_url}/chat/completions"
    t0 = time.perf_counter()
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
        body = await resp.json()
    latency = time.perf_counter() - t0
    return latency, body


async def run_t2t(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
) -> RequestResult:
    payload = {
        "model": model,
        "modalities": ["text"],
        "messages": [
            SYSTEM_PROMPT,
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
    }
    try:
        latency, body = await send_request(session, base_url, model, payload)
        if "error" in body:
            return RequestResult("t2t", latency, False, error=str(body["error"]))
        text = body["choices"][0]["message"].get("content", "")
        return RequestResult("t2t", latency, True, text=text)
    except Exception as e:
        return RequestResult("t2t", 0.0, False, error=str(e))


async def run_t2v(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
) -> RequestResult:
    payload = {
        "model": model,
        "modalities": ["text", "image"],
        "messages": [
            SYSTEM_PROMPT,
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ],
        "max_tokens": 800,
    }
    try:
        latency, body = await send_request(session, base_url, model, payload)
        if "error" in body:
            return RequestResult("t2v", latency, False, error=str(body["error"]))

        has_image = False
        text = ""
        for choice in body.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            has_image = True
            elif isinstance(content, str):
                text += content
        return RequestResult("t2v", latency, True, has_image=has_image, text=text)
    except Exception as e:
        return RequestResult("t2v", 0.0, False, error=str(e))


async def run_s2s(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    audio_b64: str,
) -> RequestResult:
    payload = {
        "model": model,
        "modalities": ["text", "audio"],
        "messages": [
            SYSTEM_PROMPT,
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": 512,
    }
    try:
        latency, body = await send_request(session, base_url, model, payload)
        if "error" in body:
            return RequestResult("s2s", latency, False, error=str(body["error"]))

        has_audio = False
        text = ""
        for choice in body.get("choices", []):
            msg = choice.get("message", {})
            audio = msg.get("audio")
            if audio and audio.get("data"):
                has_audio = True
            content = msg.get("content")
            if isinstance(content, str) and content and content != "None":
                text += content
        return RequestResult("s2s", latency, True, has_audio=has_audio, text=text)
    except Exception as e:
        return RequestResult("s2s", 0.0, False, error=str(e))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
async def run_benchmark(
    mode: str,
    base_url: str,
    model: str,
    num_prompts: int,
    concurrency: int,
    audio_b64: str,
) -> BenchmarkStats:
    stats = BenchmarkStats(mode=mode, total=num_prompts, success=0)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(i: int) -> RequestResult:
        async with semaphore:
            if mode == "t2t":
                prompt = T2T_PROMPTS[i % len(T2T_PROMPTS)]
                return await run_t2t(session, base_url, model, prompt)
            elif mode == "t2v":
                prompt = T2V_PROMPTS[i % len(T2V_PROMPTS)]
                return await run_t2v(session, base_url, model, prompt)
            else:  # s2s
                prompt = S2S_PROMPTS[i % len(S2S_PROMPTS)]
                return await run_s2s(session, base_url, model, prompt, audio_b64)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded(i) for i in range(num_prompts)]
        results = await asyncio.gather(*tasks)

    for r in results:
        if r.success:
            stats.success += 1
            stats.latencies.append(r.latency)
        else:
            print(f"  [FAIL] {r.error[:80]}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HyperCLOVAX-SEED-Omni-8B benchmark")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode",
        choices=["t2t", "t2v", "s2s", "all"],
        default="all",
        help="Benchmark mode (default: all)",
    )
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--audio-file", default=None, help="WAV file for S2S input")
    parser.add_argument("--output-json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Prepare audio
    if args.audio_file:
        with open(args.audio_file, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        print(f"Using audio file: {args.audio_file}")
    else:
        audio_b64 = make_sine_wav_b64(1.0)
        print("Using synthetic 1s 440Hz sine wave audio")

    modes = ["t2t", "t2v", "s2s"] if args.mode == "all" else [args.mode]

    print(f"\nBenchmark: {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Prompts  : {args.num_prompts} per mode, concurrency={args.concurrency}")
    print()

    all_stats = []
    for mode in modes:
        print(f"Running {mode.upper()} ({args.num_prompts} requests)...")
        stats = asyncio.run(
            run_benchmark(mode, args.base_url, args.model, args.num_prompts, args.concurrency, audio_b64)
        )
        all_stats.append(stats)
        s = stats.summary()
        print(f"  {mode.upper()} Results:")
        for k, v in s.items():
            print(f"    {k:20s}: {v}")
        print()

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([s.summary() for s in all_stats], f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
