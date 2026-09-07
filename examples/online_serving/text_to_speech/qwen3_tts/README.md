# Qwen3-TTS NPU部署及优化

基于 vLLM-Omni 的 Qwen3-TTS 在线语音合成服务，通过 OpenAI 兼容的 `/v1/audio/speech` 接口提供三种任务模式：CustomVoice（预置音色）、VoiceDesign（语音设计）和 Base（声音克隆）。

## 代码准备

```
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/vllm-project/vllm-ascend.git
git clone https://github.com/vllm-project/vllm-omni.git
```

### 模型下载

Qwen3-TTS 提供三种任务类型对应的模型，按需下载：

| 任务类型 | 模型 | 说明 |
| --- | --- | --- |
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 预置音色，支持风格/情感控制 |
| VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 通过自然语言描述生成语音 |
| Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 基于参考音频的声音克隆 |

此外还有 0.6B 的小型变体可供选择。

模型会在首次启动服务时自动从 HuggingFace 下载，也可提前手动下载：

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```
## 版本配套

```
vllm                                     0.22.0+empty                         
vllm_ascend                              dde19f7b06ed24d9e3cc9fed45595408424364a4 
vllm-omni                                main
```

## 源码编译
```
# build vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e . --no-build-isolation

# build vllm-ascend
cd ../vllm-ascend
pip install -v -e .

# build vllm-omni
cd ../vllm-omni
pip install -v -e . --no-build-isolation
```

## 启动服务化
```
vllm serve /home/Qwen3-TTS-12Hz-1.7B-Base --omni --port 8091 --allowed-local-media-path / --stage-configs-path qwen3_tts_high_concurrency.yaml
```

## YAML文件配置
```
# Qwen3-TTS high-concurrency deploy profile.
#
# This is intentionally separate from qwen3_tts.yaml. Use it only when running
# sustained high-concurrency serving on two GPUs, for example:
#
#   vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni \
#     --deploy-config vllm_omni/deploy/qwen3_tts_high_concurrency.yaml
#
# Profile validated for the c64 / PROMPTS=512 performance experiments:
# Stage 0 talker on GPU 0 with S0=64, Stage 1 Code2Wav on GPU 1 with S1=10.
#
async_chunk: true

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      shm_threshold_bytes: 65536
      codec_streaming: true
      connector_get_sleep_s: 0.01
      connector_get_max_wait_first_chunk: 3000
      connector_get_max_wait: 300
      codec_chunk_frames: 25
      codec_left_context_frames: 72
      # Stage0 code-predictor prefix CUDA graphs for the c64 hot path.
      # These keys are consumed by Qwen3-TTS talker and ignored by Code2Wav.
      code_predictor_prefix_graphs: true
      code_predictor_prefix_graph_buckets: [64]
      code_predictor_prefix_graph_seq_lens: [2, 3, 4, 5, 6, 7, 8]
      # Keep voice-clone reference context bounded so Stage1 chunk lengths are
      # stable across different reference-audio durations.
      ref_code_context_frames: 72
      # Emit only the first audio chunk early, then return to codec_chunk_frames.
      initial_codec_chunk_frames: 1
      # Common Stage1 decode buckets:
      #   no-ref first/steady chunks: 25 / 97 frames
      #   Base ref-context first/steady chunks: 73 / 169 frames
      #   decoder internal non-streaming chunks: 325 frames
      decode_cudagraph_capture_sizes: [25, 73, 97, 169, 325]
      # Keep B>1 captures opt-in; c64 e2e validation did not show a stable win.
      decode_cudagraph_batch_sizes: [1]
      decode_compile_shapes: []
      decode_batch_max_size: 8
      decode_batch_bucket_frames: []
      decode_enable_tf32: false

stages:
  - stage_id: 0
    max_num_seqs: 8
    gpu_memory_utilization: 0.3
    trust_remote_code: true
    enable_prefix_caching: false
    async_scheduling: false
    max_num_batched_tokens: 512
    max_model_len: 4096
    devices: "0"
    output_connectors:
      to_stage_1: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: 4096
      seed: 42
      repetition_penalty: 1.05
    subtalker_sampling_params:
      do_sample: true
      temperature: 0.9
      top_k: 50
      top_p: 1.0

  - stage_id: 1
    max_num_seqs: 8
    gpu_memory_utilization: 0.3
    enforce_eager: false
    compilation_config:
        cudagraph_mode: FULL
    trust_remote_code: true
    enable_prefix_caching: false
    async_scheduling: false
    max_num_batched_tokens: 65536
    max_model_len: 65536
    devices: "1"
    input_connectors:
      from_stage_0: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 65536
      seed: 42
      repetition_penalty: 1.0

platforms:
  npu:
    stages:
      - stage_id: 0
        enforce_eager: true
        compilation_config:
            cudagraph_mode: FULL

```
如果碰到mooncake问题是镜像原因
```
mv /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake \
   /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake.disabled
```

## 性能测试
```
"""Benchmark client for Qwen3-TTS via /v1/audio/speech endpoint.

Measures TTFP (Time-to-First-Packet), E2E latency, and RTF (Real-Time Factor)
across configurable concurrency levels. Saves results as JSON for plotting.

Usage:
    python bench_tts_serve.py \
        --host 127.0.0.1 --port 8000 \
        --num-prompts 50 \
        --max-concurrency 1 4 10 \
        --result-dir results/
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

PROMPTS = [
    "Hello, welcome to the voice synthesis benchmark test.Hello, welcome to the voice synthesis benchmark test.Hello, welcome to the voice synthesis benchmark test.",
]


@dataclass
class RequestResult:
    success: bool = False
    ttfp: float = 0.0  # Time to first audio packet (seconds)
    e2e: float = 0.0  # End-to-end latency (seconds)
    audio_bytes: int = 0  # Total audio bytes received
    audio_duration: float = 0.0  # Audio duration in seconds (estimated from PCM)
    rtf: float = 0.0  # Real-time factor = e2e / audio_duration
    prompt: str = ""
    error: str = ""


@dataclass
class BenchmarkResult:
    config_name: str = ""
    concurrency: int = 0
    num_prompts: int = 0
    completed: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # TTFP stats (ms)
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
    audio_throughput: float = 0.0  # audio_duration / wall_time
    request_throughput: float = 0.0  # requests / second
    # Per-request details
    per_request: list = field(default_factory=list)


def pcm_bytes_to_duration(num_bytes: int, sample_rate: int = 24000, sample_width: int = 2) -> float:
    """Convert raw PCM byte count to duration in seconds."""
    num_samples = num_bytes / sample_width
    return num_samples / sample_rate


async def send_tts_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    voice: str = "vivian",
    language: str = "English",
    pbar: tqdm | None = None,
) -> RequestResult:
    """Send a streaming TTS request and measure latency metrics."""
    payload = {
        "input": prompt,
        "task_type": "Base",
        "ref_audio": "file:///home/t00953717/shengri.wav",
        "ref_text": "祝您生日快乐噢",
        "voice": voice,
        "language": language,
        "stream": True,
        "response_format": "pcm",
    }

    result = RequestResult(prompt=prompt)
    st = time.perf_counter()

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                result.success = False
                return result

            first_chunk = True
            total_bytes = 0

            async for chunk in response.content.iter_any():
                if first_chunk and len(chunk) > 0:
                    result.ttfp = time.perf_counter() - st
                    first_chunk = False
                total_bytes += len(chunk)

            result.e2e = time.perf_counter() - st
            result.audio_bytes = total_bytes
            result.audio_duration = pcm_bytes_to_duration(total_bytes)

            if result.audio_duration > 0:
                result.rtf = result.e2e / result.audio_duration
            result.success = True

    except Exception as e:
        result.error = str(e)
        result.success = False
        result.e2e = time.perf_counter() - st

    if pbar:
        pbar.update(1)
    return result


async def run_benchmark(
    host: str,
    port: int,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int = 3,
    voice: str = "vivian",
    language: str = "English",
) -> BenchmarkResult:
    """Run benchmark at a given concurrency level."""
    api_url = f"http://{host}:{port}/v1/audio/speech"

    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        keepalive_timeout=60,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=600),
    )

    # Warmup
    if num_warmups > 0:
        print(f"  Warming up with {num_warmups} requests...")
        warmup_tasks = []
        for i in range(num_warmups):
            prompt = PROMPTS[i % len(PROMPTS)]
            warmup_tasks.append(send_tts_request(session, api_url, prompt, voice, language))
        await asyncio.gather(*warmup_tasks)
        print("  Warmup done.")

    # Build request list
    request_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_prompts)]

    # Run benchmark
    print(f"  Running {num_prompts} requests with concurrency={max_concurrency}...")
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=num_prompts, desc=f"  concurrency={max_concurrency}")

    async def limited_request(prompt):
        async with semaphore:
            return await send_tts_request(session, api_url, prompt, voice, language, pbar)

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(limited_request(p)) for p in request_prompts]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    await session.close()

    # Compute stats
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    bench = BenchmarkResult(
        concurrency=max_concurrency,
        num_prompts=num_prompts,
        completed=len(successful),
        failed=len(failed),
        duration_s=duration,
    )

    if successful:
        ttfps = [r.ttfp * 1000 for r in successful]  # convert to ms
        e2es = [r.e2e * 1000 for r in successful]
        rtfs = [r.rtf for r in successful]
        audio_durs = [r.audio_duration for r in successful]

        bench.mean_ttfp_ms = float(np.mean(ttfps))
        bench.median_ttfp_ms = float(np.median(ttfps))
        bench.std_ttfp_ms = float(np.std(ttfps))
        bench.p90_ttfp_ms = float(np.percentile(ttfps, 90))
        bench.p95_ttfp_ms = float(np.percentile(ttfps, 95))
        bench.p99_ttfp_ms = float(np.percentile(ttfps, 99))

        bench.mean_e2e_ms = float(np.mean(e2es))
        bench.median_e2e_ms = float(np.median(e2es))
        bench.std_e2e_ms = float(np.std(e2es))
        bench.p90_e2e_ms = float(np.percentile(e2es, 90))
        bench.p95_e2e_ms = float(np.percentile(e2es, 95))
        bench.p99_e2e_ms = float(np.percentile(e2es, 99))

        bench.mean_rtf = float(np.mean(rtfs))
        bench.median_rtf = float(np.median(rtfs))
        bench.std_rtf = float(np.std(rtfs))
        bench.p99_rtf = float(np.percentile(rtfs, 99))

        bench.mean_audio_duration_s = float(np.mean(audio_durs))
        bench.total_audio_duration_s = float(np.sum(audio_durs))
        bench.audio_throughput = bench.total_audio_duration_s / duration
        bench.request_throughput = len(successful) / duration

        bench.per_request = [
            {
                "ttfp_ms": r.ttfp * 1000,
                "e2e_ms": r.e2e * 1000,
                "rtf": r.rtf,
                "audio_duration_s": r.audio_duration,
                "prompt": r.prompt,
            }
            for r in successful
        ]

    # Print summary in standardized performance template
    W = 50
    print("")
    print(f"{'=' * W}")
    print(f"{'Serving Benchmark Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Successful requests:':<40}{bench.completed:<10}")
    print(f"{'Failed requests:':<40}{bench.failed:<10}")
    print(f"{'Maximum request concurrency:':<40}{max_concurrency:<10}")
    print(f"{'Benchmark duration (s):':<40}{duration:<10.2f}")
    print(f"{'Request throughput (req/s):':<40}{bench.request_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'End-to-end Latency':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean E2EL (ms):':<40}{bench.mean_e2e_ms:<10.2f}")
    print(f"{'Median E2EL (ms):':<40}{bench.median_e2e_ms:<10.2f}")
    print(f"{'P99 E2EL (ms):':<40}{bench.p99_e2e_ms:<10.2f}")
    print(f"{'=' * W}")
    print(f"{'Audio Result':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Total audio duration generated (s):':<40}{bench.total_audio_duration_s:<10.2f}")
    print(f"{'Audio throughput (audio duration/s):':<40}{bench.audio_throughput:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Time to First Packet':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_TTFP (ms):':<40}{bench.mean_ttfp_ms:<10.2f}")
    print(f"{'Median AUDIO_TTFP (ms):':<40}{bench.median_ttfp_ms:<10.2f}")
    print(f"{'P99 AUDIO_TTFP (ms):':<40}{bench.p99_ttfp_ms:<10.2f}")
    print(f"{'-' * W}")
    print(f"{'Real Time Factor':^{W}}")
    print(f"{'-' * W}")
    print(f"{'Mean AUDIO_RTF:':<40}{bench.mean_rtf:<10.3f}")
    print(f"{'Median AUDIO_RTF:':<40}{bench.median_rtf:<10.3f}")
    print(f"{'P99 AUDIO_RTF:':<40}{bench.p99_rtf:<10.3f}")
    print(f"{'=' * W}")
    print("")

    if failed:
        for r in failed[:3]:
            print(f"  [ERROR] {r.error[:200]}")

    return bench

async def main(args):
    all_results = []

    for concurrency in args.max_concurrency:
        result = await run_benchmark(
            host=args.host,
            port=args.port,
            num_prompts=args.num_prompts,
            max_concurrency=concurrency,
            num_warmups=args.num_warmups,
            voice=args.voice,
            language=args.language,
        )
        result.config_name = args.config_name
        all_results.append(asdict(result))

    # Save results
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = result_dir / f"bench_{args.config_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {result_file}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Benchmark Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts per concurrency level")
    parser.add_argument(  # noqa: E501
        "--max-concurrency", type=int, nargs="+", default=[1, 4, 10], help="Concurrency levels to test"
    )
    parser.add_argument("--num-warmups", type=int, default=3)
    parser.add_argument("--voice", type=str, default="vivian")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument(
        "--config-name", type=str, default="async_chunk", help="Label for this config (used in filenames)"
    )
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
```

启动脚本
```
python tts.py --port 8091 --num-prompts 1 --max-concurrency 1
```

## 精度测试
使用seed tts eval数据集，按照
```
https://github.com/BytedanceSpeech/seed-tts-eval/blob/main/prepare_ckpt.py
```
准备模型。

```
#!/usr/bin/env python3
import argparse
import base64
import json
import os
import string
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import scipy.signal
import soundfile as sf
import zhconv
from funasr import AutoModel
from jiwer import process_words
from tqdm import tqdm
from zhon.hanzi import punctuation

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


PUNCTUATION_ALL = punctuation + string.punctuation


def parse_args():
    parser = argparse.ArgumentParser(description="Run online Seed-TTS zh generation and WER for a dataset.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", default="/home/t00953717/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--dataset-root", default="/home/t00953717/seed-tts-eval")
    parser.add_argument("--output-dir", default="/home/t00953717/benchmark/online_seed_wer_zh_full")
    parser.add_argument("--asr-model", default="/home/t00953717/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--asr-device", default="npu:4")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true", help="Skip utterances already present in items.jsonl.")
    parser.add_argument("--save-pcm", action="store_true", help="Also save raw PCM bytes next to WAV files.")
    return parser.parse_args()


def read_seed_rows(dataset_root):
    meta = Path(dataset_root) / "zh" / "meta.lst"
    rows = []
    with open(meta, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                raise ValueError(f"Bad Seed-TTS meta line {idx + 1}: {line}")
            utterance_id, ref_text, prompt_wav_rel, target_text = parts[:4]
            prompt_wav = (Path(dataset_root) / "zh" / prompt_wav_rel).resolve()
            rows.append(
                {
                    "index": idx,
                    "meta_line": line,
                    "utterance_id": utterance_id,
                    "ref_text": ref_text,
                    "prompt_wav": str(prompt_wav),
                    "target_text": target_text,
                }
            )
    return rows

def load_done(items_jsonl):
    done = {}
    if not items_jsonl.exists():
        return done
    with open(items_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "ok":
                done[item["utterance_id"]] = item
    return done


def wav_data_url(path):
    data = Path(path).read_bytes()
    return "data:audio/wav;base64," + base64.b64encode(data).decode("ascii")


def post_pcm_stream(url, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {resp.reason}")
            chunks = []
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks), dict(resp.headers)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Request failed: {e}") from e


def save_pcm_as_wav(pcm, wav_path, sample_rate=24000):
    if not pcm:
        raise RuntimeError("Service returned empty PCM.")
    wav = np.frombuffer(pcm, dtype=np.int16)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), wav, sample_rate, subtype="PCM_16")


def load_wav_16k(wav_path):
    wav, sr = sf.read(wav_path)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
    return wav


def normalize_zh(text):
    for mark in PUNCTUATION_ALL:
        if mark != "'":
            text = text.replace(mark, "")
    return " ".join(text.replace("  ", " "))


def calc_wer(hypothesis, reference):
    truth = normalize_zh(reference)
    hypo = normalize_zh(hypothesis)
    measures = process_words(truth, hypo)
    ref_len = max(1, len(truth.split(" ")))
    return (
        measures.wer,
        measures.insertions / ref_len,
        measures.deletions / ref_len,
        measures.substitutions / ref_len,
    )


def transcribe_zh(model, wav_path):
    wav = load_wav_16k(wav_path)
    result = model.generate(input=wav, batch_size_s=300)
    return zhconv.convert(result[0]["text"], "zh-cn")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "generated_wavs"
    pcm_dir = output_dir / "generated_pcm"
    request_dir = output_dir / "requests"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    request_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pcm:
        pcm_dir.mkdir(parents=True, exist_ok=True)

    items_jsonl = output_dir / "items.jsonl"
    wer_tsv = output_dir / "wer.tsv"
    summary_json = output_dir / "summary.json"
    failed_jsonl = output_dir / "failed.jsonl"

    rows = read_seed_rows(args.dataset_root)
    selected = rows[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    done = load_done(items_jsonl) if args.resume else {}
    service_url = f"http://{args.host}:{args.port}/v1/audio/speech"
    asr_model = AutoModel(model=args.asr_model, device=args.asr_device, disable_update=True)

    if not wer_tsv.exists() or not args.resume:
        wer_tsv.write_text(
            "utt\twer\treference\tasr\tins\tdel\tsub\tgenerated_wav\n",
            encoding="utf-8",
        )

    wers = [float(item["wer"]) for item in done.values() if item.get("wer") is not None]
    started_at = time.time()
    with open(items_jsonl, "a", encoding="utf-8") as items_f, open(
        wer_tsv, "a", encoding="utf-8"
    ) as wer_f, open(failed_jsonl, "a", encoding="utf-8") as failed_f:
        for row in tqdm(selected, initial=0, total=len(selected)):
            utt = row["utterance_id"]
            if utt in done:
                continue
            wav_path = wav_dir / f"{utt}.wav"
            pcm_path = pcm_dir / f"{utt}.pcm"
            payload = {
                "model": args.model,
                "input": row["target_text"],
                "stream": True,
                "response_format": "pcm",
                "ref_audio": wav_data_url(row["prompt_wav"]),
                "ref_text": row["ref_text"],
                "task_type": "Base",
                "language": "Chinese",
                "max_new_tokens": args.max_new_tokens,
            }
            request_record = {**row, "payload": {**payload, "ref_audio": "<base64 omitted>"}}
            write_json(request_dir / f"{utt}.request.json", request_record)

            try:
                pcm, headers = post_pcm_stream(service_url, payload, args.request_timeout)
                if args.save_pcm:
                    pcm_path.write_bytes(pcm)
                save_pcm_as_wav(pcm, wav_path)
                asr = transcribe_zh(asr_model, wav_path)
                wer, ins, dele, sub = calc_wer(asr, row["target_text"])
                wers.append(wer)
                item = {
                    **row,
                    "status": "ok",
                    "service_url": service_url,
                    "generated_wav": str(wav_path),
                    "generated_pcm": str(pcm_path) if args.save_pcm else None,
                    "response_headers": headers,
                    "wer": wer,
                    "asr": asr,
                    "insertions": ins,
                    "deletions": dele,
                    "substitutions": sub,
                }
                items_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                items_f.flush()
                wer_f.write(
                    f"{utt}\t{wer}\t{row['target_text']}\t{asr}\t{ins}\t{dele}\t{sub}\t{wav_path}\n"
                )
                wer_f.flush()
            except Exception as exc:
                fail = {**row, "status": "failed", "error": repr(exc)}
                failed_f.write(json.dumps(fail, ensure_ascii=False) + "\n")
                failed_f.flush()

            summary = {
                "dataset_root": args.dataset_root,
                "total_selected": len(selected),
                "completed": len(wers),
                "failed": sum(1 for _ in open(failed_jsonl, encoding="utf-8")) if failed_jsonl.exists() else 0,
                "mean_wer": sum(wers) / len(wers) if wers else None,
                "mean_wer_percent": round((sum(wers) / len(wers)) * 100, 3) if wers else None,
                "output_dir": str(output_dir),
                "generated_wav_dir": str(wav_dir),
                "items_jsonl": str(items_jsonl),
                "wer_tsv": str(wer_tsv),
                "failed_jsonl": str(failed_jsonl),
                "elapsed_sec": round(time.time() - started_at, 3),
            }
            write_json(summary_json, summary)

    if wers:
        print(f"Mean WER: {round((sum(wers) / len(wers)) * 100, 3)}% ({len(wers)} items)")
    else:
        print("No successful items.")


if __name__ == "__main__":
    main()
```


运行代码
```
python run_online_seed_wer_batch.py
```