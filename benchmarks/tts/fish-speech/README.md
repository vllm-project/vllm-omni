# Fish Speech Benchmarks

Benchmarks for Fish Speech S2 Pro voice cache performance.

## bench_speaker_cache.py

Measures TTFP improvement from DAC-code caching by comparing two modes:

- **Round A (inline ref_audio)**: every request sends the full base64-encoded reference audio. No server-side caching.
- **Round B (uploaded voice)**: upload the voice once via `/v1/audio/voices`, then reference it by name. After the first request, the server reuses cached DAC codes.

### Prerequisites

- A running `vllm-omni serve` instance with a Fish Speech S2 Pro model
- A reference audio file (`.wav`) and its transcript

### Usage

```bash
python benchmarks/tts/fish-speech/bench_speaker_cache.py \
    --ref-audio /path/to/reference.wav \
    --ref-text "Transcript of the reference audio." \
    --num-prompts 20 \
    --port 8091
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Server host |
| `--port` | `8091` | Server port |
| `--ref-audio` | (required) | Path to reference audio file |
| `--ref-text` | (required) | Transcript of reference audio |
| `--num-prompts` | `20` | Number of prompts per round |
| `--num-warmups` | `2` | Warmup requests before measurement |
| `--voice-name` | `bench_voice` | Name for the uploaded voice |

### Metrics

- TTFP (time to first audio packet)
- E2E latency
- RTF (real-time factor)
- Audio throughput

### fish_bench_utils.py

Shared benchmark infrastructure providing:
- `RequestResult` / `BenchmarkResult` dataclasses
- `send_streaming_request()` — async HTTP client for `/v1/audio/speech` (SSE and raw audio)
- `compute_stats()` — numpy-based percentile aggregation
- `run_benchmark()` / `run_benchmark_sweep()` — concurrency sweep runner with warmup
