# VoxCPM Benchmark

This directory contains both:

- online serving benchmark through the OpenAI-compatible `/v1/audio/speech` API
- offline benchmark for `Omni` / `AsyncOmni`
- full offline smoke-matrix orchestration

Both benchmark paths report:

- TTFP: time to first PCM packet
- E2E latency
- RTF: real-time factor (`e2e / audio_duration`)

## Offline Benchmark

Single offline benchmark run:

```bash
python benchmarks/voxcpm/vllm_omni/bench_tts_offline.py \
    --model /path/to/voxcpm-model \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
    --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni." \
    --warmup-runs 1 \
    --output-dir benchmarks/voxcpm/results/offline_single
```

Streaming offline benchmark:

```bash
python benchmarks/voxcpm/vllm_omni/bench_tts_offline.py \
    --model /path/to/voxcpm-model \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml \
    --text "This is a split-stage VoxCPM streaming example running on vLLM Omni." \
    --warmup-runs 1 \
    --output-dir benchmarks/voxcpm/results/offline_streaming
```

Full fixed offline matrix, equivalent to the old `examples/offline_inference/voxcpm/test.py`:

```bash
python benchmarks/voxcpm/vllm_omni/run_offline_matrix.py \
    --model /path/to/voxcpm-model \
    --ref-audio /path/to/reference.wav \
    --ref-text "The exact transcript spoken in reference.wav." \
    --output-root benchmarks/voxcpm/results/offline_matrix
```

The full matrix covers both routes:

- streaming: `voxcpm_async_chunk.yaml`
- sync: `voxcpm.yaml`

And these six scenarios under each route:

- warmup + single TTS
- warmup + single voice cloning
- warmup + batch TTS
- warmup + batch voice cloning
- cold single TTS
- cold single voice cloning

`bench_tts_offline.py` itself no longer writes `summary.json` / `results.json`; it prints TTFP / RTF inline and saves generated WAV files only. The matrix runner keeps only per-case `run.log`.

## Start the Server

Async-chunk:

```bash
vllm serve /path/to/voxcpm-model \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml \
    --trust-remote-code \
    --enforce-eager \
    --omni \
    --port 8091
```

Non-streaming:

```bash
vllm serve /path/to/voxcpm-model \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
    --trust-remote-code \
    --enforce-eager \
    --omni \
    --port 8091
```

## Run the Benchmark

```bash
python benchmarks/voxcpm/vllm_omni/bench_tts_serve.py \
    --host 127.0.0.1 \
    --port 8091 \
    --num-prompts 20 \
    --max-concurrency 1 \
    --result-dir /tmp/voxcpm_bench
```

Voice cloning benchmark:

```bash
python benchmarks/voxcpm/vllm_omni/bench_tts_serve.py \
    --host 127.0.0.1 \
    --port 8091 \
    --num-prompts 10 \
    --max-concurrency 1 \
    --ref-audio https://example.com/reference.wav \
    --ref-text "The exact transcript spoken in the reference audio." \
    --result-dir /tmp/voxcpm_clone_bench
```

## Notes

- The benchmark uses `stream=true` and `response_format=pcm` so TTFP is measured from the first audio packet.
- `RTF < 1.0` means the server generates audio faster than real time.
- For `voxcpm_async_chunk.yaml`, keep concurrency at `1`. This matches native VoxCPM streaming more closely.
- Do not benchmark concurrent online streaming on `voxcpm_async_chunk.yaml`; use `voxcpm.yaml` for multi-request throughput runs.
- For the offline matrix mode, `--ref-audio` and `--ref-text` are required because clone cases are part of the fixed coverage set.
