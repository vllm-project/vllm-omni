# Qwen3-TTS Triton serving example

End-to-end recipe for serving [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) behind
NVIDIA Triton Inference Server.

## Motivation

Qwen3-TTS is split into two stages with very different runtime characteristics:

- **Talker** — an autoregressive Transformer that produces discrete audio
  codes. Token-by-token decoding benefits from continuous batching, paged
  KV-cache and the rest of the vLLM runtime, so we serve it with
  **vLLM-Omni** as a Python Triton backend.
- **Codec decoder** — a non-autoregressive convolutional model that turns a
  chunk of audio codes into a waveform. Each request contributes a different
  number of frames and we want to batch independent chunks together, so we
  export it to **TensorRT** with dynamic batch and dynamic sequence-length
  profiles and serve it via Triton's native `tensorrt_plan` backend with
  dynamic batching enabled.

The talker streams chunks of codes into the codec via Triton's
[BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
API, and the final waveform chunks are streamed back to the client over a
decoupled gRPC stream.

## 1. Build and run the Triton container

```bash
cd examples/online_serving/qwen3_tts_triton
docker build -t qwen3tts_triton .
docker run --rm -it --gpus all \
    --shm-size=8g \
    --network=host \
    -v "$(pwd):/workspace/server" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    qwen3tts_triton \
    /bin/bash
```

All subsequent commands are run inside the container, from
`/workspace/server`.

## 2. Export the codec decoder as a TensorRT engine (once)

```bash
python3 export_codec.py \
    --tokenizer-path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --trt-path model_repository/codec_decoder/1/model.plan \
    --trt-batch-profile 1 8 32
```

The default Triton config (`model_repository/codec_decoder/config.pbtxt`) uses dynamic batching with
`max_batch_size: 32`, so the same engine handles arbitrary batch sizes up
to 32. Codec is exported for `codec_chunk_size==30`.

## 3. Start the server

```bash
tritonserver --model-repository=model_repository
```

This loads two models:

- `qwen3_tts` — Python backend running the vLLM-Omni talker (decoupled,
  streaming).
- `codec_decoder` — TensorRT backend running the exported engine with
  dynamic batching.

## 4. Send requests

See `run_request.ipynb` for a minimal gRPC streaming client that sends text
and collects synthesized audio chunks as they arrive.

## 5. Benchmarking

Two scripts are provided:

- `benchmark_model.py` — benchmarks the **acoustic-token predictor (talker)
  only**, without the codec decoder. It spins up a single-stage
  vLLM-Omni `AsyncOmni` engine and measures throughput, TTFT and ITL on
  raw codec tokens.
- `benchmark_service.py` — benchmarks the **full Triton service end to
  end** over gRPC: text in, streamed waveform chunks out (talker + codec
  decoder + BLS plumbing). Measures throughput, real-time factor (RTF)
  and time-to-first-audio (TTFA).

Both scripts read prompts from a `<uttid>\t<text>` text file and accept a
concurrency / `--num-workers` argument so the same load can be replayed
across different batch sizes.

```bash
# Talker-only (model) benchmark
python3 benchmark_model.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text-file vctk_subset.txt \
    --num-requests 100 \
    --concurrency 1 4 8 32

# End-to-end service benchmark (Triton must be running)
python3 benchmark_service.py \
    --text-file vctk_subset.txt \
    --num-requests 100 \
    --num-workers 8
```

### Reference results (RTX A6000)

Numbers below are taken from a single RTX A6000 with the default
`max_num_seqs` / engine config used in this example. Latencies are
reported as `mean / p95` in milliseconds.

**End-to-end service** (`benchmark_service.py`, talker + codec):

| Concurrency | Throughput (req/s) | RTF    | TTFA mean / p95 (ms) |
| ----------: | -----------------: | -----: | -------------------: |
|           1 |               0.85 |  4.39x |        62.4 / 65.1   |
|           4 |               2.55 | 12.88x |       103.3 / 120.3  |
|           8 |               3.82 | 19.96x |       143.1 / 165.4  |
|          32 |               5.68 | 28.39x |       375.7 / 495.0  |

**Talker only** (`benchmark_model.py`, codec tokens only, no waveform):

| Concurrency | Throughput (req/s) | TTFT mean / p95 (ms) | ITL mean / p95 (ms) |
| ----------: | -----------------: | -------------------: | ------------------: |
|           1 |               0.66 |        16.55 / 19.00 |       17.12 / 18.81 |
|           4 |               2.32 |        38.15 / 48.70 |       19.36 / 22.51 |
|           8 |               3.76 |        47.29 / 53.66 |       23.23 / 29.94 |
|          32 |               7.91 |       126.25 / 279.79 |       41.31 / 56.48 |
