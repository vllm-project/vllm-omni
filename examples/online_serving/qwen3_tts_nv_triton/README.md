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
cd examples/online_serving/qwen3_tts_nv_triton
docker build --network=host -t qwen3tts_triton .
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
|           1 |               1.14 |  4.71x |        72.8 / 76.9   |
|           4 |               2.69 | 13.52x |       117.2 / 140.0  |
|           8 |               4.42 | 21.33x |       161.8 / 189.5  |
|          32 |               7.34 | 37.05x |       373.9 / 425.4  |

**Talker only** (`benchmark_model.py`, codec tokens only, no waveform):

| Concurrency | Throughput (req/s) | TTFT mean / p95 (ms) | ITL mean / p95 (ms) |
| ----------: | -----------------: | -------------------: | ------------------: |
|           1 |               0.73 |        28.32 / 31.28 |       15.44 / 16.70 |
|           4 |               2.59 |        46.84 / 57.45 |       17.09 / 21.19 |
|           8 |               4.39 |        55.85 / 64.12 |       19.87 / 26.98 |
|          32 |               9.89 |       100.31 / 112.5 |       33.04 / 45.13 |
