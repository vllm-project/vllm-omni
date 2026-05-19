# Qwen3-TTS Triton serving example

End-to-end recipe for serving [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
with NVIDIA Triton Inference Server.

## Motivation

Qwen3-TTS has two stages with very different runtime characteristics:

- **Talker** — autoregressive Transformer that emits discrete audio codes.
  Token-by-token decoding benefits from continuous batching and paged
  KV-cache, so we serve it with **vLLM-Omni** as a Python Triton backend.
- **Codec decoder** — non-autoregressive convolutional model that turns a
  chunk of codes into a waveform. We export it to **TensorRT** with a
  dynamic batch profile and serve it via Triton's native `tensorrt_plan`
  backend with dynamic batching enabled.

The talker streams chunks of codes into the codec via Triton's
[BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
API, and waveform chunks are streamed back to the client over a decoupled
gRPC stream.

## 1. One-time setup

Steps 1.1 and 1.2 only need to be done once per machine.

### 1.1 Export the codec decoder to ONNX (host)

The ONNX export must run in an environment that matches the original
Qwen3-TTS repo — see its
[Quickstart](https://github.com/QwenLM/Qwen3-TTS#quickstart). Create a
clean Python 3.12 env on the host:

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts onnx onnxruntime

cd examples/online_serving/text_to_speech/qwen3_tts_nv
python3 scripts/export_codec_onnx.py \
    --tokenizer-path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --onnx-path codec.onnx
```

> We plan to release a pre-exported `codec.onnx` so this step can be
> skipped.

### 1.2 Build the Triton container and TRT engine

```bash
cd examples/online_serving/text_to_speech/qwen3_tts_nv
docker build --network=host -t qwen3tts_nv .
docker run --rm -it --gpus all \
    --shm-size=8g \
    --network=host \
    -v "$(pwd):/workspace" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    qwen3tts_nv \
    /bin/bash
```

All subsequent commands run inside the container, from `/workspace`.

Build the TRT engine from the ONNX produced in step 1.1:

```bash
python3 scripts/export_codec_trt.py \
    --onnx-path codec.onnx \
    --trt-path model_repository/codec_decoder/1/model.plan
```

The default Triton config (`model_repository/codec_decoder/config.pbtxt`)
uses dynamic batching with `max_batch_size: 32`, so the same engine
handles batches up to 32. Codec is exported for `codec_chunk_size==30`.

## 2. Start the server

Run from inside the container, in `/workspace`:

```bash
tritonserver --model-repository=model_repository
```

This loads two models:

- `qwen3_tts` — Python backend running the vLLM-Omni talker (decoupled,
  streaming).
- `codec_decoder` — TensorRT backend running the exported engine with
  dynamic batching.

## 3. Usage & benchmarking

Two scripts are provided:

- `scripts/benchmark_model.py` — benchmarks the **talker only**, and
  doubles as an example of how to drive the vLLM-Omni Qwen3-TTS model
  definition directly. Spins up a single-stage `AsyncOmni` engine and
  measures throughput, TTFT and ITL on raw codec tokens.
- `scripts/benchmark_service.py` — benchmarks the **full Triton service
  end to end** over gRPC, and doubles as an example of how to issue
  requests against a running `tritonserver`. Text in, streamed waveform
  chunks out (talker + codec + BLS plumbing). Measures throughput,
  real-time factor (RTF) and time-to-first-audio (TTFA).

Both read prompts from a `<uttid>\t<text>` file and accept a
concurrency / `--num-workers` argument.

```bash
# Talker-only (model) benchmark
python3 scripts/benchmark_model.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text-file vctk_subset.txt \
    --num-requests 16 \
    --concurrency 1

# End-to-end service benchmark (Triton must be running)
python3 scripts/benchmark_service.py \
    --text-file vctk_subset.txt \
    --num-requests 16 \
    --num-workers 1
```

### Reference results (RTX A6000)

Single RTX A6000, default `max_num_seqs` / engine config. Latencies are
`mean / p95` in milliseconds.

**Talker only** (`scripts/benchmark_model.py`, codec tokens only):

| Concurrency | Throughput (req/s) | TTFT mean / p95 (ms) | ITL mean / p95 (ms) |
| ----------: | -----------------: | -------------------: | ------------------: |
|           1 |               0.99 |        26.81 / 28.19 |       15.04 / 16.54 |
|           4 |               3.41 |        51.88 / 54.31 |       16.72 / 18.84 |
|           8 |               5.60 |        61.03 / 66.51 |       19.45 / 23.05 |
|          32 |              12.79 |      105.49 / 114.59 |       33.45 / 39.13 |

**End-to-end service** (`scripts/benchmark_service.py`, talker + codec):

| Concurrency | Throughput (req/s) | RTF    | TTFA mean / p95 (ms) |
| ----------: | -----------------: | -----: | -------------------: |
|           1 |               0.81 |  4.79x |        73.5 / 75.5   |
|           4 |               2.60 | 13.75x |       116.7 / 130.2  |
|           8 |               3.74 | 19.80x |       158.8 / 184.7  |
|          32 |               7.78 | 36.62x |       342.2 / 399.2  |