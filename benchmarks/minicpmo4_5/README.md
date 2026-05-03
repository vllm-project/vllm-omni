# MiniCPM-o 4.5 Benchmark

Benchmarks for MiniCPM-o 4.5 text/image/video to audio generation, comparing
the vLLM-Omni non-async pipeline against the HuggingFace reference streaming
path.

## Prerequisites

```bash
pip install --no-build-isolation 'minicpmo-utils[all]'
pip install soundfile pillow opencv-python-headless numpy
```

Use the same visible GPUs for vLLM-Omni and HF runs when comparing speed:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

The default vLLM-Omni stage config is:

```text
vllm_omni/model_executor/stage_configs/minicpmo.yaml
```

This config places thinker on GPU 0 and talker/code2wav on GPU 1.

## Quick Start

Run all benchmark modes and modalities:

```bash
bash benchmarks/minicpmo4_5/run_benchmark.sh
```

Common options:

```bash
# vLLM-Omni only
MODE=non_async bash benchmarks/minicpmo4_5/run_benchmark.sh

# HF only
MODE=hf bash benchmarks/minicpmo4_5/run_benchmark.sh

# Text-only smoke
MODALITIES=text NUM_REPEATS=1 bash benchmarks/minicpmo4_5/run_benchmark.sh

# Custom local model path
MODEL=/path/to/MiniCPM-o-4_5 bash benchmarks/minicpmo4_5/run_benchmark.sh
```

Results are written to `bench_results/minicpmo4_5/` by default.

## Direct Python Usage

```bash
python benchmarks/minicpmo4_5/bench_minicpmo4_5.py \
  --model-path openbmb/MiniCPM-o-4_5 \
  --mode all \
  --modalities text,text+image,text+video \
  --num-repeats 3 \
  --cuda-visible-devices 0,1 \
  --output-dir bench_results/minicpmo4_5
```

## Modal Runner

The local Modal helper launches the same benchmark on a remote GPU machine:

```bash
modal run scripts/modal_launch_benchmark.py \
  --modes non_async,hf \
  --modalities text,text+image,text+video \
  --num-repeats 3
```

The Modal wrapper requests the same GPU type/count for vLLM-Omni and HF and
passes the same `CUDA_VISIBLE_DEVICES` value to both subprocesses.

## Metrics

- **Latency**: wall-clock time from request start to final output.
- **RTF**: latency divided by generated audio duration. Lower is better.
- **Audio duration**: generated audio samples divided by sample rate.
- **Output text tokens**: generated text token count when text output is exposed.

The benchmark intentionally reports full-response latency instead of TTFT. The
current non-async MiniCPM path does not expose a reliable time-to-first-audio
packet metric.

## Result Files

Each run writes a JSON report named like:

```text
minicpmo45_bench_YYYYMMDD_HHMMSS.json
```

The report includes CUDA device metadata, aggregate latency/RTF statistics, and
per-request errors with exception type and traceback when a request fails.

## Notes

- HF timing includes streaming prefill and streaming generation.
- The HF `chat(..., generate_audio=True)` audio path is not used here because it
  fails in the current tested runtime; the working HF path is
  `streaming_prefill` plus `streaming_generate`.
- The public vLLM-Omni `py_generator=True` path closes `Omni` after the first
  request, so the benchmark uses the internal generation iterator for repeated
  requests on one engine instance.
