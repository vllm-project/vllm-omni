# Benchmarks

Benchmark suites for evaluating model families and infrastructure components in vLLM-Omni, organized by modality. See per-directory READMEs for detailed usage.

## Directory layout

```
benchmarks/
├── tts/              Text-to-speech serving benchmarks + model-specific tests
├── diffusion/        Image/video generation serving benchmarks + model-specific tests
├── accuracy/         Image generation/editing quality benchmarks
├── distributed/      RDMA connector transfer tests
└── kernels/          Kernel-level micro-benchmarks and auto-tuners
```

## Benchmark families

### [TTS](tts/README.md) — Text-to-Speech

Serving benchmarks for TTS models

- **Serving benchmark**: `tts/bench_tts.py` (wraps `vllm bench serve --omni`)
- **Model registry**: `tts/model_configs.yaml` (model-specific benchmark configurations)
- **Datasets**: `tts/datasets/` (bundled smoke/design prompt sets, download instructions for full Seed-TTS corpus)
- **Key metrics**: TTFP, E2E latency, RTF, audio throughput, optional quality metrics (WER, SIM, UTMOS)
- **Model-specific directories**: `tts/fish-speech` for DAC-code cache performance

### [Diffusion](diffusion/README.md) — Image and Video Generation

Benchmarks for diffusion image/video models

- **Serving benchmark**: `diffusion/diffusion_benchmark_serving.py`
- **Key metrics**: request throughput, latency percentiles, SLO attainment, per-stage durations
- **Recipes**: `diffusion/recipes/` (reference results and directions for Qwen-Image, Wan2.2)
- **Model-specific directories**: `diffusion/glm-image/` for HuggingFace baseline and offline benchmarks

### [Accuracy](accuracy/README.md) — Image Generation and Editing Quality

Accuracy benchmarks for image generation/editing models, adapting external suites to vLLM-Omni serving and local judge-evaluation flows.

- **Layout**: `accuracy/text_to_image/` (GEBench), `accuracy/image_to_image/` (GEdit-Bench)
- **Method**: generation and judge scoring both run through local `vllm-omni serve` endpoints

### [LingBot-Video](lingbot_video/README.md) — Dense and MoE Parity

Manual cross-runtime validation for the dense LingBot-Video pipeline and the
LingBot-Video MoE transformer.

- **Dense pipeline**: decoded-video MAE, MSE, PSNR, latency, and optional steady-state timings
- **MoE transformer**: bitwise router, sparse-block, shared-expert, and full-transformer parity

### [Distributed](distributed/omni_connectors/README.md) — RDMA Connector Testing

RDMA environment setup and transfer tests for `MooncakeTransferEngineConnector`, including pytest-based single-node checks and manual cross-node benchmarks.

- **Transfer modes**: `copy`, `zerocopy`, `gpu` (GPUDirect)
- **Supports**: single-node pytest suites and manual multi-node/cross-node transfer testing

### [Kernels](kernels/README.md) — Kernel Micro-Benchmarks

Kernel-level micro-benchmarks and auto-tuners for custom operators.

- **MoT GEMM**: `kernels/mot_linear_benchmarks.py` (Triton kernel auto-tuner for Mixture-of-Tokens GEMM operations)
- **Attention backends**: `kernels/bench_attention_backends.py` (diffusion attention kernel comparison across SDPA, cuDNN, Flash, FA4, FlashInfer, SageAttn3)

### Common serving metrics framework

`vllm_omni/benchmarks/` extends `vllm bench serve --omni` with Omni-specific datasets, backends, and multimodal metrics. Key metrics include:

- **Text output**: TTFT (time to first token), TPOT (time per output token), ITL (inter-token latency)
- **Audio output**: TTFP (time to first audio packet), E2E latency, RTF (real-time factor)
- **Throughput**: request throughput, output token throughput, total token throughput, audio throughput

See `vllm_omni/benchmarks/serve.py` for the `vllm bench serve --omni` runner wrapper and `vllm_omni/benchmarks/metrics/` for Omni metric definitions.

## Adding a new benchmark

When adding a new benchmark, make sure to maintain the current structure of `benchmarks/`. This includes:

1. Include benchmarks in appropriate modality (`tts/`, `diffusion/`) or other directory (`distributed/`, `kernels/`, `accuracy/`) or add a new top-level directory.
2. For model-specific benchmarks within an existing modality, create a subdirectory under the modality (e.g., `tts/fish-speech/`, `diffusion/glm-image/`).
3. Update the subdirectory's `README.md` with: new benchmark purpose, any additional prerequisites, usage examples, CLI arguments table, and key metrics
4. For any new datasets or prompt files, place under the modality's `datasets/` directory if applicable.
