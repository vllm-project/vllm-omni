# Benchmarks

This directory contains benchmark suites for evaluating different model families and infrastructure components in vLLM-Omni. Each subfolder targets a different benchmark family with its own scripts, configs, and metrics. See the per-subfolder READMEs for detailed usage.

## Benchmark families

### [TTS](tts/README.md) — Text-to-Speech

Model-agnostic serving benchmarks for TTS models, including Qwen3-TTS and VoxCPM2.

- **Layout**: `tts/bench_tts.py` (serving benchmark driver), `tts/model_configs.yaml` (model registry), `tts/plot_results.py` (result plotting)
- **Dataset**: Seed-TTS full or text-only datasets, plus bundled smoke/design prompts under `build_dataset/`
- **Key metrics**: TTFP (time to first audio packet), E2E latency, RTF (real-time factor), throughput (audio seconds / wall-clock second)

### [Diffusion](diffusion/README.md) — Image and Video Generation

Online-serving benchmark for diffusion image/video models, sending requests to the configured vLLM serving endpoint (`/v1/chat/completions`, `/v1/images/generations`, or `/v1/videos`, depending on backend/task).

- **Tasks**: text-to-image, text-to-video, image-to-image, image-to-video, text+image-to-image, text+image-to-video
- **Datasets**: `vbench`, `trace`, `random`
- **Key metrics**: request throughput, latency percentiles, optional SLO attainment

### [GLM-Image](glm_image/README.md) — Text-to-Image and Image-to-Image

Benchmarks for GLM-Image performance across HuggingFace baseline, vLLM-Omni offline inference, and vLLM-Omni online serving.

- **Layout**: `glm_image/huggingface/` (HF baseline), `glm_image/vllm-omni/` (offline inference), `glm_image/benchmark_glm_image.py` (online serving)
- **Tasks**: text-to-image and image-to-image
- **Key metrics**: request/image throughput, latency percentiles, optional per-stage pipeline timings

### [Distributed](distributed/omni_connectors/README.md) — RDMA Connector Testing

RDMA environment setup and transfer tests for `MooncakeTransferEngineConnector`, including pytest-based single-node checks and manual cross-node benchmarks.

- **Transfer modes**: `copy`, `zerocopy`, `gpu` (GPUDirect)
- **Supports**: single-node pytest suites and manual multi-node/cross-node transfer testing

### [Accuracy](accuracy/README.md) — Image Generation and Editing Quality

Accuracy benchmarks for image generation/editing models, adapting external suites to vLLM-Omni serving and local judge-evaluation flows.

- **Layout**: `accuracy/text_to_image/` (GEBench), `accuracy/image_to_image/` (GEdit-Bench)
- **Method**: generation and judge scoring both run through local `vllm-omni serve` endpoints

### Common serving metrics framework

`vllm_omni/benchmarks/` extends `vllm bench serve --omni` with Omni-specific datasets, backends, and multimodal metrics. Key metrics include:

- **Text output**: TTFT (time to first token), TPOT (time per output token), ITL (inter-token latency)
- **Audio output**: TTFP (time to first audio packet), E2E latency, RTF (real-time factor)
- **Throughput**: request throughput, output token throughput, total token throughput, audio throughput

See `vllm_omni/benchmarks/serve.py` for the `vllm bench serve --omni` runner wrapper and `vllm_omni/benchmarks/metrics/` for Omni metric definitions.

## Adding a new benchmark

1. Create a subfolder under `benchmarks/<name>/` with scripts, configs if needed, and a `README.md`.
2. If comparing against another runtime, use clear backend subfolders where applicable, such as `huggingface/` and `vllm-omni/`, or follow the shared TTS serving benchmark pattern in `tts/`.
3. Add reusable dataset or prompt-building utilities to `build_dataset/` if applicable.
4. Update this README with a link to the new benchmark family.
