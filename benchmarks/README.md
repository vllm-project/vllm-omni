# Benchmarks

This directory contains benchmark suites for evaluating different model families and infrastructure components in vLLM-Omni. Each subfolder targets a different benchmark family with its own scripts, configs, and metrics. See the per-subfolder READMEs for detailed usage.

## Benchmark families

### [Qwen2.5-Omni](qwen2_5_omni/README.md) — Embedding Wrapper Latency

CPU-observed A/B timings for redundant text embedding removal, with real checkpoint
embeddings, warmup, paired blocks, and embedding call-count validation.

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

### [LingBot-Video](lingbot_video/README.md) — Dense and MoE Parity

Manual cross-runtime validation for the dense LingBot-Video pipeline and the
LingBot-Video MoE transformer.

- **Dense pipeline**: decoded-video MAE, MSE, PSNR, latency, and optional steady-state timings
- **MoE transformer**: bitwise router, sparse-block, shared-expert, and full-transformer parity

### [Distributed](distributed/omni_connectors/README.md) — RDMA Connector Testing

RDMA environment setup and transfer tests for `MooncakeTransferEngineConnector`, including pytest-based single-node checks and manual cross-node benchmarks.

- **Transfer modes**: `copy`, `zerocopy`, `gpu` (GPUDirect)
- **Supports**: single-node pytest suites and manual multi-node/cross-node transfer testing

### [Accuracy](accuracy/README.md) — Image Generation and Editing Quality

Accuracy benchmarks for image generation/editing models, adapting external suites to vLLM-Omni serving and local judge-evaluation flows.

- **Layout**: `accuracy/text_to_image/` (GEBench), `accuracy/image_to_image/` (GEdit-Bench)
- **Method**: generation and judge scoring both run through local `vllm-omni serve` endpoints

### [MammothModa2](mammoth_moda2/README.md) — Startup and Weight Loading

Startup / model-loading benchmark for the two-stage MammothModa2 (AR → DiT) deployment, plus a raw safetensors loading micro-benchmark.

- **Layout**: `mammoth_moda2/bench_startup.py` (single-process startup and first/steady request timing), `mammoth_moda2/parse_startup_log.py` (per-stage breakdown from engine logs), `mammoth_moda2/bench_storage_scenarios.sh` (cold/warm page cache on local disk vs. network storage), `mammoth_moda2/raw_load_bench.py` (safetensors → GPU without vLLM)
- **Key metrics**: time to engine ready, per-stage spawn/init/weight-load/profile time, first vs. steady-state request latency

### [Duplex-MPE](duplex_mpe/README.md) — Multi-Party Full-Duplex Selective Participation

Closed-loop, audio-clocked interaction benchmark (`mpeval.unified.v3`) for spoken full-duplex models served over `/v1/realtime?duplex=1` (and turn-based `/v1/realtime` probes).

- **Layout**: `duplex_mpe/evaluate.py` (`python -m benchmarks.duplex_mpe`), `duplex_mpe/realtime_session.py` (reactive WebSocket session driver), `duplex_mpe/configs/judge.example.json`
- **Dataset**: Pinned [`ChengqianMa/Duplex-MPE`](https://huggingface.co/datasets/ChengqianMa/Duplex-MPE) (`2,000` explicit/implicit pairs; `--mini` `20`-pair smoke subset)
- **Key metrics**: target response rate (`TR`), conditional/unconditional answer accuracy (`AA`), bystander silence preservation (`SP`), answering-window yield (`AY`)

### [SOVBench-O](../docs/cli/bench/serve.md#sovbench-o-benchmark) — Streaming Omni Multi-Choice Sessions

Multiple-choice questions over audio-video streams that keep advancing as a session
progresses (issue #10954): every turn reveals one new clip, the model replays the clips
accumulated since the session start, and the dialogue history is built from the model's own
previous predictions. Ports the reference protocol, prompt text, and scorer from the
StreamOV repository (SOVBench-O: 172 sessions / 969 groups / 1739 turns).

- **Layout**: `vllm_omni/benchmarks/data_modules/sovbench_o_dataset.py` (session parsing,
  per-turn prompt construction, media decode/cache), `sovbench_o_eval.py` (option-letter
  extraction + accuracy)
- **Datasets**: annotations `streamov_bench.jsonl` (pinned StreamOV repository file) and media
  `benchvideos.tar` (Hugging Face `PatrickyMing/StreamOV`)

- **Key metrics**: overall accuracy over parseable predictions, per turn type
  (Real-time / Recall / Proactive), per group size, per session length; per-turn predictions
  are saved for independent rescoring

### Common serving metrics framework

`vllm_omni/benchmarks/` extends `vllm bench serve --omni` with Omni-specific datasets, backends, and multimodal metrics. Key metrics include:

- **Text output**: TTFT (time to first token), TPOT (time per output token), ITL (inter-token latency)
- **Audio output**: TTFP (time to first audio packet), E2E latency, RTF (real-time factor)
- **Throughput**: request throughput, output token throughput, total token throughput, audio throughput

See `vllm_omni/benchmarks/serve.py` for the `vllm bench serve --omni` runner wrapper and `vllm_omni/benchmarks/metrics/` for Omni metric definitions.

The [Qwen3-Omni Seed-TTS Realtime comparison](../docs/cli/bench/serve.md#seed-tts-reference-speech-as-real-audio-input)
uses this framework to compare explicit-commit and server-VAD turns.

## Adding a new benchmark

1. Create a subfolder under `benchmarks/<name>/` with scripts, configs if needed, and a `README.md`.
2. If comparing against another runtime, use clear backend subfolders where applicable, such as `huggingface/` and `vllm-omni/`, or follow the shared TTS serving benchmark pattern in `tts/`.
3. Add reusable dataset or prompt-building utilities to `build_dataset/` if applicable.
4. Update this README with a link to the new benchmark family.
