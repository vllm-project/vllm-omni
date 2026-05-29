# GLM-Image Benchmarks

Benchmark GLM-Image T2I (text-to-image) and I2I (image-to-image) performance across three backends: HuggingFace baseline, vLLM-Omni offline, and vLLM-Omni online serving.

## Benchmarks

| Benchmark | Script | Description |
|-----------|--------|-------------|
| HuggingFace Baseline | `huggingface/inference.py` | Single-GPU transformers + diffusers pipeline |
| vLLM-Omni Offline | `vllm-omni/inference.py` | Offline inference with continuous batching |
| vLLM-Omni Online | `benchmark_glm_image.py` | Online serving via `/v1/chat/completions` |

## HuggingFace Baseline

Single-request sequential inference using the reference HuggingFace pipeline.

```bash
# T2I
CUDA_VISIBLE_DEVICES=0 python benchmarks/glm_image/huggingface/inference.py \
    --model-path /path/to/GLM-Image --mode t2i --num-prompts 10

# I2I
CUDA_VISIBLE_DEVICES=0 python benchmarks/glm_image/huggingface/inference.py \
    --model-path /path/to/GLM-Image --mode i2i --num-prompts 10
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `zai-org/GLM-Image` | Model path |
| `--mode` | `t2i` | `t2i` or `i2i` |
| `--dataset-path` | `prompt/prompt.json` | Path to prompt.json |
| `--num-prompts` | `10` | Number of images to generate |
| `--width` / `--height` | `1024` | Output image size |
| `--num-inference-steps` | `50` | Diffusion denoising steps |
| `--output-dir` | `benchmarks/glm_image/huggingface/outputs` | Output directory |
| `--output-file` | - | JSON file for metrics |

## vLLM-Omni Offline

Multi-GPU offline inference with pipeline parallelism and continuous batching.

```bash
# T2I
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/glm_image/vllm-omni/inference.py \
    --model-path /path/to/GLM-Image --mode t2i --num-prompts 10

# I2I
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/glm_image/vllm-omni/inference.py \
    --model-path /path/to/GLM-Image --mode i2i --num-prompts 10
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `zai-org/GLM-Image` | Model path |
| `--deploy-config` | - | Deploy config YAML |
| `--mode` | `t2i` | `t2i` or `i2i` |
| `--dataset-path` | `prompt/prompt.json` | Path to prompt.json |
| `--num-prompts` | `10` | Number of images to generate |
| `--width` / `--height` | `1024` | Output image size |
| `--num-inference-steps` | `50` | Diffusion denoising steps |
| `--output-dir` | `benchmarks/glm_image/vllm-omni/outputs` | Output directory |
| `--output-file` | - | JSON file for metrics |
| `--stage-init-timeout` | `600` | Stage initialization timeout (s) |

### Latency Computation

In offline mode all requests are submitted simultaneously and processed with continuous batching. The per-request latency is computed by summing the actual per-stage times (with `stage_0_gen_ms` diffed against the previous request to remove accumulated queue/scheduling wait).

## vLLM-Omni Online Serving

### Start the server

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/GLM-Image \
    --omni --port 8091 --host 0.0.0.0 \
    --served-model-name glm-image
```

### Run the benchmark

```bash
# T2I
python benchmarks/glm_image/benchmark_glm_image.py \
    --mode t2i --num-prompts 10 --model glm-image

# I2I
python benchmarks/glm_image/benchmark_glm_image.py \
    --mode i2i --num-prompts 10 --model glm-image

# Custom dataset
python benchmarks/glm_image/benchmark_glm_image.py \
    --mode i2i --dataset custom \
    --dataset-path prompts.json --num-prompts 5
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `t2i` | `t2i` or `i2i` |
| `--dataset` | `prompt` | `prompt`, `random`, or `custom` |
| `--dataset-path` | - | JSON file path (required for `custom`) |
| `--num-prompts` | `10` | Number of benchmark requests |
| `--max-concurrency` | `1` | Max concurrent requests |
| `--request-rate` | `inf` | Requests per second (Poisson arrival) |
| `--warmup-requests` | `1` | Warmup requests before measurement |
| `--width` / `--height` | `1024` | Output image size |
| `--num-inference-steps` | `50` | Diffusion denoising steps |
| `--seed` | - | Random seed |
| `--model` | `default` | Model name (must match `--served-model-name`) |
| `--host` | `localhost` | Server host |
| `--port` | `8091` | Server port |
| `--output-file` | - | JSON output file for metrics |
| `--num-input-images` | `1` | Number of input images for random I2I |

## Dataset

The default dataset is hosted on [HuggingFace](https://huggingface.co/datasets/JaredforReal/glm-image-bench) (`prompt.json`). It is automatically downloaded and cached to `prompt/prompt.json` on first run. No manual setup needed.

Each entry contains:

- `t2i_prompt`: Text prompt for text-to-image generation
- `i2i_prompt`: Text prompt for image-to-image editing
- `image_url`: Source image URL for I2I (downloaded and cached on first use)

Custom datasets use the same JSON format and can be provided via `--dataset-path`.

## Pipeline Timings

All three benchmarks report per-stage pipeline timings (in milliseconds):

| Key | Description |
|-----|-------------|
| `preprocess_ms` | Input preprocessing (tokenization, multimodal encoding) |
| `stage_0_gen_ms` | AR (autoregressive) model generation time |
| `ar2diffusion_ms` | AR output to diffusion input conversion |
| `stage_1_gen_ms` | Diffusion model denoising time |
| `queue_wait_ms` | Queue wait time before processing |

The stages are ordered by execution: `preprocess → stage_0 (AR) → ar2diffusion → stage_1 (Diffusion)`.

## Sample Results

Tested on 2x GPU with 10 prompts, 1024x1024, 50 denoising steps:

| Backend | Mode | Latency Mean (s) | Throughput (img/s) |
|---------|------|-------------------|--------------------|
| HuggingFace | T2I | 72.6 | 0.014 |
| HuggingFace | I2I | 70.9 | 0.014 |
| vLLM-Omni Offline | T2I | 35.0 | 0.044 |
| vLLM-Omni Offline | I2I | 31.0 | 0.053 |
| vLLM-Omni Online | T2I | 38.8 | 0.026 |
| vLLM-Omni Online | I2I | 34.7 | 0.029 |
