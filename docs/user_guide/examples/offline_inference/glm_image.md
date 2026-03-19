# GLM-Image Multistage End-to-End Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/glm_image>.


This example demonstrates how to run GLM-Image with the vLLM-Omni multistage architecture.

## Architecture

GLM-Image uses a 2-stage pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                     GLM-Image Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 0 (AR Model)              Stage 1 (Diffusion)        │
│  ┌─────────────────┐            ┌─────────────────────┐     │
│  │ vLLM-optimized  │            │  GlmImagePipeline   │     │
│  │ GlmImageFor     │  prior     │  ┌───────────────┐  │     │
│  │ Conditional     │──tokens───►│  │ DiT Denoiser  │  │     │
│  │ Generation      │            │  └───────────────┘  │     │
│  │ (9B AR model)   │            │         │          │     │
│  └─────────────────┘            │         ▼          │     │
│         ▲                       │  ┌───────────────┐  │     │
│         │                       │  │  VAE Decode   │──┼──► Image
│    Text/Image                   │  └───────────────┘  │     │
│      Input                      └─────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **vLLM-optimized AR**: Uses PagedAttention and tensor parallelism for faster prior token generation
- **Flexible deployment**: AR and Diffusion stages can run on different GPUs
- **Text-to-Image**: Generate images from text descriptions
- **Image-to-Image**: Edit existing images with text prompts

## Usage

### Text-to-Image

```bash
python end2end.py \
    --model-path /path/to/glm-image \
    --config-path ../../vllm_omni/model_executor/stage_configs/glm_image.yaml \
    --prompt "A beautiful sunset over the ocean with sailing boats" \
    --height 1024 \
    --width 1024 \
    --output output_t2i.png
```

### Image-to-Image (Image Editing)

```bash
python end2end.py \
    --model-path /path/to/glm-image \
    --config-path ../../vllm_omni/model_executor/stage_configs/glm_image.yaml \
    --prompt "Transform this scene into a winter wonderland" \
    --image input.png \
    --output output_i2i.png
```

### With Custom Parameters

```bash
python end2end.py \
    --model-path /path/to/glm-image \
    --config-path ../../vllm_omni/model_executor/stage_configs/glm_image.yaml \
    --prompt "A photorealistic cat sitting on a window sill" \
    --height 1024 \
    --width 1024 \
    --num-inference-steps 50 \
    --guidance-scale 1.5 \
    --seed 42 \
    --output output.png
```

## Shell Scripts

### Run Text-to-Image

```bash
./run_t2i.sh
```

### Run Image-to-Image

```bash
./run_i2i.sh --image /path/to/input.png
```

## Stage Configuration

The stage config (`glm_image.yaml`) defines:

- **Stage 0 (AR)**: Uses `GPUARWorker` with vLLM engine

  - Model: `GlmImageForConditionalGeneration`
  - Output: `token_ids` (prior tokens)

- **Stage 1 (Diffusion)**: Uses diffusion engine
  - Model: `GlmImagePipeline`
  - Output: Generated image

See `vllm_omni/model_executor/stage_configs/glm_image.yaml` for full configuration.

## Comparison with Single-Stage

| Aspect      | Single-Stage (transformers) | Multistage (vLLM)   |
| ----------- | --------------------------- | ------------------- |
| AR Model    | transformers native         | vLLM PagedAttention |
| Memory      | Higher (no KV cache opt)    | Lower (optimized)   |
| Throughput  | Lower                       | Higher              |
| Flexibility | Single GPU                  | Multi-GPU support   |

## Troubleshooting

### OOM Error

Try reducing memory usage:

```bash
# In glm_image.yaml, adjust:
gpu_memory_utilization: 0.5  # Reduce from 0.6
```

### Slow Initialization

The first run loads model weights. Subsequent runs are faster:

```bash
--stage-init-timeout 900  # Increase timeout for slow storage
```

## Requirements

- vLLM-Omni with GLM-Image support
- CUDA-capable GPU (recommended: H100/A100 with 80GB)
- GLM-Image model weights

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/glm_image/end2end.py"
    ``````
??? abstract "run_i2i.sh"
    ``````sh
    --8<-- "examples/offline_inference/glm_image/run_i2i.sh"
    ``````
??? abstract "run_t2i.sh"
    ``````sh
    --8<-- "examples/offline_inference/glm_image/run_t2i.sh"
    ``````
