# BAGEL-7B-MoT

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/bagel>.

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Architecture

BAGEL-7B-MoT is a Mixture-of-Transformers (MoT) model supporting both image generation and understanding. It offers two deployment topologies:

| Topology | Stages | Description |
| :------- | :----- | :---------- |
| **Two-stage** (default) | Stage 0 (Thinker, AR) + Stage 1 (DiT, Diffusion) | Thinker handles text/understanding via vLLM AR engine; DiT handles image generation. KV cache is transferred between stages. |
| **Single-stage** | Stage 0 (DiT, Diffusion) only | The DiT stage contains a full LLM, ViT, VAE, and tokenizer internally. All modalities are handled within a single diffusion process. |

Both topologies support all four modalities: `text2img`, `img2img`, `img2text`, `text2text`.

## Quick Start

```bash
cd examples/offline_inference/bagel

# Default two-stage mode (auto-detected)
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat"

# Single-stage mode
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat" \
                  --deploy-config vllm_omni/deploy/bagel_single_stage.yaml
```

> **Note**: These examples work with the default configuration on an **NVIDIA A100 (80GB)**. For dual-GPU setups, modify the deploy YAML to distribute stages across devices.

## Modality Control

Control the mode using the `--modality` argument:

| Modality | Input | Output | Description |
| :------- | :---- | :----- | :---------- |
| `text2img` | Text | Image | Generate images from text prompts |
| `img2img` | Image + Text | Image | Transform images using text guidance |
| `img2text` | Image + Text | Text | Generate text descriptions from images |
| `text2text` | Text | Text | Pure text generation (language model mode) |

### Text to Image (text2img)

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat" \
                  --steps 50
```

### Image to Image (img2img)

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2img \
                  --image-path /path/to/image.jpg \
                  --prompts "Let the woman wear a blue dress"
```

### Image to Text (img2text)

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2text \
                  --image-path /path/to/image.jpg \
                  --prompts "Describe this image in detail"
```

### Text to Text (text2text)

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2text \
                  --prompts "What is the capital of France?"

# Load prompts from a text file (one prompt per line):
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2text \
                  --txt-prompts /path/to/prompts.txt
```

## Think Mode

Think mode enables the model to generate `<think>...</think>` planning/reasoning tokens before producing the final output. This improves generation quality for complex prompts.

- **Two-stage**: The Thinker (AR) stage decodes think tokens, then transfers the augmented KV cache to the DiT stage for image generation.
- **Single-stage**: The DiT's internal LLM generates think tokens in-place before proceeding to denoise.

```bash
# Think + text2img: plan before generating
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A futuristic city with flying cars" \
                  --think \
                  --max-think-tokens 1000

# Think + img2img: reason about the edit
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2img \
                  --image-path /path/to/image.jpg \
                  --prompts "Make it look like a watercolor painting" \
                  --think

# Think + img2text: reason before describing
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2text \
                  --image-path /path/to/image.jpg \
                  --prompts "What is happening in this image?" \
                  --think

# Think + text2text: chain-of-thought reasoning
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2text \
                  --prompts "Solve: 23 * 47" \
                  --think
```

Think mode parameters:

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--think` | `False` | Enable thinking mode |
| `--max-think-tokens` | `1000` | Maximum tokens for think generation |
| `--do-sample` | `False` | Enable sampling (vs. greedy) for text generation |
| `--text-temperature` | `0.3` | Temperature for text generation sampling |

## Classifier-Free Guidance (CFG)

CFG controls the trade-off between prompt fidelity and diversity. These parameters apply to image generation modalities (`text2img`, `img2img`).

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A photorealistic portrait" \
                  --cfg-text-scale 6.0 \
                  --cfg-img-scale 2.0 \
                  --negative-prompt "blurry, low quality, distorted" \
                  --cfg-interval 0.4 1.0 \
                  --cfg-renorm-type global \
                  --cfg-renorm-min 0.0
```

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--cfg-text-scale` | `4.0` | Text CFG scale (higher = more prompt-adherent) |
| `--cfg-img-scale` | `1.5` | Image CFG scale (for img2img) |
| `--negative-prompt` | `None` | Negative prompt for CFG conditioning |
| `--cfg-interval` | pipeline default | CFG active interval `[start, end]` as fractions of total timesteps |
| `--cfg-renorm-type` | `None` | Renormalization type: `global`, `text_channel`, `channel` |
| `--cfg-renorm-min` | `None` | Minimum renormalization value |
| `--cfg-parallel-size` | `1` | CFG parallel size: `1` = batched (single GPU), `2` = 2-branch parallel, `3` = full 3-GPU parallel |

## Deployment Topologies

### Two-Stage (Default)

The default topology auto-detected from the model. No extra flags needed.

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat"
```

The pipeline is defined in [`bagel.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel.yaml). Stage 0 (Thinker) and Stage 1 (DiT) share GPU 0 by default. For dual-GPU setups, customize the deploy YAML and set `devices: "1"` for stage 1.

### Single-Stage

Pass the single-stage deploy config via `--deploy-config`:

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat" \
                  --deploy-config vllm_omni/deploy/bagel_single_stage.yaml
```

See [`bagel_single_stage.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel_single_stage.yaml) for configuration details. The `pipeline: bagel_single_stage` field selects the single-stage topology from the pipeline registry.

### Tensor Parallelism (TP)

For larger models or multi-GPU environments, customize the deploy YAML (see [`bagel.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/bagel.yaml)) and set per-stage `tensor_parallel_size` and `devices`:

```yaml
# Example: TP=2 on GPUs 0,1 for the Thinker stage
stages:
  - stage_id: 0
    tensor_parallel_size: 2
    devices: "0,1"
```

Then pass the custom deploy YAML:

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat" \
                  --deploy-config /path/to/custom_bagel.yaml
```

### FP8 Quantization

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat" \
                  --quantization fp8
```

## Command Line Reference

### Core Arguments

| Argument | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `--model` | string | `ByteDance-Seed/BAGEL-7B-MoT` | Model path or HuggingFace name |
| `--modality` | choice | `text2img` | `text2img`, `img2img`, `img2text`, `text2text` |
| `--prompts` | list | `None` | Input text prompts |
| `--txt-prompts` | string | `None` | Path to text file with one prompt per line |
| `--image-path` | string | `None` | Input image path (required for `img2img`/`img2text`) |
| `--output` | string | `.` | Output directory for saved images |
| `--steps` | int | `50` | Number of diffusion inference steps |
| `--seed` | int | `None` | Random seed for reproducibility |

### Think Mode Arguments

| Argument | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `--think` | flag | `False` | Enable `<think>...</think>` planning/reasoning |
| `--max-think-tokens` | int | `1000` | Maximum tokens for think generation |
| `--do-sample` | flag | `False` | Use sampling instead of greedy decoding |
| `--text-temperature` | float | `0.3` | Sampling temperature for text generation |

### CFG Arguments

| Argument | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `--cfg-text-scale` | float | `4.0` | Text CFG guidance scale |
| `--cfg-img-scale` | float | `1.5` | Image CFG guidance scale |
| `--negative-prompt` | string | `None` | Negative prompt for CFG |
| `--cfg-parallel-size` | int | `1` | CFG parallel GPU count (1, 2, or 3) |
| `--cfg-interval` | float[2] | pipeline default | CFG active window `[start, end]` |
| `--cfg-renorm-type` | string | `None` | `global`, `text_channel`, or `channel` |
| `--cfg-renorm-min` | float | `None` | Minimum renormalization value |

### Engine Arguments

| Argument | Type | Default | Description |
| :------- | :--- | :------ | :---------- |
| `--deploy-config` | string | `None` | Path to deploy YAML (auto-detected if omitted) |
| `--stage-configs-path` | string | `None` | [Deprecated] Legacy path to `stage_args` YAML; prefer `--deploy-config` |
| `--worker-backend` | choice | `process` | `process` or `ray` |
| `--ray-address` | string | `None` | Ray cluster address |
| `--quantization` | string | `None` | Quantization method (e.g. `fp8`) |
| `--log-stats` | flag | `False` | Enable statistics logging |
| `--init-timeout` | int | `300` | Initialization timeout (seconds) |
| `--batch-timeout` | int | `5` | Batch timeout (seconds) |
| `--enable-diffusion-pipeline-profiler` | flag | `False` | Profile diffusion stage durations |

## FAQ

- If you encounter OOM errors, try decreasing `max_model_len` or `gpu_memory_utilization` in the deploy YAML.

**Two-stage VRAM usage:**

| Stage | VRAM |
| :---- | :--- |
| Stage 0 (Thinker) | **15.04 GiB + KV Cache** |
| Stage 1 (DiT) | **26.50 GiB** |
| Total | **~42 GiB + KV Cache** |

**Single-stage VRAM usage:** The DiT loads the full model (~42 GiB) in one process.

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/bagel/end2end.py"
    ``````
