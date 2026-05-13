# SenseNova-U1-8B-MoT

## Architecture

SenseNova-U1 is a unified Qwen3-based LLM with Mixture-of-Tokenizers (MoT) attention. Unlike two-stage pipelines (e.g., BAGEL), it handles text encoding, optional reasoning ("think mode"), and flow-matching-based image denoising entirely within a single diffusion stage.

| Feature | Description |
| :------ | :---------- |
| **Model type** | Unified LLM (single-stage diffusion pipeline) |
| **Base LLM** | Qwen3 with MoT attention and 3D RoPE |
| **Image generation** | Flow-matching Euler sampler, no separate VAE |
| **Think mode** | Optional chain-of-thought reasoning before image generation |
| **Parallelism** | Tensor Parallelism (TP) with fused QKV and fused gate/up projections |
| **Modalities** | text2img, img2img, img2text (understanding), text2text |

## Quick Start

```bash
cd examples/offline_inference/sensenova_u1

# Text-to-image
python end2end.py --prompt "A cute cat" --think

# Image-to-image editing
python end2end.py --prompt "Turn this into an oil painting" \
                  --image input.png --think

# Image understanding (img2text)
python end2end.py --modality img2text \
                  --prompt "Describe this image in detail" \
                  --image photo.jpg

# Text-to-text (pure chat)
python end2end.py --modality text2text \
                  --prompt "What is the capital of France?"

# Custom resolution (image generation)
python end2end.py --prompt "A futuristic cityscape at sunset" \
                  --width 2048 --height 1024 --think
```

> **Note**: Default configuration works on a single **NVIDIA A100 (80GB)** or **H100** GPU.

## Text-to-Image (t2i)

Standard text-to-image with optional think mode:

```bash
python end2end.py \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light" \
    --width 1536 --height 2720 \
    --think --print-think
```

## Image-to-Image Editing (img2img)

Pass one or more `--image` paths to trigger img2img mode. The model uses the
input image(s) as visual context for editing:

```bash
# Single input image
python end2end.py \
    --prompt "Add a sunset sky in the background" \
    --image photo.jpg \
    --width 2048 --height 2048 \
    --think

# Multiple reference images
python end2end.py \
    --prompt "Combine the style of Image-1 with the content of Image-2" \
    --image style_ref.png content_ref.png \
    --width 2048 --height 2048 \
    --think
```

### Dual CFG for img2img

img2img supports dual classifier-free guidance (text CFG + image CFG):

```bash
python end2end.py \
    --prompt "Make the person smile" \
    --image portrait.jpg \
    --cfg-scale 4.0 \
    --img-cfg-scale 2.0 \
    --think
```

| `img_cfg_scale` | Behavior |
| :-------------- | :------- |
| `1.0` (default) | Text-only CFG: guidance = image-condition → full-condition |
| `== cfg_scale` | Standard CFG: guidance = unconditional → full-condition |
| Other value | Dual CFG: separate text and image guidance strengths |

## Full Parameter Reference

### General Parameters

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--modality` | `auto` | Task modality: `auto`, `text2img`, `img2img`, `img2text`, `text2text` |
| `--prompt` | "A cute cat..." | Text prompt / editing instruction / question |
| `--image` | None | Input image path(s) for img2img or img2text |

### Image Generation Parameters (text2img / img2img)

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--height` | 2048 | Height of generated image (pixels) |
| `--width` | 2048 | Width of generated image (pixels) |
| `--seed` | 42 | Random seed for reproducibility |
| `--num-steps` | 50 | Number of denoising steps |
| `--cfg-scale` | 4.0 | Text classifier-free guidance scale |
| `--img-cfg-scale` | 1.0 | Image CFG scale (img2img only, 1.0 = disabled) |
| `--cfg-norm` | "none" | CFG normalization: none, global, channel, cfg_zero_star (t2i only) |
| `--timestep-shift` | 3.0 | Timestep shift for flow-matching schedule |
| `--t-eps` | 0.02 | Epsilon for timestep schedule |
| `--think` | False | Enable think mode |
| `--print-think` | False | Print think text to stdout |

### Text Generation Parameters (text2text / img2text)

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--max-tokens` | 512 | Maximum number of tokens to generate |
| `--do-sample` | False | Use sampling instead of greedy decoding |
| `--temperature` | 0.7 | Sampling temperature (higher = more diverse) |

### Infrastructure Parameters

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `--model` | `SenseNova/SenseNova-U1-8B-MoT` | HuggingFace model ID or local path |
| `--output` | `.` | Output directory for saved images |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--enforce-eager` | False | Disable torch.compile |
| `--enable-cpu-offload` | False | Enable module-wise (sequential) CPU offload to reduce peak VRAM |

## Reducing GPU Memory Usage

For hardware with limited VRAM, enable **module-wise CPU offload** with
`--enable-cpu-offload`. The pipeline implements `SupportsModuleOffload`, so the
vision encoder (`vision_model`) and the Qwen3 LLM (`language_model`) are
swapped between CPU and GPU on demand:

- During text/vision encoding, the LLM is on CPU.
- During the diffusion loop, the vision encoder is on CPU.
- Lightweight FM modules stay resident on GPU.

This lowers peak VRAM at the cost of extra CPU<->GPU transfers (use pinned
memory) — useful for running SenseNova-U1 on consumer-grade GPUs.

```bash
# Text-to-image with CPU offload
python end2end.py \
    --prompt "A cute cat sitting on a windowsill" \
    --width 2048 --height 2048 \
    --enable-cpu-offload --think

# Image-to-image editing with CPU offload
python end2end.py \
    --prompt "Turn this into an oil painting" \
    --image input.png \
    --width 2048 --height 2048 \
    --enable-cpu-offload --think

# Image understanding (img2text) with CPU offload
python end2end.py \
    --modality img2text \
    --prompt "Describe this image in detail" \
    --image photo.jpg \
    --enable-cpu-offload
```

> **Notes**
> - CPU offload is **single-GPU only** (incompatible with `--tensor-parallel-size > 1`).
> - First-step latency is higher because of the cold-start CPU<->GPU transfers.
> - For more details and other offloading strategies, see
>   [CPU Offloading for Diffusion Models](../../../docs/user_guide/diffusion/cpu_offload_diffusion.md).

## Reproducing the E2E Test

The following command reproduces the pixel-validated CI test case:

```bash
python end2end.py \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus." \
    --width 1536 --height 2720 \
    --seed 42 --num-steps 50 \
    --cfg-scale 4.0 --timestep-shift 3.0 --cfg-norm none \
    --think --print-think \
    --output outputs
```

The corresponding pytest:

```bash
pytest -s -v tests/e2e/offline_inference/test_sensenova_u1_text2img.py \
    -m "advanced_model" --run-level "advanced_model"
```

The img2img counterpart lives at
`tests/e2e/offline_inference/test_sensenova_u1_img2img.py`.

## Online Serving

For OpenAI-compatible API serving, see
[`examples/online_serving/sensenova_u1/`](../../examples/online_serving/sensenova_u1/).
