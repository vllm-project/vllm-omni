# ERNIE-Image for text-to-image generation

## Summary

- Vendor: Baidu
- Model: `baidu/ERNIE-Image` / `baidu/ERNIE-Image-Turbo`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`baidu/ERNIE-Image` or `baidu/ERNIE-Image-Turbo` with vLLM-Omni for
high-quality text-to-image generation.

ERNIE-Image is an 8B-parameter Diffusion Transformer (DiT) model that achieves
state-of-the-art performance among open-weight text-to-image models. Key
strengths include:

- **Text rendering:** Excellent for dense, long-form, and layout-sensitive
  text — ideal for posters, infographics, and UI-like images.
- **Instruction following:** Reliably handles complex prompts with multiple
  objects, detailed relationships, and knowledge-intensive descriptions.
- **Structured generation:** Effective for posters, comics, storyboards, and
  multi-panel compositions where layout matters.
- **Style coverage:** Supports realistic photography, design-oriented imagery,
  and stylized aesthetic outputs.
- **Practical deployment:** Can run on consumer GPUs with 24GB VRAM.

Two model variants are provided:

1. **ERNIE-Image** — The SFT model with stronger general-purpose capability
   and instruction fidelity, typically using 50 inference steps.
2. **ERNIE-Image-Turbo** — Optimized by DMD and RL for faster speed and
   higher aesthetics, requiring only 8 inference steps.

## References

- Upstream model card: <https://huggingface.co/baidu/ERNIE-Image>
- Upstream model card (Turbo): <https://huggingface.co/baidu/ERNIE-Image-Turbo>
- GitHub: <https://github.com/baidu/ERNIE-Image>
- Blog: <https://ernie-image.github.io/>

## Hardware Support

This recipe currently documents tested configurations for CUDA GPU serving.
Add more sections for other hardware as community validation lands.

## GPU

### 1 x RTX 4090 (Single GPU, 24GB VRAM, Minimum Recommended)

#### Command

**ERNIE-Image (full model):**

```bash
vllm serve baidu/ERNIE-Image --omni \
  --enable-layerwise-offload \
  --port 8091
```

**ERNIE-Image-Turbo (distilled, faster):**

```bash
vllm serve baidu/ERNIE-Image-Turbo --omni \
  --enable-layerwise-offload \
  --port 8091
```

### 2 x RTX 4090 (Multi-GPU, 24GB VRAM)

#### Command

**ERNIE-Image (full model):**

```bash
vllm serve baidu/ERNIE-Image --omni \
  --tensor-parallel-size 2 \
  --enable-cpu-offload \
  --port 8091
```

**ERNIE-Image-Turbo (distilled, faster):**

```bash
vllm serve baidu/ERNIE-Image-Turbo --omni \
  --tensor-parallel-size 2 \
  --enable-cpu-offload \
  --port 8091
```

#### Verification

After the server is ready, test with a simple request:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photo of a cat sitting on a laptop keyboard, digital art style.",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

For ERNIE-Image-Turbo, reduce the inference steps:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photo of a cat sitting on a laptop keyboard, digital art style.",
    "size": "1024x1024",
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

#### Notes

- **Memory usage:** The 8B model requires significant VRAM. Use
  `--enable-cpu-offload` to reduce GPU memory footprint by offloading
  components to CPU when not in use.
- **Key flags:**
  - `--omni` — enables vLLM-Omni diffusion serving.
- **Advanced features:**
  - **TP (Tensor Parallelism):** `--tensor-parallel-size <N>` — distribute
    model weights across N GPUs.
  - **SP (Sequence Parallelism):** `--usp <N>` (Ulysses SP) and `--ring <N>`
    (Ring SP) for long-sequence workloads.
  - **HSDP:** `--use-hsdp` enables Hybrid Sharded Data Parallelism; use
    `--hsdp-shard-size` and `--hsdp-replicate-size` for fine-grained control.
  - **Cache-DiT:** `--cache-backend cache_dit` caches DiT intermediate outputs
    for faster generation; configure via `--cache-config`.
  - **Layer offload:** `--enable-layerwise-offload` offloads DiT layers to CPU
    for memory-constrained scenarios.
- **Prompt Enhancer (PE):** ERNIE-Image includes an optional 3B-parameter
  Prompt Enhancer model that expands brief user inputs into richer structured
  descriptions, improving output quality. Set `use_pe=False` in the request
  body to disable it if you prefer direct prompt processing or want to use
  larger LLMs (e.g., Gemini, ChatGPT) for prompt enhancement instead.
- **Recommended settings:**
  - ERNIE-Image: `num_inference_steps=50`, `guidance_scale=4.0`
  - ERNIE-Image-Turbo: `num_inference_steps=8`, `guidance_scale=1.0`
