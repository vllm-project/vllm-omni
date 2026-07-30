# HiDream-O1-Image

> Text-to-image, instruction-based image editing, multi-reference personalization, and layout-bbox conditioning

## Summary

- Vendor: HiDream.ai
- Models: `HiDream-ai/HiDream-O1-Image` (full, 50 steps) · `HiDream-ai/HiDream-O1-Image-Dev` (distilled, 28 steps)
- Task: Text-to-image generation, image editing, multi-reference subject personalization, layout-bbox spatial control
- Mode: Offline inference and online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving either HiDream-O1-Image variant.
HiDream-O1-Image is a Pixel-level Unified Transformer (UiT): a single ~8B-parameter model that performs
flow-matching diffusion directly on raw 32×32 pixel patches with no VAE and no separate text encoder.
It supports text-to-image, instruction-based editing (1 reference image), multi-reference subject
personalization (2–10+ reference images), and layout-bbox spatial grounding on a single H100 80 GB GPU.

**Checkpoint selection:**
- `HiDream-O1-Image` (full): 50 steps, guidance_scale 5.0, shift 3.0 — highest quality.
- `HiDream-O1-Image-Dev` (distilled): 28 steps, no CFG (guidance_scale 1.0), shift 1.0 — faster iteration.

## References

- Model: <https://huggingface.co/HiDream-ai/HiDream-O1-Image>
- Dev variant: <https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev>
- Reference implementation: <https://github.com/HiDream-ai/HiDream-O1-Image>
- Related examples: [`examples/offline_inference/hidream_o1_image/`](../../examples/offline_inference/hidream_o1_image/)
- Online serving: [`examples/online_serving/hidream_o1_image/`](../../examples/online_serving/hidream_o1_image/)

## Hardware Support

This recipe documents CUDA GPU serving configurations. No VAE, no auxiliary text encoder — all computation
is in the single UiT transformer.

## GPU

### 1x H100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an H100 80 GB GPU
- `transformers >= 4.57.1` (for `Qwen3VLForConditionalGeneration`)
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command — Dev variant (fast, no CFG)

```bash
export MODEL_NAME_OR_PATH=HiDream-ai/HiDream-O1-Image-Dev
vllm serve ${MODEL_NAME_OR_PATH} \
    --omni \
    --port 8095 \
    --tensor-parallel-size 1
```

#### Command — Full variant (high quality, CFG)

```bash
export MODEL_NAME_OR_PATH=HiDream-ai/HiDream-O1-Image
vllm serve ${MODEL_NAME_OR_PATH} \
    --omni \
    --port 8095 \
    --tensor-parallel-size 1
```

#### Offline inference — text-to-image

```bash
python examples/offline_inference/hidream_o1_image/text_to_image_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --prompt "A golden retriever running through a field of sunflowers" \
    --model-type dev \
    --height 1024 --width 1024 \
    --seed 42 \
    --output t2i_output.png
```

#### Offline inference — image editing (1 reference image)

```bash
python examples/offline_inference/hidream_o1_image/image_edit_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --prompt "Make the background a snowy mountain landscape" \
    --ref-images /path/to/photo.jpg \
    --model-type dev \
    --output edited.png
```

#### Offline inference — layout-bbox spatial control

```bash
python examples/offline_inference/hidream_o1_image/layout_control_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --prompt "A person and a dog sitting in a park" \
    --ref-images person.jpg dog.jpg \
    --layout-bboxes '[[0.0, 0.45, 0.1, 0.9], [0.55, 1.0, 0.1, 0.9]]' \
    --model-type dev \
    --output layout_output.png
```

Layout bbox format: `[x1, x2, y1, y2]` in xxyy order, values in `[0, 1]` (relative) or `[0, 100]` (percentage).
One bbox entry per reference image, in the same order.

#### Verification

After the server is ready, run the online serving client:

```bash
python examples/online_serving/hidream_o1_image/openai_client_hidream_o1.py \
    --server http://localhost:8095 \
    --prompt "A golden retriever running through a field of sunflowers" \
    --height 1024 --width 1024 \
    --output hidream_o1_recipe.png
```

Or use curl:

```bash
curl -s http://localhost:8095/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A golden retriever running through a field of sunflowers",
        "size": "1024x1024",
        "num_inference_steps": 28,
        "seed": 42
    }' | jq -r '.data[0].b64_json' | base64 -d > output_online.png
```

#### Notes

- No `--auxiliary-text-encoder` is needed — HiDream-O1 uses its own built-in Qwen3-VL decoder for text embedding.
- No `--vae_use_slicing` / `--vae_use_tiling` — the model has no VAE; it diffuses directly on raw pixel patches.
- Resolution is snapped internally to the nearest 32-px-aligned value; you can pass any `height`/`width`.
- For the `full` checkpoint with CFG (`guidance_scale > 1.0`), expect roughly 2x the VRAM and compute of the `dev` variant at the same resolution.
- `transformers >= 4.57.1` is required at runtime; the pipeline will raise a clear error if the installed version is older.

## Performance

> **Status:** Benchmark numbers TBD — to be filled once the checkpoint is available on the target cluster.
> The methodology below describes how to collect them.

### Methodology

HiDream-O1-Image has two cost buckets per request:

1. **One-time setup** (runs once per request, not per step):
   - Vision-tower `get_image_features` (only for ref-image paths — editing, personalization, layout-bbox).
   - `fix_point` position-id construction (`get_rope_index_fix_point`).
   - Patch embedding via `x_embedder`.
   At 28 Dev steps the fixed overhead can represent a significant fraction of total latency — profile it
   separately to avoid attributing it to per-step cost.

2. **Per-step cost** (`forward_generation` × num_steps):
   - The full decoder stack (Qwen3-VL backbone, 32 layers, ~8B params).
   - No VAE decode — output is directly assembled from the final hidden states.

Enable per-stage timing with `--enable-diffusion-pipeline-profiler` (backed by
`DiffusionPipelineProfilerMixin`); the pipeline reports
`transformer.get_image_features`, `transformer.forward_generation`, and
`transformer.x_embedder.forward` durations.

### Configurations to benchmark

| Config | Command flags | Expected benefit |
|--------|---------------|-----------------|
| Baseline (Dev, 28 steps) | _(none)_ | Reference |
| Cache-DiT | `--cache-backend cache_dit` | ~1.5–2× speedup per step (residual reuse) |
| TP-2 | `--tensor-parallel-size 2` | ~1.6–1.9× step throughput (linear layer sharding) |
| Ulysses-2 | `--ulysses-degree 2` | ~1.5–1.8× for long sequences (T2I at 1024²) |
| Cache-DiT + TP-2 | `--cache-backend cache_dit --tensor-parallel-size 2` | Compounded; measure combined |

### Profiling commands

```bash
# Single-GPU baseline with profiler
vllm serve HiDream-ai/HiDream-O1-Image-Dev \
    --omni \
    --enable-diffusion-pipeline-profiler

# Cache-DiT
vllm serve HiDream-ai/HiDream-O1-Image-Dev \
    --omni \
    --cache-backend cache_dit \
    --enable-diffusion-pipeline-profiler

# TP-2
vllm serve HiDream-ai/HiDream-O1-Image-Dev \
    --omni \
    --tensor-parallel-size 2 \
    --enable-diffusion-pipeline-profiler
```

### Benchmark numbers

> Numbers to be filled from H100-80GB measurements at 1024×1024, batch=1, seed=42.

| Config | forward_generation (ms/step) | Total (s, 28 steps) | Peak VRAM (GB) |
|--------|------------------------------|---------------------|----------------|
| Baseline | — | — | — |
| Cache-DiT | — | — | — |
| TP-2 | — | — | — |
| Ulysses-2 | — | — | — |
| Cache-DiT + TP-2 | — | — | — |
| CPU Offload | — | — | — |
