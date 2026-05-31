# FLUX.2-klein-9b-kv for image generation

## Summary

- Vendor: Black Forest Labs
- Model: `black-forest-labs/FLUX.2-klein-9b-kv`
- Task: Text-to-image and image-to-image generation with KV cache acceleration
- Mode: Online serving with vLLM-Omni
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve `black-forest-labs/FLUX.2-klein-9b-kv`
with vLLM-Omni for fast image generation. This variant includes KV-cache
optimization for accelerated multi-reference editing workflows, providing up
to 2.5x speedup when using the same reference images across multiple generations.

Key features:
- **9B flow model** with **8B Qwen3 text embedder**
- **Step-distilled to 4 inference steps** — sub-second generation
- **KV-cache support** for fast multi-reference image editing
- **Sequence Parallel (SP)** supported: Ulysses + Ring attention

## References

- Upstream model card: [FLUX.2-klein-9b-kv on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv)
- Diffusers pipeline: `Flux2KleinKVPipeline`

## Hardware Support

## GPU

### 1x RTX 4090 (24GB) with layer-wise offload （Minimum Recommended）

```bash
vllm serve black-forest-labs/FLUX.2-klein-9b-kv --omni --port 8091 \
  --model-class-name Flux2KleinKVPipeline \
  --enable-layerwise-offload
```

Text-to-image (JSON body):

```bash
curl -s -D >(grep -i x-request-id >&2) \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > flux_klein_kv_output.png) \
  -X POST "http://localhost:8091/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene lakeside sunrise with mist over the water",
    "size": "1024x1024",
    "n": 1,
    "num_inference_steps": 4,
    "guidance_scale": 4.0,
    "seed": 42
  }'
```

Image-to-image edit (multipart form):

```bash
curl -s -D >(grep -i x-request-id >&2) \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > flux_klein_kv_edit.png) \
  -X POST "http://localhost:8091/v1/images/edits" \
  -F "image=@path/to/reference.jpg" \
  -F "prompt=Transform into a cyberpunk cityscape at night" \
  -F "size=1024x1024" \
  -F "num_inference_steps=4" \
  -F "guidance_scale=4.0" \
  -F "seed=42"
```

Multi-image edit (multiple references):

```bash
curl -s -D >(grep -i x-request-id >&2) \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > flux_klein_kv_multi_edit.png) \
  -X POST "http://localhost:8091/v1/images/edits" \
  -F "image[]=@path/to/reference1.jpg" \
  -F "image[]=@path/to_reference2.jpg" \
  -F "prompt=Transform into a cyberpunk cityscape at night" \
  -F "size=1024x1024" \
  -F "num_inference_steps=4" \
  -F "guidance_scale=4.0" \
  -F "seed=42"
```

### 2x RTX 4090 (24GB) with Tensor Parallel （Minimum Recommended）

```bash
vllm serve black-forest-labs/FLUX.2-klein-9b-kv --omni --port 8091 \
  --model-class-name Flux2KleinKVPipeline \
  --tensor-parallel-size 2
```


### Notes

- **API endpoints:**
  - `/v1/images/generations` — text-to-image (use JSON with `-H "Content-Type: application/json" -d '{...}'`)
  - `/v1/images/edits` — image-to-image (use multipart form with `-F`)
- **Memory usage:** The 9B model requires significant VRAM (~29GB). Use
  `--enable-cpu-offload` to reduce GPU memory footprint by offloading
  components to CPU when not in use.
- **Key flags:**
  - `--omni` — enables vLLM-Omni diffusion serving.
  - `--model-class-name Flux2KleinKVPipeline` — required for KV-cache model.
- **Advanced features:**
  - **TP (Tensor Parallelism):** `--tensor-parallel-size <N>` — distribute model weights across N GPUs.
  - **SP (Sequence Parallelism):** `--ulysses-degree <N>` (Ulysses SP) and `--ring <N>` (Ring SP) for long-sequence workloads.
  - **Layer offload:** `--enable-layerwise-offload` offloads DiT layers to CPU for memory-constrained scenarios.
  - **CPU offload:** `--enable-cpu-offload` offloads model to CPU RAM.
- **Recommended settings:**
  - FLUX.2-klein-9b-kv: `num_inference_steps=4`, `guidance_scale=4.0`
- **KV-cache:** Provides up to 2.5x speedup when using the same reference image for multi-reference editing workflows
