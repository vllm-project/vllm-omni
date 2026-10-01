# Ming-Image 0.1 Design

> Text-to-image, image editing, and layer decomposition.

## Summary

- Vendor: inclusionAI
- Models: `inclusionAI/Ming-Image-0.1-Design` and `inclusionAI/Ming-Image-0.1-Design-Layer`
- Runtime: vLLM-Omni two-stage online serving
- API: OpenAI-compatible chat completions

## Architecture

Stage 0 uses a Qwen2.5-VL vision tower plus a 20-layer BailingMoeV2 language model.
It appends 256 learned image-query tokens and exports the final query states plus direct VLM states from layers 5, 12, and 20.
The query-token checkpoint is in the root sibling `mlp/` directory, so remote stage-0 materialization downloads both `mllm/` and `mlp/`.

Stage 1 projects those conditions through the checkpoint connector and runs a 30-layer Z-Image DiT with a Qwen-Image RGBA VAE.

For Design-Layer, the first returned image is the reconstructed composite and the remaining images are the requested layers.

## CUDA

### Environment

- OS: Linux
- Python: 3.10+
- CUDA: 13.0
- vLLM version: 0.29.0
- vLLM-Omni version or commit: 6daf5b30f
- 2x H100 80GB (1xH100 to be validated)

### Commands

```bash
MODEL=inclusionAI/Ming-Image-0.1-Design
vllm serve "$MODEL" --omni --deploy-config vllm_omni/deploy/ming_image.yaml --port 8091
```

For layer decomposition, set `MODEL` to `inclusionAI/Ming-Image-0.1-Design-Layer`.

## Text-to-image

Note that a prompt refiner is expected to describe the prompts with details; we will refine with more example inputs soon.

```bash
curl -s http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/Ming-Image-0.1-Design",
    "messages": [{"role": "user", "content": "A clean editorial botanical poster"}],
    "modalities": ["image"],
    "extra_body": {
      "height": 1024, "width": 1024,
      "num_inference_steps": 12, "guidance_scale": 1.0, "seed": 42
    }
  }' \
  | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
  | base64 -d > ming_design_smoke.png
```

## Image editing

Pass one input image and an editing instruction:

```bash
MODEL=inclusionAI/Ming-Image-0.1-Design
INPUT_IMAGE=/path/to/input.png

jq -n \
  --arg model "$MODEL" \
  --rawfile image <(base64 -w0 "$INPUT_IMAGE") \
  '{
    model: $model,
    messages: [{role: "user", content: [
      {type: "image_url", image_url: {url: ("data:image/png;base64," + $image)}},
      {type: "text", text: "Change the background to blue"}
    ]}],
    modalities: ["image"],
    extra_body: {
      height: 1024, width: 1024,
      num_inference_steps: 12, guidance_scale: 1.0, seed: 42
    }
  }' |
curl -sS http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @- \
  | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
  | base64 -d > ming_edit.png
```

## Layer decomposition

Use the Design-Layer checkpoint and set `INPUT_IMAGE` to a local flattened design image.
Note that the prompt should better depict each layer to be decomposed, we will refine with more example inputs soon.

The default `ming_image.yaml` uses two GPUs and decodes without tiling. For
1024x1024 output with six layers, choose one of these VAE configurations:

```bash
# Two GPUs: Stage 0 on GPU 0, Stage 1 and tiled VAE on GPU 1.
vllm serve inclusionAI/Ming-Image-0.1-Design-Layer --omni \
  --deploy-config vllm_omni/deploy/ming_image_layer_tiled.yaml --port 8091

# Three GPUs: Stage 0 on GPU 0, Stage 1 on GPUs 1 and 2;
# the VAE distributes spatial tiles across the two Stage 1 ranks.
vllm serve inclusionAI/Ming-Image-0.1-Design-Layer --omni \
  --deploy-config vllm_omni/deploy/ming_image_layer_tile_parallel.yaml --port 8091

# Three GPUs: Stage 0 on GPU 0, Stage 1 on GPUs 1 and 2;
# the VAE shards decoder features along height and exchanges boundary halos.
vllm serve inclusionAI/Ming-Image-0.1-Design-Layer --omni \
  --deploy-config vllm_omni/deploy/ming_image_layer_spatial_shard.yaml --port 8091
```

The spatial-shard configuration uses `tensor_parallel_size: 1`,
`ulysses_degree: 2`, and `vae_patch_parallel_size: 2` for Stage 1. The VAE
parallel size must equal the VAE executor process-group size (two ranks here),
and distributed initialization is required for multiple ranks. Invalid
configurations raise an error without falling back to tile decoding. To shard
along width, copy this YAML and change `vae_parallel_mode` from
`spatial_shard_height` to `spatial_shard_width`; keep the same GPU layout and
parallel sizes. A VAE instance is bound to its spatial-shard direction, process
group, and parallel size after its first multi-rank spatial-shard decode;
restart with a fresh instance to change them or return to tile mode.

Tile mode decodes overlapping spatial tiles independently and blends their
boundaries. Spatial-shard mode exchanges boundary features between decoder
layers and gathers the complete valid feature map for global attention. It
preserves the full decoding algorithm for the same latent, including small
inputs below the tiling threshold, with floating-point numerical differences
allowed. It does not introduce tile cropping or blending approximations.
Single-rank spatial-shard mode performs full, non-tiled decoding without
collectives. These equivalence claims apply to latent decoding, not reference
image encoding. The existing tiled configurations remain separate options.

Run one command at a time, then use the request below with the same input
image, prompt, seed, `num_layers=6`, 1024x1024 dimensions, and 12 steps.
Warm up the same shape and layer count before timing. Record request latency,
`vae.decode` stage duration with `enable_diffusion_pipeline_profiler`, and peak
GPU memory for each configuration. Check all seven returned images in order:
composite first, then layers 1 through 6. Compare tiled outputs numerically
between the two deployments; report untiled differences separately because
tile-boundary blending can change individual pixels. Compare spatial-shard and
full decoding using the same latent when checking decode equivalence.
`vae_patch_parallel_size` controls the VAE parallel size for both distributed
tile and spatial-shard modes. Large-model spatial-shard performance and image
quality have not yet been evaluated; multi-GPU speedup has not been measured.
Set `enable_diffusion_pipeline_profiler: true` under Stage 1 in a temporary
copy of each deployment config when collecting decode durations.

```bash
MODEL=inclusionAI/Ming-Image-0.1-Design-Layer
INPUT_IMAGE=/path/to/input.png
PROMPT="Decompose this image into 6 layers with the following specifications: Number of layers: 6 \nLayer 1: Central text ... Layer 2: ... Layer 6: ..."

jq -n \
  --arg model "$MODEL" \
  --rawfile image <(base64 -w0 "$INPUT_IMAGE") \
  --arg prompt "$PROMPT" \
  '{
    model: $model,
    messages: [{role: "user", content: [
      {type: "image_url", image_url: {url: ("data:image/png;base64," + $image)}},
      {type: "text", text: $prompt}
    ]}],
    modalities: ["image"],
    extra_body: {
      num_layers: 6,
      height: 1024, width: 1024,
      num_inference_steps: 12, guidance_scale: 2.0, seed: 42
    }
  }' |
curl -sS http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @- > response.json

jq -r '.choices[0].message.content[].image_url.url | split(",")[1]' response.json |
  nl -v 0 |
  while read -r index data; do
    printf "%s" "$data" | base64 -d > "ming_layer_${index}.png"
  done
```

## Notes

- The default deployment keeps Stage 0 eager and enables dynamic regional
  compilation with CUDA Graph Trees for the repeated Stage 1 DiT blocks.
  The first request for each new image shape or layer count pays compilation
  and graph-capture cost; warm up every production shape before measuring or
  serving latency-sensitive traffic.
- Only one reference image and one request at a time are currently supported.
- Design-Layer requires a reference image except during warmup.
- A non-empty `negative_prompt` is rejected; Ming-Image uses zero negative conditioning.
- Height and width must be divisible by 16.
- Returned images retain the checkpoint's four RGBA channels.

## Step caching

Step caching is disabled by default. To enable it for serving, pass the cache backend and its JSON threshold configuration on the command line. The step cache runs in the diffusion stage.

```bash
# Design: TeaCache 0.30
vllm serve inclusionAI/Ming-Image-0.1-Design --omni \
  --deploy-config vllm_omni/deploy/ming_image.yaml \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh":0.3}' \
  --port 8091

# Design: Cache-DiT 0.16
vllm serve inclusionAI/Ming-Image-0.1-Design --omni \
  --deploy-config vllm_omni/deploy/ming_image.yaml \
  --cache-backend cache_dit \
  --cache-config '{"residual_diff_threshold":0.16}' \
  --port 8091

# Design-Layer: TeaCache 0.13
vllm serve inclusionAI/Ming-Image-0.1-Design-Layer --omni \
  --deploy-config vllm_omni/deploy/ming_image.yaml \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh":0.13}' \
  --port 8091

# Design-Layer: Cache-DiT 0.16
vllm serve inclusionAI/Ming-Image-0.1-Design-Layer --omni \
  --deploy-config vllm_omni/deploy/ming_image.yaml \
  --cache-backend cache_dit \
  --cache-config '{"residual_diff_threshold":0.16}' \
  --port 8091
```

In these measured runs, Design TeaCache 0.30 reduced E2E latency by 6.8%, Design Cache-DiT 0.16 by 23.5%, Design-Layer TeaCache 0.13 by 18.7%, and Design-Layer Cache-DiT 0.16 by 23.3% versus no cache.

Representative online runs on 2×H100 80 GB (vLLM 0.30.0, 1024×1024, 12 steps, concurrency 1):

For Design, all profiles use one shared no-cache control: the same 1024×1024 botanical-poster prompt, CFG 1.0, 12 steps, and seed 42.

### Design

| Profile | E2E s | Stage 1 ms | req/s | SSIM / PSNR dB / LPIPS |
| --- | ---: | ---: | ---: | ---: |
| No cache | 2.219 | 1,596 | 0.451 | — |
| TeaCache 0.30 | 2.068 | 1,465 | 0.484 | 0.948 / 24.51 / 0.021 |
| Cache-DiT 0.16 | 1.698 | 1,289 | 0.589 | 0.853 / 19.11 / 0.095 |

### Design-Layer

One fixed reference image and prompt, CFG 2.0, one warmup and three measured runs per profile, sharing one no-cache control.

| Profile | E2E s | Stage 1 ms | req/s | SSIM / PSNR dB / LPIPS |
| --- | ---: | ---: | ---: | ---: |
| No cache | 12.723 | 10,813 | 0.079 | — |
| TeaCache 0.13 | 10.343 | 8,547 | 0.097 | 0.988 / 22.06 / 0.011 |
| Cache-DiT 0.16 | 9.764 | 7,986 | 0.102 | 0.991 / 28.06 / 0.007 |
