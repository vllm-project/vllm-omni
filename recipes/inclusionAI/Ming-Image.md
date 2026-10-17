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

### Original two-GPU BF16 environment

- OS: Linux
- Python: 3.10+
- CUDA: 13.0
- vLLM version: 0.29.0
- vLLM-Omni version or commit: 6daf5b30f
- 2x H100 80GB

The single-GPU FP8 comparison below uses a newer, separately specified
[environment](#fp8-quantization-stage-1-single-h100-observations).

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

## FP8 quantization (stage 1): single-H100 observations

The following historical comparison covers text-to-image generation with
`inclusionAI/Ming-Image-0.1-Design` on one H100. It does not qualify FP8 for
the two-GPU layout, image editing, or Design-Layer.

### Comparison environment and deployment

- GPU: 1x H100 80GB, with both stages on the same device.
- Runtime: vLLM 0.31.0 and vLLM-Omni `099f9b553`.
- CUDA: 13.0 through the forward-compatibility package on a driver-535 host.
- Requests: one at a time, 1024x1024, 12 inference steps,
  `guidance_scale=1.0`, and `seed=42` in both arms.

Copy `vllm_omni/deploy/ming_image.yaml` to `ming_image_bf16.yaml` and apply
the following overrides to its existing stage entries. Preserve the other
fields, including `max_num_seqs: 1`, compilation settings, connectors, and
sampling defaults.

```yaml
stages:
  - stage_id: 0
    devices: "0"
    gpu_memory_utilization: 0.60
  - stage_id: 1
    devices: "0"
    gpu_memory_utilization: 0.33
```

Copy that configuration to `ming_image_fp8.yaml` and add
`quantization: fp8` to its stage-1 entry only. This applies online FP8 to
the DiT; no pre-quantized checkpoint is required. Keep stage 0 in BF16.

Run the two configurations separately, stopping one server before starting
the other:

```bash
vllm serve inclusionAI/Ming-Image-0.1-Design --omni \
  --deploy-config ming_image_bf16.yaml --port 8091
```

```bash
vllm serve inclusionAI/Ming-Image-0.1-Design --omni \
  --deploy-config ming_image_fp8.yaml --port 8091
```

Use the [text-to-image request](#text-to-image) for each prompt in the
table below, saving BF16 and FP8 outputs separately. Warm up the requested
shape before measuring steady-state latency.

### Reported performance and image differences

These are single-request observations from the environment above, not a
throughput benchmark or a multi-run statistical comparison.

| Metric | BF16 | Stage-1 FP8 |
| --- | --- | --- |
| Stage-1 latency (steady requests) | ~2.0 s | ~1.2 s |
| Peak VRAM | 70.1 GiB | 64.3 GiB |
| Server startup in the same session on a shared host | 472 s | 654 s |

The image comparisons use the same prompt and seed in both arms. SSIM and
PSNR were recorded on 8-bit RGB images with alpha dropped and
`data_range=255`.

| Exact prompt | SSIM vs BF16 | PSNR vs BF16 |
| --- | --- | --- |
| A clean editorial botanical poster | 0.922 | 20.9 dB |
| A serene mountain lake at sunrise with mist over the water | 0.941 | 22.9 dB |
| A vibrant street food market scene at night with neon signs | 0.816 | 18.9 dB |
| A minimalist coffee brand logo with a mountain silhouette | 0.990 | 28.1 dB |

These values describe observed differences; they are not minimum quality
thresholds or a passing accuracy gate. LPIPS values are omitted because
their original backbone and package version could not be verified.
Establish quality thresholds for the intended workload before using FP8
by default; this comparison does not establish them.

If quality is insufficient, retain BF16 or test model-appropriate
`ignored_layers` patterns as described in the
[FP8 quantization guide](../../docs/user_guide/quantization/fp8.md#overview).
Layer patterns are architecture-specific; the guide's `img_mlp` example
is not a validated Ming-Image fallback from this comparison.

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
