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
- 2x H100 80GB, or 1x H100 80GB co-located (see [Single-GPU co-location](#single-gpu-co-location-1x-h100))

### Commands

```bash
MODEL=inclusionAI/Ming-Image-0.1-Design
vllm serve "$MODEL" --omni --deploy-config vllm_omni/deploy/ming_image.yaml --port 8091
```

For layer decomposition, set `MODEL` to `inclusionAI/Ming-Image-0.1-Design-Layer`.

## Single-GPU co-location (1x H100)

Both stages also fit on one 80 GB GPU. Copy `vllm_omni/deploy/ming_image.yaml`
and change three fields (keep `max_num_seqs: 1` on both stages):

```yaml
stages:
  - stage_id: 0
    devices: "0"
    gpu_memory_utilization: 0.60   # up from 0.55
  - stage_id: 1
    devices: "0"                   # was "1"
    gpu_memory_utilization: 0.33   # new
```

Validated end to end on one H100 80GB with the text-to-image smoke prompt
(1024², 12 steps, cfg 1.0, seed 42; vLLM 0.31.0, vLLM-Omni 099f9b553,
first request after server start, so stage-1 numbers include its
compilation warmup):

| Metric (single request) | 1x H100 co-located | 2x H100 (stock yaml) |
|---|---|---|
| E2E latency | 2.39 s | 3.73 s |
| Stage 0 (VLM) | 0.22 s | 1.23 s |
| Stage 1 (DiT) | 2.09 s | 2.41 s |
| Stage 0→1 transfer | 0 ms | 0 ms |
| Peak VRAM | 69.9 GiB | 44.2 GiB + 21.8 GiB |

The seed-42 output is byte-identical (SHA-256) to the two-GPU reference
run; the shared-memory connector keeps the inter-stage hop free in both
layouts. Reproduced on a second independent server start (2.74 s e2e,
0.24 s / 2.40 s per stage, 69.9 GiB peak, identical output hash).
0.60 + 0.33 leaves roughly 10 GB of headroom for the stage-1
compile workspace and KV growth; raising the two fractions much further
risks OOM during shape warmup.

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
