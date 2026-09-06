# Boogu-Image

> Text-to-image and image-editing online serving
(Boogu-Image-0.1-Base / -Edit / -Edit-Turbo)

## Summary

- Vendor: Boogu
- Model: `Boogu/Boogu-Image-0.1-Base` (text-to-image),
  `Boogu/Boogu-Image-0.1-Edit` (image editing), and
  `Boogu/Boogu-Image-0.1-Edit-Turbo` (four-step image editing)
- Task: Text-to-image generation and text-guided image editing (TI2I)
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Boogu/Boogu-Image-0.1-Base`, `Boogu/Boogu-Image-0.1-Edit`, or
`Boogu/Boogu-Image-0.1-Edit-Turbo` with vLLM-Omni's native pipeline (no
`--diffusion-load-format diffusers`, and the upstream `boogu` package is not
required).

Boogu-Image-0.1 is an Apache-2.0 unified image generation and editing model
family. The Base text-to-image checkpoint pairs a Qwen3-VL multimodal encoder
with a Diffusion Transformer (DiT) and a flow-match Euler scheduler with
time-shift. It handles photorealistic generation and Chinese/English text
rendering. The Edit and Edit-Turbo checkpoints use the same native
`BooguImagePipeline` for image editing.

## References

- Upstream model card: <https://huggingface.co/Boogu/Boogu-Image-0.1-Base>
- Edit model card: <https://huggingface.co/Boogu/Boogu-Image-0.1-Edit>
- Edit-Turbo 1K hotfix: <https://huggingface.co/Boogu/Boogu-Image-0.1-Edit-Turbo/tree/hotfix-1k-20260708>
- Project page: <https://boogu.org>
- GitHub: <https://github.com/boogu-project/Boogu-Image>
- Related example: [`examples/online_serving/text_to_image/`](../../examples/online_serving/text_to_image/README.md)

## Hardware Support

This recipe documents supported configurations for CUDA GPU serving. The native
pipeline supports single-GPU inference and classifier-free-guidance (CFG)
parallelism. CPU offload, cache acceleration, and other multi-GPU dimensions
remain unsupported (see Notes).

## GPU

### 1 x A100/H100 (Single GPU, 40GB+ VRAM)

The model footprint is roughly 34.6 GiB on GPU, so a 40GB+ card is recommended.

#### Command

```bash
vllm serve Boogu/Boogu-Image-0.1-Base --omni --port 8091
```

!!! note
    If you hit Out-of-Memory (OOM) on a smaller card, enable VAE slicing and
    tiling to reduce peak memory: `--vae-use-slicing --vae-use-tiling`.

#### Verification

After the server is ready, test with a simple request:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Boogu/Boogu-Image-0.1-Base",
    "prompt": "A mountain lake at sunset, photorealistic, cinematic lighting",
    "size": "1024x1024",
    "num_inference_steps": 28,
    "guidance_scale": 4.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

Or via the chat-completions endpoint (parameters go in `extra_body`):

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A mountain lake at sunset, photorealistic, cinematic lighting"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 28,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

#### Notes

- **Memory usage:** ~34.6 GiB on GPU. Use `--vae-use-slicing --vae-use-tiling`
  to trim peak VRAM if needed.
- **Key flags:**
    - `--omni` — enables vLLM-Omni diffusion serving.
- **Guidance:** Boogu-Image uses `guidance_scale` (mapped to the upstream
  `text_guidance_scale`); the default is `4.0`. Classifier-free guidance is
  active whenever `guidance_scale > 1.0`.
- **Recommended settings:** `num_inference_steps=28`-`50`, `guidance_scale=4.0`.
  The model's maximum native resolution is 2K.
- **Known limitations (not yet supported):** CPU offload
  (`--enable-cpu-offload` / `--enable-layerwise-offload`), Cache-DiT
  (`--cache-backend cache_dit`), and TP / SP / HSDP multi-GPU parallelism.

### 2 x H100 (CFG parallel, Base T2I)

Base T2I has two predictions per guided denoising step, so two CFG ranks map
one-to-one to the positive and negative-text branches:

```bash
vllm serve Boogu/Boogu-Image-0.1-Base \
  --omni \
  --cfg-parallel-size 2 \
  --port 8091
```

Use `guidance_scale > 1.0` to activate the parallel two-branch path. Requests
with `guidance_scale=1.0` remain valid on the same server: every rank evaluates
only the positive branch, no negative embeddings are built, and no guidance is
applied.

## Image editing (Boogu-Image-0.1-Edit)

The Edit checkpoint is served by the same native pipeline (`BooguImagePipeline`);
the image-editing (TI2I) path activates automatically when a request carries a
reference image. The Base text-to-image path is unaffected (no reference image
is sent).

### Command

```bash
vllm serve Boogu/Boogu-Image-0.1-Edit --omni --port 8091
```

For text-only or image-only guidance, use two CFG ranks:

```bash
vllm serve Boogu/Boogu-Image-0.1-Edit \
  --omni \
  --cfg-parallel-size 2 \
  --port 8091
```

Double guidance has three predictions per denoising step. It supports either
two ranks (round-robin assignment: rank 0 runs branches 0 and 2) or three ranks
(one branch per rank):

```bash
# Practical two-GPU configuration
vllm serve Boogu/Boogu-Image-0.1-Edit \
  --omni \
  --cfg-parallel-size 2 \
  --port 8091

# Maximum branch parallelism
vllm serve Boogu/Boogu-Image-0.1-Edit \
  --omni \
  --cfg-parallel-size 3 \
  --port 8091
```

### Verification

Edit an image with `/v1/images/edits` (the model-card example — change a photo
to a colored-pencil drawing). Diffusion parameters are plain multipart form
fields; add `guidance_scale_2` to enable image guidance (double-guidance path):

```bash
curl -s http://localhost:8091/v1/images/edits \
  -F model="Boogu/Boogu-Image-0.1-Edit" \
  -F image="@input.png" \
  -F prompt="Change the style to a colored pencil drawing." \
  -F num_inference_steps=28 \
  -F guidance_scale=5.0 \
  -F guidance_scale_2=2.0 \
  -F seed=42 \
  | jq -r '.data[0].b64_json' | base64 -d > edited.png
```

Or via chat completions (attach the image as a data URL; parameters go in
`extra_body`):

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
        {"type": "text", "text": "Change the style to a colored pencil drawing."}
      ]}
    ],
    "extra_body": {
      "num_inference_steps": 28,
      "guidance_scale": 5.0,
      "guidance_scale_2": 1.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > edited.png
```

### Notes

- **Single reference image:** only one input image is supported for now (the
  upstream "Only support 1 reference image for now" limit).
- **Guidance semantics:**
    - `guidance_scale` = text guidance (upstream `text_guidance_scale`, default
    `4.0`); `> 1.0` enables text CFG. Editing typically uses `5.0`.
    - `guidance_scale_2` = image guidance (upstream `image_guidance_scale`,
    default `1.0` = off). Setting it `> 1.0` enables the double-guidance path
    (3 model predictions per step), steering more strongly toward the reference
    image.
- **CFG-size recommendations:**

  | Request mode | Scales | Predictions per step | Recommended CFG size |
  | --- | --- | ---: | ---: |
  | Base T2I text guidance | `guidance_scale > 1`, no reference | 2 | 2 |
  | Edit text-only guidance | `guidance_scale > 1`, `guidance_scale_2 = 1` | 2 | 2 |
  | Edit image-only guidance | `guidance_scale = 1`, `guidance_scale_2 > 1` | 2 | 2 |
  | Edit double guidance | both scales `> 1` | 3 | 2 (practical) or 3 (maximum parallelism) |
  | CFG off | both scales `= 1` | 1 | Reuse a size-2 server safely; no speedup expected |

  Double guidance preserves Boogu's original combination:

  ```text
  pred = positive_with_reference
       + (text_scale - 1) * (positive_with_reference - negative_with_reference)
       + (image_scale - 1) * (negative_with_reference - uncond)
  ```
- **Output resolution:** the output size follows the reference image (upstream
  `align_res`, on by default for a single reference), so `height`/`width` are
  derived from the input and requested sizes are not applied for edits.
- **Same limitations** as the Base checkpoint apply except that CFG parallelism
  is supported as described above; CPU offload, Cache-DiT, TP, SP, and HSDP are
  still unsupported.
- **Offline editing:** the shared offline example
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_to_image.md)
  supports Boogu-Image-Edit directly
  (`--model Boogu/Boogu-Image-0.1-Edit --guidance-scale 5.0`, optional
  `--guidance-scale-2`).

## Fast image editing (Boogu-Image-0.1-Edit-Turbo)

Edit-Turbo is the distilled image-editing checkpoint. Its `model_index.json`
declares `BooguImagePipeline`, so it uses the same native TI2I path as Edit.
The upstream 1K hotfix is recommended for stable results and is pinned below
to avoid silently loading the older checkpoint from the repository's default
revision.

### Command

```bash
vllm serve Boogu/Boogu-Image-0.1-Edit-Turbo \
  --omni \
  --revision hotfix-1k-20260708 \
  --port 8091
```

### Verification

Edit an image with the distilled four-step settings:

```bash
curl -s http://localhost:8091/v1/images/edits \
  -F model="Boogu/Boogu-Image-0.1-Edit-Turbo" \
  -F image="@input.png" \
  -F prompt="Change the style to a colored pencil drawing." \
  -F num_inference_steps=4 \
  -F guidance_scale=1.0 \
  -F seed=42 \
  | jq -r '.data[0].b64_json' | base64 -d > edited-turbo.png
```

Or via chat completions:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
        {"type": "text", "text": "Change the style to a colored pencil drawing."}
      ]}
    ],
    "extra_body": {
      "num_inference_steps": 4,
      "guidance_scale": 1.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > edited-turbo.png
```

### Notes

- **Pinned revision:** use `hotfix-1k-20260708`; upstream recommends the 1K
  hotfix over the 1.5K variant for more stable results.
- **Recommended settings:** `num_inference_steps=4` and
  `guidance_scale=1.0`. Edit-Turbo is guidance-distilled, meaning the guidance
  behavior is already baked into the distilled checkpoint. A scale of `1.0`
  avoids applying additional classifier-free guidance; unlike the regular Edit
  checkpoint, it should not be increased to `5.0`.
- **Reference images:** the same single-reference and `align_res` behavior as
  the regular Edit checkpoint applies.
- **Known limitations:** the same single-GPU limitations as Base and Edit apply;
  CPU offload, Cache-DiT, and multi-GPU parallelism are not yet validated.

## Performance validation

The checked-in single-device A40 measurements remain the current Boogu
baseline: Base T2I is about 16.37 s request latency at concurrency 1 and Edit
text-only guidance is about 29.64 s for 512x512, 28-step requests, with roughly
36.8 GB peak memory. CFG-parallel A/B numbers must be collected on the same
hardware and software revision before reporting a speedup; do not compare a
multi-GPU result against these A40 numbers as if it were a controlled A/B.

Run the dedicated perf rows in
`tests/dfx/perf/tests/test_boogu_image_vllm_omni.json` and
`tests/dfx/perf/tests/test_boogu_image_edit_vllm_omni.json` on 2x/3x H100, then
record latency, throughput, and per-GPU peak memory beside the single-device
run. The expected shape is close to the general CFG-parallel range (roughly
1.5x-1.8x for balanced two-branch workloads); the three-branch Edit result will
depend on whether two-rank round-robin or full three-rank execution is used and
must be reported from measurement rather than assumed.
