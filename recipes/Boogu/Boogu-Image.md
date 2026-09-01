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

This recipe documents tested configurations for CUDA GPU serving. The native
pipeline runs single-GPU; multi-GPU parallelism, CPU offload, and cache
acceleration are not yet supported for this model (see Notes).

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
  (`--cache-backend cache_dit`), and multi-GPU parallelism (TP / SP / CFG /
  HSDP) are planned follow-ups and are not validated for this model yet.

## Image editing (Boogu-Image-0.1-Edit)

The Edit checkpoint is served by the same native pipeline (`BooguImagePipeline`);
the image-editing (TI2I) path activates automatically when a request carries a
reference image. The Base text-to-image path is unaffected (no reference image
is sent).

### Command

```bash
vllm serve Boogu/Boogu-Image-0.1-Edit --omni --port 8091
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
- **Output resolution:** the output size follows the reference image (upstream
  `align_res`, on by default for a single reference), so `height`/`width` are
  derived from the input and requested sizes are not applied for edits.
- **Same limitations** as the Base checkpoint apply (no CPU offload, Cache-DiT,
  or multi-GPU parallelism yet).
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
