# JoyAI-Image-Edit

> Text-guided single-image editing with the native JoyAI-Image-Edit diffusion pipeline

## Summary

- Vendor: JD / JoyImage Team
- Model: `jdopensource/JoyAI-Image-Edit-Diffusers`
- Task: Text-guided single-image editing
- Mode: Offline inference and OpenAI-compatible online serving
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run JoyAI-Image-Edit through the shared
image-to-image examples with a known-good baseline configuration. JoyAI-Image-Edit
is a large single-image editing model: it requires one reference image per
request, snaps requested sizes to Joy's supported resolution buckets, and has a
validated 1024x1024 baseline on an 80 GB NVIDIA GPU.

## References

- Upstream model:
  [`jdopensource/JoyAI-Image-Edit-Diffusers`](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers)
- Upstream project:
  [`jd-opensource/JoyAI-Image`](https://github.com/jd-opensource/JoyAI-Image)
- Related offline example:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Related online example:
  [`examples/online_serving/image_to_image/README.md`](../../examples/online_serving/image_to_image/README.md)
- User guide:
  [`docs/user_guide/examples/offline_inference/image_to_image.md`](../../docs/user_guide/examples/offline_inference/image_to_image.md),
  [`docs/user_guide/examples/online_serving/image_to_image.md`](../../docs/user_guide/examples/online_serving/image_to_image.md)

## Hardware Support

This recipe documents the CUDA configuration used to validate the PR's JoyAI
integration. Other GPU types should be treated as unvalidated until you rerun
the smoke and parity checks below.

## GPU

### 1x NVIDIA H800 PCIe 80GB

#### Environment

- OS: Linux
- Python: `3.12.3`
- Driver / runtime: NVIDIA driver `580.82.07`, CUDA runtime `13.0`
- PyTorch: `2.11.0+cu130`
- Diffusers: `0.39.0.dev0` with `JoyImageEditPipeline` available
- vLLM: `0.21.0`
- vLLM-Omni: use the commit you are deploying from
- GPU: NVIDIA H800 PCIe, 81559 MiB

#### Online Serving

Start the OpenAI-compatible server from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve jdopensource/JoyAI-Image-Edit-Diffusers \
  --omni \
  --port 8092 \
  --init-timeout 1200 \
  --stage-init-timeout 900
```

The server is ready when the log shows `Application startup complete.`

If you need extra GPU headroom, use one of the memory-saving variants below.
These variants are useful on tighter 80 GB deployments, but the latency and
quality numbers in this recipe are from the baseline command above.

```bash
# Pipeline-level CPU offload for large components.
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve jdopensource/JoyAI-Image-Edit-Diffusers \
  --omni \
  --port 8092 \
  --init-timeout 1200 \
  --stage-init-timeout 900 \
  --enable-cpu-offload
```

```bash
# DiT blockwise offload. This trades latency for lower peak GPU memory.
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve jdopensource/JoyAI-Image-Edit-Diffusers \
  --omni \
  --port 8092 \
  --init-timeout 1200 \
  --stage-init-timeout 900 \
  --enable-layerwise-offload
```

#### Online Verification

Download the public JoyAI plate example:

```bash
wget -O /tmp/joy_plate.jpg \
  https://raw.githubusercontent.com/jd-opensource/JoyAI-Image/main/test_images/test_1.jpg
```

Send one image and one prompt. JoyAI-Image-Edit supports exactly one input image
per request.

```bash
IMG_B64=$(base64 -w0 /tmp/joy_plate.jpg)

curl -s http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg img "$IMG_B64" '{
    messages: [{
      role: "user",
      content: [
        {type: "text", text: "Turn the plate blue"},
        {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $img)}}
      ]
    }],
    extra_body: {
      height: 1024,
      width: 1024,
      num_inference_steps: 30,
      true_cfg_scale: 4.0,
      seed: 0
    }
  }')" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- \
  | base64 -d > /tmp/joyai_image_edit_plate_online.png
```

Expected result: `/tmp/joyai_image_edit_plate_online.png` is a 1024x1024 PNG
where the plate color changes to blue while the surrounding scene is mostly
preserved. Use a client timeout of at least 600 seconds for first-run or cold
server checks.

#### Offline Command

Run the same smoke case without starting a server:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/offline_inference/image_to_image/image_edit.py \
  --model jdopensource/JoyAI-Image-Edit-Diffusers \
  --image /tmp/joy_plate.jpg \
  --prompt "Turn the plate blue" \
  --negative-prompt "" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --cfg-scale 4.0 \
  --seed 0 \
  --enforce-eager \
  --init-timeout 1200 \
  --stage-init-timeout 900 \
  --output /tmp/joyai_image_edit_plate_offline.png
```

For a local checkpoint mirror, replace `--model` with the local model directory.
The PR validation used the same command shape with
`/autodl-fs/data/joyai_models/JoyAI-Image-Edit-Diffusers`.

#### Validated Results

The table below is from the 1x H800 PCIe environment above. Resolution requests
were snapped to the Joy 1024x1024 bucket, batch size was 1, and CFG was 4.0.
`vLLM peak` is the observed peak GPU memory for the vLLM-Omni run.

| Case | Steps | Seed | vLLM latency | Diffusers latency | vLLM peak | Diffusers peak | SSIM | PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plate_seed123` | 30 | 123 | 50.87 s | 52.74 s | 57.52 GiB | 57.28 GiB | 0.9869 | 34.7051 |
| `plate_seed0` | 30 | 0 | 51.02 s | 52.01 s | 57.52 GiB | 57.28 GiB | 0.9946 | 41.4351 |
| `astronaut_seed0` | 40 | 0 | 67.90 s | 69.22 s | 57.52 GiB | 57.28 GiB | 0.9948 | 38.2498 |
| `crane_removal_seed0` | 40 | 0 | 68.00 s | 69.27 s | 57.52 GiB | 57.28 GiB | 0.9975 | 50.0135 |
| `object_move_seed0` | 40 | 0 | 68.06 s | 69.32 s | 57.52 GiB | 57.28 GiB | 0.9969 | 46.1138 |
| `object_rotation_seed0` | 40 | 0 | 67.94 s | 69.28 s | 57.52 GiB | 57.28 GiB | 0.9898 | 35.5365 |
| `camera_control_seed0` | 40 | 0 | 68.15 s | 69.51 s | 57.52 GiB | 57.28 GiB | 0.9952 | 41.7697 |

#### Output Size Behavior

JoyAI-Image-Edit does not use arbitrary image sizes directly. The pipeline maps
the input or requested `height` / `width` to the nearest supported Joy bucket,
then resizes and center-crops the reference image to that bucket before
denoising.

- Provide both `height` and `width`, or omit both.
- For square outputs, request `height=1024` and `width=1024`.
- A request such as `512x512` is also square and is snapped to the 1024x1024 Joy
  bucket, so the final image size will not match the literal request.
- Non-square requests are snapped to the nearest aspect-ratio bucket supported
  by the model.

#### Guidance Parameters

Use `true_cfg_scale` or `cfg_scale` for JoyAI classifier-free guidance. The
server accepts `guidance_scale` only as a Diffusers compatibility alias when
`true_cfg_scale` is not also set. If both are provided with different values,
the request is rejected.

CFG is active only when the effective true CFG scale is greater than 1 and a
negative prompt is present. The offline CLI maps `--cfg-scale` to Joy's true CFG
scale.

#### Known Limitations

- Single image only: JoyAI-Image-Edit rejects requests with zero or multiple
  input images.
- Single prompt only: batched prompt requests are rejected.
- HSDP is not supported yet. Starting with HSDP enabled raises a `ValueError`
  before the generic HSDP setup reaches the transformer.
- LoRA is not included for this JoyAI pipeline.
- Cache-DiT and TeaCache are not included for this JoyAI pipeline. Do not treat
  the generic image-to-image cache flags as validated JoyAI acceleration knobs.
- Text encoder quantization is not included.
- The validation above is CUDA-only; ROCm and NPU configurations are not
  documented as supported by this recipe.
