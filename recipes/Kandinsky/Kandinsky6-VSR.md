# Kandinsky 6 HD and Full HD Generation

Kandinsky6 T2VA (text-to-video-and-audio) and I2VA (image-to-video-and-audio)
can return HD or Full HD using SR as an internal final stage. The base video
is generated at the checkpoint's default resolution, refined to the requested
size, and returned with the generated audio.

## Request the output format

Use either the final dimensions or `target_resolution`:

| Format | Height | Width | `target_resolution` |
| --- | --- | --- | --- |
| HD | 720 | 1280 | `hd` |
| Full HD | 1080 | 1920 | `fhd` (or `fullhd`) |

```python
from PIL import Image
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers",
    enable_cpu_offload=True,
)
try:
    # T2VA: generate HD directly from text.
    hd = omni.generate(
        {"prompt": "A red fox runs through a snowy forest."},
        OmniDiffusionSamplingParams(seed=42, height=720, width=1280),
    )
    # I2VA: generate Full HD from an image and text.
    fhd = omni.generate(
        {"prompt": "The fox starts running through the forest.",
         "multi_modal_data": {"image": Image.open("fox.png").convert("RGB")}},
        OmniDiffusionSamplingParams(seed=42, extra_args={"target_resolution": "fhd"}),
    )
finally:
    omni.close()
```

The same request fields work with regular and distilled Lite/Pro generation
checkpoints. Base video frame count, base denoising settings, and audio
sampling are resolved from the generation checkpoint as usual. SR preserves
the generated frame count and audio. No source video is required.

## Existing offline examples

```bash
CUDA_VISIBLE_DEVICES=6 python examples/offline_inference/text_to_video/text_to_video.py \
  --model kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers \
  --prompt "A red fox runs through a snowy forest." \
  --enable-cpu-offload --height 720 --width 1280 --output fox_hd.mp4

CUDA_VISIBLE_DEVICES=7 python examples/offline_inference/image_to_video/image_to_video.py \
  --model kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers \
  --image fox.png --prompt "The fox starts running through the forest." \
  --enable-cpu-offload --height 1080 --width 1920 --output fox_fhd.mp4
```

Alternatively, pass `--extra-body '{"target_resolution": "fhd"}'` with omitted
height and width. The reference image is resized by the generation pipeline
to its base canvas before conditioning.

## SR checkpoints and defaults

The final stage defaults to
`kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` with two PiFlow
model evaluations per tile. Select the regular four-step Euler checkpoint
through `sr_model`:

```python
sampling = OmniDiffusionSamplingParams(
    seed=42,
    extra_args={
        "target_resolution": "fhd",
        "sr_model": "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers",
    },
)
```

Both bundles come from the [Kandinsky VSR collection](https://huggingface.co/collections/kandinskylab/kandinsky-60-vsr).
`num_inference_steps` controls base generation. `sr_num_inference_steps`
overrides SR evaluations separately. `sr_min_overlap` defaults to 0.2 and
`sr_tiles_batch_size` to 1. These fields can be supplied in `extra_body`.

HD/FHD requests center-crop to the delivery aspect ratio, refine with tiled
diffusion, and crop KVAE padding to the exact output dimensions. Omitting an
HD/FHD target keeps the existing base-resolution generation behavior.

## Execution

SR weights load lazily after the first HD/FHD base generation and are cached
on CPU between requests. The base model is temporarily moved to CPU during
refinement and restored afterward. This requires enough host RAM for the
base model and SR checkpoint. Pro generation should use component CPU offload
on an H100 80 GB GPU.

The integrated SR stage currently supports one GPU with ordinary residency
or component CPU offload. HSDP and layerwise offload are unsupported for
HD/FHD requests. The SR transformer uses compiled flex attention for NABLA
sparse attention, so the first request includes compilation time.

The SR components are adapted from Apache-licensed Diffusers source at commit
`916a5791ed5f29d966200d2eee53f06862ec4434` and work with Omni's pinned Diffusers
version. No separate Kandinsky package is required.

## Tests

```bash
python -m pytest tests/diffusion/models/kandinsky6 tests/diffusion/models/kandinsky6_sr -q
```
