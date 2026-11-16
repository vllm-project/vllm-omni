# FLUX 3 Image Klein Offline Inference

FLUX 3 Image Klein is an image-generation model supporting text-to-image (T2I) and
image-to-image (I2I).

## Model checkpoint layout

The current FLUX 3 Image Klein package layout keeps the DiT and VAE safetensors at the
package root, with text assets in subdirectories:

```text
/path/to/flux-3-image-klein/
├── config.json          # FLUX 3 Image Klein package settings
├── model.safetensors    # DiT weights
├── ae.safetensors       # Image autoencoder weights
└── text_encoder/        # Qwen3-VL text encoder and tokenizer / processor
```

The BF16 root package config determines whether sampling uses CFG or flow map.
Flow-map packages use their configured step count. They do not support
classifier-free guidance, so omit `--guidance-scale` and `--negative-prompt`.

The package root is detected from `config.json` using `model: flux-3-image-klein` or the
supported architecture fields. Native loading also applies when the
package includes a Diffusers pipeline index.

## Quick start

FLUX 3 Image Klein uses the shared offline runners. Run them from the
repository root:

```bash
# Text-to-image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model /path/to/flux-3-image-klein \
    --prompt "A fox sitting in a sunlit forest" \
    --height 1024 --width 1024 \
    --num-inference-steps 50 --guidance-scale 4.0 \
    --output flux3_t2i.png

# Image-to-image with one or more references, in input order
python examples/offline_inference/image_to_image/image_edit.py \
    --model /path/to/flux-3-image-klein \
    --prompt "dress the person from image 1 in the clothing from image 2" \
    --image person.jpg clothing.jpg \
    --guidance-scale 4.0 \
    --output flux3_i2i.png
```

Height and width must be multiples of 16. Pass the flow shift through
`--extra-body '{"flow_shift": 3.0}'` (`text_to_image.py`) or
`--extra-args '{"flow_shift": 3.0}'` (`image_edit.py`).

The shared offline runners always send an explicit guidance scale. Flow-map
(distilled) packages reject `guidance_scale`, so run them through the
[online API](../../online_serving/flux3/README.md), which omits unset sampling
fields and applies the package defaults.

## Inputs

Each request generates one image; `num_outputs_per_prompt` above one is
rejected.

| Inputs | Generation |
| --- | --- |
| `--prompt` | Text-to-image |
| `--prompt` and `--image` (up to 10 references) | Image-to-image |

When height and width are omitted for image-to-image, the reference aspect
ratio is preserved at the package's default area.

## Benchmark

Use the shared serving benchmark against a running server — see
[`benchmarks/diffusion/README.md`](../../../benchmarks/diffusion/README.md):

```bash
vllm serve /path/to/flux-3-image-klein --omni --served-model-name flux-3-image-klein --port 8099
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8099 \
    --model flux-3-image-klein \
    --task t2i \
    --dataset random \
    --num-prompts 20 \
    --num-inference-steps 50
```
