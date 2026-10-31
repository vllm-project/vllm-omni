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

```bash
# Run text-to-image with the default prompt
python end2end.py --model /path/to/flux-3-image-klein

# Run image-to-image with a reference
python end2end.py --model /path/to/flux-3-image-klein --prompt "restyle as watercolor" --image ref.jpg

# Multiple reference images, in input order
python end2end.py --model /path/to/flux-3-image-klein --prompt "dress the person from image 1 in the clothing from image 2" --image person.jpg clothing.jpg

# Custom prompt and resolution
python end2end.py --model /path/to/flux-3-image-klein \
    --prompt "A fox sitting in a sunlit forest" \
    --height 720 --width 1280 \
    --num-inference-steps 50
```

When height and width are omitted, FLUX 3 Image Klein uses the package sampler default
(`1024x1024` for text-to-image). For image-to-image, the reference image aspect
ratio is preserved at the same default area.

## Inputs

Each invocation generates one image. The prompt and optional reference image
are forwarded directly to the model:

| Inputs | Generation |
| --- | --- |
| `--prompt` (or the default prompt) | Text-to-image |
| `--prompt` and `--image` | Image-to-image |

Omitted guidance, step count, and flow shift use the model package's defaults.

See the [online serving README](../../online_serving/flux3/README.md) for
server startup and client examples.

## Outputs

Outputs default to `outputs/` in the current working directory. Use
`--output-dir` to change the location.

The generated image is saved as `image.png`. Use different output directories to
keep results from separate runs.

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
