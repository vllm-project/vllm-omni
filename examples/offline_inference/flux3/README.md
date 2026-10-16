# FLUX 3 Image Klein Offline Inference

FLUX 3 Image Klein is an image-generation model supporting text-to-image (T2I) and
image-to-image (I2I).

## Model checkpoint layout

The current FLUX 3 Image Klein package layout keeps the DiT and VAE safetensors at the
package root, with text assets in subdirectories:

```
/path/to/flux-3-image-klein/
├── config.json          # FLUX 3 Image Klein package settings
├── model.safetensors    # DiT weights
├── ae.safetensors       # Image autoencoder weights
├── text_encoder/        # Qwen3-VL text encoder and tokenizer / processor
└── variants/            # Optional DiT packages
    └── <variant>/       # Variant config.json and DiT weights
```

Packages can provide alternate DiT variants under `variants/`. Keep
`--model` pointed at the shared package root and select one with `--variant`.
Available variants depend on the checkpoint root; it need not contain every
example variant listed below. Omit the variant to load the BF16 root package,
whose config determines whether sampling uses CFG or flow map.
Any variant folder name is accepted; `--variant` must match it exactly.

The following names and configurations are examples. DiT precision and sampling
are determined by the selected package's `config.json`, not its folder name.

| Example variant | Example DiT precision | Example sampling |
|---|---|---|
| omitted | BF16 | From root config |
| `fp8r` | FP8R | From package config |
| `nvfp4` | NVFP4 | From package config |

Flow-map packages use their configured step count. They do not support
classifier-free guidance, so omit `--guidance-scale` and `--negative-prompt`. Each variant loads
its own DiT package under `variants/` and shares the root text encoder and VAE.

The package root is detected from `config.json` using `model: flux-3-image-klein` or the
supported architecture fields. Native variant loading also applies when the
package includes a Diffusers pipeline index.

## Install PrimsTS kernels

PrimsTS is an optional performance optimization for compatible FP8R and NVFP4
packages on SM100, SM103, and SM107 CUDA GPUs. FLUX 3 Image Klein runs without it using
vLLM kernels. To install it, run this from the repository root in your
inference environment:

```bash
uv pip install -e '.[prims-ts]'
```

This installs the pinned FlashInfer revision containing the kernels. PrimsTS
defaults to `auto`: compatible FP8R and NVFP4 packages use it when available
on supported CUDA GPUs (SM100, SM103, SM107). Otherwise FLUX 3 Image Klein falls
back to vLLM kernels. Installing the extra does not enable PrimsTS for BF16
packages or unsupported GPUs.

## Quick start

### FP8 attention

Use `--fp8-attention` to select FlashInfer/cuDNN FP8 attention for the DiT:

```bash
python end2end.py --model /path/to/flux-3-image-klein --fp8-attention
```

This is independent of the weight variant. Q, K, and V are quantized at runtime;
the output stays in the compute dtype. It requires FlashInfer's cuDNN FP8 backend
and cuDNN >= 9.17.1. FP8 attention is available on SM100.

### Generation commands

The variant commands below require the corresponding folder under the chosen
checkpoint root. For the base model, use `/path/to/flux-3-image-klein-base`.

```bash
# Run text-to-image with the default prompt
python end2end.py --model /path/to/flux-3-image-klein

# Run image-to-image with a reference
python end2end.py --model /path/to/flux-3-image-klein --prompt "restyle as watercolor" --image ref.jpg

# Multiple reference images, in input order
python end2end.py --model /path/to/flux-3-image-klein --prompt "dress the person from image 1 in the clothing from image 2" --image person.jpg clothing.jpg

# Run the FP8R package variant
python end2end.py --model /path/to/flux-3-image-klein --variant fp8r

# Run the nvfp4 package variant
python end2end.py --model /path/to/flux-3-image-klein --variant nvfp4

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
|---|---|
| `--prompt` (or the default prompt) | Text-to-image |
| `--prompt` and `--image` | Image-to-image |

This applies to every variant. Omitted guidance, step count, and flow shift
use the selected model package's defaults.

See the [online serving README](../../online_serving/flux3/README.md) for
server startup and client examples.

## Outputs

Outputs default to `outputs/` in the current working directory. Use
`--output-dir` to change the location.

The generated image is saved as `image.png`. Use different output directories
to keep results from separate runs.

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
