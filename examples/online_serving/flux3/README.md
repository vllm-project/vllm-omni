# FLUX 3 Image Klein Online Serving

FLUX 3 Image Klein supports image generation via the OpenAI-compatible API: T2I and I2I.

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

Variants are stored under `variants/<variant>/` and selected from the shared
package root. Available variants depend on the checkpoint root; it need not
contain every example variant listed below. Omit the variant to load the BF16
root package, whose config determines whether sampling uses CFG or flow map.
Any variant folder name is accepted; `VARIANT` must match it exactly.

The following names and configurations are examples. DiT precision and sampling
are determined by the selected package's `config.json`, not its folder name.

| Example variant | Example DiT precision | Example sampling |
|---|---|---|
| omitted | BF16 | From root config |
| `fp8r` | FP8R | From package config |
| `nvfp4` | NVFP4 | From package config |

Flow-map packages use their package sampling defaults and reject
`guidance_scale` and nonempty `negative_prompt`. Omit these overrides when
testing distilled packages online. Each variant requires its corresponding package folder.

The package root is detected from `config.json` using `model: flux-3-image-klein` or the
supported architecture fields. Native variant loading also applies when the
package includes a Diffusers pipeline index.

## Install PrimsTS kernels

PrimsTS is an optional performance optimization for compatible FP8R and NVFP4
packages on SM100, SM103, and SM107 CUDA GPUs. The server runs without it using
vLLM kernels. To install it, run this from the repository root in the server
environment before starting the server:

```bash
uv pip install -e '.[prims-ts]'
```

This installs the pinned FlashInfer revision containing the kernels. PrimsTS
defaults to `auto`: compatible FP8R and NVFP4 packages use it when available
on supported CUDA GPUs (SM100, SM103, SM107). Otherwise FLUX 3 Image Klein falls
back to vLLM kernels. Installing the extra does not enable PrimsTS for BF16
packages or unsupported GPUs.

## 1. Start the server

### FP8 attention

Enable FlashInfer/cuDNN FP8 DiT attention at server startup:

```bash
MODEL=/path/to/flux-3-image-klein bash run_server.sh --fp8-attention
```

All generation requests then use this attention configuration. The flag takes
precedence over `DIFFUSION_ATTENTION_BACKEND` and is independent of `VARIANT`.
It requires FlashInfer's cuDNN FP8 backend and cuDNN >= 9.17.1.
FP8 attention is available on SM100. Client requests need no additional parameter.

### Launch commands

The variant commands below require the corresponding folder under the chosen
checkpoint root. For the base model, use `/path/to/flux-3-image-klein-base`.

```bash
MODEL=/path/to/flux-3-image-klein ./run_server.sh

# Quantized package variant
MODEL=/path/to/flux-3-image-klein VARIANT=fp8r ./run_server.sh

# NVFP4 package variant
MODEL=/path/to/flux-3-image-klein VARIANT=nvfp4 ./run_server.sh
```

Or directly:

```bash
vllm serve /path/to/flux-3-image-klein --omni \
    --served-model-name flux-3-image-klein \
    --port 8099

# Quantized package variant
vllm serve /path/to/flux-3-image-klein --omni \
    --served-model-name flux-3-image-klein \
    --stage-overrides '{"0":{"additional_config":{"variant":"fp8r"}}}' \
    --port 8099

# NVFP4 package variant
vllm serve /path/to/flux-3-image-klein --omni \
    --served-model-name flux-3-image-klein \
    --stage-overrides '{"0":{"additional_config":{"variant":"nvfp4"}}}' \
    --port 8099
```

### Preload a LoRA adapter

Set `LORA_PATH` to preload an adapter when the server starts. `LORA_SCALE`
defaults to `1.0`:

```bash
MODEL=/path/to/flux-3-image-klein \
LORA_PATH=/absolute/path/to/adapter LORA_SCALE=1.0 \
bash run_server.sh
```

For direct `vllm serve` commands, add `--lora-path /absolute/path/to/adapter`
and optionally `--lora-scale 1.0`. Preloading uses the default PEFT backend.
Still select the adapter in each client request using the same absolute path;
requests without a LoRA selection use the base model.

## 2. Run the client

The same client works for the root package and supported variants. By default it omits
guidance scale, step count, and flow shift so the server uses the selected
package's defaults:

Each invocation sends one request. Without `--image`, the client uses the
text-to-image endpoint; supplying `--image` selects image-to-image.

```bash
# Text-to-image
python openai_client.py --model flux-3-image-klein --prompt "alpine meadow at golden hour"

# Image-to-image
python openai_client.py --model flux-3-image-klein --prompt "restyle as watercolor" --image photo.jpg

# Multiple reference images, in input order
python openai_client.py --prompt "dress the person from image 1 in the clothing from image 2" --image person.jpg clothing.jpg
```

Explicit `--steps` and `--flow-shift` override package defaults. Explicit
`--guidance-scale` and nonempty `--negative-prompt` are rejected for distilled
packages; omit them. For base packages, those options remain available.

For LoRA, add `--lora-path` and optionally `--lora-scale` (default: `1.0`) to
either client mode:

```bash
python openai_client.py --prompt "your adapter trigger, portrait in natural light" \
    --lora-path /absolute/path/to/adapter --lora-scale 1.0
```

The path must exist on the server and is sent unchanged. `--lora-name` optionally
sets the adapter label; otherwise the final path component is used.

## 3. API endpoints

| Mode | Endpoint | Method |
|---|---|---|
| T2I | `/v1/images/generations` | POST (JSON) |
| I2I | `/v1/chat/completions` | POST (JSON, image in message) |

## 4. Benchmark

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8099 \
    --model flux-3-image-klein \
    --task t2i \
    --dataset random \
    --num-prompts 20 \
    --num-inference-steps 50
```

## 5. Prompt upsampler (pup)

[black-forest-labs/flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)
is a Qwen3.5-9B vision-language fine-tune that expands a short prompt, into the detailed caption FLUX 3 Image
Klein works best with. The reference inference code runs it in front of the image model by default. It is a
standard `Qwen3_5ForConditionalGeneration` checkpoint, so it is served with the stock vLLM image;
the system prompt (`system_prompts.json` in the repo) selects text-to-image or editing mode.

```bash
PUP_PATH="<local-path>"
hf download black-forest-labs/flux-3-image-klein-pup --local-dir ${PUP_PATH}

docker run --name flux-3-image-klein-pup-vllm \
      --gpus '"device=0"' \
      --ipc=host \
      -p 8000:8000 \
      -v ${PUP_PATH}:/model:ro \
      vllm/vllm-openai:v0.29.0 \
      --model /model \
      --served-model-name flux-3-image-klein-pup \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.90 \
      --reasoning-parser qwen3
```

`openai_client_pup.py` fetches the system prompts from the Hub and prints the caption.
Without `--image` it uses the text-to-image prompt; with `--image` (repeatable, up to 10)
the editing prompt, references first. `--bbox` switches to the `t2i_bbox` / `i2i_bbox`
prompts, which return the caption followed by a JSON element table with one
`[y_min, x_min, y_max, x_max]` box (0-1000 grid) per cited element; `--boxes` passes your
own boxes as a JSON array after the request:

```bash
# Text-to-image
python openai_client_pup.py --prompt "A man walking with his dog, afternoon sun"

# Editing with a reference image
python openai_client_pup.py --prompt "put a red bandana around the dog's neck" --image dog.jpg

# Layout: caption plus bounding-box table
python openai_client_pup.py --prompt "A man walking with his dog, afternoon sun" --bbox
```

Feed the printed caption to `openai_client.py --prompt` to generate the image.
