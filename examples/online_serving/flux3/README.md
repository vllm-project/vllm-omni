# FLUX 3 Image Klein Online Serving

FLUX 3 Image Klein supports image generation via the OpenAI-compatible API: T2I and I2I.

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
Flow-map packages use their package sampling defaults and reject
`guidance_scale` and nonempty `negative_prompt`. Omit these overrides when
testing distilled packages online.

The package root is detected from `config.json` using `model: flux-3-image-klein` or the
supported architecture fields. Native loading also applies when the
package includes a Diffusers pipeline index.

## 1. Start the server

```bash
MODEL=/path/to/flux-3-image-klein ./run_server.sh
```

Or directly:

```bash
vllm serve /path/to/flux-3-image-klein --omni \
    --served-model-name flux-3-image-klein \
    --port 8099
```

## 2. Run the client

By default the client omits guidance scale, step count, and flow shift so the
server uses the package defaults:

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

## 3. API endpoints

| Mode | Endpoint | Method |
| --- | --- | --- |
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
