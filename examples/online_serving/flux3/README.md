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

## 2. Send requests

Omitted sampling fields (`num_inference_steps`, `guidance_scale`,
`flow_shift`) use the package defaults. Flow-map (distilled) packages reject
an explicit `guidance_scale` and a nonempty `negative_prompt`; base packages
accept both.

Text-to-image with the shared client, run from the repository root:

```bash
python examples/online_serving/text_to_image/openai_chat_client.py \
    --server http://localhost:8099 \
    --prompt "alpine meadow at golden hour" \
    --height 1024 --width 1024 \
    --output flux3_t2i.png
```

Or call the Images API directly:

```bash
curl -s http://localhost:8099/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{"model": "flux-3-image-klein", "prompt": "alpine meadow at golden hour",
         "size": "1024x1024", "n": 1, "response_format": "b64_json", "seed": 42}' \
    | jq -r '.data[0].b64_json' | base64 -d > flux3_t2i.png
```

Image-to-image sends the references as `image_url` parts of one chat message,
in reference order (up to 10):

```bash
IMG1=$(base64 -w0 person.jpg); IMG2=$(base64 -w0 clothing.jpg)
curl -s http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "flux-3-image-klein", "seed": 42, "height": 1024, "width": 1024,
         "messages": [{"role": "user", "content": [
           {"type": "text", "text": "dress the person from image 1 in the clothing from image 2"},
           {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG1"'"}},
           {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG2"'"}}]}]}' \
    | jq -r '.choices[0].message.content[0].image_url.url' | cut -d, -f2 | base64 -d > flux3_i2i.png
```

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

The caption request follows the model card:

- System prompt: the `t2i` entry of `system_prompts.json` (in the pup repository)
  for text-to-image, or `i2i` for editing. The `t2i_bbox` / `i2i_bbox` entries
  return the caption followed by a JSON element table with one
  `[y_min, x_min, y_max, x_max]` box (0-1000 grid) per cited element.
- Editing: send the reference images first (up to 10), then the text
  `Reference images follow in order (the first image, the second image, ...).\nUser request: <request>\nOutput the caption.`
- Sampling: `temperature: 0`, `max_tokens: 3584`,
  `chat_template_kwargs: {"enable_thinking": false}`, and `presence_penalty`
  1.5 for text-to-image or 0 for editing.

```bash
SYSTEM=$(hf download black-forest-labs/flux-3-image-klein-pup system_prompts.json | xargs jq -r .t2i)
jq -n --arg system "$SYSTEM" --arg request "A man walking with his dog, afternoon sun" '{
    model: "flux-3-image-klein-pup", max_tokens: 3584, temperature: 0, presence_penalty: 1.5,
    chat_template_kwargs: {enable_thinking: false},
    messages: [{role: "system", content: $system}, {role: "user", content: $request}]}' \
  | curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d @- \
  | jq -r '.choices[0].message.content'
```

Use the printed caption as the FLUX 3 Image Klein prompt.
