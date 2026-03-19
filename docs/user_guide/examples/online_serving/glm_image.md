# GLM-Image Online Serving

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/glm_image>.


This example demonstrates how to deploy GLM-Image for online image generation using vLLM-Omni.

## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Run examples (GLM-Image)

**Note**: These examples work with the default configuration on **2× NVIDIA A100 (80GB)** or equivalent. Stage 0 (AR) and Stage 1 (Diffusion) each use one GPU by default. For single-GPU setups, modify the stage configuration to share the same device.

### Launch the Server

```bash
# Use default configuration
vllm serve zai-org/GLM-Image --omni --port 8091
```

Or use the convenience script:

```bash
cd examples/online_serving/glm_image
bash run_server.sh
```

If you have a custom stage configs file:

```bash
vllm serve zai-org/GLM-Image --omni --port 8091 --stage-configs-path /path/to/glm_image.yaml
```

### Send Requests

Get into the glm_image folder:

```bash
cd examples/online_serving/glm_image
```

Send request via Python:

```bash
python openai_chat_client.py --prompt "A cute cat sitting on a window sill"
```

The Python client supports the following command-line arguments:

- `--prompt` (or `-p`): Text prompt for generation (default: `A beautiful sunset over the ocean with sailing boats`)
- `--output` (or `-o`): Output file path (default: `glm_image_output.png`)
- `--server` (or `-s`): Server URL (default: `http://localhost:8091`)
- `--image` (or `-i`): Input image path (for image-to-image editing)
- `--height`: Image height in pixels (default: 1024)
- `--width`: Image width in pixels (default: 1024)
- `--steps`: Number of inference steps (default: 50)
- `--guidance-scale`: Classifier-free guidance scale (default: 1.5)
- `--seed`: Random seed (default: 42)
- `--negative`: Negative prompt

## Modality Control

GLM-Image supports **text-to-image** and **image-to-image** modes.

The default yaml configuration deploys AR on GPU 0 and DiT on GPU 1. You can use the default configuration file: [`glm_image.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/model_executor/stage_configs/glm_image.yaml)

| Mode           | Input        | Output | Description                        |
| -------------- | ------------ | ------ | ---------------------------------- |
| Text-to-Image  | Text         | Image  | Generate images from text prompts  |
| Image-to-Image | Image + Text | Image  | Edit images with text instructions |

### Text-to-Image

Generate images from text prompts:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "A photorealistic mountain landscape at sunset" \
    --height 1024 \
    --width 1024 \
    --output landscape.png
```

**Using curl**

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A beautiful sunset over the ocean with sailing boats"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 1.5,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

Or use the script:

```bash
bash run_curl_text_to_image.sh "A futuristic city skyline at night"
```

### Image-to-Image (Image Editing)

Edit images with text instructions:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "Convert this image to watercolor style" \
    --image input.png \
    --output watercolor.png
```

**Using curl**

```bash
IMG_B64=$(base64 < input.png | tr -d '\n')

curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Convert this image to watercolor style"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'$IMG_B64'"}}
    ]
  }],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 1.5,
    "seed": 42
  }
}
EOF
```

Or use the script:

```bash
bash run_curl_image_edit.sh input.png "Convert to watercolor style"
```

## Generation Parameters (extra_body)

| Parameter             | Type  | Default | Description                         |
| --------------------- | ----- | ------- | ----------------------------------- |
| `height`              | int   | 1024    | Image height in pixels              |
| `width`               | int   | 1024    | Image width in pixels               |
| `num_inference_steps` | int   | 50      | Number of diffusion denoising steps |
| `guidance_scale`      | float | 1.5     | Classifier-free guidance scale      |
| `seed`                | int   | 42      | Random seed for reproducibility     |
| `negative_prompt`     | str   | None    | Negative prompt                     |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "zai-org/GLM-Image",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,..."
            }
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {}
}
```

## Extract Image

```bash
# From a saved JSON response
cat response.json | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

## Architecture

GLM-Image uses a 2-stage multistage pipeline:

```
Stage 0 (AR Model)                Stage 1 (Diffusion)
┌───────────────────┐            ┌─────────────────────┐
│ vLLM-optimized    │  prior     │  GlmImagePipeline   │
│ GlmImageFor       │──tokens──►│  ┌───────────────┐  │
│ Conditional       │            │  │ DiT Denoiser  │  │
│ Generation        │            │  └───────┬───────┘  │
│ (9B AR model)     │            │          ▼          │
└───────────────────┘            │  ┌───────────────┐  │
        ▲                        │  │  VAE Decode   │──┼──► Image
        │                        │  └───────────────┘  │
   Text / Image                  └─────────────────────┘
     Input
```

## VRAM Requirements

| Stage             | VRAM                   |
| :---------------- | :--------------------- |
| Stage-0 (AR)      | **~18 GiB + KV Cache** |
| Stage-1 (DiT+VAE) | **~20 GiB**            |
| Total             | **~38 GiB + KV Cache** |

## File Description

| File                        | Description                           |
| --------------------------- | ------------------------------------- |
| `run_server.sh`             | Server startup script                 |
| `run_curl_text_to_image.sh` | Text-to-image curl example            |
| `run_curl_image_edit.sh`    | Image-to-image (editing) curl example |
| `openai_chat_client.py`     | Python client (t2i + i2i)             |

## FAQ

- If you encounter OOM errors, adjust `gpu_memory_utilization` in the stage config:

```yaml
# In glm_image.yaml, reduce from default 0.6:
gpu_memory_utilization: 0.5
```

- The first request may be slow due to model warmup. Subsequent requests will be faster.

## Example materials

??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/glm_image/openai_chat_client.py"
    ``````
??? abstract "run_curl_image_edit.sh"
    ``````sh
    --8<-- "examples/online_serving/glm_image/run_curl_image_edit.sh"
    ``````
??? abstract "run_curl_text_to_image.sh"
    ``````sh
    --8<-- "examples/online_serving/glm_image/run_curl_text_to_image.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/glm_image/run_server.sh"
    ``````
