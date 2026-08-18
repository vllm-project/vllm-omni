# SenseNova-Vision-7B-MoT

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/sensenova>.

## Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Architecture

SenseNova-Vision-7B-MoT is a Mixture-of-Transformers (MoT) model supporting
image generation, image understanding, dense perception, and 3D reconstruction
from a single unified pipeline. It is served with a two-stage topology:

| Topology | Stages | Description |
| :------- | :----- | :---------- |
| **Two-stage** (default) | Stage 0 (Thinker, AR) + Stage 1 (DiT, Diffusion) | Thinker handles text/understanding via the vLLM AR engine; DiT handles image generation. KV cache is transferred between stages. |

The modality matrix includes `text2text`, `img2text` (captioning, dense
detection, dense OCR), `text2img` (generate), `img2img` (edit), `img2dense`
(depth / normal / segmentation), multi-image camera pose estimation, multi-view
reconstruction (`recon3d`), and the mixed `caption_generate` mode that returns
both an image and intermediate caption text.

## Launch the Server

```bash
vllm serve sensenova/SenseNova-Vision-7B-MoT --omni --port 8092 \
    --deploy-config vllm_omni/deploy/sensenova.yaml
```

Or use the convenience script:

```bash
bash examples/online_serving/sensenova/run_server.sh
```

See [`sensenova.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/deploy/sensenova.yaml) for the default two-stage deploy configuration.

## Send Requests

```bash
cd examples/online_serving/sensenova
```

### Text to Image (text2img)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "A cute corgi astronaut on the moon, cinematic" \
    --modality text2img \
    --output sensenova_text2img.png
```

**curl:**

```bash
curl http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "A cute corgi astronaut on the moon, cinematic"}]}],
    "modalities": ["image"],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "seed": 42
  }'
```

### Image to Text (img2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "What are the main objects in this scene and their relationships?" \
    --modality img2text \
    --image-url /path/to/photo.jpg
```

**curl:**

```bash
IMAGE_BASE64=$(base64 -w 0 photo.jpg)

cat <<EOF > payload.json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What are the main objects in this scene and their relationships?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
    ]
  }],
  "modalities": ["text"]
}
EOF

curl http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Mixed Text + Image (caption_generate)

The `caption_generate` mode returns both a generated image and the
intermediate caption text. The client saves the image and prints the text.

```bash
python openai_chat_client.py \
    --prompt "<image> Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer." \
    --modality mixed \
    --image-url /path/to/photo.jpg \
    --output sensenova_mixed.png
```

### Image to Image (img2img)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "Turn this image into a vibrant cartoon-style illustration." \
    --modality img2img \
    --image-url /path/to/photo.jpg \
    --output sensenova_img2img.png
```

### Text to Text (text2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

## Python Client Arguments

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--prompt` / `-p` | per-mode default | Text prompt |
| `--output` / `-o` | `sensenova_output.png` | Output file path (image results) |
| `--server` / `-s` | `http://localhost:8092` | Server URL |
| `--image-url` / `-i` | `None` | Input image URL or local path (img2text/img2img/mixed) |
| `--modality` / `-m` | `text2img` | `text2img`, `img2img`, `img2text`, `text2text`, `mixed` |
| `--height` | `1024` | Image height (image-output modes) |
| `--width` | `1024` | Image width (image-output modes) |
| `--num-steps` | `50` | Number of inference steps (image-output modes) |
| `--seed` | `42` | Random seed |
| `--cfg-text-scale` | `None` | Text CFG scale (forwarded to `extra_args`) |
| `--cfg-img-scale` | `None` | Image CFG scale (forwarded to `extra_args`) |
| `--timestep-shift` | `None` | Flow-match timestep shift (forwarded to `extra_args`) |
| `--max-think-tokens` | `None` | Max think tokens for text modes (forwarded to `extra_args`) |

## FAQ

- If you encounter OOM errors, try decreasing `max_model_len` or `gpu_memory_utilization` in the deploy YAML.
- The dense perception, camera pose, and reconstruction modes are exercised by the offline example (`examples/offline_inference/sensenova/end2end.py`), which runs the decoders client-side (e.g. `parse_bbox`, `decode_depth`, `parse_camera_pose`, `decode_point_map`).

## Example materials

??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/sensenova/openai_chat_client.py"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/sensenova/run_server.sh"
    ``````
