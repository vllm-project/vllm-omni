# SenseNova-Vision: Online serving

OpenAI-compatible chat completions API for
[`sensenova/SenseNova-Vision-7B-MoT`](https://huggingface.co/sensenova/SenseNova-Vision-7B-MoT),
served by `vllm-omni`. The server exposes the full SenseNova-Vision modality
matrix through the standard chat API: text2text, img2text, text2img, img2img,
dense perception, multi-view camera pose, recon3d, and the mixed
`caption_generate` mode (image + intermediate text).

## Launch the Server

```bash
bash examples/online_serving/sensenova/run_server.sh
# or, with overrides
MODEL=sensenova/SenseNova-Vision-7B-MoT \
DEPLOY_CONFIG=vllm_omni/deploy/sensenova.yaml \
PORT=8092 \
    bash examples/online_serving/sensenova/run_server.sh
```

Equivalently, run `vllm serve` directly:

```bash
vllm serve sensenova/SenseNova-Vision-7B-MoT \
    --omni \
    --port 8092 \
    --deploy-config vllm_omni/deploy/sensenova.yaml
```

The two-stage deploy config (Thinker + DiT sharing GPU 0) defaults to
`vllm_omni/deploy/sensenova.yaml`; point `DEPLOY_CONFIG` at a custom YAML to
change the topology or device layout.

## Send Requests

```bash
cd examples/online_serving/sensenova
```

### Text to Image (text2img)

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

### Mixed text + image (caption_generate)

```bash
python openai_chat_client.py \
    --prompt "<image> Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer." \
    --modality mixed \
    --image-url /path/to/photo.jpg \
    --output sensenova_mixed.png
```

The `caption_generate` mode returns both a generated image and the
intermediate caption text; the client saves the image and prints the text.

### Image to Image (img2img)

```bash
python openai_chat_client.py \
    --prompt "Turn this image into a vibrant cartoon-style illustration." \
    --modality img2img \
    --image-url /path/to/photo.jpg \
    --output sensenova_img2img.png
```

### Text to Text (text2text)

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
