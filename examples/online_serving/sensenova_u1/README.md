# SenseNova-U1 Online Serving

## Launch the Server

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091
```

Or use the convenience script:

```bash
cd examples/online_serving/sensenova_u1
bash run_server.sh
```

### Tensor Parallelism (TP)

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091 \
    --tensor-parallel-size 2
```

## Send Requests

```bash
cd examples/online_serving/sensenova_u1
```

### Text to Image (text2img)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "A beautiful sunset over mountains" \
    --modality text2img \
    --height 2048 --width 2048 --num-steps 50
```

**curl:**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "A beautiful sunset over mountains"}]}],
    "modalities": ["image"],
    "height": 2048,
    "width": 2048,
    "num_inference_steps": 50,
    "seed": 42
  }'
```

### Image to Image (img2img)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "Turn this into an oil painting" \
    --modality img2img \
    --image-url /path/to/input.jpg \
    --height 2048 --width 2048
```

**curl:**

```bash
IMAGE_BASE64=$(base64 -w 0 input.jpg)

cat <<EOF > payload.json
{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Turn this into an oil painting"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
      ]
    }],
    "modalities": ["image"],
    "height": 2048,
    "width": 2048,
    "num_inference_steps": 50,
    "seed": 42
}
EOF

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Image to Text (img2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text \
    --image-url /path/to/image.jpg
```

**curl:**

```bash
IMAGE_BASE64=$(base64 -w 0 image.jpg)

cat <<EOF > payload.json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image in detail"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
    ]
  }],
  "modalities": ["text"]
}
EOF

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Text to Text (text2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

**curl:**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}],
    "modalities": ["text"]
  }'
```

## Python Client Arguments

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--prompt` / `-p` | `A cute cat` | Text prompt |
| `--output` / `-o` | `sensenova_u1_output.png` | Output file path |
| `--server` / `-s` | `http://localhost:8091` | Server URL |
| `--image-url` / `-i` | `None` | Input image URL or local path (img2img/img2text) |
| `--modality` / `-m` | `text2img` | `text2img`, `img2img`, `img2text`, `text2text` |
| `--height` | `2048` | Image height (image generation only) |
| `--width` | `2048` | Image width (image generation only) |
| `--num-steps` | `50` | Number of inference steps (image generation only) |
| `--seed` | `42` | Random seed |
| `--cfg-scale` | `4.0` | CFG scale (image generation only) |
| `--think` | `False` | Enable think mode |
