# BAGEL-7B-MoT

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/bagel>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Run examples (BAGEL-7B-MoT)

**Note**: These examples work with the default configuration on an **NVIDIA A100 (80GB)**. We also tested on dual **NVIDIA RTX 5000 Ada (32GB each)**. For dual-GPU setups, please modify the stage configuration to distribute the model across devices.

### Launch the Server

```bash
# Use default configuration
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091
```

Or use the convenience script:

```bash
cd /workspace/vllm-omni/examples/online_serving/bagel
bash run_server.sh
```

If you have a custom stage configs file, launch the server with the command below:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```

### Send Multi-modal Request

Get into the bagel folder:

```bash
cd examples/online_serving/bagel
```

Send request via Python

```bash
python openai_chat_client.py --prompt "A cute cat" --modality text2img
```

The Python client supports the following command-line arguments:

- `--prompt` (or `-p`): Text prompt for generation (default: `A cute cat`)
- `--output` (or `-o`): Output file path for image results (default: `bagel_output.png`)
- `--server` (or `-s`): Server URL (default: `http://localhost:8091`)
- `--image-url` (or `-i`): Input image URL or local file path (for img2img/img2text modes)
- `--modality` (or `-m`): Task modality (default: `text2img`). Options: `text2img`, `img2img`, `img2text`, `text2text`
- `--height`: Image height in pixels (default: 512)
- `--width`: Image width in pixels (default: 512)
- `--steps`: Number of inference steps (default: 25)
- `--seed`: Random seed (default: 42)
- `--negative`: Negative prompt for image generation

Example with custom parameters:

```bash
python openai_chat_client.py \
    --prompt "A futuristic city" \
    --modality text2img \
    --height 768 \
    --width 768 \
    --steps 50 \
    --seed 42 \
    --negative "blurry, low quality"
```

## Modality Control

BAGEL-7B-MoT supports **multiple modality modes** for different use cases.

The default yaml configuration deploys Thinker and DiT on the same GPU. You can use the default configuration file: [`bagel.yaml`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/model_executor/stage_configs/bagel.yaml)

| Modality    | Input        | Output | Description                            |
| ----------- | ------------ | ------ | -------------------------------------- |
| `text2img`  | Text         | Image  | Generate images from text prompts      |
| `img2img`   | Image + Text | Image  | Transform images using text guidance   |
| `img2text`  | Image + Text | Text   | Generate text descriptions from images |
| `text2text` | Text         | Text   | Pure text generation                   |

### Text to Image (text2img)

Generate images from text prompts:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "A beautiful sunset over mountains" \
    --modality text2img \
    --output sunset.png \
    --steps 50
```

**Using curl**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<|im_start|>A beautiful sunset over mountains<|im_end|>"}]}],
    "modalities": ["image"],
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "seed": 42
  }'
```


### Image to Image (img2img)

Transform images based on text prompts:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "Make the cat stand up" \
    --modality img2img \
    --image-url /path/to/input.jpg \
    --output transformed.png
```

**Using curl**

```bash
IMAGE_BASE64=$(base64 -w 0 cat.jpg)

cat <<EOF > payload.json
{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "<|im_start|>Make the cat stand up<|im_end|>"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
      ]
    }],
    "modalities": ["image"],
    "height": 512,
    "width": 512,
    "num_inference_steps": 50,
    "seed": 42
}
EOF

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json

```

### Image to Text (img2text)

Generate text descriptions from images:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text \
    --image-url /path/to/image.jpg
```

**Using curl**

```bash
IMAGE_BASE64=$(base64 -w 0 cat.jpg)

cat <<EOF > payload.json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "<|im_start|>user\n<|image_pad|>\nDescribe this image in detail<|im_end|>\n<|im_start|>assistant\n"},
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

Pure text generation:

**Using Python client**

```bash
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

**Using curl**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"}]}]
    "modalities": ["text"]
  }'
```

## FAQ

- If you encounter an error about the backend of librosa, try to install ffmpeg with the command below.

```bash
sudo apt update
sudo apt install ffmpeg
```

- If you don’t know how much VRAM is needed for the model or encounter the OOM error, you can try to decrease the max_model_len.

| Stage               | VRAM                         |
| :------------------ | :--------------------------- |
| Stage-0 (Thinker)   | **15.04 GiB** **+ KV Cache** |
| Stage-1 (DiT)       | **26.50 GiB**                |
| Total               | **~42 GiB + KV Cache**       |

## Example materials

??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/bagel/openai_chat_client.py"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/bagel/run_server.sh"
    ``````
