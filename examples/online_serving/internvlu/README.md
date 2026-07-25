# InternVL-U Online Serving

## Launch the Server

```bash
vllm serve InternVL-U/InternVL-U --omni --port 8091
```

The default deployment resolves to `vllm_omni/deploy/internvlu_chat.yaml`
(both stages on one GPU). For text-then-image generation, serve the think
deployment instead — think mode is configured statically per deployment,
like BAGEL's think mode:

```bash
vllm serve InternVL-U/InternVL-U --omni --port 8091 \
    --deploy-config vllm_omni/deploy/internvlu_chat_think.yaml
```

Or use the convenience script:

```bash
cd examples/online_serving/internvlu
bash run_server.sh

# Think (text-then-image) deployment
THINK=1 bash run_server.sh
```

## Send Requests

```bash
cd examples/online_serving/internvlu
```

### Chat Completions

```bash
# Text-to-image
python openai_chat_client.py --prompt "A cute cat" --modality text2img

# Image editing
python openai_chat_client.py --prompt "Add a red scarf around the cat's neck" \
    --modality img2img --image-url cat.png

# Image understanding
python openai_chat_client.py --prompt "Describe this image in detail" \
    --modality img2text --image-url photo.jpg

# Text chat
python openai_chat_client.py --prompt "What is the capital of France?" \
    --modality text2text
```

**curl (text-to-image):**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "A cute cat"}]}],
    "modalities": ["image"],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 20,
    "seed": 42
  }'
```

### Images API

The OpenAI images endpoints work as well; with the think deployment the
generated description is returned as the top-level `cot_output` field:

```bash
# Generation
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL-U/InternVL-U",
    "prompt": "A cute cat",
    "size": "1024x1024",
    "num_inference_steps": 20,
    "seed": 42,
    "response_format": "b64_json"
  }'

# Editing
curl http://localhost:8091/v1/images/edits \
  -F model="InternVL-U/InternVL-U" \
  -F prompt="Add a red scarf around the cat's neck" \
  -F size="1024x1024" \
  -F num_inference_steps=20 \
  -F seed=42 \
  -F response_format=b64_json \
  -F image=@cat.png
```

### Model-Specific Parameters

InternVL-U declares these extra-body parameters (top-level request fields):

| Parameter | Default | Description |
| :-------- | :------ | :---------- |
| `all_cfg_scale` | 4.5 (3.5 in think mode) | Conditional guidance scale |
| `part_cfg_scale` | 2.0 (1.5 in think mode) | Partial guidance scale |
| `timestep_trunc` | 930 | High-timestep CFG delta-normalization cutoff |
| `flow_shift` | 3.0 | Scheduler flow shift |

Negative prompts are not supported, and one image per prompt (`n=1`).
