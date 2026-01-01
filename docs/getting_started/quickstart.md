# Quickstart

This guide will help you quickly get started with vLLM-Omni to perform:

- Offline batched inference
- Online serving using OpenAI-compatible server

## Prerequisites

- OS: Linux
- Python: 3.12

## Installation

For installation on GPU from source:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.12.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

For additional installation methods — please see the [installation guide](installation/README.md).

## Offline Inference

Text-to-image generation quickstart with vLLM-Omni:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0]["images"]
    images[0].save("coffee.png")
```

For more usages, please refer to [offline inference](../user_guide/examples/offline_inference/qwen2_5_omni.md)

## Online Serving with OpenAI-Completions API

Text-to-image generation quickstart with vLLM-Omni:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091
```

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "a cup of coffee on the table"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > coffee.png
```

For more details, please refer to [online serving](../user_guide/examples/online_serving/text_to_image.md).
