# Quickstart

This guide will help you quickly get started with vLLM-Omni to perform:

- Offline batched inference
- Online serving using OpenAI-compatible server

## Prerequisites

- OS: Linux
- Python: 3.12

## Installation

Please refer to [installation](installation/README.md)

## Offline Inference

Text-to-image generation quickstart with vLLM-Omni:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    prompt = "a cup of coffee on the table"
    images = omni.generate(prompt)
    images[0].save("coffee.png")
```

For more usages, please refer to [offline inference](../user_guide/examples/offline_inference/qwen2_5_omni.md)

## Online Serving with OpenAI-Completions API

Please refer to [online serving](../user_guide/examples/online_serving/qwen2_5_omni.md)
