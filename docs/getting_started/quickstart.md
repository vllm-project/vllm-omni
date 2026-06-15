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

# On CUDA
uv pip install vllm==0.23.0 --torch-backend=auto

# On ROCm
uv pip install vllm==0.23.0+rocm722 --extra-index-url https://wheels.vllm.ai/rocm/0.23.0/rocm722

git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

For additional installation methods — please see the [installation guide](installation/README.md).

!!! note
    It is important to install the same major & minor version of vLLM and vLLM Omni, otherwise things may not work as expected. If the versions are misaligned, you will see a warning when you import vLLM Omni.

    If you are seeing strange behavior with the `vllm` command not handling the `--omni` flag correctly, you most likely have a version mismatch with vLLM < `0.23.0` and vLLM Omni `0.23.0`, as vLLM Omni no longer hijacks the vLLM entrypoint. Updating vLLM should resolve this issue.

## Offline Inference

Text-to-image generation quickstart with vLLM-Omni:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output.images
    images[0].save("coffee.png")
```

You can pass a list of prompts and wait for the independent requests to finish,
as shown below.

!!! info

    For diffusion pipelines, each prompt becomes a separate logical request.
    The runtime may automatically batch compatible in-flight requests through
    the scheduler and runner.

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(
        model="Tongyi-MAI/Z-Image-Turbo",
        # stage_configs_path="./stage-config.yaml",  # See below
    )
    prompts = [
        "a cup of coffee on a table",
        "a toy dinosaur on a sandy beach",
        "a fox waking up in bed and yawning",
    ]
    omni_outputs = omni.generate(prompts)
    for i_prompt, prompt_output in enumerate(omni_outputs):
        this_request_output = prompt_output.request_output
        this_images = this_request_output.images
        for i_image, image in enumerate(this_images):
            image.save(f"p{i_prompt}-img{i_image}.jpg")
            print("saved to", f"p{i_prompt}-img{i_image}.jpg")
            # saved to p0-img0.jpg
            # saved to p1-img0.jpg
            # saved to p2-img0.jpg
```

!!! info

    For diffusion request-level batching controls such as `max_num_seqs` and
    `request_batch_max_wait_ms`, see
    [Request-Level Batching](../user_guide/diffusion/request_batching.md).

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
