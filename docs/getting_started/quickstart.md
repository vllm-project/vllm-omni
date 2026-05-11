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
uv pip install vllm==0.20.0 --torch-backend=auto

# On ROCm
uv pip install vllm==0.20.0+rocm721 --extra-index-url https://wheels.vllm.ai/rocm/0.20.0/rocm721

git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

For additional installation methods — please see the [installation guide](installation/README.md).


!!! note
    It is important to install the same major & minor version of vLLM and vLLM Omni, otherwise things may not work as expected. If the versions are misaligned, you will see a warning when you import vLLM Omni.

    If you are seeing strange behavior with the `vllm` command not handling the `--omni` flag correctly, you most likely have a version mismatch with vLLM < `0.20.0` and vLLM Omni `0.20.0`, as vLLM Omni no longer hijacks the vLLM entrypoint. Updating vLLM should resolve this issue.

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

You can pass a list of prompts and wait for them to process altogether, shown below.

!!! info

    However, it is not currently recommended to do so
    because not all models support batch inference,
    and batch requesting mostly does not provide significant performance improvement (despite the impression that it does).
    This feature is primarily for the sake of interface compatibility with vLLM and to allow for future improvements.

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

    For diffusion pipelines, the stage config field `stage_args.[].engine_args.max_num_seqs` is 1 by default, and the input
    list is sliced into single-item requests before feeding into the diffusion pipeline. For models that do internally support
    batched inputs, you can [modify this configuration](../configuration/stage_configs.md) to let the model accept a longer batch of prompts.

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

## Troubleshooting

If you hit issues during your first run, please check the items below before opening
an installation/bug issue. The first three problems make up the majority of new-user
reports.

### `vllm` does not recognise `--omni`

The `--omni` flag is provided by vLLM-Omni and only works when its version is aligned
with the installed vLLM. If you installed vLLM < 0.20.0 and vLLM-Omni 0.20.0, the
entrypoint will no longer be hijacked and `--omni` will look unknown. Run
`pip show vllm vllm-omni` and confirm that the major and minor versions match, then
upgrade vLLM as needed.

### `ModuleNotFoundError: No module named 'vllm_omni'`

Most often this means you ran `python …` from a different environment than the one
where vLLM-Omni was installed. Activate the same virtualenv used for `uv pip install -e .`
and re-run; on Linux/macOS that is `source .venv/bin/activate`.

### Out-of-memory (OOM) at startup

Diffusion models in particular hold large weight, scheduler, and KV-cache buffers in
GPU memory. If the engine OOMs before the first request:

- Lower `gpu_memory_utilization` (defaults to 0.9; try 0.8 or 0.7). See the
  [GPU memory configuration guide](../configuration/gpu_memory_utilization.md).
- Reduce `max_num_seqs` for the AR stage.
- Pick a smaller or quantized variant of the model.

### Generated image / audio is silent or blank

Make sure you saved the raw bytes returned by the API rather than the JSON envelope.
The serving snippet above pipes through `jq` and `base64 -d` for that reason — without
the decoding step the file will be a base64-encoded text blob, not a valid PNG or WAV.

### Running on Windows

vLLM-Omni is not natively supported on Windows. Please use WSL2 with Ubuntu 22.04+
(`wsl --install -d Ubuntu-22.04`) and follow the Linux instructions inside the WSL
shell.

If your problem isn't covered here, open an
[installation issue](https://github.com/vllm-project/vllm-omni/issues/new?template=200-installation.yml)
with the output of `python collect_env.py` attached.
