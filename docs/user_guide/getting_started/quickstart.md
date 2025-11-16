# Quickstart

This guide will help you get started with vLLM-omni.

## Installation

Use Docker to keep consistent basic environment (Optional, Recommended)

```bash
docker run --gpus all --ipc=host --network=host -v $source_dir:$container_dir --rm --name $container_name -it nvcr.io/nvidia/pytorch:25.01-py3 bash
```

Set up basic uv environment

```bash
pip install uv
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Install vLLM

```bash
uv pip install vllm==0.11.0 --torch-backend=auto
```

Install vLLM-omni

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm_omni
uv pip install -e .
```

## Run examples (Qwen2.5-omni)

Get into the example folder

```bash
cd examples/offline_inference/qwen_2_5_omni
```

Then run.

```bash
bash run.sh
```

The output audio is saved in ./output_audio

## Next Steps

- Read the [architecture documentation](../../contributing/design_documents/vllm_omni_design.md)
- Check out the [API reference](../../api/overview.md)
- Explore the [examples](../examples/index.md)
