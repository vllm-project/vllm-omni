<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="./docs/source/logos/vllm-omni-logo-text-dark.png">
    <img alt="vllm-omni" src="./docs/source/logos/vllm-omni-logo-text-dark.png" width=55%>
  </picture>
</p>
<h3 align="center">
Easy, fast, and cheap omni-modality model serving for everyone
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://tinyurl.com/vllm-omni-meeting"><b>Weekly Meeting</b></a> |
</p>

---

*Latest News* 🔥

- [2025/11] vLLM community officially released [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) in order to support omni-modality models serving.

---

## About

[vLLM](https://github.com/vllm-project/vllm) was originally designed to support large language models for text-based autoregressive generation tasks. vLLM-Omni extends its support for omni-modality model inference and serving:

- **Omni-modality**: Text, image, video, and audio data processing
- **Non-autoregressive Architectures**: extend the AR support of vLLM to Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous outputs**: from traditional text generation to multimodal outputs

<p align="center">
  <picture>
    <img alt="vllm-omni" src="./docs/source/architecture/omni-modality model architecture.png" width=55%>
  </picture>
</p>

vLLM-omni is fast with:

- Seamless AR support for efficient KV Cache management with vLLM
- Pipelined stage execution overlapping
- Fully disaggregation based on omniConnector and dynamic resource allocation across stages

vLLM-omni is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Multi-modality generation models (e.g. Qwen-image)
- Omni-modality models (e.g. Qwen-omni)

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

## 🛠️ Installation

### Environment setup

Use Docker to keep consistent basic environment (Optional, Recommended)
```bash
docker run --gpus all --ipc=host --network=host -v $source_dir:$container_dir --name $container_name -it nvcr.io/nvidia/pytorch:25.01-py3 bash
```

Set up basic uv environment
```bash
pip install uv
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Installation of vLLM (for users)

Now we build it based on vLLM v0.11.0. Please install it with command below.
```bash
uv pip install vllm==0.11.0 --torch-backend=auto
```

### Installation of vLLM (for developers)

If you want to check or debug with source code of vLLM, install stable release version of vllm with 0.11.0 from source with pre-built wheel file.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.0
```
Set up environment variables to get pre-built wheels. If there are internet problems, just download the whl file manually. And set VLLM_PRECOMPILED_WHEEL_LOCATION as your local absolute path of whl file.
```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.11.0/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl
```
Install vllm with command below (If you have no existing PyTorch).
```bash
uv pip install --editable .
```

Install vllm with command below (If you already have PyTorch).
```bash
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation --editable .
```

### Installation of vLLM-omni
Install additional requirements for vllm-omni
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm_omni
uv pip install -e .
```


## Run examples (Qwen2.5-omni)

Please check the folder of [examples](examples)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM-omni](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Weekly Meeting

- vLLM Omni Weekly Meeting: https://tinyurl.com/vllm-omni-meeting
- Wednesday, 11:30 - 12:30 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
