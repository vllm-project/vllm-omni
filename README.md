# vLLM-omni: Multi-modal Extension for vLLM

vLLM-omni is designed to extend vLLM capabilities to support multi-modality model inference and serving, particularly focusing on non-autoregressive structures and non-textual outputs.

## 🎯 Overview

Traditional vLLM systems are limited to text-based, autoregressive generation. vLLM-omni addresses this limitation by enabling support for:

- **Multi-modal Models**: Text, image, video, audio, and sensor data processing
- **Non-autoregressive Architectures**: Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous Outputs**: Beyond traditional text generation to multimodal outputs

## 🏗️ Architecture

vLLM-omni is built on a modular architecture that extends vLLM's core functionality:


## 🚀 Key Features

### Multi-Engine Support

- **Autoregressive Engine**: Traditional text generation with enhanced KV-caching
- **Diffusion Engine**: Support for DiT models and iterative generation
- **Hybrid Engine**: Combined AR+DiT processing pipelines

### Modality Processing

- **Text**: Advanced tokenization and embedding generation
- **Image**: Vision encoder integration (CLIP, etc.)
- **Audio**: Speech processing and audio embedding

## 📋 Supported Models

### AR + Diffusion Transformer (DiT) Models
- Qwen-omni (Thinker-Talker-Codec structure)
- HunyunaImage 3.0 (Ongoing)
- Qwen-Image (Ongoing)

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

### Verification for successful installation of vLLM
Just run the command below. If no error, it demonstrates that the installation is successfull.
```bash
python -c "import vllm._C"
```

### Installation of vLLM-omni
Install additional requirements for vllm-omni
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm_omni
uv pip install -r requirements.txt
```

## Run examples (Qwen2.5-omni)

Get into the example folder
```bash
cd examples/offline_inference/qwen_2_5_omni
```
Modify PYTHONPATH in run.sh as your path of vllm_omni. Then run.
```bash
bash run.sh
```
The output audio is saved in ./output_audio

## Further details

For detailed architecture design, see [vllm_omni_design.md](docs/architecture/vllm_omni_design.md) and [high_level_arch_design.md](docs/architecture/high_level_arch_design.md).
