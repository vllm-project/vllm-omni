# Offline Example of vLLM-omni for Qwen2.5-omni

## 🛠️ Installation

Use Docker to keep consistent basic environment (Optional, Recommended)
```bash
docker run --gpus all --ipc=host --network=host -v $source_dir:$container_dir --rm --name $container_name -it nvcr.io/nvidia/pytorch:25.01-py3 bash
```

Set up basic uv environment
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```
Install certain version of vllm with commitid: 808a7b69df479b6b3a16181711cac7ca28a9b941

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 808a7b69df479b6b3a16181711cac7ca28a9b941
```
Set up environment variables to get pre-built wheels. If there are internet problems, just download the whl file manually. And set VLLM_PRECOMPILED_WHEEL_LOCATION as your local absolute path of whl file.
```bash
export VLLM_COMMIT=808a7b69df479b6b3a16181711cac7ca28a9b941
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```
Install vllm with commend below.
```bash
uv pip install --editable .
```

## Run examples

Get into the example folder
```bash
cd vllm_omni
cd examples/offline_inference/qwen2_5_omni
```
Modify PYTHONPATH in run.sh as your path of vllm_omni. Then run.
```bash
bash run.sh
```
The output audio is saved in ./output_audio

## To-do list
- [x] Offline inference example for Qwen2.5-omni with single request
- [ ] Adaptation from current vllm branch to stable vllm v0.11.0
- [ ] Offline inference example for Qwen2.5-omni with streaming multiple requests
- [ ] Online inference support
- [ ] Support for other models
