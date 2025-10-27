# Offline Example of vLLM-omni for Qwen2.5-omni

## Installation

Set up basic environments
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```
Install certain version of vllm with commitid: 808a7b69df479b6b3a16181711cac7ca28a9b941

```bash
export VLLM_COMMIT=808a7b69df479b6b3a16181711cac7ca28a9b941
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
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
