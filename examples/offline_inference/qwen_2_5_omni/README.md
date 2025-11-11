# Offline Example of vLLM-omni for Qwen2.5-omni

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
### Multiple Prompts
Download dataset from [seed_tts](https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit). To get the prompt, you can:
```bash
tar -xf <Your Download Path>/seedtts_testset.tar
cp seedtts_testset/en/meta.lst examples/offline_inference/qwen_2_5_omni/meta.lst
python3 examples/offline_inference/qwen_2_5_omni/extract_prompts.py \
  --input examples/offline_inference/qwen_2_5_omni/meta.lst \
  --output examples/offline_inference/qwen_2_5_omni/top100.txt \
  --topk 100
```
Get into the example folder
```bash
cd examples/offline_inference/qwen_2_5_omni
```
Modify PYTHONPATH in run_multiple_prompts.sh as your path of vllm_omni. Then run.
```bash
bash run_multiple_prompts.sh
```
### Single Prompts
Get into the example folder
```bash
cd examples/offline_inference/qwen_2_5_omni
```
Modify PYTHONPATH in run_single_prompt.sh as your path of vllm_omni. Then run.
```bash
bash run_single_prompt.sh
```
