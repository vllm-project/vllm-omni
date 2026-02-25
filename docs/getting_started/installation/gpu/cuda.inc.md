# --8<-- [start:requirements]

- GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

vLLM-Omni depends vLLM. So please follow instructions below mainly for vLLM.

!!! note
    PyTorch installed via `conda` will statically link `NCCL` library, which can cause issues when vLLM tries to use `NCCL`. See <gh-issue:8420> for more details.

In order to be performant, vLLM has to compile many cuda kernels. The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

Therefore, it is recommended to install vLLM and vLLM-Omni with a **fresh new** environment. If either you have a different CUDA version or you want to use an existing PyTorch installation, you need to build vLLM from source. See [build-from-source-vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#build-wheel-from-source) for more details.

# --8<-- [start:pre-built-wheels]

#### Installation of vLLM

vLLM-Omni is built based on vLLM. Please install it with command below.
```bash
# vllm 0.16.0 is still under prerelease
uv pip install --prerelease=allow vllm --extra-index-url https://wheels.vllm.ai/2d5be1dd5ce2e44dfea53ea03ff61143da5137eb

# vllm 0.16.0 may have some bugs for cuda 12.9, here is how we solve them:
export FLASHINFER_CUDA_TAG="$(python3 -c 'import torch; print((torch.version.cuda or "12.4").replace(".", ""))')"

uv pip install --upgrade --force-reinstall \
  "flashinfer-python==0.6.3" \
  "flashinfer-cubin==0.6.3" \
  "flashinfer-jit-cache==0.6.3" \
  --extra-index-url "https://flashinfer.ai/whl/cu${FLASHINFER_CUDA_TAG}"

uv pip install --upgrade --force-reinstall "nvidia-cublas-cu12==12.9.1.4"
uv pip install --upgrade --force-reinstall "numpy==2.2.6"
```

#### Installation of vLLM-Omni

```bash
uv pip install vllm-omni
```

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

#### Installation of vLLM
If you do not need to modify source code of vLLM, you can directly install the stable 0.16.0 release version of the library

```bash
uv pip install vllm==0.16.0 --torch-backend=auto
```

The release 0.14.0 of vLLM is based on PyTorch 2.9.0 which requires CUDA 12.9 environment.

#### Installation of vLLM-Omni
Since vllm-omni is rapidly evolving, it's recommended to install it from source
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

<details><summary>(Optional) Installation of vLLM from source</summary>
If you want to check, modify or debug with source code of vLLM, install the library from source with the following instructions:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.16.0
```
Set up environment variables to get pre-built wheels. If there are internet problems, just download the whl file manually. And set `VLLM_PRECOMPILED_WHEEL_LOCATION` as your local absolute path of whl file.
```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://github.com/vllm-project/vllm/releases/download/v0.16.0/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl
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
</details>

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-wheel-from-source-in-docker]

# --8<-- [end:build-wheel-from-source-in-docker]

# --8<-- [start:pre-built-images]

vLLM-Omni offers an official docker image for deployment. These images are built on top of vLLM docker images and available on Docker Hub as [vllm/vllm-omni](https://hub.docker.com/r/vllm/vllm-omni/tags). The version of vLLM-Omni indicates which release of vLLM it is based on.

Here's an example deployment command that has been verified on 2 x H100's:
```bash
docker run --runtime nvidia --gpus 2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8091:8091 \
    --ipc=host \
    vllm/vllm-omni:v0.14.0 \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8091
```

!!! tip
    You can use this docker image to serve models the same way you would with in vLLM! To do so, make sure you overwrite the default entrypoint (`vllm serve --omni`) which works only for models supported in the vLLM-Omni project.

# --8<-- [end:pre-built-images]
