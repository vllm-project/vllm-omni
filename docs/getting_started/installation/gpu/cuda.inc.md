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

vLLM-Omni is built based on vLLM v0.11.0. Please install it with command below.
```bash
uv pip install vllm==0.11.0 --torch-backend=auto
```

#### Installation of vLLM-Omni

```bash
uv pip install vllm-omni
```

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

#### Installation of vLLM
If you do not need to modify source code of vLLM, you can directly install the stable 0.11.0 release version of the library

```bash
uv pip install vllm==0.11.0 --torch-backend=auto
```

#### Installation of vLLM-Omni
Install additional requirements for vLLM-Omni
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
git checkout v0.11.0
```
Set up environment variables to get pre-built wheels. If there are internet problems, just download the whl file manually. And set `VLLM_PRECOMPILED_WHEEL_LOCATION` as your local absolute path of whl file.
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
</details>

# --8<-- [end:build-wheel-from-source]
