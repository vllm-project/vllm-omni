# GPU

vLLM-Omni is a Python library that supports the following GPU variants. The library itself mainly contains python implementations for framework and models.

## Requirements

- OS: Linux
- Python: 3.12

!!! note
    vLLM-Omni is currently not natively supported on Windows.

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:requirements"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:requirements"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:requirements"

## Set up using Python

### Create a new Python environment

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

### Pre-built wheels

Note: Pre-built wheels are currently only available for vLLM-Omni 0.11.0rc1, 0.12.0rc1, 0.14.0rc1, 0.14.0. For the latest version, please [build from source](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/#build-wheel-from-source).

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:pre-built-wheels"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:pre-built-wheels"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:pre-built-wheels"

[](){ #build-from-source }

### Build wheel from source

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:build-wheel-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:build-wheel-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:build-wheel-from-source"

## Set up using Docker

### Pre-built images

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu/cuda.inc.md:pre-built-images"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:pre-built-images"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:pre-built-images"

### Build your own docker image

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu/rocm.inc.md:build-docker"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu/xpu.inc.md:build-docker"
