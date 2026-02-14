## Setting Up the Test Environment
### Creating a Container
vLLM-Omni provides an official Docker image for deployment. These images are built upon vLLM Docker images and are available on [Docker Hub](https://hub.docker.com/r/vllm/vllm-omni/tags). The version of vLLM-Omni indicates which vLLM release it is based on.
For a local test environment, you can follow the steps below to create a container:

::::{tab-set}
:::{tab-item} GPU (NVIDIA)
```bash
export IMAGE=vllm/vllm-omni:v0.14.0

docker run --runtime nvidia --gpus all \
    --name xxx \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    $IMAGE
```
:::

:::{tab-item} ROCM (AMD)
```bash
export IMAGE=vllm-omni-rocm

docker run --rm \
    --name xxx \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    $IMAGE
```
:::

:::{tab-item} NPU (Ascend)
```bash
export DEVICE0=/dev/davinci0
export DEVICE1=/dev/davinci1

export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1

docker run --rm \
    --name xxx \
    --shm-size=1g \
    --device $DEVICE0 \
    --device $DEVICE1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```
:::
::::

## Installing Dependencies
### vLLM & vLLM-Omni
vLLM-Omni is built based on vLLM. Please install it using the command below.

::::{tab-set}
:::{tab-item} GPU (NVIDIA)
```bash
uv pip install vllm==0.16.0 --torch-backend=auto
uv pip install vllm-omni
```

:::
:::{tab-item} ROCM (AMD)
```bash
uv pip install vllm==0.14.0+rocm700 --extra-index-url https://wheels.vllm.ai/rocm/0.14.0/rocm700
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=rocm uv pip install -e .
# OR
uv pip install -e . --no-build-isolation
```

:::
:::{tab-item} NPU (Ascend)
```bash
cd /vllm-workspace
git clone -b v0.14.0 https://github.com/vllm-project/vllm-omni.git

export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -v -e .
# OR pip install -v -e . --no-build-isolation
```

:::
::::
### Test Case Dependencies
When running test cases, you may need to install the following dependencies:
```bash
pip install \
    pytest>=7.0.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.0.0 \
    mypy==1.11.1 \
    pre-commit==4.0.1 \
    openai-whisper>=20250625 \
    psutil>=7.2.0 \
    soundfile>=0.13.1 \
    imageio[ffmpeg]>=0.6.0 \
    opencv-python>=4.12.0.88 \
    mooncake-transfer-engine==0.3.8.post1
```

## Running Tests
Our test scripts use the pytest framework. You can run the following commands in your local test environment to execute the corresponding test cases.

::::{tab-set}
:::{tab-item} L1 level
```bash
cd /vllm-omni/tests
pytest -s -v -m "core_model and cpu"
```

:::
:::{tab-item} L2 level
```bash
cd /vllm-omni/tests
pytest -s -v -m "core_model and not cpu" --run-level=core_model
```
If you only want to run a specific test case, you can use:
```bash
pytest -s -v test_xxxx.py --run-level=core_model
```

:::
:::{tab-item} L3 level & L4 level
```bash
cd /vllm-omni/tests
pytest -s -v -m "advanced_model" --run-level=advanced_model
```
If you only want to run a specific test case, you can use:
```bash
pytest -s -v test_xxxx.py --run-level=advanced_model
```
Note: To run performance tests, use:
```bash
pytest -s -v tests/perf/scripts/run_benchmark.py
```

:::
::::

## Adding New Test Cases
Please refer to the [L5 Layering Specification document](./CI_5levels.md).
