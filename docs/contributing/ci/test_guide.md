# Test Guide
## Setting Up the Test Environment
### Creating a Container
vLLM-Omni provides an official Docker image for deployment. These images are built upon vLLM Docker images and are available on [Docker Hub](https://hub.docker.com/r/vllm/vllm-omni/tags). The version of vLLM-Omni indicates which vLLM release it is based on.
For a local test environment, you can follow the steps below to create a container:
## Installing Dependencies
### vLLM & vLLM-Omni
vLLM-Omni is built based on vLLM. Please install it using the command below.

=== "GPU (NVIDIA)"

    ```bash
    uv pip install vllm=={version number} --torch-backend=auto
    uv pip install vllm-omni
    ```

=== "ROCM (AMD)"

    ```bash
    uv pip install vllm=={version number} --extra-index-url https://wheels.vllm.ai/rocm/{version number}
    git clone https://github.com/vllm-project/vllm-omni.git
    cd vllm-omni
    VLLM_OMNI_TARGET_DEVICE=rocm uv pip install -e .
    # OR
    uv pip install -e . --no-build-isolation
    ```

=== "NPU (Ascend)"

    ```bash
    cd /vllm-workspace
    git clone -b {version number} https://github.com/vllm-project/vllm-omni.git

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    cd vllm-omni
    VLLM_OMNI_TARGET_DEVICE=npu pip install -v -e .
    # OR pip install -v -e . --no-build-isolation
    ```

### Test Case Dependencies
When running test cases, you may need to install the following dependencies:

=== "L1 level"

    ```bash
    pip install \
    pytest>=7.0.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.0.0 \
    mypy==1.11.1 \
    pre-commit==4.0.1 \
    ```

=== "L2 and above levels"

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
Our test scripts use the pytest framework. First, please use `git clone https://github.com/vllm-project/vllm-omni.git` to download the vllm-omni source code. Then, in the root directory of vllm-omni, you can run the following commands in your local test environment to execute the corresponding test cases.

=== "L1 level"

    ```bash
    cd tests
    pytest -s -v -m "core_model and cpu"
    ```

=== "L2 level"

    ```bash
    cd tests
    pytest -s -v -m "core_model and not cpu" --run-level=core_model
    ```
    If you only want to run a specific test case, you can use:
    ```bash
    pytest -s -v test_xxxx.py --run-level=core_model
    ```

=== "L3 level & L4 level"

    ```bash
    cd tests
    pytest -s -v -m "advanced_model" --run-level=advanced_model
    ```
    If you only want to run a specific test case, you can use:
    ```bash
    pytest -s -v test_xxxx.py --run-level=advanced_model
    ```
    Note: To run performance tests, use:
    ```bash
    pytest -s -v perf/scripts/run_benchmark.py
    ```

## Adding New Test Cases
Please refer to the [L5 Layering Specification document](./CI_5levels.md).
