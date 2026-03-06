# Test Guide
## Setting Up the Test Environment
### Creating a Container
vLLM-Omni provides an official Docker image for deployment. These images are built upon vLLM Docker images and are available on [Docker Hub](https://hub.docker.com/r/vllm/vllm-omni/tags). The version of vLLM-Omni indicates which vLLM release it is based on.
For a local test environment, you can follow the steps below to create a container:
## Installing Dependencies
### vLLM & vLLM-Omni
vLLM-Omni is built based on vLLM. You can follow [install guide](../../getting_started/installation/README.md) to build your local environment.

### Test Case Dependencies
When running test cases, you may need to install the following dependencies:

```bash
uv pip install ".[dev]"
apt-get install -y ffmpeg
```

## Running Tests
Our test scripts use the pytest framework. First, please use `git clone https://github.com/vllm-project/vllm-omni.git` to download the vllm-omni source code. Then, in the root directory of vllm-omni, you can run the following commands in your local test environment to execute the corresponding test cases.

=== "L1 level"

    ```bash
    cd tests
    pytest -s -v -m "core_model and cpu"
    ```
    The latest test command is available in the "Simple Unit Test" step of this [pipeline](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-ready.yml).

=== "L2 level"

    ```bash
    cd tests
    pytest -s -v -m "core_model and not cpu" --run-level=core_model
    ```
    If you only want to run a specific test case, you can use:
    ```bash
    pytest -s -v test_xxxx.py --run-level=core_model
    ```
    If you only want to run specific test cases on a particular platform, you can use:
    ```bash
    pytest -s -v -m "core_model and distributed_cuda and L4"  --run-level=core_model
    ```
    The latest test commands for various test suites can be found in the [pipeline](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-ready.yml).

=== "L3 level & L4 level"

    ```bash
    cd tests
    pytest -s -v -m "advanced_model" --run-level=advanced_model
    ```
    If you only want to run L3 test case, you can use:
    ```bash
    pytest -s -v e2e/ --ignore-glob='*expansion.py' -m "advanced_model" --run-level=advanced_model
    ```
    If you only want to run a specific test case, you can use:
    ```bash
    pytest -s -v test_xxxx.py --run-level=advanced_model
    ```
    If you only want to run specific test cases on a particular platform, you can use:
    ```bash
    pytest -s -v -m "core_model and distributed_cuda and L4"  --run-level=core_model
    ```
    Note: To run performance tests, use:
    ```bash
    pytest -s -v perf/scripts/run_benchmark.py
    ```

    The latest L3 test commands for various test suites can be found in the [pipeline](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-merge.yml).

    The latest L4 test commands for various test suites can be found in the [pipeline](https://github.com/vllm-project/vllm-omni/blob/main/.buildkite/test-nightly.yml).

You can find more information about markers in the documentation: [marker doc](./tests_markers.md)

## Adding New Test Cases
Please refer to the [L5 Layering Specification document](./CI_5levels.md).
