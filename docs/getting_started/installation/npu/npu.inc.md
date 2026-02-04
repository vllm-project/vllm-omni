# --8<-- [start:requirements]

For detailed hardware and software requirements, please refer to the [vllm-ascend installation documentation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

# --8<-- [end:requirements]
# --8<-- [start:installation]

The recommended way to use vLLM-Omni on NPU is through the vllm-ascend pre-built Docker images:

```bash
# Update DEVICE according to your NPUs (/dev/davinci[0-7])
export DEVICE0=/dev/davinci0
export DEVICE1=/dev/davinci1
# Update the vllm-ascend image
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1-a3
export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1
docker run --rm \
    --name vllm-omni-npu \
    --shm-size=1g \
    --device $DEVICE0 \
    --device $DEVICE1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash

# Inside the container, install vLLM-Omni from source
cd /vllm-workspace
git clone -b v0.14.0 https://github.com/vllm-project/vllm-omni.git

# Remove this replace when the dispatch of requirements is ready
RUN sed -i -E 's/^([[:space:]]*)"fa3-fwd==0\.0\.1",/\1# "fa3-fwd==0.0.1",/' pyproject.toml \
 && sed -i -E 's/\bonnxruntime\b/onnxruntime-cann/g' pyproject.toml

cd vllm-omni
pip install -v -e .
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

The default workdir is `/workspace`, with vLLM, vLLM-Ascend and vLLM-Omni code placed in `/vllm-workspace` installed in development mode.

For other installation methods (pip installation, building from source, custom Docker builds), please refer to the [vllm-ascend installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

We are keeping [issue #997](https://github.com/vllm-project/vllm-omni/issues/997) up to date with the aligned versions of vLLM, vLLM-Ascend, and vLLM-Omni, and also outlining the Q1 roadmap there.

# --8<-- [end:installation]

# --8<-- [start:pre-built-images]

`vllm-ascend` offers Docker images for deployment. You can just pull the **prebuilt image** from the image repository [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and run it with bash.

Supported images as following.

| image name | Hardware | OS |
|-|-|-|
| image-tag | Atlas A2 | Ubuntu |
| image-tag-a3 | Atlas A3 | Ubuntu |

vLLM-Omni offers Docker images for Ascend NPU deployment. You can just pull the **prebuilt image** from the image repository [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags) and run it with bash.

Here's an example deployment command that has been verified on 6 x NPUs:

```bash
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-omni:v0.14.0
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-omni:v0.14.0-a3
export IMAGE=quay.io/ascend/vllm-omni:v0.14.0
docker run --rm \
    --name vllm-omni-npu \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash
```

!!! tip
    You can use this docker image to serve models the same way you would with in vLLM! To do so, make sure you overwrite the default entrypoint (`vllm serve --omni`) which works only for models supported in the vLLM-Omni project.

Or build IMAGE from **source code**:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
# A2
docker build -t vllm-omni-dev-image:latest -f ./docker/Dockerfile.npu .
# A3
# docker build -t vllm-omni-dev-image:latest -f ./docker/Dockerfile.npu.a3 .
```

# --8<-- [end:pre-built-images]
