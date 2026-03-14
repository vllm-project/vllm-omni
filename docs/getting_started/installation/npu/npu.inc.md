# --8<-- [start:requirements]

For detailed hardware and software requirements, please refer to the [vllm-ascend installation documentation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

# --8<-- [end:requirements]
# --8<-- [start:installation-release]

The recommended way to use vLLM-Omni on NPU is through the vllm-ascend pre-built Docker images:

```bash
# Update the vllm-ascend image
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1-a3
export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1
docker run --rm \
    --name vllm-omni-npu \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
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

# Because vllm-ascend will release v0.16.0rc1 after vllm-omni 0.16.0,
# we have to pin vllm-ascend at the current commit.
cd /vllm-workspace/vllm-ascend
git checkout e2175d9c7e62b437391dfee996b1375674ba7c18
pip install -v -e .

# Inside the container, install vLLM-Omni from source
cd /vllm-workspace
git clone -b v0.16.0 https://github.com/vllm-project/vllm-omni.git

cd vllm-omni
pip install -v -e .
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

The default workdir is `/workspace`, with vLLM, vLLM-Ascend and vLLM-Omni code placed in `/vllm-workspace` installed in development mode.

For other installation methods (pip installation, building from source, custom Docker builds), please refer to the [vllm-ascend installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

We are keeping [issue #886](https://github.com/vllm-project/vllm-omni/issues/886) up to date with the aligned versions of vLLM, vLLM-Ascend, and vLLM-Omni, and also outlining the Q1 roadmap there.

# --8<-- [end:installation-release]

# --8<-- [start:installation-main]

You can also build vLLM-Omni from the latest main branch if you want to use the latest features or bug fixes. (But sometimes it will break for a while. You can check [issue #886](https://github.com/vllm-project/vllm-omni/issues/886) for the status of the latest commit of vLLM-Omni main branch on NPU.)

```bash
# Pin vLLM version to 0.16.0
cd /vllm-workspace/vllm
git pull origin main
git fetch origin --tags
git checkout v0.16.0
VLLM_TARGET_DEVICE=empty pip install -v -e .

# Because vllm-ascend has not yet entered continuous development and has not been officially released, we need to pin it to a specific commit. Please note that this commit may change over time.
cd ../vllm-ascend
git pull origin main
git fetch origin --tags
git checkout e2175d9c7e62b437391dfee996b1375674ba7c18
pip install -v -e .

# Install vLLM-Omni from the latest main branch
cd ../vllm-omni
git clone https://github.com/vllm-project/vllm-omni.git
pip install -v -e . --no-build-isolation
# or VLLM_OMNI_TARGET_DEVICE=npu pip install -v -e .
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

# --8<-- [end:installation-main]

# --8<-- [start:pre-built-images]

vLLM-Omni offers Docker images for Ascend NPU deployment. You can just pull the **prebuilt image** from the image repository [ascend/vllm-omni](https://quay.io/repository/ascend/vllm-omni?tab=tags) and run it with bash.

Supported images as following.

| image name | Hardware | OS |
|-|-|-|
| image-tag | Atlas A2 | Ubuntu |
| image-tag-a3 | Atlas A3 | Ubuntu |

Here's an example deployment command that has been verified on 4 x NPUs:

```bash
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-omni:v0.16.0
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-omni:v0.16.0-a3
export IMAGE=quay.io/ascend/vllm-omni:v0.16.0
docker run --rm \
    --name vllm-omni-npu \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
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
