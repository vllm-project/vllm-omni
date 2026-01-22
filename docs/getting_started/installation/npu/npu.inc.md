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
# export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0-a3
export IMAGE=quay.io/ascend/vllm-ascend:v0.14.0
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

# Install the missing dependency of mooncake in the origin image.
apt update
apt install libjemalloc2
echo "export LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2:$LD_PRELOAD" >> ~/.bashrc
source ~/.bashrc

# Inside the container, install vLLM-Omni from source
cd /vllm-workspace
git clone -b v0.14.0 https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
pip install -v -e .
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# (Optional) Disable mooncake for stable capability
mv /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake \
   /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake.disabled
```

The default workdir is `/workspace`, with vLLM, vLLM-Ascend and vLLM-Omni code placed in `/vllm-workspace` installed in development mode.

For other installation methods (pip installation, building from source, custom Docker builds), please refer to the [vllm-ascend installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

# --8<-- [end:installation]
