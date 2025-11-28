# --8<-- [start:installation]

vLLM-Omni supports Ascend NPU through the vLLM Ascend Plugin (vllm-ascend). This is a community maintained hardware plugin for running vLLM on Ascend NPU.

# --8<-- [end:installation]
# --8<-- [start:requirements]

For detailed hardware and software requirements, please refer to the [vllm-ascend installation documentation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

**Key requirements summary:**
- **NPU**: Atlas A2/A3 training and inference series
- **CANN**: >= 8.3.RC1
- **PyTorch**: 2.7.1
- **torch-npu**: 2.7.1 (auto-installed with vllm-ascend)

# --8<-- [end:requirements]
# --8<-- [start:pre-built-images]

The recommended way to use vLLM-Omni on Ascend NPU is through the vllm-ascend pre-built Docker images:

```bash
# Update DEVICE according to your NPUs (/dev/davinci[0-7])
export DEVICE0=/dev/davinci0
export DEVICE1=/dev/davinci1
# Update the vllm-ascend image
# Atlas A2:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc2
# Atlas A3:
# export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc2-a3
export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0rc2
docker run --rm \
    --name vllm-omni-npu \
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
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
pip install -v -e .
```

The default workdir is `/workspace`, with vLLM and vLLM-Ascend code placed in `/vllm-workspace` installed in development mode.

For other installation methods (pip installation, building from source, custom Docker builds), please refer to the [vllm-ascend installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

# --8<-- [end:pre-built-images]
