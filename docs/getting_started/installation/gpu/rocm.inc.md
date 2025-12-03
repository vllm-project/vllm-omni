# --8<-- [start:requirements]

- GPU: Validated on gfx942 (It should be supported on the AMD GPUs that are supported by vLLM.)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

vLLM-Omni current recommends the steps in under setup through Docker Images.

# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-wheel-from-source-in-docker]

#### Get the correct docker environment

The v0.11.0 releases is on Oct 3, 2025, so we will pull a AMD nightly docker image close to the release period. `rocm/vllm-dev:nightly_main_20251005` is picked as it has a correct `transformers` version that supports qwen3_omni.

`docker pull rocm/vllm-dev:nightly_main_20251005`

#### Launch the docker environment

In this docker environment, we have all the dependencies for vLLM installed, like `flash-attn` and `aiter`

```bash
#!/bin/bash
docker run -it \
   --privileged \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   --name vllmomni \
   rocm/vllm-dev:nightly_main_20251005 \
   bash
```

#### Update docker dependencies

```bash
sudo apt update
sudo apt install ffmpeg -y
```

#### Installation of vLLM
Install the v0.11.0 release of vLLM as vLLM-Omni is currently depend on v0.11.0.

```bash
# uninstall existing vllm which is based on the main branch, not on the releases branch
python3 -m pip uninstall -y vllm

git clone https://github.com/vllm-project/vllm.git --branch releases/v0.11.0 vllm0110
cd vllm0110

# PYTORCH_ROCM_ARCH=<your-gpu-arch> python3 setup.py develop
PYTORCH_ROCM_ARCH=gfx942 python3 setup.py develop
```

!!! note
    vLLM release wheels based on the branch with prefix `releases/`, not from the tag as vLLM may cherry pick bugfixes after cutting a branch.


#### Installation of vLLM-Omni
Install additional requirements for vLLM-Omni
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
python3 -m pip install -e .
```

# --8<-- [end:build-wheel-from-source-in-docker]
