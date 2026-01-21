# --8<-- [start:requirements]

- GPU: Validated on gfx942 (It should be supported on the AMD GPUs that are supported by vLLM.)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

vLLM-Omni current recommends the steps in under setup through Docker Images.

# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-docker]

#### Build docker image

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-omni-rocm .
```

If you want to specify which GPU Arch to build for to cutdown build time:

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.rocm \
  --build-arg PYTORCH_ROCM_ARCH="gfx942;gfx950" \
  -t vllm-omni-rocm .
```

#### Launch the docker image

```
docker run -it \
--network=host \
--group-add=video \
--ipc=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--device /dev/kfd \
--device /dev/dri \
-v <path/to/model>:/app/model \
vllm-omni-rocm \
bash
```

# --8<-- [end:build-docker]

# --8<-- [start:pre-built-images]

vLLM-Omni offers an official docker image for deployment. These images are built on top of vLLM docker images and available on Docker Hub as [vllm/vllm-omni-rocm](https://hub.docker.com/r/vllm/vllm-omni-rocm/tags). The version of vLLM-Omni indicates which release of vLLM it is based on.

#### Launch vLLM-Omni Server
Here's an example deployment command that has been verified on 2 x MI300's:
```bash
docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v <path/to/model>:/app/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  vllm/vllm-omni-rocm:v0.14.0 \
  vllm serve --model Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

#### Launch an interactive terminal with prebuilt docker image.
If you want to run in dev environment you can launch the docker image as follows:
```bash
docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v <path/to/model>:/app/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  vllm/vllm-omni-rocm:v0.14.0 \
  bash
```

# --8<-- [end:pre-built-images]
