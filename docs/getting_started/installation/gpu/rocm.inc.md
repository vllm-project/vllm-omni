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

# --8<-- [end:pre-built-images]
