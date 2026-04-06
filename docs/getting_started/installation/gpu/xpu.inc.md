# --8<-- [start:requirements]

- GPU: Validated on Intel® Arc™ B-Series.

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

vLLM-Omni currently recommends using the Docker image setup steps below.

# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-docker]

#### Build docker image

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu --shm-size=4g .
```

#### Launch the docker image

##### Launch with OpenAI API Server

```
docker run -it -d --shm-size 10g \
  --name {container_name} \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/dri/by-path:/dev/dri/by-path \
  --device /dev/dri:/dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  vllm-omni-xpu \
  --model Qwen/Qwen2.5-Omni-3B --port 8091
```

# --8<-- [end:build-docker]

# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
