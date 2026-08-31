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

This layers vLLM-Omni on top of the published `vllm/vllm-openai-xpu:<VLLM_VERSION>`
base image, which Docker pulls automatically. To target a different vLLM release,
pass `--build-arg VLLM_VERSION=<tag>`. If that tag has not been published yet, build
the base from upstream's own Dockerfile first and point the build at it:

```bash
git clone --depth 1 --branch "$VLLM_VERSION" https://github.com/vllm-project/vllm /tmp/vllm
DOCKER_BUILDKIT=1 docker build -t vllm-openai-xpu:local --target vllm-openai \
  -f /tmp/vllm/docker/Dockerfile.xpu /tmp/vllm
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu --shm-size=4g \
  --build-arg VLLM_BASE=vllm-openai-xpu:local .
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
