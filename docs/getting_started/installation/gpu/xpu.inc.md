<!-- markdownlint-disable MD001 MD025 -->

# --8<-- [start:requirements]

- GPU: Validated on Intel® Arc™ B-Series.

# --8<-- [end:requirements]

# --8<-- [start:set-up-using-python]

vLLM-Omni currently recommends using the Docker image setup steps below.

# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

For development or Gradio demos in a source environment with the matching
upstream XPU runtime already installed, run from the Omni repository root:

```bash
python tools/build_gradio_pillow13.py --output-dir /tmp/gradio-pillow13
VLLM_OMNI_TARGET_DEVICE=xpu uv pip install --no-build-isolation \
  --excludes <(printf 'torch\ntriton-xpu\n') -e '.[dev,demo]' \
  /tmp/gradio-pillow13/gradio-6.31.0+omni.pillow13-py3-none-any.whl
uv pip check
```

Keep the upstream PyTorch and Triton packages paired. See the
[source dependency instructions](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/#source-development-and-demo-dependencies)
for the temporary Gradio backport's provenance and qualification limits.
XPU runtime acceptance remains pending.

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-docker]

#### Build docker image

```bash
export VLLM_VERSION=v0.33.0rc3
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu --shm-size=4g \
  --build-arg VLLM_VERSION="$VLLM_VERSION" .
```

This layers vLLM-Omni on top of the selected `vllm/vllm-openai-xpu:<VLLM_VERSION>`
base image. The Dockerfile targets v0.33.0rc3; when the candidate image is
unavailable, build the upstream source image as shown below. Keep its
PyTorch and Triton packages paired. The Dockerfile retains the upstream runtime instead of
replacing Triton with the older `triton-xpu` package, and checks installed
dependency requirements after adding Omni. XPU runtime validation with this
checkout remains pending. To target a different vLLM release, pass
`--build-arg VLLM_VERSION=<tag>`. If that tag has not been published yet, build
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

```bash
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
