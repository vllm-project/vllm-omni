<!-- markdownlint-disable MD001 MD025 -->

# --8<-- [start:requirements]

- GPU: Validated on gfx942 (It should be supported on the AMD GPUs that are supported by vLLM.)

# --8<-- [end:requirements]

# --8<-- [start:set-up-using-python]

For ROCm, vLLM-Omni currently recommends the setup steps through Docker Images.

vLLM-Omni depends on the matching major/minor release of vLLM. The 0.29
development line uses vLLM 0.29.x. Published 0.28.0 wheels and images use vLLM
0.28.x.

The Dockerfile's `BASE_IMAGE` pin applies only to Docker builds. The
`vllm-omni` package does not install vLLM as a dependency, so non-Docker source
installs must install the matching ROCm vLLM release explicitly before
installing vLLM-Omni, as shown below.

# --8<-- [start:pre-built-wheels]

#### Installation of vLLM

These pre-built wheel instructions install the published vLLM-Omni 0.28.0 release. For the 0.29 development line, use the source-install instructions below.

vLLM-Omni is built based on vLLM. Please install it with command below.

```bash
uv pip install vllm==0.28.0+rocm723 --extra-index-url https://wheels.vllm.ai/rocm/0.28.0/rocm723
```

#### Installation of vLLM-Omni

```bash
# we need to add --no-build-isolation as the torch
# is not obtained from pypi, we have to install using the
# torch installed in our environment
uv pip install vllm-omni==0.28.0

# Optional if want to run Qwen3 TTS
uv pip uninstall onnxruntime # should be removed before we can install onnxruntime-rocm
uv pip install onnxruntime-rocm
```

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

#### Installation of vLLM

If you do not need to modify source code of vLLM, you can directly install the stable 0.29.0 release version of the library

```bash
uv pip install vllm==0.29.0+rocm723 --extra-index-url https://wheels.vllm.ai/rocm/0.29.0/rocm723
```

The pre-built 0.29.0 vLLM wheel targets ROCm 7.2.3. If you need a different ROCm stack or want to reuse an existing PyTorch installation, build vLLM from source instead.

#### Installation of vLLM-Omni

Since vllm-omni is rapidly evolving, it's recommended to install it from source

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=rocm uv pip install -e .
# OR
uv pip install -e . --no-build-isolation
```

<details><summary>(Optional) Installation of vLLM from source</summary>
If you want to check, modify or debug with source code of vLLM, install the library from source with the following instructions:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.29.0
python3 -m pip install -r requirements/rocm.txt
python3 setup.py develop
```

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-docker]

#### Build docker image

The source build defaults to the published upstream base image
`vllm/vllm-openai-rocm:v0.29.0`, aligned with the vLLM release used by CI.
Older or custom bases must provide the vLLM APIs checked by the Dockerfile's
image-build canary. This upstream base is distinct from the prebuilt
`vllm/vllm-omni-rocm` images discussed below; their published-tag availability
is tracked in [#7405](https://github.com/vllm-project/vllm-omni/issues/7405).

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-omni-rocm .
```

To select the upstream ROCm base explicitly, pass `BASE_IMAGE`:

```bash
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile.rocm \
  --build-arg BASE_IMAGE=vllm/vllm-openai-rocm:v0.29.0 \
  -t vllm-omni-rocm .
```

#### Launch the docker image

##### Launch with OpenAI API Server

```bash
docker run --rm \
--group-add=video \
--ipc=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--device /dev/kfd \
--device /dev/dri \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=$HF_TOKEN" \
-p 8091:8091 \
--ipc=host \
vllm-omni-rocm \
--model Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8091
```

##### Launch with interactive session for development

```bash
docker run --rm -it \
--network=host \
--group-add=video \
--ipc=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--device /dev/kfd \
--device /dev/dri \
-v <path/to/model>:/app/model \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--entrypoint bash \
vllm-omni-rocm
```

# --8<-- [end:build-docker]

# --8<-- [start:pre-built-images]

vLLM-Omni offers an official docker image for deployment. These images are built on top of vLLM docker images and available on Docker Hub as [vllm/vllm-omni-rocm](https://hub.docker.com/r/vllm/vllm-omni-rocm/tags). The version of vLLM-Omni indicates which release of vLLM it is based on.

#### Launch vLLM-Omni Server

Here's an example deployment command that has been verified on 2 x MI300's:

```bash
docker run --rm \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v <path/to/model>:/app/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8091:8091 \
  vllm/vllm-omni-rocm:v0.28.0 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

#### Launch an interactive terminal with prebuilt docker image

If you want to run in dev environment you can launch the docker image as follows:

```bash
docker run --rm -it \
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
  --entrypoint bash \
  vllm/vllm-omni-rocm:v0.28.0
```

# --8<-- [end:pre-built-images]
