# --8<-- [start:requirements]

- GPU: compute capability 7.5 or higher (e.g., T4, RTX20xx, A100, L4, H100, etc.)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

vLLM-Omni depends on the matching major/minor release of vLLM. This source
checkout targets vLLM 0.32.0 at commit
`5df63a7203cdb9328d824a27b682e77c7b9928be` at the `v0.32.0` tag on `releases/v0.32.0`.
The combined Python metadata range for this checkout and vLLM is
`>=3.11,<3.14`. The tested CUDA runtime uses Python 3.12, PyTorch 2.13.0 and
CUDA 13.0 (`cu130`).
The pre-built Omni wheel instructions below install
the published 0.32.0 release with its matching vLLM dependency.

!!! note
    PyTorch installed via `conda` will statically link `NCCL` library, which can cause issues when vLLM tries to use `NCCL`. See <gh-issue:8420> for more details.

In order to be performant, vLLM has to compile many cuda kernels. The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

Therefore, it is recommended to install vLLM and vLLM-Omni with a **fresh new** environment. If either you have a different CUDA version or you want to use an existing PyTorch installation, you need to build vLLM from source. See [build-from-source-vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#build-wheel-from-source) for more details.

# --8<-- [start:pre-built-wheels]

#### Installation of vLLM

These pre-built wheel instructions install the published vLLM-Omni 0.32.0 release.

vLLM-Omni is built based on vLLM. Please install it with command below.
```bash
uv pip install vllm==0.32.0 --torch-backend=cu130
```

#### Installation of vLLM-Omni

```bash
uv pip install vllm-omni==0.32.0
```

To run Gradio demos, also install the optional extras:
```bash
uv pip install 'vllm-omni[demo]==0.32.0'
```

# --8<-- [end:pre-built-wheels]

# --8<-- [start:build-wheel-from-source]

#### Installation of vLLM
If you do not need to modify source code of vLLM, install the release-branch
0.32.0 wheel directly from the tested commit:

```bash
uv pip install --torch-backend=cu130 \
  "vllm @ https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl"
```

This command selects the Linux x86_64 CUDA 13.0 wheel and matching PyTorch
backend. It requires a CUDA 13 compatible driver. For another architecture,
CUDA variant or an existing incompatible PyTorch installation, build vLLM
from source instead. The [vLLM 0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
and its commit-index and PyPI wheels are available. The PyPI x86_64 wheel
embeds the same release commit and CUDA/PyTorch requirements. The command
above retains the exact tested artifact. The published vLLM-Omni 0.32.0
wheel is available through the pre-built wheel instructions above; use the
source checkout for development.

#### Installation of vLLM-Omni
Since vllm-omni is rapidly evolving, it's recommended to install it from source
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

To run Gradio demos, install with optional extras:
```bash
uv pip install -e '.[demo]'
```

<details><summary>(Optional) Installation of vLLM from source</summary>
If you want to check, modify or debug with source code of vLLM, install the library from source with the following instructions:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 5df63a7203cdb9328d824a27b682e77c7b9928be
```
Set up environment variables to get pre-built wheels. If there are internet problems, just download the whl file manually. And set `VLLM_PRECOMPILED_WHEEL_LOCATION` as your local absolute path of whl file.
```bash
# CUDA 13.0 release-branch wheel for the exact source commit on x86_64
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl
```
Install vllm with command below (If you have no existing PyTorch).
```bash
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=cu130
```
Install vllm with command below (If you already have PyTorch).
```bash
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
uv pip install --no-build-isolation --editable .
```
</details>

# --8<-- [end:build-wheel-from-source]

# --8<-- [start:build-wheel-from-source-in-docker]

# --8<-- [end:build-wheel-from-source-in-docker]

# --8<-- [start:pre-built-images]

vLLM-Omni offers an official docker image for deployment. These images are built on top of vLLM docker images and available on Docker Hub as [vllm/vllm-omni](https://hub.docker.com/r/vllm/vllm-omni/tags). The version of vLLM-Omni indicates which release of vLLM it is based on.

Nightly CUDA images built from the latest `main` are also available:

- [`vllm/vllm-omni:nightly`](https://hub.docker.com/r/vllm/vllm-omni/tags?name=nightly) is the rolling multi-architecture tag.
- [`vllm/vllm-omni:nightly-<commit>`](https://hub.docker.com/r/vllm/vllm-omni/tags?name=nightly-) provides commit-pinned multi-architecture tags. The newest 14 are retained.

Use a commit-pinned tag when you need a reproducible nightly environment.

The [v0.32.0 release](https://github.com/vllm-project/vllm-omni/releases/tag/v0.32.0)
provides CUDA images for `linux/amd64` and `linux/arm64`. Use the versioned
tag for a reproducible release environment rather than the rolling `latest` tag.

Example deployment on two GPUs (adjust hardware and memory capacity for your
model):
```bash
docker run --runtime nvidia --gpus 2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8091:8091 \
    --ipc=host \
    vllm/vllm-omni:v0.32.0 \
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

!!! tip
    The CUDA image does not define a default entrypoint, so include `vllm serve ... --omni` after the image name.

# --8<-- [end:pre-built-images]

# --8<-- [start:build-docker]

#### Build docker image

The production Dockerfile defaults to the published
`vllm/vllm-openai:v0.32.0` CUDA 13 image, which identifies release commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cuda -t vllm-omni-cuda .
```

The CI Dockerfile pins the final CUDA base by digest and installs the exact
0.32.0 commit-index wheel used by the tested local runtime. Full CI validation
of this final image stack remains pending:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.ci -t vllm-omni-cuda-ci .
```

#### Launch the docker image

##### Launch with OpenAI API Server

!!! note
    The model `Qwen/Qwen3-Omni-30B-A3B-Instruct` requires significant GPU memory. The example below has been verified on 2 x H100's.

```bash
docker run --runtime nvidia --gpus 2 \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8091:8091 \
  --ipc=host \
  vllm-omni-cuda \
  vllm serve --omni --model Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8091
```

By default, this mounts `$HOME/.cache/huggingface` as the model cache directory. To use a custom location, set the `HF_HOME` environment variable before running the command (e.g., `export HF_HOME=/data/models`).

##### Launch with interactive session for development

```bash
docker run --runtime nvidia --gpus all -it --rm \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8091:8091 \
  --ipc=host \
  --entrypoint bash \
  vllm-omni-cuda
```

# --8<-- [end:build-docker]
