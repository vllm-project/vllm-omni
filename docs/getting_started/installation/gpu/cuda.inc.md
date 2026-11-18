<!-- markdownlint-disable MD001 MD025 -->

# --8<-- [start:requirements]

- GPU: compute capability 7.5 or higher (e.g., T4, RTX20xx, A100, L4, H100, etc.)

# --8<-- [end:requirements]

# --8<-- [start:set-up-using-python]

vLLM-Omni depends on the matching major/minor release of vLLM. This source
checkout targets vLLM 0.33.0 at commit
`4ba787ec9959e566d3b587a796dff07673324da5` on `releases/v0.33.0` (tagged `v0.33.0`).
The combined Python metadata range for this checkout and vLLM is
`>=3.11,<3.14`. The final CUDA wheel uses PyTorch 2.14.0 and CUDA 13.0
(`cu130`); the current validation environment uses Python 3.12.
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

If you do not need to modify source code of vLLM, install the final
0.33.0 wheel directly from the frozen commit:

```bash
uv pip install --torch-backend=cu130 \
  "vllm @ https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl"
```

This command selects the Linux x86_64 CUDA 13.0 wheel and matching PyTorch
backend. It requires a CUDA 13 compatible driver. For another architecture,
CUDA variant or an existing incompatible PyTorch installation, build vLLM
from source instead. The final commit-index wheel is available; native model
and full CI validation remain pending. Install
Omni from this source checkout.

#### Installation of vLLM-Omni

Since vllm-omni is rapidly evolving, it's recommended to install it from source

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

The Torch 2.14 CUDA environment declares cuDNN 9.24 as an exact Python
package dependency. This checkout's CUDA images load a separate, pinned
cuDNN 9.25.1 runtime for correct `CUDNN_ATTN` image outputs. The normal wheel
installer verifies the pinned NVIDIA wheel hash and retains its license
outside Python site-packages.
The original Qwen-Image 2.1 IT2I test passes on L20X with this runtime;
native H100 and full CI qualification remain pending.

For a source installation, configure the same runtime before starting Python:

```bash
OMNI_CUDNN_RUNTIME_DIR="$(python -c 'import sys; print(sys.prefix)')/omni-cudnn-runtime"
uv pip install --target "$OMNI_CUDNN_RUNTIME_DIR" --no-deps --only-binary=:all: \
  --require-hashes -r requirements/cudnn-runtime.txt
export LD_PRELOAD="$OMNI_CUDNN_RUNTIME_DIR/nvidia/cudnn/lib/libcudnn.so.9${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$OMNI_CUDNN_RUNTIME_DIR/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import torch; assert torch.backends.cudnn.version() == 92501'
uv pip check
```

The output directory must be fresh. Reuse its exported library paths for
subsequent launches. See NVIDIA's [cuDNN compatibility policy](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.25.1/developer/forward-compatibility.html)
and [support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.25.1/reference/support-matrix.html)
for binary, driver and hardware requirements. Python packages and the
selected vLLM/Torch binaries remain installed with their original metadata.

To run Gradio demos from this source checkout, build the temporary Gradio
backport and pass its wheel in the same normal dependency install:

```bash
python tools/build_gradio_pillow13.py --output-dir /tmp/gradio-pillow13
uv pip install -e '.[demo]' \
  /tmp/gradio-pillow13/gradio-6.31.0+omni.pillow13-py3-none-any.whl
```

Development installs use the same built wheel:

```bash
uv pip install -e '.[dev]' \
  /tmp/gradio-pillow13/gradio-6.31.0+omni.pillow13-py3-none-any.whl
uv pip check
```

See the [source dependency instructions](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/#source-development-and-demo-dependencies)
for exact source provenance and qualification limits. This local Gradio 6.31
backport supplies the upstream Pillow 13 compatibility changes; browser,
native model and full CI acceptance remain pending.

<details><summary>(Optional) Installation of vLLM from source</summary>
If you want to check, modify or debug with source code of vLLM, install the library from source with the following instructions:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 4ba787ec9959e566d3b587a796dff07673324da5
```

Set up environment variables to get pre-built wheels. If there are internet problems, just download the whl file manually. And set `VLLM_PRECOMPILED_WHEEL_LOCATION` as your local absolute path of whl file.

```bash
# CUDA 13.0 final wheel for the exact source commit on x86_64
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
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

The production Dockerfile bootstraps from the published
`vllm/vllm-openai:v0.33.0` CUDA 13 image and verifies the frozen
vLLM 0.33.0 wheel with its matching dependencies:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cuda -t vllm-omni-cuda .
```

Both CUDA Dockerfiles pin the same bootstrap image by digest and install the
final wheel. The telemetry repair validates that exact base before removing
its unused incompatible exporter; retained exporters and LMCache remain pinned
in the normal dependency solve. Full CI validation remains pending:

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
