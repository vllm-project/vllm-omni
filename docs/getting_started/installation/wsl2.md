# NVIDIA GPUs on Windows with WSL2

Run vLLM-Omni inside a Linux distribution under WSL2. Native Windows Python
is not supported. This guide uses a single NVIDIA GPU and a small TTS model
to check the installation before attempting larger models.

## Prepare Windows and WSL2

Install an up-to-date NVIDIA **Windows** driver compatible with the CUDA
runtime you will use. Do not install a Linux display driver inside WSL;
WSL uses the Windows driver. See NVIDIA's
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

In an administrator PowerShell window, install Ubuntu if needed. Restart
Windows if prompted and complete the Linux user setup before checking WSL:

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
wsl --list --verbose
wsl --version
```

The distribution must show `VERSION 2`. To convert an existing WSL1
distribution, use `wsl --set-version <distribution-name> 2`. See Microsoft's
[installation instructions](https://learn.microsoft.com/en-us/windows/wsl/install).

Inside Ubuntu, check the GPU and available resources:

```bash
nvidia-smi
free -h
df -h ~
```

If `nvidia-smi` is not on `PATH`, try `/usr/lib/wsl/lib/nvidia-smi`.
Its `CUDA Version` field reports driver compatibility, not the runtime in
your Python environment. Check `torch.version.cuda` after installing PyTorch.

An 8GB GPU also serves the Windows desktop. Leave headroom for its existing
usage, and start with one model and one request at a time. Host RAM matters
as well: each stage runs a separate worker. If WSL exhausts RAM or swaps
heavily, reduce other workloads or adjust WSL's memory limit before retrying.
Do not treat a model's parameter count as its complete memory requirement.

## Install in a Linux Python environment

Keep the checkout, virtual environment, and model cache in the Linux
filesystem, such as `~/src`, rather than under `/mnt/c`. Run the commands
below in Ubuntu, not PowerShell.

```bash
sudo apt-get update
sudo apt-get install -y git build-essential libsndfile1 ffmpeg
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed.
Use a fresh environment and a matching vLLM/Omni release pair. The source
revision used below targets vLLM 0.33.0, PyTorch 2.14, and CUDA 13.0; do not
combine it with the published Omni 0.32 wheel.

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git checkout a96e3bcb90c81aeb2fd187dc59644b367cb4a32e
uv venv --python 3.12 .venv
source .venv/bin/activate
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_INSTALLS=1
uv pip install vllm==0.33.0 --torch-backend=cu130
VLLM_OMNI_TARGET_DEVICE=cuda uv pip install -e .
```

Configure the pinned cuDNN runtime described in the
[CUDA source-installation instructions](gpu.md#build-wheel-from-source):

```bash
OMNI_CUDNN_RUNTIME_DIR="$VIRTUAL_ENV/omni-cudnn-runtime"
uv pip install --target "$OMNI_CUDNN_RUNTIME_DIR" --no-deps --only-binary=:all: \
  --require-hashes -r requirements/cudnn-runtime.txt
export LD_PRELOAD="$OMNI_CUDNN_RUNTIME_DIR/nvidia/cudnn/lib/libcudnn.so.9${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$OMNI_CUDNN_RUNTIME_DIR/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Use a fresh runtime directory, and reapply the library exports in each shell
used to launch the server. This installs runtime libraries, not a Linux
display driver. For another Omni revision, follow that revision's matching
[GPU installation requirements](gpu.md).

## Run a small speech model

Use `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` for this check. It has built-in
voices and does not need a reference recording. Its model and speech-tokenizer
weights download approximately 2.5GB before the first launch.

Create an overlay in the repository root. Both stages inherit GPU 0 and the
model's required codec limits. Limit concurrency, disable graph capture, and
give the talker enough memory for its weights and KV cache:

```bash
cat > qwen3-tts-wsl2.yaml <<'YAML'
base_config: vllm_omni/deploy/qwen3_tts.yaml
stages:
  - stage_id: 0
    gpu_memory_utilization: 0.4
    max_num_seqs: 1
    enforce_eager: true
  - stage_id: 1
    max_num_seqs: 1
    enforce_eager: true
YAML
```

For this wheel-only setup without `nvcc`, select the built-in sampler instead
of FlashInfer's sampler, which may need CUDA compilation during startup:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_DISABLE_XET=1
MODEL_DIR=$(hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --revision 85e237c12c027371202489a0ec509ded67b5e4b5 --max-workers 1 --quiet)
vllm serve "$MODEL_DIR" \
  --omni --host 127.0.0.1 --port 8091 \
  --deploy-config qwen3-tts-wsl2.yaml
```

Keep the terminal open. In a second Ubuntu terminal, wait for `/health` to
return HTTP 200, then request a WAV file:

```bash
curl --fail http://127.0.0.1:8091/health
curl --fail-with-body http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, this is a local speech test.","voice":"vivian","language":"Auto","response_format":"wav","seed":42}' \
  --output speech.wav
```

Activate the same environment in the second terminal to check the response:

```bash
source .venv/bin/activate
python -c 'import soundfile as sf; a, sr = sf.read("speech.wav"); assert len(a) > 0; print(sr, len(a) / sr)'
```

Listen to the file as well; a valid WAV alone does not establish speech
quality. Stop the server with Ctrl+C before starting another model.

## Verified configuration and startup issues

This single-GPU configuration completed three sequential requests with finite,
non-silent, mono 24kHz WAV output. This checks installation and response
integrity, not perceptual quality or throughput.

| Component | Tested value |
| --- | --- |
| Host | Windows 11 Home build 26200; 32GB RAM; approximately 20GiB assigned to WSL |
| WSL / Linux | WSL 2.7.12.0; kernel 6.18.33.2; Ubuntu 22.04.4 |
| GPU / driver | RTX 4070 Laptop, 8188MiB; NVIDIA 616.92 |
| Environment | Python 3.12.4; vLLM 0.33.0; Torch 2.14.0+cu130; Transformers 5.19.0; cuDNN 9.25.1 |
| Omni revision | `a96e3bcb90c81aeb2fd187dc59644b367cb4a32e` |
| Model revision | `85e237c12c027371202489a0ec509ded67b5e4b5` |

Two startup failures in this environment informed the commands above:

| Failure | Resolution in this guide |
| --- | --- |
| FlashInfer sampler warmup could not find `nvcc` | Set `VLLM_USE_FLASHINFER_SAMPLER=0`. Other models or backends may still need a CUDA toolkit. |
| The inherited 0.3 talker budget left 0.20GiB for KV cache, below the 0.44GiB required for 4096 tokens | Increase the talker budget to 0.4; keep the codec's inherited 0.3 budget. This is a profile for this GPU, not a change to the shared default. |

If a worker exits with `Killed` or stalls while swapping, check WSL's host
RAM with `free -h`; GPU memory is a separate limit. Stop the server and verify
its workers exited before trying another model. For direct browser calls,
configure CORS using the [API server guide](../../serving/README.md).

For configurations near the VRAM limit, record latency and Windows Task
Manager's dedicated/shared GPU memory alongside `nvidia-smi`.
[Community testing](https://github.com/vllm-project/vllm-omni/issues/10008#issuecomment-6823239753)
reports severe slowdowns without a CUDA OOM; successful requests alone do not
show that a workload stayed within physical VRAM. This smoke check did not
evaluate shared-memory fallback under GPU memory pressure.

Multi-GPU tensor parallelism needs separate validation.
[#8135](https://github.com/vllm-project/vllm-omni/issues/8135) reports corrupted
Qwen-Image-2.1 TP=2 output on a mixed consumer Blackwell pair under WSL2;
the report does not establish WSL2 as the cause. This single-GPU TTS check
does not qualify that model or topology.
