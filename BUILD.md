# Build / setup log — LongCat-Next on 4x A100 SXM4 80GB

Reproducible steps to go from a fresh GPU pod to a working
`thinker(0) -> multi_decoder(1)` LongCat-Next pipeline. Captures every
workaround that was actually needed, in order, including the dead ends —
skip a step at your own risk, several of these exist because the naive
approach silently breaks something two steps later.

Target hardware: 4x NVIDIA A100-SXM4-80GB (works the same on H100-80GB).
Base image used: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
(only Python 3.11 + system CUDA 12.4 toolchain matter from this image —
everything else gets replaced below).

## 0. Repos

```bash
mkdir -p /workspace && cd /workspace
git clone -b feat/longcat-next-integration https://github.com/gangula-karthik/vllm-omni.git
git clone -b feat/longcat-next https://github.com/gangula-karthik/vllm.git
```

Put these on a **local (non-network-mounted) disk** if you have the choice.
On Runpod, `/workspace` is a network volume (MooseFS/FUSE) — cloning here is
fine (small, one-shot), but the Python venv and pip/uv caches must NOT live
here (see step 2) or you will hit intermittent I/O stalls and outright
`RuntimeError: Task error ... Background writer channel closed` /
`Disk quota exceeded` failures under load.

## 1. uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

`uv` itself installs to `/root/.local/bin` — also local disk, so it survives
independently of whatever happens to the venv.

## 2. Venv — must be on local disk

```bash
mkdir -p /root/venvs
uv venv --python 3.11 /root/venvs/dev
source /root/venvs/dev/bin/activate
export UV_CACHE_DIR=/root/.cache/uv
```

**Do not** put the venv or `UV_CACHE_DIR` under `/workspace`. Two failure
modes observed there: (a) large multi-file installs (torch, vllm) randomly
stall in uninterruptible disk-wait (`D` state) for minutes; (b) any
`--volume-in-gb` resize on the pod (see step 6) triggers a full container
recreation, which wipes `/root` but preserves `/workspace` — so a venv on
`/workspace` would survive a resize, but one on `/root` won't. Either way,
budget for rebuilding the venv from scratch at least once per session; it's
fast once uv's package cache is warm (~2 min for vllm+vllm-omni).

## 3. vllm — editable, precompiled-wheel fast path

```bash
cd /workspace/vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

This downloads a prebuilt vllm wheel (matched to the nearest upstream
`main` commit) for the compiled extensions, and only builds the Python
layer locally — full from-source compilation is not needed since none of
our changes touch CUDA kernels. Pulls in `torch==2.11.0+cu130` and friends
as a side effect.

## 4. vllm-omni — editable, no-deps + separate requirements install

```bash
cd /workspace/vllm-omni
uv pip install --editable . --no-deps
uv pip install -r requirements/cuda.txt
```

`--no-deps` on the first call matters: vllm-omni's own dependency
resolution does not pin vllm (it expects you to bring your own, per
setup.py), but resolving the full extras list in one shot can otherwise
drag in a conflicting vllm/torch version. Installing `requirements/cuda.txt`
separately avoids that.

## 5. flash-attn (hard requirement, not optional)

LongCat-Next's HF `modeling_longcat_next*.py` imports `flash_attn` directly
at module load time — without it the thinker/decoder workers fail with
`ImportError: This modeling file requires ... flash_attn` as soon as they
try to construct the model, not at inference time. There is no way to skip
this and still run the pipeline.

Building it from source needs a CUDA toolchain, and this is the fiddliest
part of the whole setup:

```bash
# nvcc frontend: 13.0.88 matches the headers torch's pinned
# nvidia-cuda-runtime==13.0.96 ships (anything much newer here fails an
# internal CCCL header/compiler-version compatibility check)
uv pip install nvidia-cuda-nvcc==13.0.88
cp /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas /root/ptxas_13.0.88.bak

# ptxas: 13.0.88's own bundled ptxas has an internal version skew bug —
# its cicc frontend emits PTX ISA .version 9.3 but its own ptxas only
# accepts up to 9.0 ("Unsupported .version 9.3; current version is '9.0'").
# Swap in the newer package's ptxas binary (newer ptxas accepts older/newer
# PTX ISA fine; only the frontend/headers need to match torch's runtime).
uv pip install nvidia-cuda-nvcc==13.3.73
cp /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas /root/ptxas_13.3.73.bak
uv pip install nvidia-cuda-nvcc==13.0.88   # reinstall to restore matching headers
cp /root/ptxas_13.3.73.bak /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/bin/ptxas

# lib64 doesn't exist in this pip package layout (everything is under lib/)
# but flash-attn's / flashinfer's build scripts hardcode -L.../lib64 and
# link -lcudart / -lcuda unversioned, so:
ln -sfn /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib \
        /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib64
ln -sf /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13 \
       /root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so
# libcuda.so is provided by the driver at /usr/lib/x86_64-linux-gnu/ already.

export CUDA_HOME=/root/venvs/dev/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80   # A100 = sm_80 only; NOT torch's
                                   # TORCH_CUDA_ARCH_LIST, which flash-attn's
                                   # setup.py ignores. Restricting away from
                                   # the default "80;90;100;120" also avoids
                                   # rebuilding kernels for archs you don't
                                   # have (much faster + sidesteps the same
                                   # ptxas PTX-version issue for sm_100/120).
export MAX_JOBS=32
uv pip install flash-attn --no-build-isolation
```

Without the `lib64` symlink, `flash-attn` itself actually links fine (its
own `-L.../lib` is correct), but `flashinfer`'s JIT-compiled sampling
kernel (built lazily at first inference call, not at install time) fails
the same way — so this bites you later, mid-generation, if skipped.

## 6. Model download

```bash
mkdir -p /workspace/models
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_MAX_WORKERS=4
uvx --with hf_xet hf download meituan-longcat/LongCat-Next \
    --local-dir /workspace/models/LongCat-Next --max-workers 4
```

~160GB. Two things that will bite you:

- **Pod volume quota**: `df -h /workspace` reports the underlying cluster's
  free space (hundreds of TB), which is *not* your pod's actual quota — that
  quota is set at pod-creation time via `--volume-in-gb`. A download that
  exceeds it fails with `IO Error: Disk quota exceeded (os error 122)` even
  though `df` shows plenty of room. Resize with
  `runpodctl pod update <id> --volume-in-gb 300` (or whatever you need) —
  **this recreates the container** (new hostname, `/root` wiped, `/workspace`
  preserved), so expect to redo steps 1-5's venv/tool installs afterward.
- **hf_xet transient failures**: the Xet downloader can hang indefinitely
  (CLOSE_WAIT sockets, one thread pinned at 100% CPU, no progress) or crash
  outright (`RuntimeError: Task error: File reconstruction error: Internal
  Writer Error: Background writer channel closed`). Both are recoverable —
  kill and rerun the same `hf download` command; it resumes from the
  `.cache/huggingface/download/*.incomplete` files already on disk rather
  than restarting. Reducing `--max-workers` from the default reduces (but
  doesn't eliminate) how often this happens.

## 7. Deploy config

The repo ships 5-GPU (`longcat_next_5gpu_a40_multi_decoder.yaml`) and 4-GPU
80GB variants for the 3-stage pipeline, but no 4-GPU 80GB variant of the
**2-stage** `multi_decoder` pipeline — which is the one you actually want,
because the 3-stage `longcat_next` pipeline has a real bug: the
orchestrator's `_forward_to_next_stage` always forwards `src_stage_id + 1`'s
own output, never consulting a stage's declared `input_sources`, so the
audio decoder (stage 2) always receives the image decoder's output (stage
1), never the thinker's (stage 0) — it can't ever see real audio codes.
`vllm_omni/deploy/longcat_next_4gpu_80gb_multi_decoder.yaml` in this repo
fills that gap: thinker TP=4 across all 4 GPUs, `LongcatNextMultiDecoder`
colocated with thinker rank 3 on GPU 3. On 80GB cards there's enough
headroom to skip the 5th "decoder-only" GPU the a40 variant needs — set
thinker's `gpu_memory_utilization` low enough (0.65 here) to leave ~25-28GB
free on GPU 3 for the decoder's own weights.

## 8. Code fixes required (already committed on this branch)

Getting the yaml to actually run end-to-end needed two real code fixes in
`vllm-omni`, not just infra work — see git log on
`feat/longcat-next-integration` for the full diffs:

- `vllm_omni/model_executor/models/registry.py`: `LongcatNextMultiDecoder`
  was never registered in `_OMNI_MODELS`, so vllm's `ModelRegistry` rejected
  it as an unsupported architecture.
- `vllm_omni/core/sched/omni_generation_scheduler.py`: this vllm fork
  changed `Scheduler._free_request()` to return a
  `(kv_transfer_params, ec_transfer_params)` tuple (for a new EC-connector
  feature); `omni_generation_scheduler.py` was still treating it as a
  single value, so the tuple itself got wire-serialized into the
  `kv_transfer_params` slot and blew up on decode
  (`msgspec.ValidationError: Expected object | null, got array`). Fixed the
  three call sites to unpack both values.
  (`omni_ar_scheduler.py` has its own fully-custom `_free_request` override
  that never returns a tuple — do not "fix" that one the same way.)

## 9. Run

```bash
cd /workspace/vllm-omni
python pbs/scripts/longcat_next_wired_e2e.py \
    /workspace/models/LongCat-Next \
    vllm_omni/deploy/longcat_next_4gpu_80gb_multi_decoder.yaml \
    /workspace/results --modality audio   # or --modality image
```

Both modalities reach `verdict: PASS` (correct output *shape*/*file
written*) with this setup, but that only validates pipeline wiring — see
`pbs/scripts/longcat_next_debug_quality.py` for evidence that generation
*quality* has separate, unresolved bugs (garbage text on a plain
non-multimodal prompt, and an audio-codes accumulation bug that drops all
but the last generated frame before it reaches the client).
