# MiniMax-H3 on NVIDIA Rubin

This recipe runs MiniMax-H3 in BF16 on four NVIDIA Rubin GPUs in one host
through the synchronous `Omni.generate` API. It uses Ulysses sequence parallelism,
text-encoder tensor parallelism, and tiled VAE decode, with one checkpoint
partition loaded at a time.

The recommended profile uses `TRTLLM_ATTN` with eager execution. It measured
19.81 s median for 1344×768, 50-step T2VA on four GPUs; see the validation
section for the workload, memory, and output checks.

No Rubin-specific Omni runtime patch is needed for the tested offline path.
The HTTP server is incompatible with the tested vLLM preview versions;
see [HTTP compatibility](#http-compatibility) before attempting serving.

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: Offline video-and-audio generation
- Hardware: 4x NVIDIA VR NVL72 (Rubin, compute capability 10.7 / SM107)
- Maintainer: Community

See [MiniMax-H3.md](MiniMax-H3.md) for model behavior, reference-input limits,
and the HTTP API contract. T2VA and FL2VA use the `FL2VA` partition; Ref2VA
uses the `Ref2VA` partition.

## Prerequisites

### Hardware and capacity

Use four GPUs visible to a single host, with working NVIDIA container
support and peer communication. Inspect their memory and connectivity:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
```

| Resource | Starting configuration |
| --- | --- |
| GPUs | Four VR NVL72 GPUs, 286,524 MiB each, SM107 |
| Checkpoint storage | Approximately 135 GiB per complete partition; 270 GiB for both |
| GPU memory | Resident BF16 weights plus workload-dependent activations; measure each rank's peak |
| Host memory | 512 GiB allocated for validation; this is not a measured minimum |
| Concurrent requests | One during initial validation |

The test host used aarch64 CPUs, driver `620.69`, a 2,300 W power limit per
GPU, and NVLink between every GPU pair (`NV36` in `nvidia-smi topo -m`).
The process allocation had 32 CPU threads, with `OMP_NUM_THREADS=8` and no
explicit NUMA binding. x86 hosts and other Rubin configurations were not tested.

Ulysses splits the attention sequence, not DiT weights. With DiT TP1 each
rank retains the active DiT. Text-encoder TP4 shards the Qwen3-VL encoder
across all four ranks. Larger output shapes and Ref2VA video inputs need
additional activation memory. This recipe does not establish a minimum
Rubin HBM capacity or a multi-host launch configuration.

### Rubin software environment

Use vLLM's Rubin environment as the base. The published preview tag is
`vllm/vllm-openai:cu134-nightly`; see the
[vLLM Rubin announcement](https://vllm.ai/blog/2026-10-09-vera-rubin-preview).
For a custom build, follow
[vLLM's Rubin installation instructions](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.cuda.inc.md#preview-building-vllms-docker-image-from-source-for-nvidia-rubin-gpu-architecture).
The documented preview uses CUDA 13.4, NCCL 2.32.3 or newer, and a
Rubin-compatible source build of Triton with matching `ptxas`.

Use an Omni revision compatible with the vLLM engine inside that image.
The two projects' latest `main` revisions are not automatically a compatible
pair. Record the Omni commit and image digest when qualifying a deployment.
The stock Omni CUDA Dockerfile may install its own vLLM wheel and NCCL pins;
the commands below install Omni directly into the Rubin environment to
preserve its existing stack.

Validation used Slurm/Pyxis with the image below. For Docker, prepare the
same source revision and mount the checkout and model cache:

```bash
git clone --branch main https://github.com/vllm-project/vllm-omni.git vllm-omni-rubin
cd vllm-omni-rubin
git checkout 7467d4b59f4290334aae8f84777e989e1cb1bbeb
git rev-parse HEAD

export RUBIN_VLLM_IMAGE=vllm/vllm-openai@sha256:bec82f0f8415716ccee67dc06d7d0064a9834682a5dc1c972bb422a79f7d535d
docker pull "${RUBIN_VLLM_IMAGE}"
docker image inspect --format '{{json .RepoDigests}}' "${RUBIN_VLLM_IMAGE}"
docker run --rm -it \
  --gpus '"device=0,1,2,3"' \
  --ipc=host \
  --env HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -v "$PWD:/workspace/vllm-omni" \
  -w /workspace/vllm-omni \
  --entrypoint bash \
  "${RUBIN_VLLM_IMAGE}"
```

Inside the container, install the media tools and Omni. Freeze the existing
GPU runtime packages before resolving Omni's dependencies. Allow Python
dependencies such as Accelerate and Transformers to follow Omni's requirements:

```bash
apt-get update
apt-get install -y --no-install-recommends ffmpeg libsndfile1

export OMNI_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
uv pip freeze --python "${OMNI_PYTHON}" > /tmp/rubin-base-environment.txt
"${OMNI_PYTHON}" - <<'PY' > /tmp/rubin-runtime-constraints.txt
from importlib.metadata import distributions, version

runtime = {
    "vllm", "torch", "torchvision", "torchaudio", "triton", "pytorch-triton",
    "nixl", "apache-tvm-ffi", "tilelang", "quack-kernels", "humming-kernels",
    "tokenspeed-mla",
}
for package in sorted(distributions(), key=lambda p: p.metadata["Name"].lower()):
    name = package.metadata["Name"].lower().replace("_", "-")
    if name in runtime or name.startswith(("nvidia-", "cuda-", "flashinfer-")):
        print(f"{name}=={package.version}")
with open("/tmp/rubin-runtime-overrides.txt", "w") as overrides:
    for name in ("triton", "nvidia-nccl-cu13"):
        overrides.write(f"{name}=={version(name)}\n")
PY
VLLM_OMNI_TARGET_DEVICE=cuda uv pip install \
  --python "${OMNI_PYTHON}" \
  --prerelease=allow \
  --constraint /tmp/rubin-runtime-constraints.txt \
  --override /tmp/rubin-runtime-overrides.txt \
  -e . 'diffusers==0.40.0' 'transformers==5.19.0' \
  'accelerate==1.12.0' 'av==19.0.2'

git rev-parse HEAD
uv pip freeze --python "${OMNI_PYTHON}" > /tmp/rubin-omni-environment.txt
ffmpeg -version
ffprobe -version
```

The narrow overrides retain the base image's source-built Triton and NCCL,
which can intentionally differ from the versions declared by its PyTorch
wheel. Constraints alone reject that existing combination during resolution.

If dependency resolution or Omni's vLLM imports fail, select a compatible
image/source pair before continuing. Avoid replacing the base's PyTorch,
Triton, FlashInfer, or NCCL independently. The baseline below does not
require Omni's optional `[fa4]` extra.

The image above is `cu134-nightly-1648fefab91204f3b15bf8062217316f07f2c4bb`.
The newer `cf5cea2da` image fails to import this Omni revision because vLLM
removed `VllmConfig.use_v2_model_runner`. Updating both projects to their
latest revisions does not, by itself, establish compatibility.

The tested environment was:

| Component | Version |
| --- | --- |
| vLLM-Omni source | `7467d4b59f4290334aae8f84777e989e1cb1bbeb` (main, unmodified runtime) |
| vLLM | `0.33.0rc1.dev992+g1648fefab.d20261107.cu134` |
| PyTorch / CUDA | `2.16.0.dev20261009+cu134` / 13.4 |
| Triton | `3.8.0+git3f6e4113` |
| FlashInfer | 0.7.3, including the SM107 JIT cache |
| Loaded NCCL | 2.32.3 |
| Transformers / Diffusers / Accelerate | 5.19.0 / 0.40.0 / 1.12.0 |
| Python / PyAV / Hugging Face Hub | 3.12.3 / 19.0.2 / 1.33.0 |
| Model revision | `0919fb684102ef115126612f3fbb84d1027e7317` |

The arm64 image manifest was
`sha256:4f4c2f746b03a1a4d0a16086c6d6fd3ddecc5f59749787bb1715bc538b199dec`.

### Check the runtime

Run this inside the container before loading weights. It checks all four
devices and exercises Triton compilation on each GPU; the H3 run will also
exercise the distributed communication paths.

```bash
python3 - <<'PY'
from importlib.metadata import version
import ctypes

import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 4, "Expose exactly four GPUs for this profile"
assert tuple(map(int, torch.version.cuda.split('.')[:2])) >= (13, 4)
nccl = ctypes.CDLL("libnccl.so.2")
nccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
nccl.ncclGetVersion.restype = ctypes.c_int
nccl_version = ctypes.c_int()
assert nccl.ncclGetVersion(ctypes.byref(nccl_version)) == 0
assert nccl_version.value >= 23203, "NCCL runtime 2.32.3+ is required"
for name in ("torch", "vllm", "vllm-omni", "triton", "flashinfer-python"):
    print(name, version(name))
print("CUDA", torch.version.cuda, "NCCL runtime", nccl_version.value)
compiled = torch.compile(lambda x: x.sin() + x)
for index in range(torch.cuda.device_count()):
    assert torch.cuda.get_device_capability(index) == (10, 7), "Expected SM107"
    with torch.cuda.device(index):
        print(index, torch.cuda.get_device_name(index))
        x = torch.randn(256, device=f"cuda:{index}")
        torch.testing.assert_close(compiled(x), x.sin() + x)

import vllm_omni
from vllm_omni.entrypoints.cli.main import main

print("Runtime smoke check passed")
PY
```

`torch.cuda.nccl.version()` reports PyTorch's build-time NCCL headers. The
check above queries the loaded library through `ncclGetVersion` instead.

For authenticated Hugging Face downloads, pass `HF_TOKEN` from the host or
run `hf auth login` inside the container. Pre-download the pinned partition
before initializing the engine; the test runs used a populated cache:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 0919fb684102ef115126612f3fbb84d1027e7317 --include 'FL2VA/**'
# Also fetch this partition before running Ref2VA.
hf download MiniMaxAI/MiniMax-H3 \
  --revision 0919fb684102ef115126612f3fbb84d1027e7317 --include 'Ref2VA/**'
```

The Hugging Face cache deduplicates shared blobs between the partitions;
the pinned pair occupies about 196 GiB of unique model data. Budget more
space for runtime packages and outputs.

## Four-GPU offline inference

| Component | Parallelism |
| --- | ---: |
| DiT tensor parallelism | 1 |
| Ulysses sequence parallelism | 4 |
| Ring parallelism | 1 |
| Text-encoder tensor parallelism | 4 |
| VAE patch parallelism | 4, native tile mode |
| CFG parallelism | 1 |

Save the following as `h3_rubin.py` in the mounted checkout. It runs a
warmup followed by three sequential requests, writes an MP4 for each, and
records generation wall time, stage timings, and GPU memory sampled every
500 ms. `H3_SMOKE=1` instead runs one small, two-step execution check.

```python
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    task = os.getenv("H3_TASK", "t2va")
    reference = os.getenv("H3_REFERENCE")
    smoke = os.getenv("H3_SMOKE", "0") == "1"
    out = Path(os.getenv("H3_OUTPUT", "h3-rubin-output"))
    out.mkdir(parents=True, exist_ok=False)
    prompt = {
        "prompt": "A small robot walks along a quiet beach at sunrise, "
        "with soft footsteps and synchronized ocean waves."
    }
    if reference:
        prompt["multi_modal_data"] = {"image": Image.open(reference).convert("RGB")}
    if task in ("fl2va", "ref2va") and not reference:
        raise ValueError("Set H3_REFERENCE to a local image for this task")
    omni = Omni(
        model="MiniMaxAI/MiniMax-H3",
        revision="0919fb684102ef115126612f3fbb84d1027e7317",
        trust_remote_code=True,
        task_type="ref2va" if task == "ref2va" else "fl2va",
        num_gpus=4, tensor_parallel_size=1, ulysses_degree=4, ring_degree=1,
        text_encoder_tp_size=4, vae_patch_parallel_size=4,
        vae_parallel_mode="tile", vae_use_tiling=True, cfg_parallel_size=1,
        diffusion_attention_backend=os.getenv("H3_BACKEND", "TRTLLM_ATTN"),
        enforce_eager=os.getenv("H3_COMPILE", "0") != "1",
        flow_shift=12, enable_diffusion_pipeline_profiler=True,
        init_timeout=1200, stage_init_timeout=1200,
    )
    records = []
    try:
        for label in (("smoke",) if smoke else ("warmup", "run1", "run2", "run3")):
            params = OmniDiffusionSamplingParams(
                width=672 if smoke else 1344, height=384 if smoke else 768,
                num_inference_steps=2 if smoke else 50, fps=24,
                seed=1101, guidance_scale=1.0,
                extra_args={
                    "task": task, "duration": 4.0 if smoke else 5.0,
                    "aspect_ratio": "16:9", "audio_flow_shift": 3.0,
                    "preencode_mp4": True,
                    "video_codec_options": {"crf": "0", "preset": "ultrafast"},
                },
            )
            gpu_log = out / f"{label}.gpu.csv"
            with gpu_log.open("w") as log:
                monitor = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits", "-lms", "500",
                ], stdout=log)
                start = time.perf_counter()
                try:
                    result = omni.generate(prompt, params, use_tqdm=False)[0]
                    elapsed = time.perf_counter() - start
                finally:
                    monitor.terminate()
                    monitor.wait()
            videos = result.images or result.multimodal_output["video"]
            assert isinstance(videos[0], bytes), "Expected pre-encoded MP4"
            (out / f"{label}.mp4").write_bytes(videos[0])
            peaks = {}
            for gpu, memory in csv.reader(gpu_log.read_text().splitlines()):
                gpu = gpu.strip()
                peaks[gpu] = max(peaks.get(gpu, 0), float(memory))
            record = {
                "label": label, "warmup": label in ("smoke", "warmup"),
                "generation_wall_s": elapsed,
                "stage_durations": result.stage_durations,
                "sampled_peak_memory_mib_by_gpu": peaks,
            }
            records.append(record)
            (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
```

Run inside the container. Use a new output directory for each invocation:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8

H3_SMOKE=1 H3_OUTPUT=h3-smoke python3 h3_rubin.py
H3_OUTPUT=h3-trtllm python3 h3_rubin.py
# Reproduce the SDPA reference profile.
H3_BACKEND=TORCH_SDPA H3_OUTPUT=h3-sdpa python3 h3_rubin.py
```

The full request uses T2VA, 1344×768, 24 FPS, `duration=5.0`, 50 steps,
`flow_shift=12`, `audio_flow_shift=3.0`, and `seed=1101`. The video is
encoded in the worker with `crf=0` and `preset=ultrafast`. The first full
request is a warmup; exclude it when summarizing `results.json`.

Inspect and decode every output:

```bash
for video in h3-trtllm/*.mp4; do
  ffprobe -v error -count_frames -show_entries \
    stream=codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels:format=duration \
    -of json "${video}"
  ffmpeg -v error -xerror -i "${video}" -map 0:v:0 -map 0:a:0 -f null -
done
```

Expect H.264 video and stereo 32 kHz AAC audio. H3 aligns frames to a
`17n+5` grid: this five-second request generates 124 frames at 24 FPS.
Audio alignment can make the MP4 duration slightly longer than the video.
The two-step smoke request checks execution and encoding, not perceptual
quality. Inspect full-step video and listen to its audio separately.

### Reference-conditioned tasks

For FL2VA, supply an image as the first frame. For Ref2VA, the same script
loads the `Ref2VA` partition and uses the image as a reference. Keep one
partition loaded at a time:

```bash
ffmpeg -v error -i h3-trtllm/run1.mp4 -frames:v 1 -update 1 h3-reference.png
H3_TASK=fl2va H3_REFERENCE="$PWD/h3-reference.png" \
  H3_OUTPUT=h3-fl2va python3 h3_rubin.py
H3_TASK=ref2va H3_REFERENCE="$PWD/h3-reference.png" \
  H3_OUTPUT=h3-ref2va python3 h3_rubin.py
```

These commands cover one image reference. Video references, multiple
references, and image-plus-audio conditioning require additional input
handling; see the [shared model guide](MiniMax-H3.md).

### Attention and compilation

`TRTLLM_ATTN` selects FlashInfer's trtllm-gen attention kernels and is the
default in the script above. Keep eager execution for this pinned stack.
The regional-compilation experiment hit Dynamo's recompilation limit and
partially fell back to eager execution, with comparable measured latency.
To reproduce that experiment, warm the exact output shape and exclude
compilation time from the measurements:

```bash
H3_BACKEND=TRTLLM_ATTN H3_COMPILE=1 \
  H3_OUTPUT=h3-trtllm-compiled python3 h3_rubin.py
```

Use the validation table below to distinguish tested configurations from
additional tuning options. Online FP8, quantized attention, Skip-Softmax,
and Cache-DiT were not qualified by these BF16 runs.

## Target-hardware validation

Measured on November 10, 2026, using the pinned environment above, BF16,
TP1 × Ulysses4, text-encoder TP4, VAE tile parallelism 4, and one request at
a time. T2VA profiles used a 672×384, two-step smoke request, a full-size
warmup, and three measured requests. Full requests use 1344×768, 50 steps,
`duration=5.0`, and the exact prompt and seed in the script above.

FL2VA used the first decoded frame of that profile's T2VA run 1 as its image
input. It ran in the same process after T2VA, with one full FL2VA warmup
and one measured request. Ref2VA uses a separate process and partition,
with a smoke request, full warmup, and three measured requests against one
image reference. The raw data identifies each reference image by SHA-256.

| Profile | Task | Requests | Median generation | Range | Mean denoise, 50 steps | Mean per step | Max sampled GPU memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDPA, eager | T2VA | 3 | 110.039 s | 110.038–110.048 s | 108.874 s | 2.177 s | 96.88 GiB |
| TRTLLM, eager | T2VA | 3 | 19.811 s | 19.777–19.849 s | 18.614 s | 0.372 s | 97.26 GiB |
| TRTLLM, eager | FL2VA | 1 | 21.535 s | — | 20.267 s | 0.405 s | 98.31 GiB |
| TRTLLM, eager | REF2VA | 3 | 21.461 s | 21.400–21.476 s | 20.172 s | 0.403 s | 97.54 GiB |
| TRTLLM, compile requested† | T2VA | 3 | 19.834 s | 19.833–19.896 s | 18.656 s | 0.373 s | 97.18 GiB |
| TRTLLM, compile requested† | FL2VA | 1 | 21.529 s | — | 20.262 s | 0.405 s | 98.23 GiB |

† Regional compilation was requested for 52 DiT blocks, but PyTorch hit
`torch._dynamo.config.recompile_limit=8` on a `self.layer_index` guard and
some graphs fell back to eager execution. These rows describe that observed
configuration, not a fully compiled model. No Dynamo configuration override
or Omni runtime patch was applied.

`generation` is wall time around `Omni.generate`, including output transfer
and worker-side MP4 encoding. It excludes model loading, output validation,
and all warmups. It is not HTTP end-to-end latency. Denoise is the
`MiniMaxH3Pipeline.diffuse` stage duration; per-step time divides it by 50.
FL2VA's single measured request is an execution check, not a latency distribution.

Memory is `nvidia-smi memory.used` sampled every 500 ms during measured
requests, taking the maximum over the requests. It includes process and
CUDA-library allocations and is distinct from PyTorch allocator peaks.
Sub-sample transient peaks may be missed.

| Profile / task | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | ---: | ---: | ---: | ---: |
| SDPA, eager / T2VA | 99,205 MiB | 98,769 MiB | 98,769 MiB | 98,769 MiB |
| TRTLLM, eager / T2VA | 99,599 MiB | 99,163 MiB | 99,163 MiB | 99,163 MiB |
| TRTLLM, eager / FL2VA | 100,673 MiB | 100,373 MiB | 100,373 MiB | 100,373 MiB |
| TRTLLM, eager / REF2VA | 99,885 MiB | 99,661 MiB | 99,661 MiB | 99,659 MiB |
| TRTLLM, compile requested† / T2VA | 99,517 MiB | 99,179 MiB | 99,179 MiB | 99,179 MiB |
| TRTLLM, compile requested† / FL2VA | 100,591 MiB | 100,391 MiB | 100,391 MiB | 100,391 MiB |

All measured outputs passed full video-and-audio decoding, with
1344×768 H.264 video, 124 frames at 24 FPS, and stereo 32 kHz AAC audio.
MP4 duration was 5.175 s, and decoded audio contained finite, nonzero samples.
The three SDPA T2VA MP4s were byte-identical; their SHA-256 was
`1959a4746a37209affbf987a48a1107f17e6fb027b597024577ca7b5e8bdd66f`.

Constructing the SDPA `Omni` engine took 366.74 s with weights already cached.
Its small smoke request took 8.79 s and the full warmup took 110.24 s.
The first eager TRTLLM smoke request took 45.41 s, including first-use
FlashInfer kernel compilation. These times are excluded from the table.
Independent BF16 matrix multiplication, NCCL all-reduce/all-to-all, and
Triton compilation checks also passed on all four GPUs.

### Numerical comparison with SDPA

The comparisons below use the first measured T2VA output at the same prompt,
seed, shape, and step count. Video PSNR compares aligned decoded frames in
their native pixel format. Audio spectral cosine uses channel-averaged
STFT magnitudes, a 1024-sample Hann window, and a 256-sample hop; RMS ratio
is candidate audio RMS divided by reference audio RMS. No alignment search
or perceptual-quality pass threshold was applied.

| Candidate | Aggregate video PSNR | Audio spectral cosine | Audio RMS ratio |
| --- | ---: | ---: | ---: |
| TRTLLM, compile requested† | 30.76 dB | 0.9895 | 0.948 |
| TRTLLM, eager | 28.19 dB | 0.9875 | 0.941 |

See [the machine-readable inference data](MiniMax-H3-Rubin-results.json)
for individual requests, warmups, stage timings, output hashes, reference
image hashes, and runtime provenance. Stage keys ending in `_ms` are in
milliseconds; the named pipeline stages are in seconds.

## HTTP compatibility

The offline results do not qualify `/v1/videos` serving. Two real startup
failures were observed with unmodified Omni `7467d4b5`:

| vLLM Rubin image revision | Failure |
| --- | --- |
| `cf5cea2da` (November 8 nightly) | Omni import fails: `VllmConfig.use_v2_model_runner` was removed |
| `1648fefab` (November 7 nightly, used above) | HTTP startup fails: `AsyncOmni` lacks the abstract methods `checkpoint_prepare`, `checkpoint_restore`, and `checkpoint_state` |

Source inspection found the checkpoint methods in all nine earlier generic
Rubin nightly revisions checked, from October 29 through November 7. The offline
`Omni` class does not inherit the HTTP `EngineClient` interface. A compatible
HTTP integration must be tested separately; updating both repositories to
moving `main` revisions is insufficient.

## Known limitations

- This is a four-GPU, single-host profile with one request at a time.
- Keep Ring and CFG parallelism at 1, and VAE patch parallelism at the full
  DiT group size (4) with the native `tile` mode.
- Measurements use one prompt and seed. Successful decoding and nonzero
  audio do not establish prompt fidelity, perceptual quality, or parity
  with another hardware recipe.
- Longer clips, larger canvases, and reference inputs can increase memory.
  The observed peak does not establish a minimum supported HBM capacity.
- HTTP serving and x86 Rubin hosts are not validated by the offline run.

## Additional resources

- [MiniMax H3 model and API guide](MiniMax-H3.md)
- [Diffusion attention backends](../../docs/user_guide/diffusion/attention_backends.md)
- [Diffusion parallelism](../../docs/user_guide/diffusion/parallelism/overview.md)
- [vLLM Rubin dependency manifest](https://github.com/vllm-project/vllm/blob/main/requirements/rubin-prerelease.txt)
