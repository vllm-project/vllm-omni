# MAGI-2 Preview — NPU

## Summary

- Vendor: SandAI
- Model: `sand-ai/MAGI-2-preview`
- Runtime: Native vLLM-Omni `Magi2Pipeline`; no SandAI runtime import
- Task and mode: Online text-to-video-and-audio (T2VA)
- Hardware: Four NPU devices, 112 GiB per device; or eight 910B3 devices,
  64 GiB per device
- Deployment: Resident BF16, TP1/SP4 (Ulysses) at 112 GiB, TP1/SP8 at 64 GiB;
  eager execution
- Maintainer: Community

## Supported model contract

This recipe covers the released Preview stage and the bounded NPU profile
below. It does not expand the qualification scope of the separate
[CUDA recipe](MAGI-2-preview-L20X.md).

| Contract | Provided profile |
| --- | --- |
| Task | T2VA through `POST /v1/videos/sync` |
| Input | One text prompt; no image, video, or audio condition |
| Video | `272p`: 448x256, 125 frames, 12.5 fps, 10 seconds |
| Audio | Generated with the video; 44.1 kHz stereo |
| Sampling | 100 denoising steps, seed 42, one request |
| Deployment | Four 112 GiB devices: TP=1, SP=4. Eight 64 GiB 910B3 devices: TP=1, SP=8. BF16, `--enforce-eager` |
| Refiner | The separate 1080p refiner is not included |

## References

- [Official model card](https://huggingface.co/sand-ai/MAGI-2-preview)
- [Shared online video examples](../../examples/online_serving/text_to_video/README.md)
- [Supported models](../../docs/models/supported_models.md)
- [Diffusion feature matrix](../../docs/user_guide/diffusion_features.md)
- [NPU support issue](https://github.com/vllm-project/vllm-omni/issues/8119)
- [Operator-boundary RFC](https://github.com/vllm-project/vllm-omni/issues/6305)

## Checkpoint and setup

Use a complete local `sand-ai/MAGI-2-preview` checkpoint containing
`preview/`, `text_encoder/`, `vae/`, `turbo_vae/`, and
`stable-audio-open-1.0/`. The refiner directory is not required. The PR uses
pre-existing local assets; their Hub revision was not recorded. Pin and
record the checkpoint revision when preparing a new deployment rather than
assuming that a later download is identical.

Use the matched Ascend environment below with the PR checkout installed.
Keep its PyTorch, `torch_npu`, CANN, driver, and vLLM-Ascend stack matched;
these commands do not install or replace that stack. Make `ffmpeg` and
`ffprobe` available for muxing and media verification.

```bash
export REPO_DIR=/path/to/vllm-omni
export MODEL_PATH=/path/to/MAGI-2-preview
export OUTPUT_DIR=/path/to/validation-output
export NPU_DEVICES=0,1,2,3  # Four idle devices; use eight (0-7) with SP8 on 64 GiB devices.
export PORT=8091
mkdir -p "$OUTPUT_DIR"
```

## Hardware

| Item | Value |
| --- | --- |
| Accelerator | NPU |
| Per-device memory | 112 GiB |
| Device count | 4 |
| Device interconnect | Not recorded; no interconnect-specific performance claim |
| Host memory | About 1.48 TiB total capacity; not a peak-PSS measurement |
| Scope | One bounded online T2VA configuration, not a throughput benchmark |

Native auxiliary-component CPU staging remains active. The launch disables
pinned CPU staging with `pin_cpu_memory=false`; it does not enable
transformer layerwise offload.

### 64 GiB-class devices (8 x 910B3)

The profile above assumes 112 GiB per device. On the smaller 910B3 device the
resident SP4 layout does **not** fit: with `TP=1/SP=4` the transformer reaches
60.49 GiB of the 60.96 GiB usable HBM and fails while loading the remaining
shard (`torch.OutOfMemoryError: NPU out of memory`). Widening to eight devices
(`TP=1/SP=8`) halves the per-rank MoE-head shard and completes the request
without an allocation failure.

| Item | Value |
| --- | --- |
| Accelerator | Ascend 910B3 |
| Per-device memory | 65,536 MB (64 GiB); 60.96 GiB usable |
| Device count | 8 |
| Host | `ascend131`, openEuler 22.03 LTS-SP1, aarch64 |
| Host memory | 2,014 GiB total; 256 cores |
| Scope | One bounded online T2VA configuration, not a throughput benchmark |

Launch with the eight-device topology. Keep every other flag identical to the
four-device command in [Online serving](#online-serving); only `--num-gpus`
and `--ulysses-degree` change:

```bash
vllm serve "$MODEL_PATH" --omni --model-class-name Magi2Pipeline \
  --num-gpus 8 --tensor-parallel-size 1 --ulysses-degree 8 \
  --enforce-eager --host 127.0.0.1 --port "$PORT" \
  --stage-init-timeout 3600 --init-timeout 3600 \
  --stage-overrides '{"0":{"pin_cpu_memory":false}}'
```

`TP=1/SP=4` is not a viable fallback on this device class; use eight devices.
Memory-reducing features are outside what this recipe qualifies.

#### Validation record

Measured on vLLM-Omni commit `7476b8b8e` with the released checkpoint
`2dea51b64db47ee5b4402d36fd90829a0c58913b`, one request, devices 0-7:

| Check | Result |
| --- | --- |
| Startup | `/health` returned 200; device mapping `0->0 … 7->7` |
| Response | HTTP 200, `content-type: video/mp4`, 2,078,762 bytes |
| Wall time | 336.7 s for the 272p / 100-step request |
| Video | 125 frames, 448x256, 12.5 fps, duration 10.00 s |
| Audio | AAC 44.1 kHz stereo |
| Decode | `ffmpeg -f null -` completed without error |
| Sampling | `steps=100 seed=42`, as recorded by `serving_video.py` |

The four-device run used for comparison failed before serving any request, so
it produced no media artifact. Both runs used the same container, checkpoint
and request body; only `--num-gpus` and `--ulysses-degree` differed.

#### Environment note for this validation

The run used the `quay.io/atlas-ci/vllm-ascend:v0.32.0` base image (vLLM
0.32.0, vLLM-Ascend `450e55698`, PyTorch 2.10.0+cpu, torch_npu 2.10.0.post4,
transformers 5.19.0, CANN 9.1.0) with this checkout on `PYTHONPATH` and the
vLLM-Omni dependency set supplied out-of-band, because the box cannot reach
the package index. That image is not one of the qualified images listed above;
treat it as an additional data point rather than a replacement.

Exporting `PYTHONPATH` without preserving the image's CANN entries breaks the
first operator compile, not the model load: the server still reports `/health`
200 and then fails the request with `AOE failed to call InitCannKB` and
`Failed to init tbe`. Prefix the checkout path as the four-device command
below does instead of replacing `PYTHONPATH` outright.

## Software environment

| Item | Value |
| --- | --- |
| Container image | `quay.io/ascend/vllm-omni:nightly-20261021-24f0cf3-a5` |
| Image ID | `sha256:b67691b85280a9d59cbef51cced3f721a599fd6f1cae2e1b134adf3f4f122f24` |
| Container OS | Ubuntu 22.04.5 LTS |
| Host OS | openEuler 24.03 LTS-SP2 |
| Python | 3.12.13 |
| Ascend runtime | CANN 9.1.0; driver supplied by the host |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| vLLM | 0.32.0+empty, commit `5df63a72` |
| vLLM-Ascend | Commit `450e55698` |
| vLLM-Omni | This PR; merged upstream `main` commit `0da1d831` |

The `+cpu` suffix is the PyTorch wheel version in the Ascend image; NPU
execution uses `torch_npu`.

## Online serving

Run from a shell initialized for the matched CANN environment. Retain its
CANN Python paths, but never include a CPU-test bootstrap in service paths.
The server binds only to localhost.

```bash
cd "$REPO_DIR"
TASK_CANN_PATHS=$(python -c 'import os; print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if "Ascend" in p))')
export PYTHONPATH="$REPO_DIR${TASK_CANN_PATHS:+:$TASK_CANN_PATHS}"
export VLLM_TARGET_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES"
export LD_PRELOAD=${JEMALLOC_PATH:-/lib/x86_64-linux-gnu/libjemalloc.so.2}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MAGI2_DETERMINISTIC=0

vllm serve "$MODEL_PATH" --omni --model-class-name Magi2Pipeline \
  --num-gpus 4 --tensor-parallel-size 1 --ulysses-degree 4 \
  --enforce-eager --host 127.0.0.1 --port "$PORT" \
  --stage-init-timeout 3600 --init-timeout 3600 \
  --stage-overrides '{"0":{"pin_cpu_memory":false}}' \
  > "$OUTPUT_DIR/server.log" 2>&1
```

After `/health` returns HTTP 200, send the request from another shell with
the same `PORT` and `OUTPUT_DIR`:

```bash
curl --noproxy '*' --fail --silent --show-error \
  "http://127.0.0.1:$PORT/health" > /dev/null
curl --noproxy '*' --fail --silent --show-error \
  --connect-timeout 10 --max-time 7200 \
  --dump-header "$OUTPUT_DIR/video.headers" \
  --write-out 'HTTP=%{http_code}\nTIME_TOTAL=%{time_total}\nSIZE_DOWNLOAD=%{size_download}\n' \
  -X POST "http://127.0.0.1:$PORT/v1/videos/sync" \
  -H 'Accept: video/mp4' \
  --form-string 'prompt=A red fox walks through fresh snow while wind moves the pine branches.' \
  --form-string 'seconds=10' --form-string 'size=448x256' \
  --form-string 'num_frames=125' --form-string 'fps=12.5' \
  --form-string 'num_inference_steps=100' --form-string 'seed=42' \
  --form-string 'extra_params={"resolution":"272p"}' \
  --output "$OUTPUT_DIR/magi2_272p_100steps_seed42.mp4" \
  > "$OUTPUT_DIR/client.log" 2>&1
```

## Verification

```bash
ffprobe -v error -count_frames \
  -show_entries stream=codec_type,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration \
  -of json "$OUTPUT_DIR/magi2_272p_100steps_seed42.mp4"
ffmpeg -v error -i "$OUTPUT_DIR/magi2_272p_100steps_seed42.mp4" \
  -f null -
```

Require HTTP 200, a completed 100-step generation in the server log, full
media decoding, 125 video frames at 448x256 and 12.5 fps, 10-second duration,
and a 44.1 kHz stereo audio stream. A valid media container alone does not
establish visual accuracy or prove that all denoising steps ran.

## Supported features and qualification boundary

| Feature | NPU profile in this recipe | Guide |
| --- | --- | --- |
| Online T2VA | 272p, 100 steps, one request | [Video examples](../../examples/online_serving/text_to_video/README.md) |
| Ulysses sequence parallel | Resident SP4 with TP1 on 112 GiB devices; SP8 with TP1 on 64 GiB 910B3 devices | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) |
| Auxiliary CPU staging | Active; pinned staging disabled | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| Offline / I2VA / 540p | Not qualified by this NPU run | [Shared example](../../examples/offline_inference/text_to_video/text_to_video.py) |
| Other TP/SP/DP/CFG/HSDP/DLO layouts | Not qualified by this NPU run | [Parallelism](../../docs/user_guide/diffusion/parallelism/overview.md) |
| Cache, quantization, graph execution | Not qualified by this NPU run | [Feature matrix](../../docs/user_guide/diffusion_features.md) |

## Qualification evidence

Record the tested revision, commands, results, and generated video for the
workload above. Do not substitute historical test counts or timings for a
changed revision's results. No peak-HBM, peak-host-PSS,
steady-state performance, or GPU-runtime claim is made by this NPU recipe.
Generated output is a functional example, not an independent ground-truth
reference. Unqualified profiles remain outside this recipe's support claim.
