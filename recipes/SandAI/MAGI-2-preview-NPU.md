# MAGI-2 Preview — NPU

## Summary

- Vendor: SandAI
- Model: `sand-ai/MAGI-2-preview`
- Runtime: Native vLLM-Omni `Magi2Pipeline`; no SandAI runtime import
- Task and mode: Online text-to-video-and-audio (T2VA)
- Hardware: Four NPU devices, 112 GiB per device
- Deployment: Resident BF16, TP1/SP4 (Ulysses), eager execution
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
| Deployment | Four devices; TP=1, SP=4; BF16; `--enforce-eager` |
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
export NPU_DEVICES=0,1,2,3  # Replace with an allocated, idle four-device group.
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

## Software environment

| Item | Value |
| --- | --- |
| Container OS | Ubuntu 22.04.5 LTS |
| Host OS | openEuler 24.03 LTS-SP2 |
| Python | 3.12.13 |
| Ascend runtime | CANN 9.1; driver supplied by the host |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| vLLM | 0.31.0, commit `db9527a4` |
| vLLM-Ascend | Commit `e2591273` |
| vLLM-Omni | This PR; rebuilt on upstream `main` commit `328e9c67` |

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
| Ulysses sequence parallel | Four-device resident SP4 with TP1 | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) |
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
