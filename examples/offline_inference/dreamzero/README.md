# DreamZero Offline Benchmark

This directory contains a small offline benchmark for DreamZero. It runs local
`Omni` inference, measures per-request generation latency, decodes the predicted
video latents, and writes artifacts for visual and numeric checks.

The written MP4/GIF files default to 5 FPS for playback. This playback metadata
does not affect latency or throughput; real inference FPS is reported in the
JSON summary as `model_video_fps`.

## Assets

Download the example camera videos before running the benchmark:

```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets \
  --repo-type dataset \
  --local-dir outputs/dreamzero/assets
```

The bundled assets currently contain enough frames for the default two
action-producing requests: one initial single-frame request plus one 4-frame
chunk. Use longer videos with `--num-requests N` for longer steady-state runs.

## Run

Two-GPU CFG-parallel run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,2 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
  --output-dir outputs/dreamzero/benchmark \
  --output-stem dreamzero_gpu1_2_tp1_cfg2 \
  --save-input-video \
  --save-side-by-side
```

Single-GPU baseline:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config vllm_omni/deploy/dreamzero.yaml \
  --output-dir outputs/dreamzero/benchmark \
  --output-stem dreamzero_gpu1_tp1_cfg1_baseline \
  --save-side-by-side
```

Two-GPU tensor-parallel run. The PR does not add a bundled TP2 deploy file; use
an equivalent local YAML for benchmarking:

```yaml
# /tmp/dreamzero_tp2_cfg1.yaml
pipeline: dreamzero
async_chunk: false
distributed_executor_backend: mp
dtype: bfloat16

stages:
  - stage_id: 0
    devices: "0,1"
    max_num_seqs: 1
    enforce_eager: true
    model_class_name: DreamZeroPipeline
    parallel_config:
      tensor_parallel_size: 2
      cfg_parallel_size: 1
    model_config:
      default_robot_embodiment: roboarena
      policy_server_config:
        image_resolution: [180, 320]
        n_external_cameras: 2
        needs_wrist_camera: true
        needs_stereo_camera: false
        needs_session_id: true
        action_space: joint_position
```

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,3 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config /tmp/dreamzero_tp2_cfg1.yaml \
  --output-dir outputs/dreamzero/benchmark \
  --output-stem dreamzero_gpu1_3_tp2_cfg1 \
  --save-side-by-side
```

Optional action parity check against a previous run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,2 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
  --output-dir outputs/dreamzero/benchmark \
  --output-stem dreamzero_gpu1_2_tp1_cfg2_checked \
  --reference-actions outputs/dreamzero/benchmark/dreamzero_gpu1_tp1_cfg1_baseline_actions.npz \
  --accuracy-atol 1e-3
```

The script defaults `DIFFUSION_ATTENTION_BACKEND` to `TORCH_SDPA` unless the
environment already sets another backend.

## Main Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--deploy-config` | required | DreamZero deploy YAML. |
| `--model` | `GEAR-Dreams/DreamZero-DROID` | Hugging Face model id or local model path. |
| `--video-dir` | `outputs/dreamzero/assets` | Directory with the three camera MP4 files. |
| `--num-requests` | `2` | Total generate calls, including the initial request. |
| `--save-input-video` | off | Also write the stitched input camera video. |
| `--save-side-by-side` | off | Write input and prediction side by side. |
| `--save-gif` | off | Also write a GIF of the prediction. |
| `--reference-actions` | unset | Compare generated actions with a previous NPZ. |
| `--accuracy-atol` | `1e-3` | Max absolute action error allowed for reference comparison. |
| `--profiler-config` | unset | JSON `ProfilerConfig` for torch/cuda profiling. Diagnostic only. |
| `--profile-request-index` | `1` | Request index profiled when `--profiler-config` is set. The default profiles `chunk_0` after the initial request. |

## One-Chunk Torch Profiler

Use torch profiler only for diagnosis. Keep the normal benchmark run without
profiler as the source of latency/FPS numbers, then run one profiled request to
inspect operator and communication breakdown:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=1,2 \
python examples/offline_inference/dreamzero/benchmark_prediction_video.py \
  --deploy-config /tmp/dreamzero_tp2_cfg1.yaml \
  --output-dir /tmp/dreamzero_profile/output_gpu1_2 \
  --output-stem tp2_cfg1_profile_gpu1_2 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/dreamzero_profile/tp2_cfg1_gpu1_2","torch_profiler_use_gzip":false,"torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"torch_profiler_with_memory":false,"torch_profiler_with_flops":false,"torch_profiler_dump_cuda_time_total":true,"delay_iterations":0,"max_iterations":0,"wait_iterations":0,"warmup_iterations":0,"active_iterations":1}' \
  --profile-request-index 1
```

`--profile-request-index 1` starts profiling after the initial request and stops
immediately after `chunk_0`, so model loading and session setup stay out of the
trace.

## Output

For `--output-stem dreamzero_gpu1_2_tp1_cfg2`, the benchmark writes:

- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2.mp4`
- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2_side_by_side.mp4`
- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2_actions.npz`
- `outputs/dreamzero/benchmark/dreamzero_gpu1_2_tp1_cfg2_summary.json`

The saved video files use 5 FPS playback by default. Change `--fps` only if you
want a different playback speed for visual inspection.

Useful summary fields:

| Field | Meaning |
| --- | --- |
| `latency_s.first_request` | First request wall time, including initial cache setup. |
| `latency_s.steady_state` | Stats over requests after the first one. |
| `latency_s.model_generate_total` | Sum of `omni.generate(...)` request times. |
| `latency_s.decode_video` | VAE decode latency for the concatenated prediction latents. |
| `throughput.model_request_hz` | Requests per second over model generation time. |
| `throughput.model_video_fps` | Decoded video frames per second over model generation time. |
| `throughput.model_plus_decode_video_fps` | Decoded video frames per second including VAE decode. |
| `throughput.model_action_hz` | Generated actions per second over model generation time. This is not robot control Hz. |

## Current Local Result

Measured on June 5, 2026 with:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, driver `590.48.01`
- TP1/CFG2 run used GPUs `1,2`
- TP1/CFG1 baseline used GPU `1`
- TP2/CFG1 run used GPUs `1,3`; GPU `2` was busy during this run
- CPU: 2x AMD EPYC 9355 32-Core Processor, 128 logical CPUs
- Host memory: 1.5 TiB
- Backend: `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`
- Mode: eager, no torch compile
- Workload: 2 requests, 17 decoded output frames

| Mode | GPUs | First latency | Steady latency | Total generate | Decode | Model video FPS | Model+decode video FPS | Model action Hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP1 CFG1 | 1 | 7.434s | 8.047s | 15.481s | 0.411s | 1.098 | 1.070 | 3.101 |
| TP2 CFG1 | 2 | 7.528s | 7.676s | 15.205s | 0.402s | 1.118 | 1.089 | 3.157 |
| TP1 CFG2 | 2 | 4.655s | 4.102s | 8.758s | 0.411s | 1.941 | 1.854 | 5.481 |

The generated prediction videos were readable by OpenCV:

| Artifact | Frames | FPS | Size |
| --- | ---: | ---: | --- |
| `dreamzero_gpu1_2_tp1_cfg2.mp4` | 17 | 5.0 | 640x352 |
| `dreamzero_gpu1_2_tp1_cfg2_side_by_side.mp4` | 17 | 5.0 | 1280x352 |
| `dreamzero_gpu1_3_tp2_cfg1.mp4` | 17 | 5.0 | 640x352 |
| `dreamzero_gpu1_3_tp2_cfg1_side_by_side.mp4` | 17 | 5.0 | 1280x352 |

Action parity check between TP1/CFG2 and TP1/CFG1 baseline:

| Chunk | Shape | Max abs error | RMSE | Passed |
| ---: | --- | ---: | ---: | --- |
| 0 | 24x8 | 0.0 | 0.0 | yes |
| 1 | 24x8 | 0.0 | 0.0 | yes |

Action comparison between TP2/CFG1 and TP1/CFG1 baseline at `atol=1e-3`:

| Chunk | Shape | Max abs error | RMSE | Passed |
| ---: | --- | ---: | ---: | --- |
| 0 | 24x8 | 0.06497 | 0.01822 | no |
| 1 | 24x8 | 0.00821 | 0.00224 | no |

## TP vs CFG Diagnostic Result

Measured on June 5, 2026 on GPUs `1,2` with `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`.
Baseline latency/FPS below is from non-profiler runs; breakdown is from a
separate one-chunk torch profiler run.

| Mode | First latency | Steady chunk latency | Total generate | Model video FPS | Model action Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP2 CFG1 | 7.652s | 7.604s | 15.256s | 1.114 | 3.146 |
| TP1 CFG2 | 4.582s | 4.033s | 8.615s | 1.973 | 5.572 |

Critical-rank one-chunk breakdown:

| Mode | GPU compute | GPU communication | CPU wait / launch gap |
| --- | ---: | ---: | ---: |
| TP2 CFG1 | 4.211s / 60.7% | 2.326s / 33.5% | 0.405s / 5.8% |
| TP1 CFG2 | 3.663s / 88.1% | 0.231s / 5.6% | 0.264s / 6.3% |

Action parity against the TP1/CFG1 baseline:

| Mode | Chunk 0 max abs error | Chunk 1 max abs error | Passed `atol=1e-3` |
| --- | ---: | ---: | --- |
| TP1 CFG2 | 0.0 | 0.0 | yes |
| TP2 CFG1 | 0.06497 | 0.00821 | no |
