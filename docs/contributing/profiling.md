# Profiling Diffusion Models

> **Warning:** Profiling is for development and debugging only. It adds significant overhead and should not be enabled in production.

Diffusion profiling supports two backends through `profiler_config`:

- `torch`: detailed CPU/CUDA traces, operator tables, and optional memory snapshots
- `cuda`: low-overhead CUDA range control for NVIDIA Nsight Systems (`nsys`)

## 1. Configure `profiler_config`

Use `profiler_config` to enable profiling for a diffusion model. For diffusion usage, pass it directly to `Omni(...)` or `vllm serve`.

Minimal torch-profiler config:

```yaml
profiler_config:
  profiler: torch
  torch_profiler_dir: ./perf
```

Supported fields:

| Field | Description |
|---|---|
| `profiler` | Profiler backend. Supported values: `torch`, `cuda`. Use `torch` for `trace.json`, Excel operator tables, and optional memory snapshots. Use `cuda` for Nsight Systems only. |
| `torch_profiler_dir` | Output directory for torch-profiler artifacts. Required when `profiler: torch`. |
| `torch_profiler_use_gzip` | Compress `trace_rank*.json` into `trace_rank*.json.gz`. |
| `torch_profiler_record_shapes` | Record input shapes and add a `by_shape` sheet to `ops_rank*.xlsx`. |
| `torch_profiler_with_stack` | Record call stacks, add a `by_stack` sheet to `ops_rank*.xlsx`, and export `stacks_cpu_rank*.txt` and `stacks_cuda_rank*.txt`. |
| `torch_profiler_with_memory` | Enable memory profiling and attempt to dump `memory_snapshot_rank*.pickle`. The pickle is only generated when the current backend supports memory history and snapshot APIs. |
| `torch_profiler_with_flops` | Enable FLOPs collection in `torch.profiler`. This does not add a separate output file. |
| `torch_profiler_dump_cuda_time_total` | Export an additional text summary `profiler_out_<rank>.txt` sorted by `self_cuda_time_total`. |
| `delay_iterations` | Number of worker iterations to skip before profiling starts. |
| `max_iterations` | Maximum number of worker iterations to capture before auto-stop. |
| `wait_iterations` | Torch-profiler wait iterations before warmup. |
| `warmup_iterations` | Torch-profiler warmup iterations. |
| `active_iterations` | Torch-profiler active iterations. |

For detailed explanations of the fields, please refer to upstream vLLM implementation [vllm/config/profiler.py](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/config/profiler.py)

### Minimal configurations by output

Only collect trace output:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
}
```

Outputs:

- `trace_rank*.json`
- `ops_rank*.xlsx` with a `summary` sheet

Collect compressed trace output:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_use_gzip": True,
}
```

Outputs:

- `trace_rank*.json.gz`
- `ops_rank*.xlsx` with a `summary` sheet

Collect trace and full operator tables:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_stack": True,
}
```

Outputs:

- `trace_rank*.json`
- `ops_rank*.xlsx` with `summary`, `by_shape`, and `by_stack`
- `stacks_cpu_rank*.txt`
- `stacks_cuda_rank*.txt`

Collect trace, operator tables, and memory snapshots:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_stack": True,
    "torch_profiler_with_memory": True,
}
```

Outputs:

- `trace_rank*.json`
- `ops_rank*.xlsx` with `summary`, `by_shape`, and `by_stack`
- `stacks_cpu_rank*.txt`
- `stacks_cuda_rank*.txt`
- `memory_snapshot_rank*.pickle` when supported by the current backend

### Full torch-profiler configuration

If you want to enable the commonly used torch-profiler options together:

```python
profiler_config = {
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_use_gzip": False,
    "torch_profiler_record_shapes": True,
    "torch_profiler_with_stack": True,
    "torch_profiler_with_memory": True,
    "torch_profiler_with_flops": False,
    "torch_profiler_dump_cuda_time_total": False,
    "delay_iterations": 0,
    "max_iterations": 0,
    "wait_iterations": 0,
    "warmup_iterations": 0,
    "active_iterations": 1,
}
```

## 2. Profiling Diffusion with PyTorch Profiler

Single-stage diffusion models use `start_profile()` / `stop_profile()` controls. The profiler only writes artifacts after profiling has been started and then stopped.

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "./perf",
    },
)

omni.start_profile()
...
omni.stop_profile()
```

For diffusion offline example scripts under `examples/offline_inference/`, pass `--profiler-config` as a JSON object. The script enables profiling when this argument is set and wraps generation with `start_profile()` / `stop_profile()`.

Example:

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image input.jpg \
  --prompt "A cat playing with yarn" \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "./perf",
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_stack": true
  }'
```

Examples:

1. [Image edit example](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py)
2. [Image to video example](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video)

## 3. Profiling Diffusion with Nsight Systems (`nsys`)

For Nsight Systems, use `profiler: cuda` and wrap the process with `nsys profile`.

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  -o diffusion_trace \
  python image_to_video.py ...
```

The Python process being profiled must create the diffusion engine with:

```python
profiler_config = {"profiler": "cuda"}
```

Then call `start_profile()` before the requests you want to capture and `stop_profile()` after them. The diffusion worker processes open and close the CUDA capture range themselves, so `nsys` sees the actual GPU work instead of only the parent process.

## 4. Profiling Online Serving

When `profiler_config.profiler` is set for a diffusion model, the server exposes:

- `POST /start_profile`
- `POST /stop_profile`

### Start the server

Single-stage diffusion serving with torch profiler:

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --omni \
  --port 8091 \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "/tmp/vllm_profile_wan22_i2v",
    "torch_profiler_with_stack": true,
    "torch_profiler_with_flops": false,
    "torch_profiler_use_gzip": true,
    "torch_profiler_dump_cuda_time_total": false,
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_memory": true,
    "delay_iterations": 0,
    "max_iterations": 0,
    "wait_iterations": 0,
    "warmup_iterations": 0,
    "active_iterations": 1
  }'
```

Single-stage diffusion serving with Nsight Systems:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  -o serving_trace \
  vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --omni \
    --port 8091 \
    --profiler-config '{"profiler": "cuda"}'
```

### Control capture

Example profiling flow for an online Qwen-Image request:

```bash
# Start profiling.
curl -X POST http://localhost:8091/start_profile

# Send a Qwen-Image generation request while profiling is active.
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "A red vintage bicycle parked beside a quiet canal at sunset"
  }'

# Stop profiling and flush profiler artifacts.
curl -X POST http://localhost:8091/stop_profile
```

## 5. Diffusion Pipeline Profiler

For lightweight per-stage pipeline timing such as `vae.decode` or `diffuse`, see [Diffusion Pipeline Profiler](model/adding_diffusion_model.md#diffusion-pipeline-profiler-performance-profiling). That utility logs stage durations only and does not generate torch-profiler artifacts such as `trace.json`, Excel tables, or memory snapshots.

## 6. Analyze Results

Torch-profiler output:

- Chrome/Perfetto trace: `trace_rank*.json` or `trace_rank*.json.gz`
- Excel workbook: `ops_rank*.xlsx` with `summary`, and optional `by_shape` / `by_stack` sheets
- Stack exports: `stacks_cpu_rank*.txt` and `stacks_cuda_rank*.txt` when stack capture is enabled
- Memory snapshot: `memory_snapshot_rank*.pickle` when memory capture is enabled and supported by the backend
- Optional CUDA-time text summary: `profiler_out_<rank>.txt` when `torch_profiler_dump_cuda_time_total` is enabled

CUDA profiler / Nsight Systems output:

- `.nsys-rep` report files written by `nsys -o ...`

Recommended viewers:

- [Perfetto](https://ui.perfetto.dev/) for torch traces
- `nsys stats <report>.nsys-rep` for CLI summaries
- Nsight Systems GUI for CUDA kernel timelines

For upstream background on the underlying vLLM profiling infrastructure, see the [vLLM profiling guide](https://docs.vllm.ai/en/stable/contributing/profiling/).
