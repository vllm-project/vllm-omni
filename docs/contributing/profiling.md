# Profiling vLLM-Omni

> **Warning:** Profiling is for development and debugging only. It adds significant overhead and should not be enabled in production.

vLLM-Omni supports two profiler backends through `profiler_config`:

- `torch`: detailed CPU/CUDA traces written to `torch_profiler_dir`
- `cuda`: low-overhead CUDA range control for NVIDIA Nsight Systems (`nsys`)

## 1. Configure Profiling

Use the same `profiler_config` shape everywhere:

```yaml
profiler_config:
  profiler: torch
  torch_profiler_dir: ./perf
```

Supported fields:

| Field | Description |
|---|---|
| `profiler` | Profiler backend. Supported values: `torch`, `cuda`. |
| `torch_profiler_dir` | Output directory for torch traces. Required when `profiler: torch`. |
| `delay_iterations` | Number of worker iterations to skip before profiling starts. |
| `max_iterations` | Maximum number of worker iterations to capture before auto-stop. |
| `warmup_iterations` | Torch-profiler warmup iterations. |
| `active_iterations` | Torch-profiler active iterations. |
| `wait_iterations` | Torch-profiler wait iterations before warmup. |

For multi-stage omni pipelines, put `profiler_config` under the target stage's `engine_args`.

```yaml
stage_args:
  - stage_id: 0
    stage_type: llm
    engine_args:
      profiler_config:
        profiler: torch
        torch_profiler_dir: ./perf
```

For single-stage diffusion usage, pass `profiler_config` directly to `Omni(...)` or `vllm serve`.

## 2. Profiling Omni Pipelines

It is usually best to profile only the stages you need.

```python
# Profile all stages.
omni.start_profile()

# Profile selected stages only.
omni.start_profile(stages=[0, 2])
...
omni.stop_profile(stages=[0, 2])
```

Always stop the same stage set that you started. If only some stages have `profiler_config`, pass an explicit `stages=[...]` list instead of relying on the default "all stages" behavior.

Examples:

1. [Qwen2.5-Omni end2end](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py)
2. [Qwen3-Omni end2end](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py)

## 3. Profiling Single-Stage Diffusion

Single-stage diffusion models use the same `start_profile()` / `stop_profile()` controls, but you must provide `profiler_config` explicitly.

### PyTorch profiler

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

### Nsight Systems (`nsys`)

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
profiler_config={"profiler": "cuda"}
```

Then call `start_profile()` before the requests you want to capture and `stop_profile()` after them. The diffusion worker processes open and close the CUDA capture range themselves, so `nsys` sees the actual GPU work instead of only the parent process.

Examples:

1. [Image edit example](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py)
2. [Image to video example](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video)

## 4. Profiling Online Serving

When any stage has `profiler_config.profiler` set, the server exposes:

- `POST /start_profile`
- `POST /stop_profile`

### Start the server

Multi-stage omni serving:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
  --omni \
  --port 8091
```

(The default deploy config at `vllm_omni/deploy/qwen2_5_omni.yaml` is loaded automatically. Pass `--deploy-config /path/to/custom.yaml` to override.)

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
    "torch_profiler_dump_cuda_time_total": true,
    "torch_profiler_record_shapes": true,
    "torch_profiler_with_memory": true,
    "delay_iterations": 0,
    "max_iterations": 0
  }'
```

Useful optional fields:

- `torch_profiler_with_stack`: export `by_stack` operator views and stack text files.
- `torch_profiler_record_shapes`: export `by_shape` operator views.
- `torch_profiler_with_memory`: dump `memory_snapshot_rank*.pickle` when the backend supports memory history.
- `torch_profiler_use_gzip`: write the trace as `trace_rank*.json.gz`.

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

```bash
# Start profiling on all profiled stages.
curl -X POST http://localhost:8091/start_profile

# Start profiling on selected stages.
curl -X POST http://localhost:8091/start_profile \
  -H "Content-Type: application/json" \
  -d '{"stages": [0]}'

# Stop profiling.
curl -X POST http://localhost:8091/stop_profile
```

For mixed-stage pipelines, use explicit `stages` and pass the same stage list to both endpoints.

## 5. Analyze Results

Torch profiler output:

- Chrome/Perfetto trace: `trace_rank*.json` or `trace_rank*.json.gz`
- Excel workbook: `ops_rank*.xlsx` with `summary`, and optional `by_shape` / `by_stack` sheets
- Stack exports: `stacks_cpu_rank*.txt` and `stacks_cuda_rank*.txt` when stack capture is enabled
- Memory snapshot: `memory_snapshot_rank*.pickle` when memory capture is enabled and supported by the backend
- Optional aggregated CUDA-time tables under the same session directory

CUDA profiler / Nsight Systems output:

- `.nsys-rep` report files written by `nsys -o ...`

Recommended viewers:

- [Perfetto](https://ui.perfetto.dev/) for torch traces
- `nsys stats <report>.nsys-rep` for CLI summaries
- Nsight Systems GUI for CUDA kernel timelines

vLLM-Omni reuses the vLLM profiling infrastructure where possible. For the upstream reference, see the [vLLM profiling guide](https://docs.vllm.ai/en/stable/contributing/profiling/).
