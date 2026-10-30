# Denoise Step Graphs

`diffusion_use_graph` records one denoise step of the diffusion transformer as a
device graph and replays it for the later steps. A step is a fixed sequence of
kernels over fixed buffers, so replay skips the per-kernel launch cost. The
output stays numerically equivalent to eager: in our checks it was identical or
within eager's own run-to-run variation.

It is off by default.

## Supported pipelines and platforms

| Pipeline | Graph runner name in logs |
| ---------- | --------------------------- |
| Wan2.1 / Wan2.2 (`Wan22Pipeline`) | `Wan DiT` |
| Z-Image (`ZImagePipeline`) | `ZImage DiT` |
| HunyuanImage3 (`HunyuanImage3Pipeline`) | `HunyuanImage3 DiT` |

Capture and replay use the platform's graphed-forward class
(`current_omni_platform.get_graphed_forward_cls()`: CUDA graphs on CUDA, XPU
graphs on Intel XPU), so the same pipeline code runs on both. Other pipelines log
`diffusion_use_graph=True is ignored: <pipeline> does not support denoise graphs`
and run eager.

## Configuration

Offline:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Tongyi-MAI/Z-Image-Turbo", diffusion_use_graph=True)
```

Per-stage deploy configuration:

```yaml
stages:
  - stage_id: 0
    diffusion_use_graph: true
```

The log states the decision for each pipeline, for example
`ZImage DiT: diffusion_use_graph=True`, followed by
`ZImage DiT XPU graph captured for ...` (`CUDA graph` on CUDA) on the first
request.

## When graphs are refused

A replay repeats exactly what was recorded, so anything that changes a step's
work from outside the step's tensor arguments would be frozen at its capture
value. In these cases the runner is not built, a warning starting with
`diffusion_use_graph ignored because` is logged, and the request runs eager:

- `--enforce-eager`.
- A paged diffusion KV mode (`diffusion_kv_mode` other than `dense_legacy`).
- `diffusion_kv_cache_skip_steps` together with a quantized
  `diffusion_kv_cache_dtype`: KV quantization is switched on and off by step
  index. Skip steps without a quantized KV dtype have no effect and do not
  block graphs.
- `TRTLLM_ATTN` with `skip_softmax.disabled_until_timestep > 0`: attention
  changes with the timestep.
- A cache backend (`cache_backend` other than `none`): it decides per step
  whether the transformer runs.
- Any CPU offload strategy, including layerwise, distributed layerwise, and
  `diffusion_offload_config`.
- Pipeline parallelism.

Changing the active LoRA adapter, its scale, or its buffers drops the captured
graphs, and the next step recaptures.

## Shapes, prompts, and reuse

Graphs are keyed by the shapes of the step's inputs and its non-tensor
arguments. Each runner keeps at most two captured graphs and evicts the least
recently used one when a third shape arrives. Each new shape pays one capture on
its first step. If a capture fails on any rank, every rank turns graphs off for
that runner and continues eager.

The prompt length is part of the shape for Z-Image, which trims the prompt
embeddings to the token count. HunyuanImage3 keeps its prompt KV storage across
requests, but reallocates it, and drops its graphs, when a longer prompt
arrives. Workloads with many distinct prompt lengths therefore capture on
almost every request.

Measured on one Intel Arc Pro B70 with Z-Image-Turbo, 1024x1024, 8 steps:

| Workload            | Eager median | Graph median | Captures               |
|---------------------|--------------|--------------|------------------------|
| 8 distinct prompts  | 4.43 s       | 5.18 s       | 1 per request          |
| One prompt repeated | 4.45 s       | 4.44 s       | 1 on the first request |

Graphs pay off when a step is launch-bound (many small kernels relative to the
work per kernel) and the same shapes repeat across requests. Benchmark your own
workload with and without the option before enabling it in production.

## Memory

Captured graphs keep their memory pools allocated after the denoise loop,
including during VAE decode. Expect extra peak memory per captured shape and
leave headroom. For example, Wan2.2-T2V-A14B with TP4, 480x832, 17 frames peaked
at 32.6 GB per card with graphs against 31.6 GB without. At 33 frames the graph
run ran out of memory in VAE decode where the eager run fit.

## Related options

`enable_cuda_graph_decode` captures the KV-cache decode steps of Qwen-Image-2.1
and is independent of this option. When `diffusion_use_graph` is on, the
regional `torch.compile` path disables Inductor's own CUDA graphs so the two
graph layers do not stack.
