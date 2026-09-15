# FastVideo VSA

FastVideo Variable Sparse Attention (VSA) accelerates the `FastVideo/FastWan2.2-TI2V-5B-Diffusers` model's
self-attention by partitioning the post-patch latent grid into spatiotemporal
blocks. For every query block, VSA scores the key/value blocks and computes
attention only against the selected top-k blocks.

The supported checkpoint provides both text-to-video and image-to-video modes through `Wan22Pipeline`; the separate Wan I2V-14B, S2V, and VACE pipelines are outside this backend's supported scope.

MiniMax-H3 served with a FastH3 VSA adapter uses a second, H3-specific route
through the same backend. See the
[MiniMax-H3 recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3) for its geometry,
supported topologies, and commands.

VSA is a CUDA-only, explicitly selected backend. It requires the
`fastvideo-kernel` package and currently supports non-causal self-attention
with equal query and key/value sequence lengths. Unsupported shapes, masks,
dtypes, sequence-parallel execution, or kernel failures fall back to
`TORCH_SDPA` and emit a warning with the reason. On the Wan route, an active
sequence-parallel context is one of those fallbacks; the H3 route supports pure
Ulysses and rejects ring or all-gather sequence parallelism at startup.

## Enable the backend

For online serving, select the backend with the existing attention backend
flag. Use `--fastvideo-vsa-topk` to set the number of key/value blocks retained
for every query block:

```bash
vllm-omni serve <model> \
  --diffusion-attention-backend FASTVIDEO_VSA \
  --fastvideo-vsa-topk 64
```

The backwards-compatible environment variable selects the backend with the
default `topk=64`:

```bash
export DIFFUSION_ATTENTION_BACKEND=FASTVIDEO_VSA
vllm-omni serve <model>
```

To tune top-k, pass the CLI backend and top-k flags together as shown above,
or use the equivalent structured configuration:

```bash
vllm-omni serve <model> \
  --diffusion-attention-config \
  '{"default":{"backend":"FASTVIDEO_VSA","fastvideo_vsa_topk":64}}'
```

Do not combine `--diffusion-attention-backend` with an explicit
`diffusion_attention_config.default.backend`. The top-k value must be positive
and is valid only when the default backend is `FASTVIDEO_VSA`.

For a deploy YAML stage, use either the shorthand fields:

```yaml
stages:
  - stage_id: 0
    diffusion_attention_backend: FASTVIDEO_VSA
    fastvideo_vsa_topk: 64
```

or the structured configuration:

```yaml
stages:
  - stage_id: 0
    diffusion_attention_config:
      default:
        backend: FASTVIDEO_VSA
        fastvideo_vsa_topk: 64
```

## Choose top-k

At runtime the backend logs the sequence shape and derived block count:

```text
FASTVIDEO_VSA routing: seq_len=27280, dit_seq_shape=(31, 22, 40),
block_size=(4, 8, 8), num_blocks=120, topk=64, keep_ratio=53.3%,
checkpoint_mode=native, route=VSA
```

Use `num_blocks` as the upper bound when tuning top-k. Top-k is the number of
key/value blocks retained for each query block, not the number of tokens or
the total number of blocks processed by the layer.

- A smaller top-k increases sparsity and may reduce attention computation, but
  it can remove relevant blocks and reduce visual quality or temporal
  consistency. Routing and padding overhead can also make a smaller value
  slower for some shapes.
- A larger top-k retains more context and generally approaches dense-attention
  quality, but reduces the potential speedup and increases memory traffic.
- `topk > num_blocks` is invalid for the runtime shape and falls back to SDPA.
- For a native checkpoint, `topk == num_blocks` routes to SDPA because scoring
  every block provides no sparsity benefit.
- For a FastVideo DMD checkpoint, `topk == num_blocks` stays on the VSA
  all-block path so the checkpoint keeps its trained compensation semantics.

There is no universally optimal value. Resolution, frame count, GPU, kernel
version, and checkpoint all affect both quality and latency. Start from the
logged `num_blocks`, test several keep ratios on the target workload, and
compare output quality against the same checkpoint running dense attention.

## Checkpoint behavior

The backend does not expose a user-selectable gate mode. Wan checkpoints that
contain learned `to_gate_compress` weights use that projection automatically.
When those weights are absent, the unused projection is removed and VSA uses
the sparse branch without learned compensation.

FastVideo DMD checkpoints use their fixed distilled timestep schedule. Native
Wan checkpoints keep their normal scheduler and inference-step configuration;
selecting VSA does not turn a native checkpoint into a distilled model.

## Verify routing and fallback

Check the startup and first-forward logs instead of assuming that selecting
the backend guarantees sparse execution:

- `route=VSA` means top-k block selection is active.
- `route=VSA_ALL_BLOCKS` means the FastVideo DMD checkpoint retained all
  blocks through the VSA kernel.
- `route=SDPA` or `FASTVIDEO_VSA falling back to SDPA: ...` means dense SDPA
  executed; the warning includes the reason.

The Wan route requires CUDA tensors in FP16 or BF16, 256-token blocks,
standard `head_size**-0.5` scaling, equal Q/K/V head counts, no attention mask,
and no active sequence-parallel context. The MiniMax-H3 route uses 64-token
`(4, 4, 4)` blocks and supports pure Ulysses. NPU and XPU paths do not execute
the FastVideo VSA CUDA kernel.
