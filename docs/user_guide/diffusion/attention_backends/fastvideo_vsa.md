# FastVideo VSA

FastVideo Variable Sparse Attention (VSA) partitions the post-patch latent grid into spatiotemporal
blocks. For every query block, VSA scores the key/value blocks and computes
attention only against the selected top-k blocks.

## Supported models

| Model / checkpoint | Required adapter | Tasks | Sequence parallelism |
| --- | --- | --- | --- |
| `FastVideo/FastWan2.2-TI2V-5B-Diffusers` | None | T2V, I2V through `Wan22Pipeline` | Disabled |
| `MiniMaxAI/MiniMax-H3` | [FastH3 VSA (recipe)](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#fasth3-adapter) | T2VA | Disabled or pure Ulysses |
| Wan I2V-14B, S2V, VACE | — | Unsupported | — |

## Installation

Install vLLM-Omni from this checkout with the `vsa` extra:

```bash
uv pip install -e '.[vsa]'
```

The extra installs the tested kernel dependency automatically. Prebuilt kernels require
Linux, Python 3.12, and glibc 2.34 or newer (x86-64 or aarch64). The full
FastVideo framework and provider environment variables are not required.

`FASTVIDEO_VSA` selects the attention algorithm. H3 automatically selects the
tile64 provider using the input device, dtype, and installed kernels. BF16
inputs on SM120/SM121 use FlashInfer when its tile64 API is available;
otherwise H3 uses FastVideo. The `vsa` extra installs
`fastvideo-kernel==0.3.4` for that path. Wan's tile256 route keeps FastVideo.
The default precision remains BF16; auto selection never enables Sage itself.

## Enable the backend

For online serving, select the backend with the existing attention backend
flag. Use `--fastvideo-vsa-topk` to set the number of key/value blocks retained
for every query block:

```bash
vllm serve <model> --omni \
  --diffusion-attention-backend FASTVIDEO_VSA \
  --fastvideo-vsa-topk 64
```

<details markdown="1">
<summary>Alternative configuration formats</summary>

Equivalent structured configuration:

```bash
vllm serve <model> --omni \
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

</details>

## Choose top-k

H3 uses 64-token video blocks and keeps
all prefix blocks; see the [FastH3 VSA recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#fasth3-adapter).

### Wan top-k behavior

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

VSA is a CUDA-only, explicitly selected backend. The FastVideo provider requires
the `fastvideo-kernel` package and currently supports non-causal self-attention
with equal query and key/value sequence lengths. Unsupported shapes, masks,
dtypes, sequence-parallel execution, or kernel failures fall back to
`TORCH_SDPA` and emit a warning with the reason. On the Wan route, an active
sequence-parallel context is one of those fallbacks; the H3 route supports pure
Ulysses and rejects ring or all-gather sequence parallelism at startup.
H3 accelerator faults propagate instead of attempting dense recovery.

On the Wan route, check the startup and first-forward logs:

- `route=VSA` means top-k block selection is active.
- `route=VSA_ALL_BLOCKS` means the FastVideo DMD checkpoint retained all
  blocks through the VSA kernel.
- `route=SDPA` or `FASTVIDEO_VSA falling back to SDPA: ...` means dense SDPA
  executed; the warning includes the reason.

For H3, check `FastH3 adapter active` at startup and
`FASTVIDEO_VSA H3 routing` during DiT execution, as described in the
[model recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#fasth3-adapter).

The Wan route requires CUDA tensors in FP16 or BF16, 256-token blocks,
standard `head_size**-0.5` scaling, equal Q/K/V head counts, no attention mask,
and no active sequence-parallel context. The MiniMax-H3 route uses 64-token
`(4, 4, 4)` blocks and supports pure Ulysses. NPU and XPU paths do not execute
the FastVideo VSA CUDA kernel.

## Automatic provider selection

MiniMax-H3 defaults to `fastvideo_vsa_provider=auto`: supported BF16 inputs
use FlashInfer on SM120/SM121 when its kernels are installed, otherwise
FastVideo. Wan continues to use FastVideo. Existing video API requests do not
need to change.

For approximate Sage attention on SM120, select the precision explicitly:

```bash
--diffusion-attention-config '{"default":{"backend":"FASTVIDEO_VSA","fastvideo_vsa_precision":"sage","fastvideo_vsa_topk":64},"per_role":{"minimax_h3.token_refiner":{"backend":"TORCH_SDPA"}}}'
```

Sage uses QK INT8 / PV FP8 arithmetic and requires a FlashInfer build containing
[#5127](https://github.com/flashinfer-ai/flashinfer/pull/5127). The default
precision remains BF16. Unsupported Sage requests fail clearly.

The `vsa` extra supplies FastVideo; it does not install this optional FlashInfer
Sage API. Check the FlashInfer build in the serving environment before choosing
it explicitly:

```bash
python -c 'from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_fwd, bsa_attn_sm120_blk64_sage_fwd'
```

This checks API availability. Execution also requires the compatible GPU and
CUDA build described above. Provider and precision settings can be supplied in
either `diffusion_attention_config.default` or a `per_role` entry alongside
`backend: FASTVIDEO_VSA`. They do not change AdaLN weight placement; see
[H3 AdaLN offload](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#default-adaln-weight-offload)
for its separate SM120 default and overrides.

Set `fastvideo_vsa_provider` to `fastvideo` or `flashinfer` to choose a fixed
provider. Explicit FlashInfer selection requires compatible hardware and
installed kernels. Logs show the selected provider and precision. GPU testing
covers SM120; SM121 BF16 support follows the upstream kernel requirements.
