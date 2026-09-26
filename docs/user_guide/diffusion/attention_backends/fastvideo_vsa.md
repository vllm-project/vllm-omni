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

`FASTVIDEO_VSA` selects the attention algorithm. By default, the H3 integration
on SM120 executes FastVideo's 64-token Triton block-sparse kernel. The `vsa`
extra installs `fastvideo-kernel==0.3.4` for this execution path. FlashInfer is
an explicitly selected alternative described below.

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

VSA is a CUDA-only, explicitly selected backend. Its default provider requires
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

## FlashInfer tile64 provider

MiniMax-H3 can use FlashInfer for its model-owned VSA tile64 layout. Select the
provider and precision explicitly through the existing attention configuration:

```bash
--diffusion-attention-config '{"default":{"backend":"FASTVIDEO_VSA","fastvideo_vsa_provider":"flashinfer","fastvideo_vsa_precision":"sage"},"per_role":{"minimax_h3.token_refiner":{"backend":"TORCH_SDPA"}}}'
```

`bf16` selects the FlashInfer BF16 block-sparse kernel; `sage` selects QK INT8
and PV FP8 arithmetic. Sage changes numerical precision and is approximate.
Both precisions require BF16 inputs and head dimension 128. BF16 dispatch
accepts SM120/SM121, matching the upstream kernel contract; Sage requires
SM120. GPU qualification in this PR covers SM120 only; SM121 is not tested.
Sparse selection, prefix exemptions, tile edge sizes and compression-gate
correction remain owned by the H3 VSA implementation. RDMA is an independent
transport selection and is not enabled by this option.

The FlashInfer build must expose `bsa_attn_sm120_blk64_sage_fwd` and
`bsa_attn_sm120_blk64_fwd`. The Sage implementation was merged in FlashInfer
[#5127](https://github.com/flashinfer-ai/flashinfer/pull/5127).
Selecting FlashInfer does not require the FastVideo kernel package. Other VSA
layouts use the existing logged dense fallback; no FlashInfer tile256 path is
claimed. Missing provider APIs fail at construction.

## Reuse across models and hardware

The reusable operators live under `vllm_omni.diffusion.attention.ops`. A model
does not need to inherit the H3 implementation to use them:

| Component | Shared module | Reuse boundary |
| --- | --- | --- |
| 3-D partition, edge sizes, padding and inverse indices | `video_tiles.py` | Shared by H3 tile64 and Wan tile256; ordinary PyTorch operations with no CUDA kernel dependency |
| Pooling, prefix-aware top-k map and map-to-index conversion | `block_sparse.py` | Model-independent tensor operations; the caller chooses its routing policy and tile size |
| Explicit tile64 provider dispatch and FastVideo paired-CTA padding | `block_sparse.py` | Other models can pass their own sparse map; FastVideo hardware support is delegated to its existing CUDA providers |
| FlashInfer BF16/Sage ABI and capability checks | `flashinfer_block_sparse.py` | BF16 BSHD, head dimension 128, equal Q/K/V head counts; rectangular Q/K lengths and per-batch/head sparse layouts |
| Sage quantization and prepared-Q execution | `sage_quantization.py`, `sage_block_sparse_attention.py` | Reusable across models on SM120; the INT8/FP8 scale layout and V permutation are specific to this kernel ABI |

For an existing 64-token sparse layout, call the shared operator directly:

```python
from vllm_omni.diffusion.attention.ops.block_sparse import block_sparse_attn_bshd

output = block_sparse_attn_bshd(
    query, key, value,  # BF16 [batch, sequence, heads, 128]
    block_map,  # bool [batch, heads, ceil(Sq/64), ceil(Sk/64)]
    block_sizes,  # int32 [Kblocks], [B,Kblocks], [B,H,Kblocks], or None
    softmax_scale=128**-0.5,
    provider="flashinfer",
    precision="sage",  # or "bf16"
)
```

`block_sizes` counts valid tokens at the beginning of each key tile; inputs
with edge padding must provide these sizes. The caller restores original token
order and discards padded query rows. The shared operator does not select
blocks, add a causal mask or apply a learned compression gate.

H3's multimodal prefix segmentation, target-video layout and trained gate
remain in the model adapter. Wan already reuses the tiling helpers, but its
256-token sparse kernel and learned correction do not become a FlashInfer
tile64 integration automatically. Qwen-Image, Flux/Flux2 and HunyuanVideo 1.5
have transformer configurations with head dimension 128, making them
candidates for the shared FlashInfer operator. They still need model-specific
block selection, text/image or text/video masking, and quality validation;
this PR does not enable sparse attention for these models. Likewise, shared
PyTorch metadata helpers do not make the CUDA kernels usable on ROCm, NPU or
XPU. No FlashInfer SM80/SM90/SM100 execution or Sage SM121 support is claimed.
