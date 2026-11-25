# MiniMax-H3 SLA on Ascend

`SLA_ATTN` calls `mindiesd.layers.SparseLinearAttention` for MiniMax-H3's
main DiT blocks. MindIE-SD computes both the sparse attention branch and the
learned linear branch, including `proj_l`. The token refiner stays dense.
This integration does not train, merge checkpoints, or change the denoising
schedule. It does not enable QAT or W4A4F4.

## Prerequisites

Use an Ascend NPU environment with compatible vLLM-Ascend, torch-npu, CANN,
and MindIE-SD. The installed MindIE-SD must export
`mindiesd.layers.SparseLinearAttention` and include its SLA dependencies,
including triton-ascend. An older installation with only dense attention is
insufficient. Missing dependencies and unsupported shapes fail explicitly.

## Configuration

Add this configuration to the existing H3 serving command:

```bash
--diffusion-attention-config \
  '{"default":{"backend":"FLASH_ATTN"},"per_role":{"self":{"backend":"SLA_ATTN","sla":{"topk":0.05,"blkq":64,"blkk":64,"feature_map":"softmax"}},"minimax_h3.token_refiner":{"backend":"FLASH_ATTN"}}}'
```

Use per-role selection: selecting `SLA_ATTN` as the global backend also sends
the text refiner to SLA and is rejected. The backend never falls back to dense
attention when its provider fails. A first successful provider call logs
`Executed MindIE-SD SparseLinearAttention` with the configured top-k and block
sizes.

| Option | Default | Meaning |
| --- | --- | --- |
| `topk` | `0.05` | Fraction of K blocks retained, in `(0, 1]` |
| `blkq`, `blkk` | `64` | Query/key block sizes; each accepts `64` or `128` |
| `feature_map` | `softmax` | Linear branch feature map: `softmax`, `elu`, or `relu` |
| `zero_init` | `false` | Allow absent projection tensors and initialize them to zero for wiring checks |
| `precision` | `bf16` | Sparse branch precision: `bf16` or block-scaled E4M3 `fp8`; the linear correction stays BF16 |

`topk=0.05` requests nominal 95% sparsity for the sparse branch. Block rounding
changes the actual retained fraction, and the linear branch still runs. This
provider rounds the selected count down. Inputs that would select zero KV
blocks are rejected before the provider call. With `topk=0.05`, at least 20 KV
blocks are needed (for `blkk=64`, more than 1216 valid tokens).
This is MindIE-SD's global top-k policy; token-tag-specific dense protection or
reference-video routing from training code is not reproduced here. Match the
feature map and block sizes to the delivered checkpoint's inference contract.

## Projection weights

The transformer checkpoint must contain both tensors for every main block:

```text
blocks.<index>.attn.sla_module.proj_l.weight   [head_dim, head_dim]
blocks.<index>.attn.sla_module.proj_l.bias     [head_dim]
```

These are registered model parameters and move with their block during
offload. They are replicated across head-parallel ranks, rather than sharded
along the head dimension. Native projection names also work alongside a
Diffusers-format base. Package them with the transformer weights and update
the safetensors index if applicable; no separate projection file is loaded
automatically. Missing projection tensors fail by default. Supplying SLA
tensors with a dense backend also fails rather than silently skipping them.

To exercise SLA using an official base without projection tensors, add
`"zero_init":true` inside the `sla` block. This permits wiring validation;
it does not establish generation quality. Existing projection tensors still
load normally when present. A four-step request also requires the appropriate
four-step model weights and scheduler configuration independently of SLA.

## Supported execution

The initial path supports BF16, noncausal, square self-attention for one packed
request at a time, locally or with pure Ulysses (`ring_degree=1`). Structural
tail padding is removed before both SLA branches and restored as zero output
rows. Multiple requests use H3's existing per-request execution path.

Ring/AllGather-KV, attention schedules, reference isolation, reference-KV
reuse/compaction, VDN windows, arbitrary masks and quantized KV are unsupported.
Configuration or execution rejects these paths. This support statement does
not establish multi-card numerical parity or performance.

## Validation

CPU contract tests require the project's normal test environment, including
PyTorch and vLLM. The CPU provider is a test double; these tests verify H3
parameter registration/loading, projection effect, layout and padding, and
explicit failures:

```bash
pytest -q tests/diffusion/attention/test_sla_attn.py tests/diffusion/models/minimax_h3/test_minimax_h3_sla.py
pytest -q tests/diffusion/attention/test_sla_attn.py tests/diffusion/models/minimax_h3/test_minimax_h3_sla.py -m 'core_model and cpu' --run-level core_model
```

On one Ascend A5 NPU, this weight-free smoke test invokes the real provider, checks its
input layout, and changes the projection bias to verify that the learned
branch contributes to output. A second case exercises the H3 loader and shared
attention dispatch with real MindIE-SD; it substitutes only QKV/output projection
constructors to avoid requiring a TP process group:

```bash
pytest -q tests/diffusion/attention/test_sla_attn_npu.py -m 'core_model and npu and A5' --run-level core_model
```

Then validate a complete H3 request with the delivered transformer checkpoint,
check the provider execution log, and assess video/audio quality and measured
sparsity separately. Neither CPU tests nor zero projection smoke tests replace
that checkpoint acceptance run.

## FP8 sparse branch

Set `sla.precision` to `fp8` to quantize the sparse branch Q/K/V to
block-scaled E4M3 and execute MindIE-SD quantized BlockSparseAttention.
The existing block selection, feature maps, `proj_l` loading and linear
correction are retained. BF16 is still the default.

```json
{
  "default": {"backend": "FLASH_ATTN"},
  "per_role": {
    "self": {"backend": "SLA_ATTN", "sla": {"topk": 0.05, "precision": "fp8"}},
    "minimax_h3.token_refiner": {"backend": "FLASH_ATTN"}
  }
}
```

The NPU runtime must provide MindIE-SD BlockSparseAttention with FP8 and
dequantization-scale support; missing support raises an error. Attention
inputs and the linear correction remain BF16. Projection weights are still
required unless `zero_init=true` is explicitly selected for wiring checks.
This precision option does not change linear-layer weight quantization.
