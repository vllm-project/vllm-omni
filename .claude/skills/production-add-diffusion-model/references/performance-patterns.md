# Performance Patterns and Code Examples

## Contents

1. [Measurement loop](#measurement-loop)
2. [Faster BF16 attention](#faster-bf16-attention)
3. [Sparse attention](#sparse-attention)
4. [Fusion-first operator patterns](#fusion-first-operator-patterns)
5. [Remove redundant computation](#remove-redundant-computation)
6. [USP communication optimization](#usp-communication-optimization)
7. [Text encoder and VAE disaggregation](#text-encoder-and-vae-disaggregation)
8. [Performance acceptance](#performance-acceptance)

## Measurement loop

Do not start with a list of optimizations. Start with a fixed correctness and
performance workload:

```text
model/checkpoint revision + prompt/media hashes + seed + dimensions/frames +
scheduler/sigma/steps/guidance + task + dtype + backend + topology + hardware
```

Capture:

- stage time: load, encode, conditioning, denoise, VAE decode/postprocess;
- GPU trace: kernel duration/count, launch gaps, allocations, H2D, collectives;
- CPU trace: preprocessing, per-layer sync, Python dispatch, temporary files;
- memory: per-rank HBM peaks and process-tree host PSS;
- output: device preparation, D2H/IPC, payload bytes, codec wall/process CPU,
  event-loop wait, client materialization, and signed residual;
- serving: queue time, throughput, errors, request mix, and distribution samples.

For a fixed-work single-request A/B, run one explicit warmup policy and at
least three measured repetitions, keep every raw run, and report median or mean
with range. Do not derive p95/p99 from three samples. For tail latency, declare
the arrival model and collect enough successful requests for the percentile and
confidence claimed. Implement one optimization at a time, re-run parity, and
report both local kernel and end-to-end deltas. A kernel win that moves no E2E
metric is not a production speed claim.

Use the pipeline profiler already exposed by the target revision when
available, and use PyTorch/CUDA/ROCm/NPU/XPU profiling tools appropriate to the
platform. Do not force every platform through a CUDA-only workflow.

## Faster BF16 attention

### Shared attention wiring

Use the shared role-aware attention layer and give it the true local head
counts and QKV layout:

```python
import torch

from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm_omni.diffusion.attention.layer import Attention

class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        head_dim,
        num_heads,
        num_kv_heads,
        quant_config=None,
        prefix="",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.attention = Attention(
            num_heads=self.qkv_proj.num_heads,
            num_kv_heads=self.qkv_proj.num_kv_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
            qkv_layout="BSND",
            role="self",
            prefix=prefix,
        )
        self.out_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(self, x, attn_metadata=None):
        qkv, _ = self.qkv_proj(x)
        q_size = self.qkv_proj.num_heads * self.head_dim
        kv_size = self.qkv_proj.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.unflatten(-1, (self.qkv_proj.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.qkv_proj.num_kv_heads, self.head_dim))
        v = v.unflatten(-1, (self.qkv_proj.num_kv_heads, self.head_dim))
        out = self.attention(q, k, v, attn_metadata)
        out = out.flatten(-2)
        out, _ = self.out_proj(out)
        return out
```

Add Q/K norm and RoPE in the official order before the `Attention` call. Keep
checkpoint mapping for the fused QKV and output projection complete.

### Backend qualification

For each role (`self`, cross-attention category, token refiner, etc.), record:

- actual selected backend from logs/trace;
- supported dtype, head size, QKV layout, mask/metadata, packed varlen behavior;
- platform and compute capability;
- TP/SP/Ring/AllGather-KV restrictions;
- fallback behavior and whether fallback changes the claimed performance.

Use unequal packed samples and compare with dense SDPA. Verify `cu_seqlens`,
max sequence length, padding exclusion, modality boundaries, CFG ownership, and
output slicing. Do not pass an `attn_mask` and assume it is consumed. Inspect
the backend contract and trace.

TRTLLM, Flash Attention, cuDNN, NPU varlen, and other backends are independent
qualification rows. The shared `Attention` class does not make every backend
packed-safe or faster automatically. Ring and hybrid paths require their own
boundary parity. AllGather-KV and specific backend combinations may fail fast;
keep that behavior visible in docs.

## Sparse attention

Sparse attention is a backend/model/platform optimization, not a synonym for
low-precision dense attention. Enable it only after dense BF16 parity.

For every candidate task/shape/topology:

1. State the sparsity policy, selected backend/kernel, block/head dimensions,
   supported hardware, external kernel/package revision, and fallback.
2. Prove the sparse kernel executed and report realized sparsity; an enabled
   flag is insufficient.
3. Compare same-seed per-block outputs, denoise trajectory, and final artifact
   with dense attention.
4. Add modality-specific metrics: temporal consistency and audio/video sync for
   video, spectral/audio quality for audio, and prompt/image metrics as relevant.
5. Report attention time and full E2E time. Include preprocessing/index-build
   overhead, peak memory, and the exact dense control.
6. Validate each task and shape. Do not extrapolate a T2V result to I2V/Ref2V,
   or NPU sparse support to CUDA/ROCm/XPU.

If Ring bypasses the selected backend or a sparse setting would be silently
ignored, reject the combination. Prefer a visible dense fallback only when the
recipe labels the row `limited` and does not claim sparse speed.

## Fusion-first operator patterns

Prefer current shared/custom operators with explicit guards and native
fallbacks. Prove parity at operator, block, trajectory, and artifact levels,
then use a trace to prove fewer launches/bytes or lower latency.

### Q/K RMSNorm + RoPE

```python
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk

self.norm_q = RMSNorm(head_dim, eps=eps)
self.norm_k = RMSNorm(head_dim, eps=eps)
# Example only: set these booleans from the official checkpoint contract.
use_neox_layout = False
cos_sin_are_half_dim = True
self.rope = RotaryEmbedding(
    is_neox_style=use_neox_layout,
    half_head_dim=cos_sin_are_half_dim,
)

q = self.norm_q(q)
k = self.norm_k(k)
q, k = apply_rope_to_qk(self.rope, q, k, (cos, sin))
```

Replace the illustrative booleans with the official layout in real code.
Preserve partial rotary dimensions by splitting the rotated prefix and concatenating
the pass-through suffix. If 3D cos/sin includes a batch axis, the current
shared path assumes the values are shared across the batch; per-sample
different positions require a proven representation/path.

For packed `[tokens, heads, head_dim]` tensors with a non-interleaved
`[cos | sin]` table, the current shared fused boundary is:

```python
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

q, k = fused_qk_norm_rope(
    q,
    k,
    self.norm_q.weight,
    self.norm_k.weight,
    rope_table,
    self.norm_q.variance_epsilon,
)
```

Its CUDA fast path currently specializes BF16 `head_dim=128` and
`rotary_dim=96`; other valid inputs take the eager implementation. The separate
shared RMSNorm and RotaryEmbedding still provide the general cross-platform
baseline. Treat NPU/MUSA and other vendor fusions as independent qualification
rows, and do not generalize a model-specific Triton kernel without compatible
shape, layout, frequency partition, dtype, and platform evidence.

### Fused QKV projection

Use `QKVParallelLinear` as shown above. Qualification includes:

- `num_heads` and `num_kv_heads` divisibility by TP;
- local split sizes from the layer, never reusing global counts after TP;
- Q/K/V checkpoint order conversion before the vLLM loader;
- all Q, K, and V shards present;
- quantization scale/loader mapping when FP8 or another method is enabled.

### Gate-up projection + fused SwiGLU

```python
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)

self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,
    output_sizes=[intermediate_size, intermediate_size],
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.gate_up_proj",
)
self.act_fn = SiluAndMul()
self.down_proj = RowParallelLinear(
    input_size=intermediate_size,
    output_size=hidden_size,
    bias=False,
    input_is_parallel=True,
    quant_config=quant_config,
    prefix=f"{prefix}.down_proj",
)

gate_up, _ = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x, _ = self.down_proj(x)
```

Map both checkpoint `gate_proj` and `up_proj` shards into the merged parameter
and require both. A merged projection followed by `F.silu(gate) * up` is not
the same claim as the `SiluAndMul` fused activation kernel; identify what the
trace actually uses.

### AdaLN

Use a shared layer only when its mathematical/return contract matches the
official model:

```python
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm

self.adaln = AdaLayerNorm(
    hidden_size,
    elementwise_affine=False,
    eps=1e-6,
)
x = self.adaln(x, scale[:, None, :], shift[:, None, :])
```

`AdaLayerNormZero`, `AdaLayerNormZeroSingle`, and continuous variants include
different projections, broadcasts, and tuple outputs; they are not drop-in
replacements. A shared AdaLN may have an NPU fast path while using native ops on
CUDA/HIP/XPU. Do not call it a universal fused AdaLN kernel.

For sigma schedules with a small finite set of timesteps, profile whether
precomputing the timestep/AdaLN projection once per request is worthwhile.
Cache keys must include the exact schedule/config/device/dtype and stay bounded.

### Layout, residual, and other candidates

After profiling, inspect:

- bias/dropout/add or gated residual chains;
- norm + linear or norm + modulation boundaries;
- repeated transpose/reshape/contiguous around attention;
- VAE normalization/activation/convolution boundaries;
- MoE routing and fused experts for models that actually use MoE.

For fused residual-plus-norm or modulation chains, preserve the unfused
materialization boundary. If the residual was stored in BF16 before RMSNorm,
round to BF16 and promote again inside the fused kernel; using the unrounded
FP32 temporary changes the denoising trajectory even when the stored residual
looks correct.

Keep DLO materialize/release hooks, process-group collectives, request cache
transitions, and cleanup outside compiled/fused regions unless lifecycle safety
is explicitly tested. Compile only stable regions and retain eager fallback.

## Remove redundant computation

### Hoist request invariants

Audit the denoise loop for work that depends only on the request rather than
the current timestep:

```python
# Before the denoise loop.
condition = self.condition_proj(text_embeddings)
refined_condition = self.token_refiner(condition, packed_metadata)
rope_tables = self.build_rope_tables(position_ids)
attn_metadata = self.build_attention_metadata(layout)

for step, sigma in enumerate(sigmas):
    timestep_state = self.prepare_timestep_state(sigma)
    noise = self.transformer(
        latents,
        refined_condition,
        rope_tables=rope_tables,
        attn_metadata=attn_metadata,
        timestep_state=timestep_state,
    )
    latents = self.scheduler_step(noise, sigma, latents)
```

Candidates include prompt conditioning, token-refiner output, reference
encoding/scans, position IDs/RoPE tables, packed masks/metadata, static CFG
branches, and finite-schedule AdaLN projections. First prove they are invariant
for all official tasks, LoRA/adapters, prompt-update features, shapes, and
schedules. In continuous batching, store request-specific results on request
state rather than the pipeline singleton.

### Eliminate per-layer synchronization

Avoid device-to-host scalar extraction such as `.item()` inside every layer:

```python
# Bad: forces a GPU/CPU synchronization in every attention layer.
max_len = int(cu_seqlens[-1].item())

# Better: compute validated host metadata once during request packing and pass it.
packed = PackedLayout(
    cu_seqlens=cu_seqlens,
    max_seqlen=max(sample_lengths),
)
for block in self.blocks:
    hidden = block(hidden, packed_layout=packed)
```

Trace to verify the synchronization disappears. Do not replace a correct
dynamic value with a stale constant merely to remove `.item()`.

### Remove unused masks and allocations

If packed attention consumes `cu_seqlens` and ignores a dense mask, do not
construct the dense mask. Reuse immutable request metadata and preallocated
buffers when safe. Prefer direct indexed scatter/gather over constructing large
temporary tensors only when duplicate-index semantics and gradients (normally
disabled for inference) remain correct.

Any cross-request cache needs a bounded lifetime and a complete key containing
shape, layout, device, dtype, schedule, task, adapter/LoRA, and any value that
changes the tensor. Otherwise keep it request-scoped.

### VAE and reference work

Profile duplicate VAE decode/postprocess, repeated reference decoding/scans,
unnecessary host-device round trips, and full-frame materialization. Apply VAE
tiling, slicing, patch parallelism, or on-demand staging only with seam,
ordering, color/range, and memory parity tests.

## USP communication optimization

“USP works” is not evidence that USP communication is optimized. For Ulysses,
Ring, hybrid, and AllGather-KV separately, profile:

- collective count, bytes, duration, stream, and overlap with compute;
- Q/K/V and output all-to-all boundaries;
- pack/unpack, padding, metadata broadcast, transpose, and contiguous copies;
- head divisibility or advanced uneven-head support;
- scaling efficiency from 1 to N devices on the same workload/interconnect;
- load balance for unequal packed samples.

Declare `_sp_plan` split/gather boundaries around meaningful module outputs,
not arbitrary Python statements. Keep packed metadata sharded/gathered in the
same ownership model as tokens.

When the target revision offers accelerated Ulysses transport such as
`--ulysses-a2a-permute`, qualify it separately from regular Ulysses. Prove the
strict scatter-heads/gather-sequence layout selected the fast path, record
one-time JIT/readiness cost, prewarm the maximum workspace shape before CUDA
graph capture, keep the grow-only workspace on one stream, and exercise cleanup
before process-group destruction. Compare collective and E2E deltas after the
dense backend and topology are frozen; keep unsupported layouts on an explicit
regular-Ulysses fallback.

For GQA, validate `Q_heads % KV_heads == 0`, pad KV heads first, and derive Q
padding from the original ratio. Exercise both contiguous and row-strided QKV
layouts; a fast copy path must preserve bitwise values and packed boundaries.

Cross-attention sometimes uses replicated text K/V while only video/audio Q is
sequence-sharded. In that specific case, `skip_sequence_parallel=True` can
avoid incorrect or wasteful Q/K/V redistribution:

```python
self.cross_attention = Attention(
    num_heads=local_heads,
    num_kv_heads=local_kv_heads,
    head_size=head_dim,
    softmax_scale=head_dim**-0.5,
    causal=False,
    role="cross",
    skip_sequence_parallel=True,
    prefix=f"{prefix}.cross_attn",
)
```

Use it only after proving K/V are replicated and Q/output ownership is correct.
Otherwise it disables needed communication. Report whether the optimization
reduces collective bytes/time and improves E2E scaling.

## Text encoder and VAE disaggregation

Disaggregation is a measured architecture decision. First collect:

| Question | Required measurement |
|---|---|
| Is the stage large enough? | Encode/decode share at target concurrency and shape |
| Is transfer economical? | Tensor/media bytes, serialization, connector latency and bandwidth |
| Is there reuse? | Prompt/reference reuse rate, cacheability, fan-out |
| Can it scale independently? | Queue saturation and resource utilization per stage |
| Is the contract stable? | Shape, dtype, mask/tag/position metadata, ordering |

### Text encoder candidate

Define an explicit tensor contract containing hidden states, masks, tags or
position IDs, dtype, shape, task/checkpoint revision, and request ID. Support
disabling the local encoder so the DiT cannot accidentally recompute it. Test
all tasks, prompt lengths, partial TP group membership, retries, backpressure,
abort, connector cleanup, and independent replica scaling.

### VAE candidate

Separate VAE encode and decode contracts if both exist. Define latent shape,
scale/shift, dtype/range, tiling/chunk order, output media ordering, and stage
metadata. Measure latent/media transfer against local decode cost. Test tiling
seams, RNG/determinism where relevant, chunk ordering, codec/postprocess,
backpressure, abort, and worker failure.

Do not label VAE tiling, patch parallelism, compile, or offload as VAE
disaggregation. Likewise, a separate text-encoder TP group is not a separate
stage. If no reusable generic stage/connector exists in the target revision,
produce a feasibility result or RFC and keep the capability `not tested` rather
than inventing a model-local production architecture.

## Performance acceptance

For each accepted optimization, require:

- same fixed workload and correctness artifacts before/after;
- actual fast-path execution proven by logs/trace/counters;
- local metric plus E2E latency/throughput and memory;
- at least three raw fixed-work repetitions after explicit warmup, or a
  declared arrival-load sample set large enough for every tail percentile;
- named hardware, backend, topology, dtype, and task scope;
- fallback/rejection tests for unsupported shapes/platforms;
- no regression in abort/error cleanup or long-run memory trend.

Update the feature matrix narrowly. Do not convert a profile-driven direction
into a generic “supported” feature merely because a model-specific PR exists.
