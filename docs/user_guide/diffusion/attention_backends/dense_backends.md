# Dense Attention Backends

Use these backends to run dense attention. Optional quantized or sparse modes
remain inactive unless they are explicitly configured. Start with
`TORCH_SDPA` for correctness comparisons, then benchmark the fastest compatible
kernel for your model, shape, and hardware.

For selection precedence and per-role configuration, see the
[attention backend overview](../attention_backends.md).

## `TORCH_SDPA`

`TORCH_SDPA` calls PyTorch `scaled_dot_product_attention` and lets PyTorch's
dispatcher choose the implementation. It is always available and is the most
conservative reference when validating another backend.

```bash
vllm-omni serve <model> --diffusion-attention-backend TORCH_SDPA
```

## `FLASH_ATTN`

`FLASH_ATTN` uses the installed FlashAttention implementation. On Blackwell it
is FA4 only (`flash_attn.cute` / `vllm-omni[fa4]`). If FA4 is unavailable,
explicit `FLASH_ATTN` is rejected and automatic selection continues to another
compatible Blackwell backend. Hopper-only FA2/FA3 wheels are not used, even if
they import. On Hopper, Ada, and Ampere it is the preferred automatic route
when a compatible package is installed.

```bash
vllm-omni serve <model> --diffusion-attention-backend FLASH_ATTN
```

### FlashAttention 4 on Blackwell

Install the optional CUDA 13 extra:

```bash
pip install 'vllm-omni[fa4]'
```

Version `4.0.0b18` is required; earlier beta wheels had known JIT failures on
Blackwell. On Blackwell, `FLASH_ATTN` is FA4 only: Hopper-only FA2/FA3
wheels are not used, even if they import. If FA4 is missing, explicit
`FLASH_ATTN` is rejected and automatic selection continues to other
Blackwell backends.

## `TRTLLM_ATTN`

`TRTLLM_ATTN` runs FlashInfer's trtllm-gen FMHA kernels and is the platform
default on datacenter Blackwell for models that declare a compatible attention
path. Selected without a `quant` or `skip_softmax` block, it runs dense BF16
attention at FA4-level performance.

```bash
vllm serve <model> --omni \
  --diffusion-attention-backend TRTLLM_ATTN
```

See [TRTLLM Attention](trtllm.md) for its requirements and for the optional
SAGE quantization and Skip-Softmax modes.

## `CUDNN_ATTN`

`CUDNN_ATTN` pins PyTorch SDPA to `CUDNN_ATTENTION`. It is particularly useful
for mask-heavy DiTs and is automatically preferred on Blackwell when cuDNN
9.5 or newer is available and the higher-priority TRTLLM route is not
compatible.

```bash
vllm-omni serve <model> --diffusion-attention-backend CUDNN_ATTN
```

### LTX-2.0 limitation

LTX-2 audio attention has a symbolic head dimension during `torch.compile`
tracing. The cuDNN SDPA selector rejects that symbolic dimension and Dynamo
aborts compilation. This is tracked in
[issue #3121](https://github.com/vllm-project/vllm-omni/issues/3121).

Use `FLASHINFER_ATTN` or `TORCH_SDPA` as a workaround:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASHINFER_ATTN \
  python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2 ...
```

## `FLASHINFER_ATTN`

`FLASHINFER_ATTN` uses FlashInfer's batch-prefill wrapper. It is an explicit
option on CUDA platforms and an automatic Blackwell fallback when FlashInfer
is installed but cuDNN is too old for `CUDNN_ATTN`.

On Blackwell, `auto` resolves to FlashInfer cute-dsl, which cannot run a
nontrivial custom mask. Automatic selection may fall back to SDPA for those
masks. An explicit `FLASHINFER_ATTN` selection does not: sequence-parallel
auto-padding that would create a padding mask is rejected at capability
preflight. Use `TORCH_SDPA`, pin `quant.flashinfer_backend` to `fa2`/`fa3`
where those kernels exist, or choose a mask-capable backend.

```bash
vllm-omni serve <model> --diffusion-attention-backend FLASHINFER_ATTN
```

### FlashInfer quantized attention

The backend accepts an `AttentionSpec.quant` block. For QK16/V8, keep Q and K
in FP16 or BF16 and use FP8 E4M3 for V:

```python
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    AttnQuantSpec,
    OmniDiffusionConfig,
)

config = OmniDiffusionConfig(
    diffusion_attention_config=AttentionConfig(
        default=AttentionSpec(
            backend="FLASHINFER_ATTN",
            quant=AttnQuantSpec(
                dtype_qk="bfloat16",
                dtype_vo="fp8_e4m3",
            ),
        ),
    ),
    ...,
)
```

`dtype_qk` controls Q and K; `dtype_vo` controls V. Mixed-dtype
configurations require FlashInfer 0.6.16rc1 or newer. The shared quantization
schema is also consumed by TRTLLM, but each backend validates its own allowed
fields and values; see [TRTLLM SAGE quantization](trtllm.md#sage-quantization).

### SM120 FP8 Skip-Softmax

FlashInfer builds containing [PR #4859](https://github.com/flashinfer-ai/flashinfer/pull/4859)
support opt-in Skip-Softmax on SM120 (compute capability 12.0). Select the
`cute-dsl-prims` variant and explicitly enable FP8 Q/K/V:

```bash
vllm-omni serve <model> --diffusion-attention-config '{
  "per_role": {
    "self": {
      "backend": "FLASHINFER_ATTN",
      "quant": {
        "flashinfer_backend": "cute-dsl-prims",
        "dtype_qk": "fp8_e4m3",
        "dtype_vo": "fp8_e4m3"
      },
      "skip_softmax": {"threshold": 0.0001}
    }
  }
}'
```

The example threshold is a starting point for validation, not a model quality
guarantee. `per_role.self` selects Wan 2.2 self-attention and MiniMax H3's
packed multimodal attention. H3's token refiner also inherits this category;
an explicit `per_role["minimax_h3.token_refiner"]` entry can select a separate
backend for it. Wan cross-attention keeps its usual backend.

This path calls `sm120_fmha_fp8_ragged_prefill` directly, following the
[reviewer's integration guidance](https://github.com/flashinfer-ai/flashinfer/pull/4859#issuecomment-5536049203).
`threshold` is the final e-based value: no multiplication or division by
sequence length is applied. Omitting `skip_softmax` selects the dense FP8
kernel. Setting `threshold` to zero selects the skip-enabled kernel but skips
no tiles, providing a useful control for measuring the skip-test overhead.

Inputs are FP16/BF16 tensors in `(batch, sequence, heads, head_dim)` layout.
They are cast directly to FP8 E4M3, without per-block scaling; outputs keep the
query dtype. FP8 quantization is approximate even with skipping disabled, and
inputs must be representable in E4M3's finite range. Validate FP8 against the
original model first, then sweep skip thresholds against the dense FP8 result.
Supported head dimensions are 64, 128 and 256, including GQA and bottom-right
causal attention. Custom masks, ring sequence parallelism and `target_sparsity`
are rejected. Ulysses can use the backend when
the attention call does not require a padding mask.

For eager execution, `skip_softmax` can include
`{"threshold": 0.5, "disabled_until_timestep": 0.94}`. The backend uses dense
FP8 while the pipeline's normalized `denoise_timestep` is greater than 0.94,
then enables threshold 0.5 at or below 0.94. This is a noise-timestep cutoff,
not a fraction of denoising steps. Missing or non-finite progress keeps the
operation dense and logs a warning. The gate also applies to runtime threshold
overrides. Set `enforce_eager=True`: CUDA Graph capture with this host-side
gate is rejected because replay does not re-evaluate Python progress.
Ungated caller-owned threshold tensors retain CUDA Graph support.

MiniMax H3's packed inputs are supported through contiguous CUDA int32
`cu_seqlens_q`/`cu_seqlens_k` and a Python integer `max_seqlen_q` in metadata.
The physical batch dimension is one; offsets define the logical requests and
must be monotonic, start at zero, and end at the corresponding token count.
The caller maintains those invariants and the launch bound. Single-request
`packed_padding` excludes alignment rows from attention and zeros their output;
multiple requests use separate ragged segments, including any explicit padding
segment. This variant advertises packed capabilities only when selected
explicitly; other FlashInfer variants retain their existing behavior.

For runtime per-request thresholds, callers may supply
`AttentionMetadata(extra={"skip_softmax_threshold": thresholds})`. This
overrides the configured scalar, including when the override is `None`.
`thresholds` must be a contiguous CUDA float32 tensor with shape `[batch_size]`
on the Q device. Its values must be finite and non-negative; callers ensure
this without a device-to-host check in the attention path. The caller must
keep the tensor alive at the same address and update it in place between
CUDA Graph replays. A captured Python float is static. Switching between
`None` and a supplied threshold requires a different graph; writing zeros to
the existing tensor disables skipping within the captured skip-enabled graph.
For packed inputs, `batch_size` is the number of launched segments, not the
physical tensor batch: `cu_seqlens_q.numel() - 1` (one with `packed_padding`).
An explicit padding segment in a multi-request batch needs its own threshold
entry. Packed offsets must also retain their addresses through graph replay.

Run the integration checks and a synthetic threshold sweep with:

```bash
python -m pytest tests/diffusion/attention/test_flashinfer_attn.py \
  tests/diffusion/attention/test_flashinfer_sm120.py -v
python benchmarks/kernels/benchmark_flashinfer_skip_softmax.py \
  --lengths 4096 16384 --heads 56 --head-dim 128 --output skip-softmax.json
```

The 56-head shape matches MiniMax H3. For Wan 2.2 A14B use `--heads 40`;
for TI2V-5B use `--heads 24`. All three use head dimension 128. These runs use
synthetic activations; they do not replace checkpoint generation and quality
comparisons at the intended resolution, frame count, and denoising schedule.

The benchmark includes Q/K/V conversion in its CUDA Graph timings, compares
skipping with dense FP8, and reports error against an FP32 reference. Random
inputs measure little/no-skip overhead; controlled inputs demonstrate potential
speedup. Neither case substitutes for end-to-end model quality evaluation.
