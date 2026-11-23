# TRTLLM Attention

`TRTLLM_ATTN` runs FlashInfer's trtllm-gen FMHA kernels on datacenter
Blackwell GPUs. Selected on its own it computes dense BF16 attention, and its
dense performance is on par with
[FlashAttention-4](dense_backends.md#flashattention-4-on-blackwell). It also
provides two opt-in, lossy acceleration modes that can be enabled independently
or together:

| Mode | Config key | What it changes |
| --- | --- | --- |
| [Skip-Softmax](#skip-softmax) | `skip_softmax` | Skips the Softmax and `PV` work of KV tiles whose scores are too low to matter |
| [SAGE quantization](#sage-quantization) | `quant` | Runs `QK^T` in INT8 or FP8 and `PV` in FP8 instead of BF16 |

Both keys are set through `--diffusion-attention-config`, as JSON or as
vLLM-style dotted flags; the
[attention backend overview](../attention_backends.md#configuration) covers
both syntaxes and per-role resolution. The examples on this page use JSON.

Attention computes scores `S = QK^T`, probabilities `P = softmax(S)`, and the
output `O = PV`. SAGE lowers the precision of both matrix multiplications.
Skip-Softmax keeps `QK^T` dense and removes the Softmax and `PV` work for
tiles that `QK^T` shows to be unimportant. The two modes therefore compose:
SAGE makes every tile cheaper, and Skip-Softmax reduces the number of tiles
that reach the second half of the kernel.

## Requirements

`TRTLLM_ATTN` requires all of the following:

- an `sm100a` or `sm103a` GPU (B200, B300, GB200, GB300); workstation Blackwell
  (`sm120`/`sm121`) is not supported;
- `head_dim=128`;
- a FlashInfer build that exposes the trtllm-gen kernels (0.6.16rc1 or newer
  for SAGE);
- an attention path that is mask-free or provides packed-padding metadata.
  Structural suffix padding is expressed through that metadata rather than an
  `attn_mask` tensor.

An explicit selection that violates these requirements raises at startup
instead of silently falling back to another backend.

### Sequence parallelism

`TRTLLM_ATTN` runs under no sequence parallelism or under pure Ulysses.
Ulysses redistributes the sequence and attention heads around the attention
call, but the local computation still goes through the configured backend, so
both optional modes work unchanged. Ring and AllGather-KV do not:

- Ring runs its own distributed attention and bypasses the backend. Combining
  Ring with a `skip_softmax` key raises; a `quant` key would be silently
  ignored, so do not combine them either.
- AllGather-KV changes the Q/KV distribution and is rejected when
  `TRTLLM_ATTN` is selected.

Configure eight-way Ulysses alone as:

```bash
--usp 8 --ring 1 --allgather-degree 1
```

A degree of `1` disables that sequence-parallel mode. It does not limit the
server to one GPU or affect tensor, pipeline, or VAE parallelism.

## Quick start

On datacenter Blackwell the platform selects `TRTLLM_ATTN` by default when the
model declares a compatible path. To select it explicitly, for dense BF16:

```bash
vllm serve <model> --omni \
  --diffusion-attention-backend TRTLLM_ATTN
```

To enable both optimizations with a conservative Skip-Softmax setting and FP8
SAGE:

```bash
vllm serve <model> --omni \
  --diffusion-attention-config '{
    "default": {
      "backend": "TRTLLM_ATTN",
      "quant": {"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16},
      "skip_softmax": {"threshold": 0.05, "disabled_until_timestep": 0.97}
    }
  }'
```

The startup log reports the selection; look for one of:

```text
Defaulting to diffusion attention backend TRTLLM_ATTN (datacenter Blackwell ..., head_dim 128)
Resolved diffusion attention backend 'TRTLLM_ATTN' for role='self' via attention_config.default
```

## Skip-Softmax

Skip-Softmax (BLASST) skips the Softmax and `PV` work of KV tiles whose
scores fall far below the running row maximum. `QK^T` always runs, and how
many tiles qualify depends on the input. The
[feature design](../../../design/feature/skip_softmax.md) gives the skip test.

### Configuration keys

| Key | Range | Meaning |
| --- | --- | --- |
| `threshold` | `>= 0`; useful values in `(0, 1)` | Skip threshold, independent of sequence length. Calibration-free. |
| `target_sparsity` | `[0, 1]` | Requested operating point on the checkpoint's calibrated curve. Requires calibration metadata. |
| `disabled_until_timestep` | `[0, 1]`; default `0` | Keeps attention dense while the normalized timestep `t > D`; `0` disables the gate. |

`threshold` and `target_sparsity` are two ways to set the same kernel
threshold; setting both is a configuration error. Exactly one of them enables
Skip-Softmax.

### Direct threshold

Set `threshold` when the checkpoint carries no calibration. `threshold=0`
skips nothing; larger values skip more tiles and lower output fidelity. Values
around `0.05` are a reasonable first try for video DiTs; tune against dense
output on the same prompt and seed. The value is independent of sequence
length; see the
[feature design](../../../design/feature/skip_softmax.md#from-configuration-to-the-kernel-threshold)
for how it reaches the kernel.

```bash
vllm serve <model> --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{
    "threshold":0.05,"disabled_until_timestep":0.97}}}'
```

### Calibrated target sparsity

A fixed `threshold` does not produce a fixed fraction of skipped tiles because
the score distribution changes with the model, prompt, and shape.
[NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer)
can calibrate a per-model curve from sparsity to threshold and store it in the
checkpoint's transformer `config.json` under `sparse_attention_config`.
`target_sparsity` selects a point on that curve. The achieved sparsity still
varies per prompt, layer, and denoising step; the calibration makes the
requested value a meaningful target, not a guarantee.

What vLLM-Omni takes from the checkpoint:

- The curve coefficients. Only the `a * exp(b * target_sparsity)` form ModelOpt
  writes is supported; another formula is rejected at startup.
- The `ignore` list: attention layers matching those patterns stay dense.
- For multi-expert Diffusers checkpoints, `transformer_2/config.json` is read
  separately; if it is missing, `transformer_2` stays dense with a warning.

Checkpoint-level `target_sparsity` and `disabled_until_timestep` defaults are
not consumed; set them in the vLLM-Omni configuration. The
[feature design](../../../design/feature/skip_softmax.md#skip-softmax-calibration-config)
documents the checkpoint format and how the coefficients reach the kernel.

To calibrate a checkpoint yourself, follow the
[ModelOpt Skip-Softmax example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers/sparsity),
which exports a copy of the checkpoint with the coefficients embedded. The
fitted curve depends on the attention statistics of the calibration shape, so
calibrate at the resolution and frame count you will serve.

Requesting `target_sparsity` for a checkpoint without calibration is a startup
error that names the `threshold` alternative. The
[ModelOpt Wan2.2 FP8 checkpoint](https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-FP8/blob/main/transformer/config.json)
is a calibrated example:

```bash
vllm serve nvidia/Wan2.2-T2V-A14B-Diffusers-FP8 --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{
    "target_sparsity":0.75,"disabled_until_timestep":0.86}}}'
```

### Timestep gating

The early, high-noise denoising steps fix the global layout of the output, and
their errors propagate through every later step. `disabled_until_timestep`
keeps those steps dense and enables Skip-Softmax once the normalized timestep
`t` reaches the configured cutoff. The default `0` is a sentinel that turns the
gate off: Skip-Softmax runs on every step and no timestep needs to be
published. `1.0` also leaves no step dense, but goes through the gate and
therefore requires the pipeline to publish `t`.

`t` is the scheduler's own timestep normalized to `[0, 1]`, published by the
pipeline for each denoising step; for rectified-flow models it is the current
sigma. It runs from near `1.0` down to `0.0`, but not linearly in the step
index: flow-shifted schedules spend many of their steps at high `t`. The number
of dense steps a given cutoff produces therefore depends on the model's
schedule and step count, and follows from the actual timestep sequence:

```text
dense_steps = count(t[i] > disabled_until_timestep)
```

Derive the cutoff from the schedule of the model you are serving rather than
reusing another model's value; the
[feature design](../../../design/feature/skip_softmax.md#mapping-the-cutoff-to-denoising-steps)
works through an example.

A pipeline that does not publish `t` stays dense whenever
`disabled_until_timestep > 0` is set, and logs a warning once. Pipelines
publish it through `DenoiseProgressMixin.record_denoise_step`.

## SAGE quantization

SAGE quantization runs both attention matrix multiplications in low precision:
Q and K are quantized to INT8 or FP8 E4M3 for `QK^T`, and P and V use FP8 E4M3
for `PV`. P is quantized inside the FMHA kernel; V is quantized per channel
before the kernel call.
vLLM-Omni exposes the Q/K dtype and the Q/K scale granularity. The P and V
formats are fixed by the kernel, so the `quant` key has no V dtype for this
backend. This mode is distinct from the standalone
[SageAttention backends](sage.md), which use their own kernels.

FP8 Q/K kernels exist on `sm100a` and `sm103a`; INT8 Q/K kernels exist on
`sm100a` only.

| Key | Values | Meaning |
| --- | --- | --- |
| `dtype_qk` | `int8`, `fp8_e4m3` | Q/K quantization dtype. Setting it enables SAGE. |
| `q_block_size` | `1`, `4`, `16` | Consecutive query tokens sharing one Q scale; default `1` |
| `k_block_size` | `1`, `4`, `16` | Consecutive key tokens sharing one K scale; default `16` |

Start with the defaults, `q_block_size=1` and `k_block_size=16`, and try
smaller K blocks only if output quality does not meet expectations; smaller
blocks give finer scales at some cost in speed. When a KV sequence in a call is
shorter than `k_block_size`, that call falls back to dense attention and a
warning is logged once.

```bash
vllm serve <model> --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","quant":{
    "dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":16}}}'
```

The `quant` key is shared with `FLASHINFER_ATTN`, but each backend validates
its own fields: `float16`/`bfloat16` Q/K dtypes and `dtype_vo` are
`FLASHINFER_ATTN` options and are rejected here.

## Composing both modes

`skip_softmax` and `quant` may appear in the same `AttentionSpec`. Their
quality effects compound, so establish a dense baseline, enable one mode at a
time, and then evaluate the combination on the same prompts and seeds.

Modes configured in `default` apply to every attention role that has no
`per_role` entry. A `per_role` spec replaces the whole spec for that role and
does not inherit `quant` or `skip_softmax` from `default`, so
`{"backend":"TRTLLM_ATTN"}` is the way to keep a short or sensitive attention
site dense while the long DiT sequence uses both modes:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni \
  --diffusion-attention-config '{
    "default": {
      "backend": "TRTLLM_ATTN",
      "quant": {"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16},
      "skip_softmax": {"threshold": 0.05, "disabled_until_timestep": 0.97}
    },
    "per_role": {
      "minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}
    }
  }'
```

Role names are declared by each model; the
[attention backend overview](../attention_backends.md#configuration) covers
the resolution order and the equivalent Python API.

End-to-end speedup depends on the share of step time spent in attention, the
sequence length, the chosen Q/K precision, and the tile sparsity the input
actually yields. Benchmark the exact workload rather than extrapolating from
another model.
