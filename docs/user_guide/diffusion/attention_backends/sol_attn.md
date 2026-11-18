# Sol-Attn

`SOL_ATTN` integrates NVlabs' training-free sparse self-attention kernel.
It is an explicit opt-in and does not change the platform default.
This integration covers the attention kernel, not the full Sol-Engine stack.

## Installation and scope

Install the optional upstream package into the vLLM-Omni environment using
the validated revision (Sol-Attn 0.5.0):

```bash
pip install 'git+https://github.com/NVlabs/Sana.git@f4035e0a976d828ed41c7b0b799e9712b47cd60d#subdirectory=techniques/sparse_backends'
```

Requirements:

- NVIDIA CUDA SM80 or newer, BF16 activations, and `head_dim=128`.
- Noncausal self-attention with matching Q/K/V shapes and equal head counts.
- One real sequence per attention call. MiniMax-H3's structural suffix padding
  is supported through producer-owned `PackedPaddingMetadata`; multiple real
  packed documents are not supported.
- Local attention or pure Ulysses (`--ring 1`). Ring and AllGather-KV are rejected.
- No arbitrary masks, GQA/MQA, paged KV, quantized KV, or compact reference-KV
  attention with unequal Q/KV lengths.

Architecture dispatch is delegated to upstream `sol_attn`, which provides
CuTe DSL kernels for Hopper SM90 and Blackwell SM100/SM120. Install the
[upstream runtime dependencies](https://github.com/NVlabs/Sana/blob/f4035e0a976d828ed41c7b0b799e9712b47cd60d/techniques/sparse_backends/README.md#requirements)
for these specialized kernels; upstream falls back to Triton when the optional
CuTe runtime is unavailable. The default `kv_splits="auto"` matches upstream's
integration policy: four splits on SM90 with CuTe and at least 65,536 valid
tokens, one split otherwise. Explicit splits of 2 or 4 require the Hopper
CuTe path. The kernel API receives the resolved integer, not `"auto"`.

Full-model compilation and outer HSDP boundaries are outside this backend's
compilation guarantee.

## MiniMax-H3 configuration

For the legacy native SOL policy, add this configuration to the online server
command in the
[MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md):

```bash
--diffusion-attention-config '{
  "default": {
    "backend": "SOL_ATTN",
    "sol_attn": {
      "tau": 1.0,
      "dense_steps": 10,
      "dense_layers": "0,1",
      "prefix_sink": true,
      "kv_splits": 1
    }
  },
  "per_role": {
    "minimax_h3.token_refiner": {"backend": "TORCH_SDPA"}
  }
}'
```

`prefix_sink` derives the exact prefix from the model's video layout rather
than assuming a fixed number of audio/text tokens. For Ref2VA it conservatively
protects every row before the target-video span, including reference videos.
With the default native provider, the prefix is kept exact as K/V and
recomputed densely as queries. Larger reference prefixes increase dense work.

## Experimental SOL-BSA composition

Set `"provider": "cudnn"` or `"provider": "triton"` inside the existing
`sol_attn` block to execute SOL's selected blocks through the shared BF16
block-sparse provider. This remains **training-free SOL**, not VSA, and does
not require VSA-trained weights or enable their compression gate.

Both choices require the pinned SOL package above. The `cudnn` provider also
requires `vllm-omni[cudnn-bsa]` (`nvidia-cudnn-frontend>=1.31.0`); `triton`
requires `vllm-omni[vsa]` for its shared FastVideo kernel. Installing that
kernel dependency does not select the VSA algorithm. Unsupported hardware or
missing dependencies fail explicitly, including when dense guards are active.
Triton requires `softmax_scale=head_dim**-0.5`. Both BSA choices require
`kv_splits=1` or `auto`, which resolves to one on this path.

The composition includes the omitted-block centroid residual, weighted using
the BSA softmax normalization. It does not simply discard unselected blocks.
Sink blocks are protected as both queries and K/V in the same BSA call, avoiding
the native path's second dense-query pass. Query protection rounds outward to
64-token blocks, so its approximation boundary differs from the native path.
The default `sink_mode="prefix"` still includes reference videos. The explicit
`text_audio` mode below replaces that policy without changing the legacy default.

This is an **experimental correctness-first path**, with a chunked PyTorch
residual rather than a fused residual kernel. It is not yet performance- or
model-quality-validated; routing metadata still grows quadratically with the
number of blocks. Do not assume it is faster than native SOL or dense attention.
The default remains `provider="native"`. This option does not enable INT8 QK
or change PV precision.

### Text/audio-only protection, all denoising steps

This profile aligns the sink policy with
[upstream Sol-H3 Ref2VA](https://github.com/NVlabs/Sana/blob/cf67a3acad7511acb85053e56a05e5dd24c82ba0/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py).
It is available for T2VA, FL2VA, and Ref2VA with native checkpoints. The online
examples below use `num_inference_steps=50`, without a distilled adapter.

All 50 steps and all main-DiT layers dispatch SOL-BSA; the text token refiner
remains dense. No four-step LoRA, quantized transport, or timestep reduction is
enabled. SP4 and AdaLN reuse settings are independent of attention routing.

H3 builds a request-local `TextAudioBlockMask` from the actual packed text and
audio positions, including reference audio and target audio. It is reused across
steps and layers, and remains in global sequence coordinates under Ulysses.
Q and KV share a **64-token** block size. Any block containing text/audio is
protected along both axes in one BSA pass; visual reference blocks are otherwise
eligible for sparse routing. Mixed boundary blocks also protect their visual
rows, and the ragged final block excludes padding from attention.

`text_audio` overrides `prefix_sink`, rejects explicit sink ranges and the
`native` provider, and requires matching producer-owned metadata even during
dense guard steps. Missing metadata never silently reverts to prefix protection.
The same policy is available with `provider="triton"`. Disabling guards does
not make every block approximate: protected, neighboring, and route-selected
blocks remain exact.

## MiniMax-H3 online serving

With the dependencies above installed, start T2VA or FL2VA:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni \
  --task-type fl2va \
  --enforce-eager --vae-use-tiling \
  --diffusion-attention-config '{
    "default": {"backend": "CUDNN_ATTN"},
    "per_role": {
      "minimax_h3.main": {
        "backend": "SOL_ATTN",
        "sol_attn": {"provider": "cudnn", "sink_mode": "text_audio", "dense_steps": 0, "dense_layers": []}
      },
      "minimax_h3.token_refiner": {"backend": "TORCH_SDPA"}
    }
  }'
```

GPU selection and parallelism are deployment choices, not SOL requirements.
Without parallelism options this uses one GPU; multi-GPU deployments must use
supported parallel degrees and provide sufficient memory for the model.

- **T2VA / FL2VA:** use the command above; set request `extra_params.task` to
  `t2va` or `fl2va` respectively.
- **Ref2VA:** replace `/FL2VA` with `/Ref2VA` and `--task-type fl2va` with
  `--task-type ref2va`; set request `extra_params.task` to `ref2va`.
- Set request `num_inference_steps=50`. This profile uses block64, protects
  text/audio blocks, and enables SOL-BSA for every main-DiT layer and step.
- Exact AdaLN runtime reuse is already on by default; no cache flag is needed.
  This is not whole-schedule precomputation before the request.

Submit requests through the
[online video API](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md#http-api-examples).
Kernel/attention-forward checks do not yet validate HTTP serving, full-model
quality, or speedup for this experimental profile.

## Other models

The backend is not tied to MiniMax-H3. Other models can opt in when their
attention calls meet the geometry, dtype and metadata requirements above.
Wan exposes separate `self` and `cross` roles, so select SOL only for
self-attention and leave cross-attention dense:

```bash
--diffusion-attention-config '{
  "default": {"backend": "TORCH_SDPA"},
  "per_role": {
    "self": {
      "backend": "SOL_ATTN",
      "sol_attn": {"dense_steps": 10, "dense_layers": "0,1"}
    }
  }
}'
```

Do not apply SOL globally to models that also use cross-attention, masks,
GQA, or incompatible head dimensions. Models must publish denoising-step
metadata and parseable layer prefixes for the default guards to permit sparse
execution.

## Controls

Set these fields in `AttentionSpec.sol_attn`, under `default` or `per_role`:

| Key | Default | Meaning |
| --- | --- | --- |
| `provider` | `native` | Native fused SOL, or experimental `cudnn`/`triton` BF16 SOL-BSA composition. |
| `tau` | `1.0` | Finite nonnegative routing threshold multiplier; higher values generally select fewer exact blocks. |
| `thresh_type` | `diag` | Threshold estimator: `diag` ignores cross-dimension covariance; `exact` includes it. Neither changes attention precision or means fully dense attention. |
| `dense_steps` | `10` | First N zero-based denoising steps stay dense. Missing step metadata stays dense when N is positive. |
| `dense_layers` | `"0,1"` | Nonnegative layer index, list, or range string such as `"0-3,8"`; empty/null disables it. Unidentified layers stay dense when this guard is enabled. |
| `prefix_sink` | `true` | Derive a protected prefix from `video_layout` when available; without layout, use the explicit sink settings. |
| `sink_mode` | `prefix` | Legacy prefix/range policy, or `text_audio`: BSA-only block64 protection from model-owned text/audio positions; overrides `prefix_sink` and rejects explicit sink ranges. |
| `sink_tokens` | `0` | Additional exact K/V and dense-query sink length, clamped to valid rows. BSA rounds both axes outward to whole tiles. |
| `sink_start` | `0` | Sink offset; `null` selects the last `sink_tokens` valid rows. With `prefix_sink`, the union is conservatively covered by one contiguous range. |
| `kv_splits` | `auto` | Native: `auto` selects 4 on SM90 with CuTe and at least 65,536 valid tokens, otherwise 1. Explicit `2/4` require SM90, CuTe DSL, and at least one 4096-token route group per split. BSA: only `1` or `auto` (one split). |

`SOL_ATTN` without a `sol_attn` block uses these defaults. Schedules with at
most ten denoising steps remain entirely dense with `dense_steps=10`.

Sparse attention is approximate, not dense-equivalent. Validate video and audio
quality against a matching dense baseline before reducing the guards or
deploying a configuration. Dense guards use PyTorch SDPA (cuDNN on SM120/SM121),
which can also differ numerically from `FLASH_ATTN`.
