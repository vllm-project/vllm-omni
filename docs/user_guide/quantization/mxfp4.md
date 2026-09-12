# W4A4 MXFP4 Quantization

## Overview

W4A4 MXFP4 (Microscaling FP4) quantizes both weights and activations to FP4
(`float4_e2m1fn_x2`, packed 2 values per byte) using the OCP MX format: groups
of 32 K-dimension elements share a single `float8_e8m0fnu` exponent scale.

vLLM-Omni provides two quantization methods with different scale structures:

| Method | Scale structure | Mode | Use case |
| -------- | ---------------- | ------ | ---------- |
| `mxfp4` | Single-scale (per-32 fine only) | Online + pre-quantized single-scale weights | Online baseline, or an existing single-scale checkpoint in the layout below |
| `mxfp4_dualscale` | Dual-scale (fine per-32 + coarse per-512 + per-channel `mul_scale`) | Online + Offline | Production; better accuracy; offline recommended |

!!! tip "Existing W4A4-only DualScale deployments"
    For W4A4-only deployments, the existing `mxfp4_dualscale` offline mode uses a
    pre-quantized checkpoint produced by msModelSlim. Offline checkpoints load
    calibrated `mul_scale` tensors from disk, providing measurably better accuracy
    than any online method. The one-time preprocessing cost amortises across all
    subsequent inference runs.

    W4A8 layer/step fallback requires single-scale `mxfp4`, online or offline.
    DualScale supports W4A4 and BF16 routing only.

!!! warning "Online single-scale ≠ Offline dual-scale"
    `mxfp4_dualscale` offline mode uses `NPUMxfp4DualScaleLinearMethod`:
    fine scale (per-32 K), coarse scale (per-512 K), and per-input-channel
    `mul_scale` from calibration — all loaded from the checkpoint.
    `mxfp4_dualscale` online mode uses `NPUMxfp4DualScaleOnlineLinearMethod`:
    dual-level scales computed on the fly from BF16 weights; no calibration
    `mul_scale` is available. Loading an offline checkpoint with the online
    method (or vice versa) will produce incorrect results or shape errors.

## Hardware Support

| Device | Support |
| -------- | --------- |
| NVIDIA Blackwell GPU (SM 100+) | ⭕ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ⭕ |
| NVIDIA Ampere GPU (SM 80+) | ⭕ |
| AMD ROCm (gfx950 / MI355X) | ✅ |
| Intel XPU | ⭕ |
| Ascend NPU (Atlas 950 A5) | ✅ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this guide.

## Model Type Support

### Diffusion Model (Wan2.2)

| Model | Online | Offline | Notes |
| ------- | -------- | --------- | ------- |
| Wan2.2-T2V-A14B | `mxfp4` / `mxfp4_dualscale` | `mxfp4_dualscale` | MoE cascade (`transformer` + `transformer_2`); both transformers quantized with the same config |
| Wan2.2-I2V-A14B | `mxfp4` / `mxfp4_dualscale` | `mxfp4_dualscale` | MoE cascade; same scheme as T2V-A14B |
| Wan2.2-TI2V-5B | ❌ | ❌ | Parameter count too small; W4A4 causes unacceptable accuracy loss |

The choice between `mxfp4` and `mxfp4_dualscale` in **online mode** is about
quantization quality, not model compatibility — both work on cascade (A14B) and
single-transformer models alike, the same as `mxfp8` online:

- `mxfp4`: single-scale, lower overhead, simpler compute; online and pre-quantized weights
- `mxfp4_dualscale`: dual-scale + optional BF16 fallback, better accuracy, online **and** offline

The existing `merge_mxfp4_dualscale_checkpoint.py` produces **offline
`mxfp4_dualscale`** checkpoints. Single-scale weights must satisfy the separate
layout described below; changing a DualScale checkpoint's method name to
`mxfp4` does not convert its weights. The single-scale importer
`merge_mxfp4_checkpoint.py` uses single-scale exports, not DualScale
reconstruction. Real exported checkpoint compatibility and model quality
require validation of that checkpoint; the synthetic tests do not establish
C7 export or video acceptance.

!!! note "Per-layer BF16 fallback in offline cascade models"
    The A14B offline checkpoint uses `quant_method: mxfp4_dualscale`. Most
    linear layers are stored as W4A4 MXFP4 DualScale; precision-sensitive layers
    retain their original BF16 weights and are listed in `ignored_layers` inside
    each transformer's `config.json`. The two transformers may have different
    `ignored_layers` sets — the pipeline reads each transformer's own `config.json`
    and rebuilds the config locally when they differ, so routing is always
    per-transformer-accurate.

!!! warning "TI2V-5B not supported"
    Wan2.2-TI2V-5B is excluded from W4A4 quantization. Its smaller parameter
    count makes it significantly more sensitive to 4-bit quantization noise,
    resulting in unacceptable accuracy loss. Use [MXFP8](mxfp8.md) for TI2V-5B.

## Configuration

### W4A8 fallback at selected layers and denoising steps (Ascend NPU)

For single-scale `mxfp4`, set `w4a8_fallback_steps` to a list of zero-based
request denoising loop indices (not scheduler timestep values).
Set `w4a8_fallback_layers` to exact runtime Linear paths, for example
`["blocks.10.attn1.to_qkv", "blocks.12.ffn.net_2"]`. A selected layer uses W4A8
at every step; at a selected step, all other quantized layers also use W4A8.
The rules are combined with **OR**. W4A8 uses MXFP8 activations with MXFP4
weights. Omitting both lists or passing `[]` keeps the existing W4A4 behavior. Existing
`ignored_layers` take priority: **BF16 > (layer OR step) W4A8 > W4A4**.
DualScale's existing leading BF16 block rule is unchanged; it does not enable
W4A8.

```python
from vllm_omni import Omni

omni = Omni(
    model="<Wan2.2-T2V-A14B-Diffusers>",
    quantization_config={
        "method": "mxfp4",
        "w4a8_fallback_layers": ["blocks.10.attn1.to_qkv"],
        "w4a8_fallback_steps": [0, 1, 38, 39],
    },
)
```

Both policies also work with offline single-scale `mxfp4`. Any nonempty
W4A8 list with `mxfp4_dualscale`, online or offline, raises an explicit error
at configuration construction. Empty lists preserve its existing W4A4 path.
Each expert's checkpoint still determines its serialized
format and BF16 layer routing. A serialized checkpoint config is the complete
storage contract: `ignored_layers`, or its `modules_to_not_convert` alias,
selects the BF16 layers, and omitting both means no explicit BF16 layers. A
method-only checkpoint value such as `"mxfp4"` or
`{"quant_method": "mxfp4"}` carries no storage policy and leaves a caller's
online `ignored_layers` unchanged. An explicitly supplied quantization config
retains its W4A8 layer and step lists when the loader rebuilds either expert's
config from disk, including empty lists that disable checkpoint policies.
Within an explicit runtime config, an omitted W4A8 list defaults to `[]`;
individual omitted fields do not inherit saved policies. Without an explicit
runtime config, auto-detection uses the saved checkpoint policies.

Layer names are relative to each expert (`blocks.10...`, without a
`transformer.` or `transformer_2.` prefix). A global config applies the same
list to both experts; per-component `transformer`/`transformer_2` configs can
select different lists. Wan selects these expert names exactly: a `transformer`
entry does not match `transformer_2`; an omitted expert uses `default` if set.
A component mapped to `null` (or an unmatched component
without a default) stays unquantized and requires a BF16 checkpoint; loading a
quantized checkpoint for that disabled component raises an error. A top-level
`quantization_config=None` retains checkpoint auto-detection.
Names are exact, with no regex, wildcard, block expansion,
or checkpoint-name remapping. Wan validates them against its runtime Linear
modules and logs the selected paths/count and BF16 overrides per PP rank.
Use `attn1.to_qkv` for fused self-attention; `attn1.to_q`, `to_k`, or `to_v`
checkpoint names raise an error. Cross-attention `attn2.to_q/to_k/to_v` remains
separate. Layer-only fallback does not require a denoise ForwardContext.

The text-to-video example accepts the JSON directly:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
python examples/offline_inference/text_to_video/text_to_video.py \
    --model /path/to/Wan2.2-T2V-A14B-Diffusers \
    --quantization-config '{"method":"mxfp4","w4a8_fallback_steps":[0,2]}' \
    --tensor-parallel-size 1 --ulysses-degree 2 --cfg-parallel-size 2 \
    --enforce-eager --height 480 --width 832 --num-frames 81 \
    --num-inference-steps 40 --seed 42 --output wan22-w4a8.mp4
```

Use either `--quantization` or `--quantization-config` in this example.
Serving accepts the same JSON through `--diffusion-quantization-config`.

The list uses one request's loop index across both high- and low-noise experts,
and both CFG branches see the same index. A new request starts at step zero.
Indices outside the request's step range are never selected. A pipeline must
publish `ForwardContext.denoise_step_idx`; requesting step fallback without
that context raises an error instead of silently ignoring the list.

Single-scale W4A4 and W4A8 reuse the same prepared `weight` and `weight_scale`.
Online loading quantizes the original BF16/FP16 weight once; offline loading
packs the numeric FP4 values and reshapes the saved E8M0 scale without
requantizing the weight. Only activation precision changes during forward.
There is no DualScale decoding, second W4 representation or `w4a8_weight` /
`w4a8_weight_scale` cache. No weight quantization runs inside the denoising loop.

The mixed FP8/FP4 GEMM uses BF16 output and a matching BF16 row bias for
compatibility with the target A5 runtime. An FP16 caller receives that result
converted back to FP16; its output shape and dtype are preserved.

The single-scale fallback validation target is eager Wan2.2 T2V A14B on A5.
Quality scores, optimal fallback lists, real C7 exports and additional
compile/offload combinations require separate evaluation. Existing W4A4-only
DualScale still needs complete 512-channel input groups; use TP=1 with
sequence/CFG parallelism for its initial multi-card run.

### `mxfp4` — Pre-Quantized Single-Scale Weights (Ascend NPU)

Set the following fields in each transformer's `config.json` when a
single-scale checkpoint uses calibrated Smooth scaling:

```json
{
  "quantization_config": {
    "quant_method": "mxfp4",
    "is_checkpoint_mxfp4_serialized": true,
    "require_smooth_scale": true,
    "ignored_layers": []
  }
}
```

Each quantized Linear must provide tensors under its model parameter name:

| Tensor | Checkpoint layout | Meaning |
| -------- | ------------------- | --------- |
| `weight` | `(N, K)` numeric FP4 values | Packed to FP4 after loading |
| `weight_scale` | `(N, ceil(K / 32))`, `uint8` | Raw E8M0 exponent bytes, required |
| `mul_scale` | `(K,)`, FP32 | Smooth pre-scale, required when declared above |

Both W4A4 and W4A8 quantize `x * mul_scale`. The Smooth tensor is converted
to the layer's activation dtype after loading. Row-parallel layers load the
corresponding input-channel shard. Pre-fused QKV weights must use one shared
Smooth tensor for the fused projection.

For backward compatibility, `require_smooth_scale` defaults to `false`:
an absent tensor uses identity scaling, while a present tensor is still loaded.
Without a declaration, an absent Smooth tensor cannot be distinguished from a
legacy unsmoothed checkpoint. Calibrated checkpoints should declare
`require_smooth_scale: true` so missing or misnamed Smooth tensors fail loading;
incorrect tensor shapes, non-floating tensors, NaN/Inf and nonpositive values
also fail. Values that overflow or underflow to zero in the activation dtype
are rejected during weight processing. The flag requires serialized single-scale
weights and is rejected for online quantization. It does not make DualScale
scales optional or change missing-parameter policies for other backends.
Either the checkpoint or an explicit caller can require Smooth: a `true`
declaration is retained when per-expert metadata rebuilds the configuration,
including when the other source omits the flag or sets it to `false`.

### Native Ascend single-scale checkpoints

For Wan2.2-T2V-A14B, `native_checkpoint_path` accepts an msModelSlim
`mindie_format_saver` output root containing `high_noise_model/` and
`low_noise_model/`, each with one `quant_model_description*.json` and one
`quant_model_weight*.safetensors`. Pass the original **BF16 Diffusers** model
root as `model`; it supplies pipeline configuration, T5/VAE and explicitly
FLOAT parameters omitted by the quantized export.

```python
omni = Omni(
    model="/models/Wan2.2-T2V-A14B-Diffusers-BF16",
    quantization_config={
        "method": "mxfp4",
        "native_checkpoint_path": "/models/Wan2.2-native-MXFP4",
        "mxfp4_scale_alg": 2,
        "require_smooth_scale": True,
        "w4a8_fallback_steps": [0, 1, 38, 39],
    },
)
```

This path is offline automatically. Omni validates descriptions, maps each
expert's names and loads packed `uint8[N,K/2]` W4 plus `uint8[N,K/32]` scale
bytes directly. No user conversion step or expanded weight directory is needed.
The existing Wan loader fuses Q/K/V along output channels; row-parallel loaders
slice packed weights, group scales and the logical input-channel Smooth tensor.
Every input partition must contain complete groups of 32.

Only `W4A4_MXFP4` and `FLOAT` labels are accepted. Quantized native layers
are limited to the block attention projections and FFN linears managed by
Omni. Root condition embeddings and the output head must remain FLOAT. Missing quantized weights or
scales fail; they cannot be substituted from the original BF16 model. Native
FLOAT descriptors determine the floating layers. Ignoring a packed layer is
rejected. Smooth `.linear.*` / `.div.mul_scale` pairs must be complete, finite,
positive and identical across fused Q/K/V. Other savers, rank directories,
sharded native runtime exports, mixed quantization types and rotations are not
supported. The existing numeric single-scale checkpoint path remains supported;
the optional CPU conversion CLI is not required by this runtime path.

Native headers identify storage, **not the generation algorithm**. C7 7.25,
search disabled and calibration provenance still require the producer's receipt.
A checkpoint without `mul_scale` does not demonstrate Smooth validation.

### `mxfp4` — Single-Scale Online Mode

Online mode quantizes BF16/FP16 weights once at loading. On NPU,
`mxfp4_scale_alg` selects both online W4 preparation and runtime A4:

| Setting | W4 / A4 algorithm | NPU parameters |
| --- | --- | --- |
| `0` (default) | OCP MX | `scale_alg=0` |
| `2` | C7 | `scale_alg=2, dst_type_max=7.25` |

Both use FP4 E2M1 with E8M0 scales, `axis=-1`, `block_size=32`,
`round_mode="rint"`. W4A8 always quantizes the activation to
`float8_e4m3fn` with `scale_alg=0` and the same grouping/rounding, without
C7 parameters. A4 and A8 share the prepared W4 and scale. C7 is currently
NPU-only; ROCm retains its existing AITER online algorithm.

An explicit C7 online configuration is
`{"method": "mxfp4", "mxfp4_scale_alg": 2}`. For offline checkpoints the
setting affects **only A4 activations**. Existing weights are never requantized:
a C7 checkpoint can run C7 W4 + C7 A4 or C7 W4 + OCP A4, and both fall back to
C7 W4 + OCP A8. Record the weight-generation algorithm separately from the
runtime activation setting. Smooth remains active with either setting. The
formal C7 precision recipe explicitly selects `2`; FA is unchanged.

A single block scale
(`float8_e8m0fnu`, one per 32 K elements) is computed on the fly; no
calibration `mul_scale` is available. Applies equally to single-transformer
and cascade (A14B) models — both transformers in a cascade receive the same
quantization config automatically.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="mxfp4")
outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

```bash
# Single-transformer or cascade model — same command
python text_to_video.py --model <your-model> --quantization mxfp4

# Online serving
vllm serve <your-model> --omni --quantization mxfp4
```

### `mxfp4_dualscale` — DualScale Online Mode

Online DualScale mode computes both fine and coarse scales on the fly from BF16
weights using `npu_dynamic_dual_level_mx_quant`. Applies equally to
single-transformer and cascade (A14B) models. Compared to `mxfp4` online,
DualScale provides better quantization accuracy at higher compute cost.

The default configuration keeps the leading 5 transformer blocks in BF16
(`num_bf16_fallback_layers=5`). Accuracy evaluation on Wan2.2-A14B shows this
is sufficient to meet quality requirements and is the recommended setting.

```python
omni = Omni(model="<your-model>", quantization="mxfp4_dualscale")
```

```bash
python text_to_video.py --model <your-model> --quantization mxfp4_dualscale
```

If accuracy debugging identifies additional precision-sensitive layers, they
can be pinned to BF16 via the Python API:

```python
omni = Omni(
    model="<your-model>",
    quantization_config={
        "method": "mxfp4_dualscale",
        "ignored_layers": ["blocks.10.attn1.to_q"],   # explicit per-layer override
    },
)
```

BF16 fallback routing in online mode applies two rules in priority order:

1. **`ignored_layers`** (explicit per-layer override): any layer whose prefix
   matches is kept in BF16 regardless of block index.
2. **`num_bf16_fallback_layers`** (coarse leading-block rule): the first N
   transformer blocks (`blocks.0` … `blocks.N-1`) fall back to BF16. Defaults
   to `5` (recommended). Layers outside `blocks.N.*`
   (e.g. `condition_embedder`) are always quantized.

### `mxfp4_dualscale` — DualScale Offline Mode (Recommended)

Offline mode loads a pre-quantized DualScale checkpoint from msModelSlim. A
preprocessing step converts the raw quantized output to the diffusers format
expected by vLLM-Omni and injects the quantization config into each
`transformer/config.json` so that vLLM-Omni auto-detects the offline path
without a `--quantization` flag.

BF16 fallback layers may be interleaved anywhere in the transformer — they are
not restricted to leading blocks. The merge script detects them from
`quant_model_description.json` and writes their prefixes into `ignored_layers`
inside `config.json`. At runtime, each layer's prefix is matched against
`ignored_layers` to decide BF16 vs. MXFP4 DualScale.

#### Checkpoint tensor layout

Each quantized linear layer stores four tensors:

| Tensor | Shape | dtype | Description |
| -------- | ------- | ------- | ------------- |
| `weight` | `(N, K)` | float8_e4m3fn | Numeric FP4 values; packed into two values per byte after loading |
| `weight_scale` | `(N, K//32)` | uint8 | Fine block scale (`float8_e8m0fnu` bit pattern) |
| `weight_dual_scale` | `(N, K//512, 1)` | float32 | Coarse block scale |
| `mul_scale` | `(K,)` | float32 | Per-input-channel smooth pre-scale (from calibration) |

BF16 fallback layers have no quantization tensors; only the original `weight`
(and optional `bias`) are present, loaded directly from the base checkpoint.

#### Step 1 — Quantize with msModelSlim

```bash
msmodelslim quant \
  --model_path /path/to/Wan2.2-T2V-A14B-Diffusers \
  --save_path  /path/to/wan2_2_t2v_quantized_raw \
  --device npu \
  --model_type Wan2_2 \
  --config_path /path/to/wan2_2_w4a4_mxfp4_dualscale.yaml \
  --trust_remote_code True
```

After this step, `--save_path` contains raw quantized safetensors files,
scale files, and a metadata JSON (`quant_model_description*.json`).

For cascade MoE models (T2V-A14B, I2V-A14B), msModelSlim outputs two
subdirectories: `high_noise_model/` (transformer) and `low_noise_model/`
(transformer_2).

#### Step 2 — Preprocess with merge_mxfp4_dualscale_checkpoint.py

The script (`vllm_omni/quantization/tools/merge_mxfp4_dualscale_checkpoint.py`):

1. Copies the original diffusers model to `--output-path` (VAE, text encoder,
   scheduler, etc. are preserved).
2. Remaps tensor names from msModelSlim convention to diffusers convention and
   strips `.linear.` / `.div.` wrappers added by the quantization tool.
3. Overlays MXFP4 tensors (weight, fine/coarse scales, `mul_scale`) onto the
   BF16 base checkpoint. Non-quantized layers keep their original BF16 weights.
4. Detects all linear layers that remain in BF16 and writes their prefixes into
   `ignored_layers` in `config.json`.
5. Injects `quantization_config` so vLLM-Omni auto-detects offline MXFP4
   DualScale.

For cascade MoE models, steps 2–5 run separately for each transformer.

```bash
python vllm_omni/quantization/tools/merge_mxfp4_dualscale_checkpoint.py \
  --model-type     Wan2.2-T2V-A14B \
  --original-model /path/to/Wan2.2-T2V-A14B-Diffusers \
  --quant-path     /path/to/wan2_2_t2v_quantized_raw \
  --output-path    /path/to/Wan2.2-T2V-A14B-MXFP4-DualScale
```

| Argument | Description |
| ---------- | ------------- |
| `--model-type` | Model variant: `Wan2.2-T2V-A14B` or `Wan2.2-I2V-A14B` |
| `--original-model` | Root directory of the original BF16 diffusers model |
| `--quant-path` | Root directory of the msModelSlim quantized output |
| `--output-path` | Output directory for the merged model (created by the script) |

The script outputs a complete diffusers model directory at `--output-path`,
with each transformer subfolder containing:

- `diffusion_pytorch_model.safetensors` — MXFP4 weights + scale tensors, with BF16 fallback layers from the base checkpoint
- `config.json` — original transformer config with `quantization_config` injected
- `quant_model_description.json` — quantization metadata (reference only)

The `quantization_config` injected into `config.json` for each transformer:

```json
{
  "quant_method": "mxfp4_dualscale",
  "is_checkpoint_serialized": true,
  "ignored_layers": [
    "blocks.0.attn1.to_qkv",
    "blocks.0.attn1.to_out",
    "proj_out"
  ]
}
```

`ignored_layers` lists every linear layer that retains its original BF16 weight,
using vllm-omni model parameter names (QKV-fused, FFN underscored, `to_out`
unindexed). The exact entries are determined by the quantization tool (msModelSlim)
and may differ between `transformer` and `transformer_2` in a cascade model.

#### Step 3 — Serve

```bash
python text_to_video.py --model /path/to/Wan2.2-T2V-A14B-MXFP4-DualScale

# Online serving
vllm serve /path/to/Wan2.2-T2V-A14B-MXFP4-DualScale --omni
```

```python
omni = Omni(model="/path/to/Wan2.2-T2V-A14B-MXFP4-DualScale")
```

!!! note
    No `--quantization` flag is needed for offline mode. The preprocessing
    script injects `quantization_config` into each `transformer/config.json`,
    which vLLM-Omni reads automatically to activate the correct offline path.

## Parameters

### `mxfp4` (single-scale, online + offline)

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `method` | str | — | `"mxfp4"` |
| `mxfp4_scale_alg` | int | `0` | `0`: OCP MX; `2`: C7 A4 quantization with `dst_type_max=7.25`; A8 remains OCP MX |
| `native_checkpoint_path` | str or null | `null` | Local native single-scale checkpoint directory, loaded alongside the original floating model |
| `require_smooth_scale` | bool | `false` | Require calibrated Smooth tensors for offline single-scale weights |
| `ignored_layers` | list[str] | `[]` | Layer prefixes to keep in BF16 |
| `w4a8_fallback_layers` | list[str] | `[]` | Exact runtime Linear paths that always use A8; NPU only |
| `w4a8_fallback_steps` | list[int] | `[]` | Zero-based denoise steps that use A8; NPU only |

### `mxfp4_dualscale` (dual-scale, online + offline)

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `method` | str | — | `"mxfp4_dualscale"` |
| `is_checkpoint_serialized` | bool | `False` | `True` for offline DualScale checkpoints; auto-set from `config.json` when using the preprocessing script |
| `ignored_layers` | list[str] | `[]` | Layer prefixes to keep in BF16. **Works in both modes**: offline — populated by the merge script for interleaved sensitive layers; online — user-supplied for explicit per-layer precision override |
| `num_bf16_fallback_layers` | int | `5` | **Online mode only**: leading N transformer blocks (`blocks.0` … `blocks.N-1`) kept in BF16. Applied after `ignored_layers`; ignored in offline mode. Default of `5` is the evaluated recommended value for Wan2.2-A14B |
| `w4a8_fallback_layers` | list[str] | `[]` | Must remain empty; DualScale does not support W4A8 fallback |
| `w4a8_fallback_steps` | list[int] | `[]` | Must remain empty; DualScale does not support W4A8 fallback |

#### BF16 fallback priority (online mode)

```text
for each linear layer:
    if prefix in ignored_layers               → BF16  (explicit override, highest priority)
    elif block_idx < num_bf16_fallback_layers → BF16  (coarse leading-block rule)
    else                                      → MXFP4 DualScale online
```

Layers outside `blocks.N.*` (e.g. `condition_embedder.*`) are always quantized
unless they appear in `ignored_layers`.

## Validation and Notes

1. **Online single-scale (`mxfp4`)** quantizes BF16 weights at load time using
   `npu_dynamic_mx_quant` (single-scale). No calibration `mul_scale` is
   available — all output partitions receive an identity pre-scale. Offline
   single-scale checkpoints use the separate layout and native loading path
   described above.

2. **Online dual-scale (`mxfp4_dualscale`, `is_checkpoint_serialized=False`)**
   quantizes BF16 weights using `npu_dynamic_dual_level_mx_quant` (fine + coarse
   scales computed on the fly). No calibration `mul_scale`; leading blocks or
   explicit `ignored_layers` stay in BF16 for accuracy.

3. **Offline dual-scale (`mxfp4_dualscale`, `is_checkpoint_serialized=True`)** —
   **recommended for production** — loads four tensors per quantized layer: FP4
   weight, fine scale (`uint8` reinterpreted as `float8_e8m0fnu`), coarse scale
   (`float32`), and per-input-channel `mul_scale` (`float32`). BF16 fallback
   layers have no quantization tensors and are routed via `ignored_layers`.

4. **Scale dtype**: fine scales are stored as `uint8` in safetensors (same bit
   layout as `float8_e8m0fnu`) and reinterpreted at load time without a lossy
   float32 round-trip.

5. **Cascade model config propagation**: in a cascade model (transformer +
   transformer_2), vLLM-Omni reads each transformer's own `config.json` and
   rebuilds the quant config locally when `ignored_layers` differs between
   transformers, ensuring per-layer routing is accurate for each. The first
   transformer's config is propagated to `od_config` so the second transformer
   can reuse it as a starting point.

6. **Self-attention QKV fusion**: Q, K, V projection weights are fused into a
   single `QKVParallelLinear` layer at runtime. `ignored_layers` entries use the
   fused name (`attn1.to_qkv`), written automatically by the merge script.

7. W4A4 carries higher quantization noise than W8A8 (16 vs 256 levels). The
   DualScale offline method mitigates this with calibrated `mul_scale` smooth
   quantization. Use `ignored_layers` and `num_bf16_fallback_layers` to trade
   off compression vs. accuracy for precision-sensitive layers.

## Adapting MXFP4 for a New Model

This section is aimed at developers who want to add MXFP4 support to a model
other than Wan2.2. The three integration points are: (1) discovering the correct
runtime layer names, (2) wiring `ignored_layers` into the model, and (3) writing
a merge script for offline checkpoints.

### Step 1 — Discover runtime layer names

`ignored_layers` entries must match the **runtime parameter names** used inside
vllm-omni, which may differ from the names stored in the diffusers checkpoint.
The canonical source of truth is the model's own `named_parameters()`.

```python
from vllm_omni import Omni

# Load the model without quantization to inspect parameter names.
omni = Omni(model="/path/to/your-model")  # no --quantization flag
for name, _ in omni.pipeline.transformer.named_parameters():
    if "weight" in name and "scale" not in name:
        print(name)
```

Compare the printed names against the diffusers checkpoint keys
(`safetensors.safe_open` or `torch.load`) to identify any renames your model
applies. Common patterns that differ in Wan2.2 (and may appear in other
models):

| Diffusers checkpoint name | vllm-omni runtime name | Reason |
| --------------------------- | ------------------------ | -------- |
| `attn1.to_q`, `attn1.to_k`, `attn1.to_v` | `attn1.to_qkv` | Self-attention Q/K/V fused into `QKVParallelLinear` |
| `ffn.net.0.proj` | `ffn.net_0.proj` | Dots in sub-module names replaced with underscores |
| `ffn.net.2` | `ffn.net_2` | Same underscore rule |
| `to_out.0` | `to_out` | Sequential index stripped |

If your model has different fusion patterns, inspect `packed_modules_mapping`
on the model class — this dict records how checkpoint keys are mapped to
fused runtime parameters.

!!! warning "Partial QKV fallback is not allowed"
    If your model fuses Q, K, V into a single layer, `ignored_layers` must
    include **all three or none**. A partial fallback (e.g. `to_q` in BF16 but
    `to_k`, `to_v` quantized) cannot be expressed at runtime because they share
    one `QKVParallelLinear`. The merge script enforces this and raises an error
    if only some of the trio appear as non-quantized.

### Step 2 — Add ignored_layers to the model

#### Online mode

Pass `ignored_layers` directly in the quantization config using the **runtime
names** discovered in Step 1. No code changes to the model are required.

```python
omni = Omni(
    model="/path/to/your-model",
    quantization={
        "method": "mxfp4_dualscale",
        "ignored_layers": [
            "blocks.0.attn1.to_qkv",   # runtime name, not diffusers name
            "blocks.0.attn1.to_out",
            "blocks.0.ffn.net_0.proj",
        ],
    },
)
```

```bash
# CLI does not support list-typed ignored_layers directly.
# Use the Python API or set ignored_layers in config.json (offline).
python your_script.py --model /path/to/your-model --quantization mxfp4_dualscale
```

The `num_bf16_fallback_layers` coarse rule is an alternative to listing layers
individually: set it to N to keep all linear layers in blocks 0 … N-1 in BF16.
The right value depends on the model's sensitivity; evaluate on a validation
set and pick the smallest N that meets your accuracy target.

#### Offline mode

For offline checkpoints, `ignored_layers` is written into each transformer's
`config.json` by the merge script (see Step 3). No manual editing is needed if
the merge script is correct. The injected block:

```json
{
  "quant_method": "mxfp4_dualscale",
  "is_checkpoint_serialized": true,
  "ignored_layers": [
    "blocks.0.attn1.to_qkv",
    "blocks.0.attn1.to_out"
  ]
}
```

To add a layer manually (e.g. to pin an additional layer to BF16 without
re-running the merge script), edit `config.json` inside the transformer
subfolder. Use runtime names, not diffusers checkpoint names.

### Step 3 — Write a merge script for offline mode

The merge script for a new model mirrors
`vllm_omni/quantization/tools/merge_mxfp4_dualscale_checkpoint.py`. The four
things it must do:

1. **Remap tensor names** from the quantization tool convention to diffusers
   convention (strip wrappers like `.linear.`, `.div.`; fix any prefix
   differences).

2. **Collect ignored_layers**: after loading, enumerate all `*.weight` keys that
   have no corresponding `*.weight_scale` (i.e. layers the tool left in BF16).
   Convert diffusers names to vllm-omni runtime names (fuse QKV, rename FFN
   sub-modules, etc.). Write the result to `config.json`.

3. **Inject `quantization_config`** into `config.json`:
   ```python
   config["quantization_config"] = {
       "quant_method":              "mxfp4_dualscale",
       "is_checkpoint_serialized":  True,
       "ignored_layers":            ignored_layers,   # runtime names
   }
   ```

4. **Save** the merged safetensors and the updated `config.json`.

The key helper to implement is the diffusers-to-runtime name translator
(equivalent to `_diffusers_to_vllm_ignored` in the Wan2.2 merge script).
For each non-quantized diffusers weight key, apply your model's specific
renaming rules and collect the results.

!!! tip "Validate before serving"
    After producing the offline checkpoint, load it without a `--quantization`
    flag and verify that vLLM-Omni auto-detects the correct method. Check that
    the layer count reported in the startup log matches expectations: quantized
    layer count + `ignored_layers` count should equal total linear layer count.
    Any mismatch indicates a name-mapping bug in the merge script.
