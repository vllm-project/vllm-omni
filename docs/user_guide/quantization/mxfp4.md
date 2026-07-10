# W4A4 MXFP4 Quantization

## Overview

W4A4 MXFP4 (Microscaling FP4) quantizes both weights and activations to FP4
(`float4_e2m1fn_x2`, packed 2 values per byte) using the OCP MX format: groups
of 32 K-dimension elements share a single `float8_e8m0fnu` exponent scale.

vLLM-Omni provides two quantization methods with different scale structures:

| Method | Scale structure | Mode | Use case |
|--------|----------------|------|----------|
| `mxfp4` | Single-scale (per-32 fine only) | Online only | Quick accuracy baseline; no checkpoint prep needed |
| `mxfp4_dualscale` | Dual-scale (fine per-32 + coarse per-512 + per-channel `mul_scale`) | Online + Offline | Production; better accuracy; offline recommended |

!!! tip "Recommended: `mxfp4_dualscale` offline"
    For production deployments, use the `mxfp4_dualscale` offline mode with a
    pre-quantized checkpoint produced by msModelSlim. Offline checkpoints load
    calibrated `mul_scale` tensors from disk, providing measurably better accuracy
    than any online method. The one-time preprocessing cost amortises across all
    subsequent inference runs.

    Use `mxfp4` online only for quick experimentation where preprocessing time
    is not acceptable and accuracy loss is tolerable.

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
|--------|---------|
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
|-------|--------|---------|-------|
| Wan2.2-T2V-A14B | `mxfp4` / `mxfp4_dualscale` | `mxfp4_dualscale` | MoE cascade (`transformer` + `transformer_2`); both transformers quantized with the same config |
| Wan2.2-I2V-A14B | `mxfp4` / `mxfp4_dualscale` | `mxfp4_dualscale` | MoE cascade; same scheme as T2V-A14B |
| Wan2.2-TI2V-5B | ❌ | ❌ | Parameter count too small; W4A4 causes unacceptable accuracy loss |

The choice between `mxfp4` and `mxfp4_dualscale` in **online mode** is about
quantization quality, not model compatibility — both work on cascade (A14B) and
single-transformer models alike, the same as `mxfp8` online:

- `mxfp4`: single-scale, lower overhead, simpler compute, online only
- `mxfp4_dualscale`: dual-scale + optional BF16 fallback, better accuracy, online **and** offline

**Offline** checkpoints for A14B are always in `mxfp4_dualscale` format (produced
by the merge script); there is no offline `mxfp4` single-scale format.

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

### `mxfp4` — Single-Scale Online Mode

Online mode requires no pre-processing. vLLM-Omni quantizes BF16 weights to
MXFP4 at load time using `npu_dynamic_mx_quant`. A single block scale
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
|--------|-------|-------|-------------|
| `weight` | `(N, K)` | float8_e4m3fn | FP4 packed (2 values per byte) |
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
|----------|-------------|
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

### `mxfp4` (single-scale, online only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | `"mxfp4"` |
| `ignored_layers` | list[str] | `[]` | Layer prefixes to keep in BF16 |

### `mxfp4_dualscale` (dual-scale, online + offline)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | `"mxfp4_dualscale"` |
| `is_checkpoint_serialized` | bool | `False` | `True` for offline DualScale checkpoints; auto-set from `config.json` when using the preprocessing script |
| `ignored_layers` | list[str] | `[]` | Layer prefixes to keep in BF16. **Works in both modes**: offline — populated by the merge script for interleaved sensitive layers; online — user-supplied for explicit per-layer precision override |
| `num_bf16_fallback_layers` | int | `5` | **Online mode only**: leading N transformer blocks (`blocks.0` … `blocks.N-1`) kept in BF16. Applied after `ignored_layers`; ignored in offline mode. Default of `5` is the evaluated recommended value for Wan2.2-A14B |

#### BF16 fallback priority (online mode)

```
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
   available — all output partitions receive an identity pre-scale. No offline
   checkpoint format exists for this method.

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
|---------------------------|------------------------|--------|
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
