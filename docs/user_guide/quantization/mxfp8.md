# W8A8 MXFP8 Quantization

## Overview

W8A8 MXFP8 (Microscaling FP8) quantizes both weights and activations to FP8
using the OCP MX format: groups of 32 K-dimension elements share a single
`float8_e8m0fnu` exponent scale. This gives better accuracy than channel-wise
FP8 while keeping the same 8-bit weight footprint.

This method supports three modes:

| Mode | Description |
|------|-------------|
| **Online** | BF16 weights are quantized to MXFP8 at load time — no pre-processing needed |
| **Offline (Native)** | msModelSlim-exported MXFP8 weights converted to diffusers format via `merge_mxfp8_checkpoint.py` — weights and scales are loaded directly from the preprocessed checkpoint |
| **Offline (AutoRound)** | AutoRound MXFP8 checkpoints with `data_type="mx_fp"` — auto-detected from `config.json` |

## Hardware Support

| Device | Online | Offline (Native) | Offline (AutoRound) |
|--------|--------|------------------|---------------------|
| NVIDIA Blackwell GPU (SM 100+) | ⭕ | ⭕ | ⭕ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ⭕ | ⭕ | ⭕ |
| NVIDIA Ampere GPU (SM 80+) | ⭕ | ⭕ | ⭕ |
| AMD ROCm | ⭕ | ⭕ | ⭕ |
| Intel XPU | ✅ | ❌ | ✅ |
| Ascend NPU (Atlas 950 A5) | ✅ | ✅ | ⭕ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this guide.

**Note**: Intel XPU only supports AutoRound MXFP8 for offline mode. Use AutoRound quantized checkpoints or online mode for XPU.

## Model Type Support

### Diffusion Model (Wan2.2)

| Model | Mode | Notes |
|-------|------|-------|
| Wan2.2-T2V-A14B | Online + Offline | MoE cascade; quantizes two transformers (`transformer` + `transformer_2`) |
| Wan2.2-I2V-A14B | Online + Offline | MoE cascade; quantizes two transformers (`transformer` + `transformer_2`) |
| Wan2.2-TI2V-5B | Online + Offline | Single transformer |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Status | Notes |
|-------|--------|-------|
| Qwen3-Omni | Not validated | — |
| Qwen3-TTS | Not validated | — |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Status | Notes |
|-------|--------|-------|
| BAGEL | Not validated | — |
| GLM-Image | Not validated | — |

## Configuration

### Online Mode

Online mode requires no pre-processing. vLLM-Omni quantizes BF16 weights to
MXFP8 at load time.

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="mxfp8")

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_video.py --model <your-model> --quantization mxfp8

# Online serving
vllm serve <your-model> --omni --quantization mxfp8
```

### Offline Mode (Native)

Native offline mode loads a pre-quantized checkpoint from msModelSlim. A preprocessing
step converts the raw quantized output to the diffusers format expected by
vLLM-Omni and injects the quantization config into `transformer/config.json` so
that vLLM-Omni auto-detects the offline path without a `--quantization` flag.

#### Step 1 — Quantize with msModelSlim

```bash
msmodelslim quant \
  --model_path /path/to/Wan2.2-TI2V-5B-Diffusers \
  --save_path  /path/to/wan2_2_ti2v_quantized_raw \
  --device npu \
  --model_type Wan2_2 \
  --config_path /path/to/wan2_2_w8a8f8_mxfp.yaml \
  --trust_remote_code True
```

After this step, `--save_path` contains the raw quantized safetensors files and
a metadata JSON (`quant_model_description*.json`).

For cascade MoE models (T2V-A14B, I2V-A14B), msModelSlim outputs two
subdirectories: `high_noise_model/` and `low_noise_model/`.

#### Step 2 — Preprocess with merge_mxfp8_checkpoint.py

The script (`vllm_omni/quantization/tools/merge_mxfp8_checkpoint.py`):

1. Copies the original diffusers model to `--output-path` (VAE, text encoder,
   scheduler, etc. are preserved).
2. Remaps tensor names from msModelSlim convention to diffusers convention.
3. Saves the converted weights as `diffusion_pytorch_model.safetensors`.
4. Copies the original `transformer/config.json` and injects
   `quantization_config` so that vLLM-Omni auto-detects offline MXFP8.

For cascade MoE models, steps 2–4 run separately for `high_noise_model/` →
`transformer/` and `low_noise_model/` → `transformer_2/`.

```bash
python vllm_omni/quantization/tools/merge_mxfp8_checkpoint.py \
  --model-type     Wan2.2-TI2V-5B \
  --original-model /path/to/Wan2.2-TI2V-5B-Diffusers \
  --quant-path     /path/to/wan2_2_ti2v_quantized_raw \
  --output-path    /path/to/Wan2.2-TI2V-5B-MXFP8
```

| Argument | Description |
|----------|-------------|
| `--model-type` | Model variant: `Wan2.2-T2V-A14B`, `Wan2.2-I2V-A14B`, or `Wan2.2-TI2V-5B` |
| `--original-model` | Root directory of the original BF16 diffusers model |
| `--quant-path` | Root directory of the msModelSlim quantized output |
| `--output-path` | Output directory for the merged model (created by the script) |

The script outputs a complete diffusers model directory at `--output-path`,
with each transformer subfolder containing:

- `diffusion_pytorch_model.safetensors` — converted FP8 weights
- `config.json` — original transformer config with `quantization_config` injected
- `quant_model_description.json` — renamed quantization metadata (reference only)

#### Step 3 — Serve

```bash
python text_to_video.py --model /path/to/Wan2.2-TI2V-5B-MXFP8

# Online serving
vllm serve /path/to/Wan2.2-TI2V-5B-MXFP8 --omni
```

Python API:

```python
omni = Omni(model="/path/to/Wan2.2-TI2V-5B-MXFP8")
```

!!! note
    No `--quantization` flag is needed for native offline mode. The preprocessing
    script injects `quantization_config` into each `transformer/config.json`,
    which vLLM-Omni reads automatically to activate the offline MXFP8 method.

### Offline Mode (AutoRound)

AutoRound MXFP8 checkpoints declare `quant_method="auto-round"` with `data_type="mx_fp"` in their `config.json`. These are automatically detected and served by vLLM's INC MXFP8 scheme through `OmniINCConfig` (`vllm_omni/quantization/inc_config.py`).

To use an AutoRound MXFP8 checkpoint:

```bash
python text_to_video.py --model <autoround-mxfp8-model>

# Online serving
vllm serve <autoround-mxfp8-model> --omni
```

Python API:

```python
omni = Omni(model="<autoround-mxfp8-model>")
```

!!! note
    AutoRound MXFP8 checkpoints are auto-detected from `config.json` and do not require a `--quantization` flag. The config must include:
    ```json
    {
      "quantization_config": {
        "quant_method": "auto-round",
        "data_type": "mx_fp",
        ...
      }
    }
    ```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Must be `"mxfp8"` |
| `is_checkpoint_mxfp8_serialized` | bool | `False` | `True` for offline pre-quantized checkpoints; auto-set from `config.json` when using the preprocessing script |
| `ignored_layers` | list[str] | `[]` | Layer name substrings to keep in BF16 (e.g. `"to_out"` matches `blocks.0.attn1.to_out.0`) |
| `prefix_bf16_steps` | int | `0` | Online NPU only; use the original BF16 weights for denoising indices `0 <= step < K`; `0` disables the fallback |
| `prefix_bf16_master_mode` | `cpu`, `resident`, or `dlo` | `cpu` | Storage policy for the retained BF16 weight copy when prefix fallback is enabled |

## Validation and Notes

On NPU, `VLLM_OMNI_MXFP8_SCALE_ALG` selects the `scale_alg` passed to
`torch_npu.npu_dynamic_mx_quant` for activations and online weights. It defaults
to `0`; set it before starting the process, for example:

```bash
export VLLM_OMNI_MXFP8_SCALE_ALG=1
```

The installed torch_npu/CANN version must support the selected algorithm and
the `scale_alg` keyword. Offline checkpoint weights are not requantized.

### NPU Online Prefix BF16 Fallback

Pass these options in the diffusion quantization config to compute the first K
denoising steps with a saved BF16 weight copy, then resume MXFP8 W8A8 Linear
computation:

```bash
vllm serve <model> --omni \
  --diffusion-quantization-config '{"method":"mxfp8","prefix_bf16_steps":5,"prefix_bf16_master_mode":"resident"}'
```

`prefix_bf16_steps` defaults to `0` (disabled) and must be non-negative. K
refers to zero-based `ForwardContext.denoise_step_idx` indices `0 <= step < K`,
so multiple CFG passes at the same step all use BF16. The step is read from
`ForwardContext` for each Linear call. If the pipeline does not publish the
step, inference continues on the MXFP8 path. A positive K is accepted only for
online MXFP8 weights; pre-quantized checkpoints do not contain the BF16 master.

`prefix_bf16_master_mode` defaults to `cpu`, which transfers the saved weight
to the activation device on each fallback call. The master is stored
transposed and contiguous, with shape `(in_features, out_features)`.
`resident` copies that master to its loading NPU and requires activations on
that device. Both modes retain a full BF16 copy of each local online Linear
weight; resident mode can significantly increase accelerator memory usage. K
larger than the actual denoising step count uses BF16 for every denoising step.

The master copy is a plain tensor attribute: it is excluded from `state_dict`
and is not moved by `module.to()`. In `cpu` and `resident` modes it is not
managed by layerwise offload. Resident
fallback is only supported while its loading device remains the activation
device; model migration, weight reload, sleep/wake, and distributed offload of
this copy are not supported. Reload the model to change fallback settings or
weights. The saved weight is BF16; at fallback time it is converted to the
activation dtype before matmul, and the result is cast back to that dtype. Bias
is added separately. Attention quantization is unaffected.

#### Experimental DLO prefix storage

Use `prefix_bf16_master_mode="dlo"` with distributed layerwise offload to
stream original BF16 masters for `0 <= step < K` and quantized weights/scales
afterwards. The offloader retains separate CPU backing for both representations
and uses its existing two device slots, sized to the maximum per-dtype block
requirement across both layouts. This first implementation does not alias
allocations across different dtypes; its device storage is bounded by block
size rather than total model size. Common parameters such as bias and norms
are included in both host layouts. No runtime requantization is performed.

The prefix provider is explicitly opt-in. Existing `cpu`, `resident`, and K=0
paths are unchanged. Phase checks invalidate speculative tail-to-head prefetch
at K and at the start of another request. CPU offload copies do not modify the
retained original BF16 bits, shape, or stride. Disable restores both formats.

Use pinned host memory and NUMA placement appropriate for the selected devices
to overlap transfers with compute. BF16-prefix shards use ordinary CPU backing;
when pinning is enabled, two pinned host staging slots are shared across all
prefix hook groups and sized to the largest local shard per dtype. Before
reusing a slot, the offloader waits for its previous transfer to complete.
This bounds the extra pinned-memory requirement independently of model depth,
but adds a CPU-to-CPU copy during prefix steps. Quantized backing retains its
existing pinning policy. HBM savings do not guarantee unchanged
latency: the prefix now transfers BF16 blocks that were previously resident.
Small modules outside DLO streaming rings use the existing CPU-copy fallback;
this also applies if no distributed offloader is installed. This prototype
does not support prefix storage combined with rank-local mmap/HWR. It requires
sequential, consistent step context across the participating workers; mixed
step batches, model migration, weight reload, and sleep/wake are not validated.

### Weight Loading Notes

1. Online mode quantizes BF16 weights at load time using
   `npu_dynamic_mx_quant`. This adds a one-time overhead on the first load
   but requires no checkpoint preparation.
2. Offline mode loads FP8 weights directly from the checkpoint. Scales are
   stored as `uint8` bytes in safetensors (same bit layout as
   `float8_e8m0fnu`) and are reinterpreted at load time without a dtype
   conversion.
3. If the offline checkpoint was produced with the old `merge_mxfp8_checkpoint.py`
   interface (arguments `--quant-dir`, `--orig-dir`, `--meta-json`,
   `--output-dir`), regenerate it with the current script. The old script
   wrote a separate `quantization_config.json` that is not read by vLLM-Omni;
   the current script injects the config directly into `transformer/config.json`.
