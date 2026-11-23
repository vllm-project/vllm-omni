# W8A8 MXFP8 Quantization

## Overview

W8A8 MXFP8 (Microscaling FP8) quantizes both weights and activations to FP8
using the OCP MX format: groups of 32 K-dimension elements share a single
`float8_e8m0fnu` exponent scale. The finer scaling can reduce quantization error
relative to coarser FP8 scaling; validate accuracy on the intended workload.
Weights use eight bits per element plus block-scale storage.

This method supports three modes:

| Mode | Description |
| ------ | ------------- |
| **Online** | BF16 weights are quantized to MXFP8 at load time — no pre-processing needed |
| **Offline (Native)** | msModelSlim-exported MXFP8 weights converted to diffusers format via `merge_mxfp8_checkpoint.py` — weights and scales are loaded directly from the preprocessed checkpoint |
| **Offline (AutoRound)** | AutoRound MXFP8 checkpoints with `data_type="mx_fp"` — auto-detected from `config.json` |

## Hardware Support

| Device | Online | Offline (Native) | Offline (AutoRound) |
| -------- | -------- | ------------------ | --------------------- |
| NVIDIA Blackwell/Rubin GPU (SM 100, 103, 107) | ✅ | ❌ | ⭕ |
| NVIDIA Ada/Hopper GPU (SM 89/90) | ❌ | ❌ | ⭕ |
| NVIDIA Ampere GPU (SM 80/86) | ❌ | ❌ | ⭕ |
| AMD ROCm | ⭕ | ⭕ | ⭕ |
| Intel XPU | ✅ | ❌ | ✅ |
| Ascend NPU (Atlas 950 A5) | ✅ | ✅ | ⭕ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this guide.

**Note**: Intel XPU only supports AutoRound MXFP8 for offline mode. Use AutoRound quantized checkpoints or online mode for XPU.

CUDA online mode reuses vLLM's MXFP8 online linear method and kernel selection.
It requires a vLLM build with CUDA MXFP8 support and its FlashInfer CuTe-DSL
dependencies. Other CUDA compute capabilities are rejected explicitly. This
path handles diffusion linear layers; it does not add CUDA MXFP8 MoE support.
Native serialized MXFP8 checkpoints are not supported on CUDA.

CUDA validation covers SM107 on Vera Rubin: BF16/FP16 input, 2D/3D and strided
activations, bias, fused QKV/MLP loading, row-parallel weight loading, and an
H3-Image SP4 serving run. SM100/SM103 are enabled through the upstream kernel
path but were not included in that MXFP8 hardware validation.

## Model Type Support

### Image Model (MiniMax-H3-Image)

Online MXFP8 was validated on four Rubin GPUs with DiT TP1, Ulysses SP4, encoder
TP4 and the released six-step schedule: six generation/editing workloads and
160 diverse 4096×4096 requests. The image pipeline applies the global
quantization setting to the DiT; the encoder and VAE retain native precision.
These checks establish serving functionality, not a universal quality bound.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --usp 4 --tensor-parallel-size 1 \
  --text-encoder-tp-size 4 --quantization mxfp8
```

### Diffusion Model (Wan2.2)

| Model | Mode | Notes |
| ------- | ------ | ------- |
| Wan2.2-T2V-A14B | Online + Offline | MoE cascade; quantizes two transformers (`transformer` + `transformer_2`) |
| Wan2.2-I2V-A14B | Online + Offline | MoE cascade; quantizes two transformers (`transformer` + `transformer_2`) |
| Wan2.2-TI2V-5B | Online + Offline | Single transformer |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Status | Notes |
| ------- | -------- | ------- |
| Qwen3-Omni | Not validated | — |
| Qwen3-TTS | Not validated | — |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Status | Notes |
| ------- | -------- | ------- |
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
| ---------- | ------------- |
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
| ----------- | ------ | --------- | ------------- |
| `method` | str | — | Must be `"mxfp8"` |
| `is_checkpoint_mxfp8_serialized` | bool | `False` | `True` for offline pre-quantized checkpoints; auto-set from `config.json` when using the preprocessing script |
| `ignored_layers` | list[str] | `[]` | Layer name substrings to keep in BF16 (e.g. `"to_out"` matches `blocks.0.attn1.to_out.0`) |

## Validation and Notes

1. Online mode quantizes BF16/FP16 weights at load time using the selected
   platform's implementation (`npu_dynamic_mx_quant` on NPU, vLLM's online
   MXFP8 path on CUDA/XPU). This adds a one-time loading overhead but requires
   no checkpoint preparation.
2. Offline mode loads FP8 weights directly from the checkpoint. Scales are
   stored as `uint8` bytes in safetensors (same bit layout as
   `float8_e8m0fnu`) and are reinterpreted at load time without a dtype
   conversion.
3. If the offline checkpoint was produced with the old `merge_mxfp8_checkpoint.py`
   interface (arguments `--quant-dir`, `--orig-dir`, `--meta-json`,
   `--output-dir`), regenerate it with the current script. The old script
   wrote a separate `quantization_config.json` that is not read by vLLM-Omni;
   the current script injects the config directly into `transformer/config.json`.
