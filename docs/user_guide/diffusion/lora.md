# LoRA (Low-Rank Adaptation) Guide

LoRA (Low-Rank Adaptation) enables fine-tuning diffusion models by adding trainable low-rank matrices to existing model weights. vLLM-Omni supports PEFT-style LoRA adapters, allowing you to customize model behavior without modifying the base model weights.

## Overview

vLLM-Omni exposes two complementary LoRA flows for diffusion models:

1. **Init-time LoRA**: a single adapter is pre-loaded when `Omni` starts and is applied to every request. Lowest runtime overhead; best when all requests should share the same adapter.
2. **Per-request LoRA**: zero or more adapters are attached to each request via `sampling_params.lora_requests`. Supports switching adapters between requests and composing multiple adapters in a single forward pass (multi-LoRA).

Adapters are managed by an LRU cache so repeated activations avoid redundant weight reloads.

## LoRA Adapter Format

LoRA adapters must be in **PEFT (Parameter-Efficient Fine-Tuning)** format. A typical adapter directory:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

`adapter_config.json` contains:
- `r`: LoRA rank
- `lora_alpha`: LoRA alpha scaling factor
- `target_modules`: list of module names the adapter applies to

!!! note "Server-side Path Requirement"
    The LoRA adapter path must be readable on the **server** machine. If your client and server are on different hosts, ensure the adapter is accessible via a shared mount or copied to the server.


## Init-time LoRA

### How It Works

Passing `lora_path` to `Omni(...)` instructs the engine to register a single adapter at startup and activate it as the only adapter for every request. The adapter occupies one slot of the LoRA cache for the lifetime of the process.

### Usage

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    lora_path="/path/to/lora_adapter",
    lora_scale=1.0,  # optional, default 1.0
)

outputs = omni.generate(
    "A piece of cheesecake",
    OmniDiffusionSamplingParams(height=1024, width=1024, num_inference_steps=9),
)
images = outputs[0].request_output.images
```

The CLI wrapper `examples/offline_inference/text_to_image/text_to_image.py` exposes these two kwargs as `--lora-path` and `--lora-scale`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "A piece of cheesecake" \
  --lora-path /path/to/lora_adapter \
  --lora-scale 1.0 \
  --output outputs/cheesecake.png
```

### Limitations

- Exactly one adapter, chosen at init. The adapter cannot be swapped or disabled for individual requests — restart `Omni` to change it.
- Mutually exclusive with `--lora-paths` in the example CLI. Use per-request LoRA when you need different adapters on different requests.


## Per-request LoRA

### How It Works

Each request carries its own adapter set via `OmniDiffusionSamplingParams`:

```python
sampling_params = OmniDiffusionSamplingParams(
    ...,
    lora_requests=[req_a, req_b],  # list of LoRARequest
    lora_scales=[1.0, 0.5],        # same length as lora_requests
)
```

- `lora_requests=[]` (or omitted) → no LoRA applied to this request.
- `lora_requests=[req]` → single adapter at the given scale.
- `lora_requests=[req_a, req_b, ...]` → multi-LoRA: all listed adapters are activated simultaneously, each in its own cache slot, and their deltas are summed during the forward pass.

The cache is sized by `max_loras` (defaults to 1). Set `Omni(..., max_loras=N)` when you plan to activate up to `N` adapters concurrently — requests exceeding this limit are rejected. The example CLI at `examples/offline_inference/text_to_image/text_to_image.py` auto-sizes this to `max(len(--lora-paths), 1)` when `--max-loras` is omitted.

### Scale Semantics

- `lora_scales[i]` multiplies adapter `i`'s contribution to the output delta.
- `lora_scales[i] == 0.0` skips the adapter for this activation: it is filtered out before the `max_loras` check, so it does not consume a slot and produces no delta. Subsequent requests that re-enable the adapter will reload it into the cache if it was evicted.
- When `lora_requests` is set and `lora_scales` is omitted, every adapter defaults to scale `1.0`.

### Usage

**Single adapter (per-request):**

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

omni = Omni(model="Tongyi-MAI/Z-Image-Turbo", max_loras=1)

req = LoRARequest(
    lora_name="style_a",
    lora_int_id=stable_lora_int_id("/path/to/style_a"),
    lora_path="/path/to/style_a",
)

outputs = omni.generate(
    "A piece of cheesecake",
    OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=9,
        lora_requests=[req],
        lora_scales=[1.0],
    ),
)
```

**Multi-LoRA composition:**

```python
omni = Omni(model="Tongyi-MAI/Z-Image-Turbo", max_loras=2)

req_a = LoRARequest(lora_name="style_a", lora_int_id=stable_lora_int_id("/lora/a"), lora_path="/lora/a")
req_b = LoRARequest(lora_name="style_b", lora_int_id=stable_lora_int_id("/lora/b"), lora_path="/lora/b")

outputs = omni.generate(
    "A piece of cheesecake",
    OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=9,
        lora_requests=[req_a, req_b],
        lora_scales=[1.0, 0.5],
    ),
)
```

**Switching adapters between requests** — issue separate `omni.generate(...)` calls with different `OmniDiffusionSamplingParams`. `sampling_params_list` on `omni.generate` is stage-indexed (one entry per pipeline stage) and is shared across all prompts in a batch, so per-prompt adapter variance within a single batch call is not supported through that path.

**CLI:**

The example CLI exposes `--lora-paths` + `--lora-scales` for per-request composition, and `--axis` for Cartesian-product XYZ plots that can put any parameter on any of the three axes. Supported axis types are `prompt`, `lora_scale[i]` (i-th `--lora-paths` entry), `guidance_scale`, `num_inference_steps`, and `seed`. X is columns, Y is rows, Z writes one `grid_z{k}.png` per value.

```bash
# Compose two adapters on one prompt
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "A piece of cheesecake" \
  --lora-paths /lora/a /lora/b \
  --lora-scales 1.0 0.5 \
  --max-loras 2 \
  --output-dir outputs/composed/

# 2×2 LoRA-scale grid across 2 prompts (Z): produces grid_z00.png + grid_z01.png
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --lora-paths /lora/a /lora/b \
  --max-loras 2 \
  --axis "x=lora_scale[0]:0|1" \
  --axis "y=lora_scale[1]:0|1" \
  --axis "z=prompt:a girl|a cat" \
  --output-dir outputs/axis_test/
```

### Limitations

- Up to `max_loras` adapters per request. Requests that exceed the limit fail fast before inference.
- All adapters in one request share the same forward pass; they must target compatible modules (scheme enforced by PEFT's `target_modules` field). Adapters targeting disjoint modules compose trivially; overlapping modules add linearly.
- `max_loras` sizes the cache at init and is not resizable at runtime.


## Wan2.2 LightX2V Offline Assembly

This workflow is LoRA-adjacent: it uses external LightX2V conversion plus
`Wan2.2-Distill-Loras` to bake converted Wan2.2 I2V checkpoints into a local
Diffusers directory, instead of loading LoRA adapters at runtime.

### Required assets

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Diffusers skeleton: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- Optional external converter from the LightX2V project (not shipped in this repository)
- Optional LoRA weights: `lightx2v/Wan2.2-Distill-Loras`

### Step 1: Optional - convert high/low-noise DiT weights with LightX2V

Install or clone LightX2V from the upstream repository
(`https://github.com/ModelTC/LightX2V`). After cloning, the converter used
below is available at `<lightx2v_root>/tools/convert/converter.py`.

```bash
python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/high_noise_model \
  --output /tmp/wan22_lightx2v/high_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file

python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/low_noise_model \
  --output /tmp/wan22_lightx2v/low_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file
```

If you are not using LightX2V, skip this step and either keep the original
Diffusers weights from the skeleton or point Step 2 at any other converted
`transformer/` and `transformer_2/` checkpoints.

### Step 2: Assemble a final Diffusers-style directory

```bash
python tools/wan22/assemble_wan22_i2v_diffusers.py \
  --diffusers-skeleton /path/to/Wan2.2-I2V-A14B-Diffusers \
  --transformer-weight /tmp/wan22_lightx2v/high_noise_out \
  --transformer-2-weight /tmp/wan22_lightx2v/low_noise_out \
  --output-dir /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --asset-mode symlink \
  --overwrite
```

`--transformer-weight` and `--transformer-2-weight` are optional. If you omit
them, the tool keeps the original weights from the Diffusers skeleton.

### Step 3: Run offline inference

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --image /path/to/input.jpg \
  --prompt "A cat playing with yarn" \
  --num-frames 81 \
  --num-inference-steps 4 \
  --tensor-parallel-size 4 \
  --height 480 \
  --width 832 \
  --flow-shift 12 \
  --sample-solver euler \
  --guidance-scale 1.0 \
  --guidance-scale-high 1.0 \
  --boundary-ratio 0.875
```

Notes:

- This route avoids runtime LoRA loading changes in vLLM-Omni when you choose to bake converted weights into a local Diffusers directory.
- Output quality and speed depend on the replacement checkpoints and sampling params you choose.


## See Also

- [Text-to-Image Offline Example](../examples/offline_inference/text_to_image.md#lora) - Complete offline LoRA example
- [Text-to-Image Online Example](../examples/online_serving/text_to_image.md#lora) - Complete online LoRA example
