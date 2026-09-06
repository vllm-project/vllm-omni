# LoRA (Low-Rank Adaptation) Guide

LoRA (Low-Rank Adaptation) enables fine-tuning diffusion models by adding trainable low-rank matrices to existing model weights. vLLM-Omni supports two LoRA backends: **PEFT** for PEFT-style adapters, and **Distill** usually for few-steps inference. **PEFT** backend allows you to customize model behavior without modifying the base model weights. **Distill** backend fuses LoRA weights into the base model at initialization.

## Overview

LoRA adapters are lightweight, model-specific fine-tuning weights that can be applied to diffusion models in two ways:

- **PEFT backend** (`--lora-backend peft`, default): Loads a PEFT-format adapter folder via `DiffusionLoRAManager`. Adapters are cached (LRU) and activated per request via `LoRARequest`. It uses a unified LoRA handling mechanisms similar to vLLM with LRU cache management.
- **Distill backend** (`--lora-backend distill`): Calls `pipeline.load_lora_weights` once at initialization to fuse one or more concrete checkpoint files directly into the base weights. Typically used for distilled few-step LoRAs (e.g. Lightning, LightX2V).

## LoRA Adapter Format

### PEFT (Parameter-Efficient Fine-Tuning) format (default)

A typical PEFT-format LoRA adapter directory structure:

```text
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

The `adapter_config.json` file contains metadata about the LoRA adapter, including:

- `r`: LoRA rank
- `lora_alpha`: LoRA alpha scaling factor
- `target_modules`: List of module names to apply LoRA to

## Quick Start

### Offline Inference

#### PEFT backend: pre-loaded LoRA

Load a PEFT-format LoRA adapter at initialization. The adapter is pre-loaded into the cache and can be activated per request:

```python
from vllm_omni import Omni
from vllm_omni.lora.request import LoRARequest

lora_path = "/path/to/lora_adapter"

omni = Omni(
    model="stabilityai/stable-diffusion-3.5-medium",
    lora_path=lora_path,
    lora_backend="peft",  # default, can be omitted
)

lora_request = LoRARequest(
    lora_name="preloaded",
    lora_int_id=1,
    lora_path=lora_path
)

outputs = omni.generate(
    prompt="A piece of cheesecake",
    lora_request=lora_request,
    lora_scale=2.0, # optional arg, default 1.0
)
```

#### Distill backend: fuse distilled LoRA at init

For distilled few-step LoRAs, pass `lora_backend="distill"` together with one or more concrete `.safetensors` files. The weights are fused into the base model once at init; subsequent `generate()` calls do not need a `LoRARequest`.

##### Supported pipelines

For Qwen-Image and Wan pipelines, the distill backend calls `pipeline.load_lora_weights(...)` during worker initialization.

| Pipeline | Supported distilled LoRA repo | Notes |
| ---------- | -------------------------------- | ------- |
| `QwenImagePipeline` | `lightx2v/Qwen-Image-2512-Lightning` | Used with Qwen-Image-2512 Lightning-style few-step inference. |
| `Wan22Pipeline` | `lightx2v/Wan2.1-Distill-Loras`, `lightx2v/Wan2.2-Distill-Loras` | Wan2.1 uses one LoRA file. For dual-transformer Wan2.2 MoE, pass high-noise then low-noise LoRA files. |
| `Wan22I2VPipeline` | `lightx2v/Wan2.2-Distill-Loras` | For dual-transformer Wan2.2 MoE, pass high-noise then low-noise LoRA files. |

Other diffusion pipelines are not currently listed as supporting distilled LoRA. Use the PEFT backend for request-time adapters, or bake converted weights into a local Diffusers directory before serving.

Single-file example (Qwen-Image-Lightning):

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image-2512",
    lora_path="/path/to/Qwen-Image-2512-Lightning.safetensors",
    lora_backend="distill",
)

outputs = omni.generate(prompt="A piece of cheesecake")
```

Multi-file example (Wan2.2 MoE, high + low noise):

```python
from vllm_omni import Omni

omni = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    lora_path=[
        "/path/to/wan2.2_high_noise_lora.safetensors",   # -> transformer
        "/path/to/wan2.2_low_noise_lora.safetensors",    # -> transformer_2
    ],
    lora_backend="distill",
)
```

The CLI examples under `examples/offline_inference/` accept the same flags, e.g.:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --lora-backend distill \
  --lora-path /path/to/high.safetensors /path/to/low.safetensors \
  --prompt "A cat playing with yarn"
```

##### Online serving

Distilled LoRAs can also be fused when an online diffusion server starts. The
adapter remains active for the lifetime of that server, so requests do not pass
a per-request `lora` field.

```bash
vllm serve Qwen/Qwen-Image-2512 \
  --omni \
  --port 8091 \
  --lora-backend distill \
  --lora-path /path/to/Qwen-Image-2512-Lightning-4steps.safetensors
```

Then send a normal generation request with the sampling settings expected by
the distilled checkpoint:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A Chinese female college student, around 20 years old, with a very short haircut that conveys a gentle, artistic vibe. Her hair naturally falls to partially cover her cheeks, projecting a tomboyish yet charming demeanor. She has cool-toned fair skin and delicate features, with a slightly shy yet subtly confident expression—her mouth crooked in a playful, youthful smirk. She wears an off-shoulder top, revealing one shoulder, with a well-proportioned figure. The image is framed as a close-up selfie: she dominates the foreground, while the background clearly shows her dormitory—a neatly made bed with white linens on the top bunk, a tidy study desk with organized stationery, and wooden cabinets and drawers. The photo is captured on a smartphone under soft, even ambient lighting, with natural tones, high clarity, and a bright, lively atmosphere full of youthful, everyday energy."}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 4,
      "true_cfg_scale": 1.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > qwen_image_distill_lora_output.png
```

For Wan2.2 MoE serving, pass the high-noise checkpoint first and the low-noise
checkpoint second to `--lora-path`.

!!! note "Server-side Path Requirement"
    The LoRA adapter path (`local_path`) must be readable on the **server** machine. If your client and server are on different machines, ensure the LoRA adapter is accessible via a shared mount or copied to the server.

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
- If you only need to fuse distilled LoRAs into a Wan2.2 checkpoint at load time (without the full LightX2V convert + assemble pipeline), you can instead pass them directly via `--lora-backend distill --lora-path <high>.safetensors <low>.safetensors`. See the [Distill backend](#distill-backend-fuse-distilled-lora-at-init) section above.

## MiniMax-H3 adapter-declared schedules

MiniMax-H3 supports two few-step mechanisms that must not be conflated:

- **Checkpoint-pinned schedule**: a merged release writes `base_schedule` into
  `model_index.json`. Requests must pass `num_inference_steps` as the interval
  count (for example `4` for `[1.0, 0.7, 0.4, 0.15, 0.0]`).
- **Runtime Turbo LoRA**: LightX2V Turbo artifacts request five sigma points and
  enforce `flow_shift=6`, `audio_flow_shift=3`.
- **Runtime native LoRA**: FlashGen-style artifacts declare
  `key_format=minimax-h3-native` and embed `base_schedule` in safetensors
  metadata. When active, the adapter schedule overrides the base checkpoint and
  requests must use the interval-count contract (`num_inference_steps=4`).
  Request-mode generation may omit the field and take the count from the adapter
  schedule; step execution requires it explicitly, because the step scheduler
  derives the total step count from the request before the adapter schedule is
  known.

Native artifacts also declare `qkv_layout=grouped`. The H3 loader reorders fused
`qkv_proj` LoRA rows with the same `_reorder_grouped_qkv_to_qkv` helper used for
base-weight loading, then binds the packed Q/K/V slices through the legacy PEFT
manager without modifying `DiffusionLoRAManager`.

Both runtime adapters run with distributed layerwise offload in request-mode
generation, where the manager keeps the LoRA A/B buffers resident on the compute
device while DLO streams the base blocks. MiniMax-H3 step execution still
rejects DLO, so `--step-execution` cannot be combined with
`--enable-distributed-layerwise-offload`. Model-level CPU offload and standard
layerwise offload are rejected in every mode, because the dynamic LoRA tensors
are neither parameters nor registered buffers and therefore do not participate
in those weight lifecycles.

## See Also

- [Text-to-Image Offline Example](../examples/offline_inference/text_to_image.md#lora) - Complete offline LoRA example
- [Text-to-Image Online Example](../examples/online_serving/text_to_image.md#lora) - Complete online LoRA example
