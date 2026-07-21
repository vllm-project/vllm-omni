# Robot-Policy

This unified example script predicts robot actions from task prompt.

- `robot_policy.py`: command-line script for single-shot / AR trajectory predictions with advanced options.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Local CLI Usage](#local-cli-usage)
- [Key Arguments](#key-arguments)


## Overview

`robot_policy.py` loads a robot-policy DiT model, builds observations from a task prompt and a directory of organized assets, and outputs a predicted action sequence saved as a `.npz` file by default. Model-specific behavior is registered per model under `vllm_omni/model_extras/`.

### Supported Models

| Model | Obs. Resolution | Peak VRAM (GiB) | Model Weights (GiB) | Notes |
|-------|----------------|-----------------|---------------------|-------|
| `GEAR-Dreams/DreamZero-DROID` | 180 × 320 | 71.29 | 64.8 | VLA robot policy; deploy config auto-resolved; AR-Diffusion engine required usage |

!!! info
    Peak VRAM: based on basic single-card usage, batch size = 1, without any acceleration/optimization features. Some model weights may need one card with 80 GiB VRAM or more.

Default model: `GEAR-Dreams/DreamZero-DROID`.

## Inference Modes

`robot_policy.py` does **not** require a CLI flag to pick the execution mode — it is inferred
automatically from the return type of the model's `build_robot_observations`:

- **Single-shot** — the observation builder returns a single `dict`. One forward pass produces
  the full action sequence in one shot. Used by e.g. `InternVLA-A1`.
- **Autoregressive (AR)** — the observation builder returns an iterable of `dict` objects, one
  per rollout step. The script loops over the observations, generating one action chunk per
  step. Used by e.g. `GEAR-Dreams/DreamZero-DROID` (requires the AR-Diffusion engine).

## Prerequisites

### DreamZero

Download default example assets with following command:

```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets --repo-type dataset --local-dir outputs/dreamzero/assets
```

## Local CLI Usage

### DreamZero

```bash
python examples/offline_inference/robot_policy/robot_policy.py \
  --model GEAR-Dreams/DreamZero-DROID \
  --model-class-name DreamZeroPipeline \
  --deploy-config vllm_omni/deploy/dreamzero.yaml \
  --data-dir outputs/dreamzero/assets \
  --task "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan" \
  --extra-body '{"session_id": "dreamzero_example", "num_chunks": 15, "repeat_chunk_observations": false}'
```

**NOTE:**
- `DreamZero` uses `cache-backend = "step_cache"` by default, assigning any other value to `--cache-backend` will overwrite the default cache backend.

## Key Arguments

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--model` | str | `GEAR-Dreams/DreamZero-DROID` | Robot policy model ID or local path |
| `--model-class-name` | str | `None` | Override model class name (e.g., `DreamZeroPipeline`) |
| `--task` | str | `""` | Task prompt string that controls the robot trajectory planning |
| `--deploy-config` | str | `None` | Deploy config YAML |
| `--data-dir` | str | `None` | Directory containing organized assets needed by examples |
| `--seed` | int | `42` | Random seed for deterministic sampling |
| `--dtype` | str | `bfloat16` | dtype maybe used for some models |
| `--device` | str | `cuda` | device maybe used for some models |
| `--output` | str | `robot_policy_output.npz` | Path to save the generated robot action sequence |

| `--vae-use-slicing` | bool | False | Enable VAE slicing for memory optimization |
| `--vae-use-tiling` | bool | False | Enable VAE tiling for memory optimization |
| `--enable-cpu-offload` | bool | False | Enable module-wise (sequential) CPU offload to reduce peak VRAM |
| `--enable-layerwise-offload` | bool | `False` | Enable layerwise offloading on DiT |
| `--cfg-parallel-size` | int | `1` | Set to `2` to enable CFG Parallel |
| `--tensor-parallel-size` | int | `1` | Tensor parallel size (effective for models that support TP, e.g. DreamZero) |
| `--ulysses-degree` | int | `1` | Ulysses sequence parallel degree |
| `--ring-degree` | int | `1` | Ring sequence parallel degree |
| `--cache-backend` | str | `None` | Cache backend |
| `--use-hsdp` | bool | False | Enable Hybrid Sharded Data Parallel |
| `--hsdp-shard-size` | int | `-1` | GPUs per shard group (-1 auto-calculates) |
| `--hsdp-replicate-size` | int | `1` | Number of replica groups for HSDP |
