# Krea 2 for text-to-image (base + LoRA)

## Summary

- Vendor: Krea
- Model: `krea/Krea-2-Turbo` (few-step distilled) and `krea/Krea-2-Raw`
- Task: Text-to-image (T2I)
- Mode: Offline engine and online OpenAI-compatible serving
- Maintainer: Community

## When to use this recipe

Use this recipe for a known-good starting point to run the Krea 2 single-stream
MMDiT with vLLM-Omni on a **single 80 GB NVIDIA H100** — both the offline
`text_to_image.py` entrypoint and the online `/v1/images/generations` route,
including PEFT LoRA adapters. `Krea2Pipeline` is auto-detected from
`model_index.json`; guidance is checkpoint-aware (distilled Turbo runs no-CFG,
Raw runs CFG).

## References

- Related example under `examples/`:
  [`examples/offline_inference/text_to_image/README.md`](../../examples/offline_inference/text_to_image/README.md)
  (see the "Krea 2" section)
- Feature support:
  [`docs/user_guide/diffusion_features.md`](../../docs/user_guide/diffusion_features.md)
- Related PR: [#4730](https://github.com/vllm-project/vllm-omni/pull/4730)

## Hardware Support

This recipe documents a **single-GPU** CUDA layout on H100 80 GB. Add more
platforms (ROCm / NPU) or multi-GPU layouts as community validation lands.

## GPU

### 1× H100 80GB

#### Environment

Versions from a working editable install (activate `vllm-omni/.venv` or your equivalent):

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA driver **595.71.05** (CUDA 13.x)
- torch: **2.11.0+cu130**
- vLLM: **0.24.0**
- vLLM-Omni: editable install from this repo (Git **`960ce8f6`**)
- Transformers: **5.12.1**

Krea 2 weights are ~30 GB and fit comfortably on a single 80 GB H100.

#### Command

Offline — few-step distilled (Turbo) checkpoint (2048×2048, no CFG):

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model krea/Krea-2-Turbo \
  --prompt "a fox in the snow" \
  --num-inference-steps 8 \
  --guidance-scale 0.0 \
  --height 2048 --width 2048 \
  --output krea2_turbo.png
```

Offline — Raw checkpoint (1024×1024, CFG enabled):

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model krea/Krea-2-Raw \
  --prompt "a fox in the snow" \
  --num-inference-steps 28 \
  --guidance-scale 4.5 \
  --height 1024 --width 1024 \
  --output krea2_raw.png
```

LoRA (offline) — pass a PEFT adapter with `--lora-path` / `--lora-scale`.
`NagaSaiAbhinay/Krea-2-vllm-darkbrush-LoRA` is a vLLM-Omni-compatible repackaging
of `krea/Krea-2-LoRA-darkbrush`; its trigger word is `monochrome ink wash style`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model krea/Krea-2-Turbo \
  --prompt "a fox in the snow, monochrome ink wash style" \
  --num-inference-steps 8 --guidance-scale 0.0 \
  --lora-path NagaSaiAbhinay/Krea-2-vllm-darkbrush-LoRA --lora-scale 1.0 \
  --output krea2_darkbrush.png
```

Online — serve and query the OpenAI-compatible image route:

```bash
vllm serve krea/Krea-2-Turbo --omni --enforce-eager --port 8091
```

```bash
curl -s http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a fox in the snow, monochrome ink wash style",
    "size": "1024x1024",
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "seed": 42,
    "lora": {"name": "darkbrush", "path": "NagaSaiAbhinay/Krea-2-vllm-darkbrush-LoRA", "scale": 1.0}
  }' | jq -r '.data[0].b64_json' | base64 -d > krea2_online.png
```

Cache-DiT acceleration (offline) — add `--cache-backend cache_dit`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model krea/Krea-2-Turbo --prompt "a fox in the snow" \
  --num-inference-steps 8 --guidance-scale 0.0 \
  --cache-backend cache_dit --output krea2_cachedit.png
```

#### Verification

```bash
# Offline: the script prints "Saved generated image to <path>"; confirm the file exists.
ls -lh krea2_turbo.png
```

For the online route, decode the returned `b64_json` (see the `curl` above) and
confirm a valid 1024×1024 PNG is written.

#### Notes

- Key flags: distilled Turbo → few steps (~8) with `guidance_scale=0.0`; Raw →
  more steps (~28) with `guidance_scale>0` (CFG). `guidance-scale` follows the
  Krea 2 convention `velocity = cond + guidance_scale * (cond - uncond)`.
- LoRA: adapters must be PEFT format (`adapter_config.json` +
  `adapter_model.safetensors`); the projections are `ReplicatedLinear`
  (264 modules). Offline uses `--lora-path` (which sets
  `OmniDiffusionSamplingParams.lora_request` / `lora_scale`); online uses the
  top-level `lora` object. Style LoRAs like darkbrush need their trigger word in
  the prompt.
- Cache-DiT is supported; `has_separate_cfg` is checkpoint-aware (False for Turbo
  no-CFG, True for Raw CFG).
- Known limitations: TP / SP / CFG-Parallel are not yet wired for Krea 2. HSDP,
  layerwise CPU offload, and VAE-patch-parallel (decode) are supported.
