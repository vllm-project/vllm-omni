# Omni Examples

Examples showcasing multi-stage AR + DiT pipelines powered by vLLM-omni.

## AR ➜ DiT with Diffusers Backend

```bash
# Optional: customise models
export AR_MODEL=Qwen/Qwen3-0.6B
export DIT_MODEL=./models/stable-diffusion-2-1/

# Run with defaults (20 steps @ 512x512)
python examples/omni/ar_dit_diffusers.py

# Faster run
python examples/omni/ar_dit_diffusers.py --steps 14 --height 384 --width 384 --guidance 4.5
```

Arguments:
- `--ar-model`: Override AR stage model (default `Qwen/Qwen3-0.6B`).
- `--dit-model`: Override DiT stage model (default env `DIT_MODEL`, else diffusers repo).
- `--steps`: Number of diffusion steps (default 20).
- `--guidance`: Guidance scale (default 5.0).
- `--height` / `--width`: Output resolution (default 512).
- `--seed`: Optional RNG seed.
- `--prompt`, `--temperature`, `--max-tokens`: Control AR stage generation.
- `--output`: Destination path for the generated image file.

Generated images are saved next to the script as `omni_dit_output.png` by default.

See the script for additional tweaks (scheduler, prompt chaining, etc.).
