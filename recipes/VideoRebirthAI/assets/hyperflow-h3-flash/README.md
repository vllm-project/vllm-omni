# HyperFlow H3 Flash experiment assets (PR evidence)

Prompt:

```text
A clay fox walks through a snowy forest, soft stop-motion style.
```

Shared parameters: `seed=42`, `num_inference_steps=6`, `480×480`, CFG off,
**124** frames / 5.175 s at 24 FPS, 1× NVIDIA L20X.

| File | Source |
| --- | --- |
| `omni_t2va_clayfox_seed42_steps6_f124.mp4` | vLLM-Omni `/v1/videos/sync` (`duration=5.0`) |
| `official_t2va_clayfox_seed42_steps6_f124.mp4` | official `python -m hyperflow_h3_flash` |
| `omni_f124_t1.jpg` / `official_t1.jpg` | stills at t≈1 s |

See [HyperFlow-H3-Flash.md](../../HyperFlow-H3-Flash.md#experimental-results).
