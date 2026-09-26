# Ovis-Image text encoder FP8

Ovis-Image supports online FP8 for the Qwen3 text encoder through an explicit
component configuration:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="ATH-MaaS/Ovis-Image-7B",
    dtype="bfloat16",
    quantization_config={"text_encoder": {"method": "fp8"}},
)
```

This converts attention and MLP projections under `text_encoder.layers`.
Embeddings, normalization, the diffusion transformer and VAE retain their
original precision. A global `"fp8"` setting does not opt this encoder in.
Only unquantized BF16/FP16 checkpoints with dynamic activation scaling are
supported. Use full projection prefixes in `ignored_layers`, for example
`text_encoder.layers.0.self_attn.q_proj`.

## Reproduce the image comparison

Use a local copy of the official checkpoint in `OVIS_MODEL`. Set `OVIS_CFG=2`
and expose two GPUs to repeat with CFG parallelism. Each mode performs two
warmups and six measured requests, with identical seeds and 30 denoising steps.

```bash
OVIS_MODEL=/path/to/Ovis-Image-7B OVIS_CFG=1 CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import os
import statistics
import time
from pathlib import Path

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

prompts = [
    "A red ceramic teapot and two blue cups on a wooden table, soft window light, detailed product photograph.",
    'A storefront with a clear sign reading "OPEN", a bicycle parked beside the door, watercolor illustration.',
    "Three yellow rubber ducks floating in a turquoise pool, overhead photograph, realistic ripples.",
]
for mode in ("bf16", "fp8"):
    engine = Omni(
        model=os.environ["OVIS_MODEL"], dtype="bfloat16", enforce_eager=True,
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=int(os.environ["OVIS_CFG"])),
        quantization_config={"text_encoder": {"method": "fp8"}} if mode == "fp8" else None,
        enable_diffusion_pipeline_profiler=True,
    )
    output = Path(f"ovis-{mode}")
    output.mkdir(exist_ok=True)
    timings = []
    for index in range(8):
        prompt = prompts[0 if index < 2 else (index - 2) % 3]
        params = OmniDiffusionSamplingParams(
            width=768, height=512, num_inference_steps=30,
            guidance_scale=4.0, seed=142, num_outputs_per_prompt=1,
        )
        start = time.perf_counter()
        result = engine.generate(
            {"prompt": prompt, "negative_prompt": "blurry, distorted", "modalities": ["image"]},
            sampling_params_list=[params],
        )
        elapsed = time.perf_counter() - start
        extract_images_from_outputs(result)[0].save(output / f"{index}.png")
        if index >= 2:
            timings.append(elapsed)
    print(mode, "mean", statistics.mean(timings), "stdev", statistics.stdev(timings))
    engine.close()
PY
```

## L20 validation

The official checkpoint at revision `41be1c5821a92c970d63d7eb595a2fd3fe32b22e`
completed the comparison above on one L20 and on two L20s with CFG parallelism.
The rank-0 snapshot verified 196 FP8 decoder projections and BF16 DiT/VAE
parameters. Memory figures below report rank 0, not the maximum across ranks.

| Configuration | BF16 rank-0 peak allocated | FP8 rank-0 peak allocated |
| --- | --- | --- |
| 1×L20 | 18.008 GiB | 16.690 GiB |
| 2×L20, CFG=2 | 17.999 GiB | 16.680 GiB |

Three prompt pairs retained the requested objects, counts and OPEN text; paired
image SSIM was 0.9667, 0.6998 and 0.9380. The storefront sign layout changed.
Single-card and CFG-parallel PNG files matched exactly within each precision mode.
These samples establish smoke coverage, not general quality equivalence.

The text encoder was slower with FP8 at this request size. Shared GPU workloads
caused substantial E2E timing variation, so these runs establish memory savings
and functional coverage without a reliable latency speedup claim.
