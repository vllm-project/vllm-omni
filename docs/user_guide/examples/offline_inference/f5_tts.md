# F5-TTS (Text-To-Speech)

F5-TTS is a flow-matching based text-to-speech model that generates high-quality speech with voice cloning support.

## Supported Models

| Model ID | Vocoder | Notes |
|----------|---------|-------|
| `SWivid/F5-TTS/F5TTS_v1_Base` | Vocos | Recommended default |
| `SWivid/F5-TTS/F5TTS_v1_Base_no_zero_init` | Vocos | Alternative checkpoint |
| `SWivid/F5-TTS/F5TTS_Base` | Vocos | Older version, pe_attn_head=1 |
| `SWivid/F5-TTS/F5TTS_Base_bigvgan` | BigVGAN | Uses BigVGAN vocoder |

## Offline Inference

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="SWivid/F5-TTS/F5TTS_v1_Base",
    enforce_eager=True,
)

params = OmniDiffusionSamplingParams(
    num_inference_steps=32,
    guidance_scale=2.0,
    seed=42,
)
# Required so the pipeline honors a non-default guidance_scale.
params.guidance_scale_provided = True

output = omni.generate(
    {
        "prompt": "Hello, this is a test of F5 TTS synthesis.",
        "additional_information": {
            "ref_audio": ["https://example.com/ref_audio.wav"],
            "ref_text": "Reference text matching the audio.",
        },
    },
    params,
)
```

## With Cache Acceleration

### Cache-DiT (recommended)

Block-level caching skips redundant DiT blocks within each ODE step —
measured **1.62x speedup with zero quality loss** (Colab L4, seed-tts-eval
EN, 200 samples):

```python
omni = Omni(
    model="SWivid/F5-TTS/F5TTS_v1_Base",
    enforce_eager=True,
    cache_backend="cache_dit",
    cache_config={
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
    },
)
```

### TeaCache (not supported for F5-TTS)

TeaCache (step-level caching) is **fundamentally incompatible** with
F5-TTS's flow-matching + sway sampling: exhaustive tuning (48+
configurations) found no setting that achieves both zero quality loss and
a speedup. Do not use `cache_backend="tea_cache"` with F5-TTS. See
`docs/user_guide/diffusion/cache_acceleration/teacache.md`.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inference_steps` | 32 | ODE solver steps (16 for faster, 32 for quality) |
| `guidance_scale` | 2.0 | Classifier-free guidance strength |
| `seed` | None | Random seed for reproducibility |

## Notes

- F5-TTS uses flow-matching (not standard diffusion), with an Euler ODE solver
- Reference audio is recommended to be 3-12 seconds
- The model outputs 24kHz mono audio
- All F5-TTS variants use float32 for the transformer (automatically cast)
