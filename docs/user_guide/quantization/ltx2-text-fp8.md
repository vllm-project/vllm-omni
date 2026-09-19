# LTX-2 Gemma3 text encoder FP8 (experimental)

Enable online FP8 explicitly for the Gemma3 text encoder:

```python
from vllm_omni import Omni

omni = Omni(
    model="Lightricks/LTX-2",
    dtype="bfloat16",
    quantization_config={"text_encoder": {"method": "fp8"}},
)
```

This converts only the Gemma3 language-model decoder's attention and MLP
projections. Embeddings, norms, the language-model output head, vision tower,
connectors, DiT and VAEs retain their original precision. The diffusion loader
finalizes FP8 weights through its existing post-load path.

The initial implementation accepts BF16/FP16 source weights and dynamic online
FP8 only. Global DiT quantization does not opt the encoder in. Gemma4-based
LTX-2.5 models reject this explicit encoder setting.

Official `Lightricks/LTX-2` video/audio generation was validated on one NVIDIA
L20 with layerwise CPU offload, VAE tiling, eager execution and `TORCH_SDPA`.
Both BF16 and encoder FP8 produced 121 frames at 512×768 and 24 FPS, with a
24 kHz stereo audio track. The workload used 40 steps, CFG 4 and seed 42.
Runtime inspection confirmed 336 FP8 decoder projections; the remaining
components retained BF16.

| Metric | BF16 | Encoder FP8 |
| --- | ---: | ---: |
| First request, excluding model loading and MP4 export (s) | 390.80 | 442.61 |
| Observed speed ratio (BF16 / variant) | 1.000× | 0.883× |
| Peak allocated GPU memory during generation (GiB) | 37.10 | 27.09 |
| Peak reserved GPU memory during generation (GiB) | 40.50 | 30.49 |

These are single requests on different L20s on a shared host, with no explicit
warmup or repetition series. They establish a working full-model path and a
lower generation allocation peak for this workload, not a steady-state speedup
or lower checkpoint-loading peak. Initial loading still uses full-precision
checkpoint weights.

For the tested dog-in-meadow prompt, both videos retained the subject and
motion, with visible texture differences. Mean decoded-frame SSIM was 0.5414
and audio waveform cosine similarity was 0.9254. These measurements from one
seeded prompt do not establish general perceptual quality. Encoder TP,
multi-GPU execution and steady-state latency remain unvalidated.
