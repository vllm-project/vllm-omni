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
seeded prompt do not establish general perceptual quality. The encoder remains
replicated; encoder TP is not implemented.

## Encoder layerwise offload

To stream both Gemma3 decoder blocks and the video Transformer from CPU,
select both components. This also keeps the unquantized Transformer on CPU
during initialization when encoder-only FP8 is enabled.

```bash
CUDA_VISIBLE_DEVICES=0,1 DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Lightricks/LTX-2 --model-class-name LTX2Pipeline \
  --quantization-config '{"text_encoder":{"method":"fp8"}}' \
  --enable-layerwise-offload \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --ulysses-degree 2 --vae-use-tiling --enforce-eager \
  --height 512 --width 768 --num-frames 17 --num-inference-steps 40 \
  --guidance-scale 4 --fps 24 --seed 42 \
  --prompt "A golden retriever walks through a grassy meadow in gentle sunlight. Birds chirp in the distance." \
  --output ltx2_encoder_fp8.mp4
```

The text encoder is replicated across workers; `ulysses_degree` shards video
Transformer computation. Block offload reduces encoder residency in both
precision modes, so its GPU-memory comparison differs from keeping the
complete encoder resident.

### Repeated two-GPU validation

Two L20 workers with Ulysses=2, the component offload configuration above,
17 frames, 512×768, 40 steps, CFG 4, 24 FPS and seed 42. Each independent
process completed one warmup followed by three measured requests. The prompt
is the dog/meadow prompt in the command above. Source: `05da0e18e`.

| Metric | BF16 | Encoder FP8 | Comparison |
| --- | ---: | ---: | --- |
| E2E mean ± sample SD (s) | 133.029 ± 1.045 | 131.108 ± 0.093 | 1.015× |
| Text encoder mean (s) | 1.750 | 0.934 | 1.875× |
| Rank-0 peak allocated (GiB) | 12.397 | 12.188 | 0.209 GiB lower |
| Rank-0 peak reserved (GiB) | 15.932 | 15.457 | 0.475 GiB lower |

Measured E2E samples were 131.846 / 133.825 / 133.417 s for BF16 and
131.143 / 131.002 / 131.178 s for FP8. Timings exclude model loading and MP4
export. These are observations from three requests on a shared host, not an
isolated general speedup claim. The profiler and eager mode were enabled in
both arms. Memory numbers are rank-0 PyTorch allocator peaks, not total
process memory or a maximum across workers.

Both MP4 files decode to 17 RGB frames and stereo 24 kHz audio. Frame SSIM is
0.9447 and audio waveform cosine is 0.9750; the dog, meadow and motion are
retained. The repeated BF16 MP4 matches the earlier two-GPU BF16 output
byte-for-byte. Shutdown emitted worker termination warnings after successful
export in both precision modes.
