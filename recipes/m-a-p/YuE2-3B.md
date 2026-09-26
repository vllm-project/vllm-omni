# YuE2-3B

> Text-to-music: style + lyrics (optionally an ABC score) in, a 48 kHz stereo song out

## Summary

- Vendor: m-a-p (Multimodal Art Projection)
- Model: `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`
- Task: Text-to-music generation with an editable symbolic plan (ABC notation)
- Mode: Offline generation and online `/v1/audio/speech` serving (adapter `tts_adapters/yue2.py`)
- Integration issue: [#7661](https://github.com/vllm-project/vllm-omni/issues/7661)

## When to use this recipe

Use it to generate songs from a style description and lyrics. Unlike MiniMax
Music 3, YuE2-3B first writes an ABC score (chords and melody, `cot=full` /
`cot=melody`) that you can inspect, edit and resubmit (`--abc-file`), then
renders it with the lyrics. `cot=off` skips the score and generates music
directly. Weights are **CC BY-NC 4.0** (non-commercial); this recipe loads
user-downloaded weights and bundles none.

## References

- Model card: [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
- Upstream: [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) (pinned `bd90e4c`)
- Offline example: `examples/offline_inference/yue2/end2end.py`
- Deploy config: `vllm_omni/deploy/yue2.yaml` (auto-discovered for `model_type=yue2`)

## Architecture

| Component | Spec |
| --- | --- |
| Backbone | Qwen3-1.7B-class decoder, 28 layers, hidden 2048, GQA 16/8, q/k-norm, vocab 184,704, context 24,576 |
| MoT | NAR path (`nar_self_attn`/`nar_mlp` per layer) shares embed_tokens, final norm and lm_head with the AR path |
| Frames | Single codebook: one codec token per frame, 25 frames/s, span [151853, 184521) |
| Sampling | temperature/top-k/top-p + windowed repetition penalty, seeded; model-owned (phase-masked vocab) |
| Guidance | Off by default (1.0 full/melody, 1.01 off); not exercised |
| Acoustic | 32-step midpoint flow matching over 64-dim latents on the NAR path, chunked ~12k frames |
| VAE | `m-a-p/YuE2-Vae` decoder-only, tiled (1024 core + 16 halo frames), FP32 |
| Output | 48 kHz stereo, whole song delivered when the semantic request finishes |

Generation is one or two requests on a single AR stage: the abc phase writes
the score, the semantic phase generates codec frames and, on its last step,
solves the ODE and decodes the whole song in-engine (weights load once; the
VAE loads at startup from `$YUE2_VAE` or the hub id).

## Running

```bash
export YUE2_VAE=/models/YuE2-Vae   # or omit and let it resolve from the hub
python examples/offline_inference/yue2/end2end.py \
    --model /models/YuE2-3B \
    --style "A melancholic lo-fi hip-hop track at 85 BPM in F minor" \
    --lyrics "[Verse]
Walking down the empty street at midnight
[Chorus]
And I keep on walking" \
    --cot full --seed 831001 --max-frames 200 --output song.wav
```

`--max-frames 200` caps the song at 8 s (25 frames/s). The song also ends on
its own end token; `truncated` in the report line says which happened.

## Notes

- **Memory (RTX 4090, 24 GB, `gpu_memory_utilization: 0.70`):** the engine
  reserves ~15.8 GiB after startup; with 4 concurrent requests pinned to the
  9000-frame cap (worst case), peak usage during the terminal NAR/VAE
  finishing pass reaches ~20.9 GiB, leaving ~3 GiB of headroom. At 0.85 the
  same workload OOMs inside the finishing pass, so the deploy yaml pins 0.70.
- **Known limitations:** 4 concurrent full-length requests are the verified
  shape (`max_num_seqs: 4`); CUDA graph capture is a
  follow-up; the abc phase runs eagerly
  after prefill (its tokens are the product, not audio).
- Audio is not expected to match the upstream torch reference bit-for-bit
  (fused vs eager kernels). Measured against the upstream torch reference
  (bd90e4c, same style/lyrics/seed 831001, cot=off, 200 frames): prompt token
  ids 63/63 bit-identical, and the sampled semantic stream agrees
  bit-for-bit for the first 55 frames before kernel numerics diverge — well
  past the "first frames agree" acceptance bar of the MiniMax Music 3
  precedent.
