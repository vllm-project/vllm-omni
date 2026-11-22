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
the score, the semantic phase generates codec frames, then queues the ODE and VAE on a
high-priority CUDA stream. Other requests keep decoding while the finishing
request emits HOLD tokens (up to 4096 steps, within its token/context budget).
The scheduler is unchanged. Weights load once; the VAE loads at startup from
`$YUE2_VAE` or the hub id.

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

`--max-frames 200` caps the song at 8 s (25 frames/s). The end token is
masked for the preset's first 200 steps, so a 200-frame budget always
truncates; a longer budget lets the song end on its own end token, and
`truncated` in the report line says which happened. Serving follows the
same rule: omit `max_new_tokens` (or set it above 200) for a natural
ending.

## Notes

The lifecycle e2e regression forces KV-cache preemption with chunked and
unchunked recompute, mixes ABC and semantic sampling, and cancels a request
after NAR work has started. It checks history alignment, released synthesis
buffers, and successful generation after cancellation. Run with one H100/H200
and cached YuE2-3B and YuE2-Vae weights:

```bash
export YUE2_MODEL_DIR=/path/to/YuE2-3B
export YUE2_VAE=/path/to/YuE2-Vae
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/e2e/online_serving/test_yue2.py \
    -k preemption_and_synthesis_abort -m 'slow and tts' --run-level full_model -q -s
```

The test retains the module's weekly TTS routing; it is not a per-PR CI gate.

- **Memory:** `gpu_memory_utilization` budgets the vLLM engine; NAR K/V,
  acoustic graph buffers and VAE activations also need room during synthesis.
  The 0.70 / 4-slot defaults require validation on the target card. The earlier
  RTX 4090 measurement (15.8 GiB after startup, 20.9 GiB peak at 9000 frames)
  predates the async/compiled implementation; it does not validate this
  version or long ABC prefixes on a 4090. Reduce the fraction if the target
  workload needs more synthesis memory, especially for long ABC prefixes and
  the 9000-frame cap. Releasing completed chunks bounds live NAR buffers,
  while the shared graph allocator retains reserved storage for reuse.
- **CUDA graphs:** the default deploy keeps the AR backbone eager. Both ABC
  and semantic sampling use prewarmed power-of-two graph buckets. Custom
  sampler captures have a bounded cache and fall back to eager sampling when
  it fills. The NAR velocity pass uses a chunk graph on the serialized synthesis
  stream. Chunk graphs share one allocator pool; completed chunks release
  their engines and buffers after their own events. FA3 runs on Hopper; other
  CUDA cards use SDPA, which is also warmed at startup.
- **Serving:** run `vllm serve m-a-p/YuE2-3B --omni` with the default
  `yue2.yaml`, which uses eager AR with 4 slots. Validate memory capacity
  for the target card and workload.
  For the tested single-H200 graph settings, reuse this deploy with
  `--stage-overrides '{"0":{"max_num_seqs":32,"gpu_memory_utilization":0.5,"enforce_eager":false,"compilation_config":{"cudagraph_mode":"FULL_AND_PIECEWISE"}}}'`.
  Decode uses FULL and mixed prefill uses PIECEWISE; VAE and the Python
  synthesis queue remain outside these graphs.
  Audio is delivered as a whole song; time to first audio equals completion
  latency. Aborting a running song stops future work units and waits only for
  the at-most-two submitted units before releasing its buffers.
- **Preemption:** before drawing for a resumed row, model-owned sampling
  reconciles the codec history and terminal flags with the engine's retained
  sequence. Discarded draws cannot advance its penalty window or frame budget.
- Audio is not expected to match the upstream torch reference bit-for-bit
  (fused vs eager kernels). Measured against the upstream torch reference
  (bd90e4c, same style/lyrics/seed 831001, cot=off, 200 frames): prompt token
  ids 63/63 bit-identical, and the sampled semantic stream agrees
  bit-for-bit for the first 55 frames before kernel numerics diverge — well
  past the "first frames agree" acceptance bar of the MiniMax Music 3
  precedent.
