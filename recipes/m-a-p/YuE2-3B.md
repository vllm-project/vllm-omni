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

To start from a recording, use the [SheetSage2 H200 recipe](SheetSage2-H200.md)
to transcribe it into ABC in a separate environment, then pass the score to
YuE2 with `--abc-file`. It includes full-song frame budgets and offline cover
validation details.

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
| Guidance | Paired semantic CFG, default scale 1.0 (disabled) |
| Acoustic | 32-step midpoint flow matching over 64-dim latents on the NAR path, chunked ~12k frames |
| VAE | `m-a-p/YuE2-Vae` decoder-only, tiled (1024 core + 16 halo frames), FP32 |
| Output | 48 kHz stereo, whole song delivered when the semantic request finishes |

The single `yue2` pipeline uses one worker and one set of model weights.
Semantic completion releases the request's native AR slot and paged KV
blocks. NAR and VAE finish in a model-owned acoustic queue, while the native
request stays alive until complete audio is returned or cancellation cleans
it up. Synthesis does not consume HOLD tokens. The VAE is loaded at startup
from `hf_overrides.yue2_vae_path`.

### Automatic ABC and paired CFG

YuE2 supports automatic ABC: with `cot=full` or `cot=melody`, omit
`abc` to generate the score and then semantic tokens within one request.
Supplying `abc` uses that score directly; `cot=off` skips ABC. ABC and semantic
sampling retain their independent presets and restart at the same request
seed. Offline outputs include `abc_token_ids` and `meta.abc_truncated` when
ABC was generated. The speech endpoint returns the final audio and reports
budget truncation through its existing response metadata.

```bash
vllm serve m-a-p/YuE2-3B --omni \
    --deploy-config vllm_omni/deploy/yue2.yaml
```

```bash
curl http://localhost:8000/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"m-a-p/YuE2-3B","input":"[Verse]\nWalking home tonight", "instructions":"Gentle acoustic folk", "seed":831001, "max_new_tokens":200, "response_format":"wav", "extra_params":{"cot":"full","abc_max_tokens":4096}}' \
    --output song.wav
```

`max_new_tokens` is the semantic frame budget (up to 9000); `abc_max_tokens`
is a separate ABC budget (1–4096). The prompt plus both budgets must fit
`max_model_len`. A truncated nonempty ABC can still proceed to synthesis;
an empty or invalid score fails that request. The existing offline script
continues to use two calls. Native callers can use
`vllm_omni.model_executor.models.yue2.request.prepare_song` to obtain a
token prompt and `SamplingParams` for one native request.

Set `extra_params.cfg_scale`, for example `1.5`, to enable paired semantic
CFG on the same endpoint. The default is `1.0` (no negative branch); finite
values from 0 to 20 are accepted. The negative branch shares the exact ABC,
uses a separate native AR slot and paged KV, and receives the token sampled
once from guided logits on the positive branch. Only the positive branch
synthesizes audio.

CFG needs at least two AR slots, complete prefills, sufficient prefill token
budget for both actual prefixes, and enough KV capacity to reserve both
branches through their output budgets. A pair that cannot fit fails its
request instead of waiting forever. Set deployment capacity for the expected
request lengths; four AR slots allow at most two guided semantic songs.
Ordinary CFG=1 songs retain native AR admission and support one-slot servers.

This implementation requires one GPU (`TP=PP=1`), model runner v1,
synchronous FCFS scheduling, prefix caching disabled, and no external KV
connector or speculative decoding. Chunked prefill remains available for
ordinary requests; CFG requests require it disabled. The default config
keeps the existing 24576 context, four AR slots and eager backbone, with
chunked prefill disabled. Existing two-call ABC/semantic clients and external
scores remain supported; pipeline, architecture and HTTP endpoint names stay
`yue2`, `Yue2ForCausalLM` and `/v1/audio/speech`.

### Acoustic batching and memory

Set `hf_overrides.yue2_nar_batch_size: 2` in `yue2.yaml` to batch up to two
ready songs; VAE decoding stays at one tile at a time. NAR packs real tokens with separate
attention spans; each request writes its conditioning KV directly into the
final packed storage. This removes an intermediate KV copy, not all memory
movement in synthesis. A single-song batch retains the original NAR path.

Set these fields under the stage's `hf_overrides` and restart to apply them:

| Setting | Default | Meaning |
| --- | --- | --- |
| `yue2_nar_batch_size` | 1 | Positive batch cap, independent of AR `max_num_seqs`; use 2 to enable packing |
| `yue2_nar_batch_wait_ms` | 0 | Maximum wait from the oldest ready song; full batches start immediately, B1 never waits |
| `yue2_nar_batch_max_tokens` | 0 | Aggregate conditioning plus NAR token budget; 0 disables this limit |
| `yue2_vae_batch_size` | 1 | VAE tile batch cap; retain 1 for lower peak memory |
| `yue2_vae_tile_frames` | 1024 | Core frames per VAE tile, with the existing halo |

The token budget preserves FIFO order. An oversized first request runs alone;
the budget is not a hard memory limit or an OOM recovery mechanism. Waiting
time includes time spent behind the previous acoustic job. B2 requires two
ready songs and may rarely trigger when AR is the bottleneck. Packed kernels
can change floating-point results and subsequent generated audio; do not
assume bitwise equivalence across batching configurations.

AR paged KV, NAR conditioning KV, graphs and VAE activations all contribute to
peak memory. The synthesis job waits for the final NAR completion event and
releases its KV and graph buffers before allocating VAE workspace. The shared
graph allocator pool retains reserved storage for reuse.
If NAR graph capture runs out of memory, it discards the failed graph,
synchronizes and clears unused cached memory, then attempts capture once more.
This recovery also covers an OOM wrapped by CUDA's capture-invalidated error;
other errors and a second OOM propagate to the synthesis job.
Successful captures and graph replays do not perform this reclamation.

The supplied four-slot / 0.70 memory-utilization settings are starting points,
not a 24 GiB deployment guarantee. For long songs on a 24 GiB card, start with
one AR slot, an explicit 3 GiB KV cache and two client requests, keeping
NAR1/VAE1 and the default zero wait/token budget. This configuration completed
the recorded full-CoT 9000-frame workload; it is not a guarantee for other
prompts or environments. CFG requires at least two AR slots and therefore
needs a separate memory assessment. Tune AR slots, client concurrency and
acoustic caps together on the actual workload.

## Checkpoints and revisions

Both checkpoints accept a local directory or HF repository ID. The deployment
YAML selects the VAE independently of the main model:

```yaml
hf_overrides:
  architectures: [Yue2ForCausalLM]
  yue2_vae_path: m-a-p/YuE2-Vae
  yue2_vae_revision: main
```

Use a commit SHA for each revision to pin the weights. `vllm serve --revision`
selects the main checkpoint and its `qwen.tiktoken`; the VAE uses
`yue2_vae_revision`. The offline example accepts `--revision`, `--vae`,
`--vae-revision`, and `--deploy-config` for the same settings. HF cache and
authentication follow the engine configuration. An explicit `yue2_vae_path`
takes precedence over the existing `YUE2_VAE` environment variable. With the
default YAML value `null`, that variable remains supported; if unset, the VAE
loads from `m-a-p/YuE2-Vae`.

## Running

```bash
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
