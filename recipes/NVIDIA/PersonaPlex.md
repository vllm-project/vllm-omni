# PersonaPlex

> Live full-duplex speech-to-speech (the model listens while it speaks) with
> persona and voice control, on the native vLLM-Omni engine.

## Summary

- Vendor: NVIDIA
- Model: [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
  (gated; a Moshi finetune)
- Task: Full-duplex speech-to-speech. 24 kHz mic audio in, agent speech +
  inner-monologue text out, in 80 ms lockstep. Persona (role text) and voice
  (zero-shot clone) are set per session.
- Mode: Live duplex WebSocket serving through the unified duplex stack
  (`/v1/duplex` and `/v1/realtime?duplex=1`)
- Maintainer: [`@linyueqian`](https://github.com/linyueqian)

## When to use this recipe

Use this recipe to run `nvidia/personaplex-7b-v1` as a live voice agent on one GPU.
The integration is **moshi-free**: the temporal transformer, depformer, and
streaming Mimi codec are all native vLLM-Omni modules (Mimi runs on
`transformers.MimiModel`), so no vendored fork is installed. Decoding is greedy,
matching the reference implementation frame for frame on the golden replays used
as the acceptance gate.

PersonaPlex is the first Moshi-class (pure-lockstep) model on the vLLM-Omni
full-duplex serving stack: it plugs into the generic `/v1/duplex` handler
through the standard plugin seams (`duplex_serving_adapter` /
`duplex_runtime_extension` in its `pipeline.py`) with a model-specific
package at `vllm_omni/model_executor/models/personaplex/duplex/`.

## References

- Duplex serving package:
  [`vllm_omni/model_executor/models/personaplex/duplex/`](../../vllm_omni/model_executor/models/personaplex/duplex/)
- Native model modules (temporal / depformer / streaming Mimi):
  [`vllm_omni/model_executor/models/personaplex/`](../../vllm_omni/model_executor/models/personaplex/)
- Online serving example:
  [`examples/online_serving/personaplex/`](../../examples/online_serving/personaplex/)
- Staged pipeline (talker -> Mimi code2wav, async-chunk streaming):
  [`vllm_omni/deploy/personaplex.yaml`](../../vllm_omni/deploy/personaplex.yaml)
- Integration PR: [vllm-project/vllm-omni#4771](https://github.com/vllm-project/vllm-omni/pull/4771)
- Framework context: [RFC #3745](https://github.com/vllm-project/vllm-omni/issues/3745)
  (duplex adapter patterns), [#1335](https://github.com/vllm-project/vllm-omni/issues/1335)
  (full-duplex target), [PR #3907](https://github.com/vllm-project/vllm-omni/pull/3907)
  (fullduplex core contracts this adapter conforms to)
- Upstream model card:
  [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)

## Why it is different from MiniCPM-o / JoyVL duplex

| | MiniCPM-o 4.5 (#3907) | JoyVL | **PersonaPlex (this)** |
| --- | --- | --- | --- |
| Cadence | 1 s chunk groups | ~1 fps frames | **80 ms lockstep (12.5 Hz)** |
| Turn control | learned `⟨listen⟩`/`⟨speak⟩` | `</silence>`/`</response>` | **none, pure lockstep** |
| Per step | variable-length token group | text decision | **1 user frame in -> 1 agent frame + 1 text token** |
| Barge-in | at chunk boundary | n/a | **native (model always hears the user)** |
| Session state | chunk-group KV | per-tick HTTP | **persistent ring KV + streaming Mimi state** |

This is the lockstep ("parallel-frame joint") shape of full duplex: audio flows
continuously in both directions with no client commits or external turn signals.
On the unified stack this is expressed purely through the capability payload
(`supports_client_commit=false`, `supports_external_turn_signal=false`,
`supports_barge_in=false`); every session on a PersonaPlex deployment is native
duplex (`is_enabled()` is unconditionally true). Turn-based models
(MiniCPM-o 4.5) are unaffected.

## Architecture (Moshi RQ-Transformer, all native)

- **Mimi codec** 24 kHz, 12.5 Hz, 1920 samples/frame, 8 active codebooks
  (card 2048). Streaming encode + decode via `transformers.MimiModel` with the
  PersonaPlex reference checkpoint weights.
- **Temporal transformer** (Helium backbone) 4096-d / 32 layers / 32 heads,
  sliding window 3000 frames (a 4-minute rolling context) held in a ring KV cache
  with per-slot recycle.
- **Depformer** 1024-d / 6 layers / 16 heads, autoregressive over 16 codebooks
  per frame (8 vocoded), reset each frame.
- Per-frame token column `[B, 17, 1]`: row 0 = inner-monologue text,
  rows 1-8 = agent audio, rows 9-16 = user audio.
- Persona + voice are injected once at session open through the same lockstep
  step: voice clone forces the agent stream from reference-audio Mimi codes;
  persona forces the text stream from `<system> ... <system>` tokens.

## Hardware Support

Verified on GPU (Hopper-class). The serving path is plain PyTorch eager plus the
native modules, so other CUDA GPUs with enough memory are expected to work.

## GPU

### 1x 141 GB Hopper-class GPU (H20-class, verified)

#### Environment

- OS: Linux
- Python: 3.10+ (CI covers 3.11 / 3.12)
- vLLM-Omni: current `main`
- `export HF_TOKEN=...` with access to the gated
  `nvidia/personaplex-7b-v1` repo (accept the license on the model page)

First run auto-downloads from the model repo: `model.safetensors`,
`tokenizer_spm_32k_3.model`, `voices.tgz`, and the Mimi reference checkpoint;
`kyutai/mimi` is fetched via `transformers` for the codec module graph.

#### Command

Live duplex serving through the unified stack:

```bash
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m vllm_omni.entrypoints.cli.main serve \
  /path/to/personaplex-7b-v1 \
  --omni \
  --deploy-config vllm_omni/deploy/personaplex.yaml
```

This exposes `WS /v1/duplex` (native duplex dialect) and
`WS /v1/realtime?duplex=1` (OpenAI Realtime projection; client API and wire
protocol in [`docs/serving/realtime_duplex_api.md`](../../docs/serving/realtime_duplex_api.md)).
Voice and persona are set per session via `extra_body`.

#### Verification

```bash
# GPU-free contract tests (stage0 runtime + unified serving adapter)
pytest tests/model_executor/models/personaplex/duplex/ -q

# GPU e2e: paced 24 kHz PCM over /v1/realtime?duplex=1, two concurrent
# sessions, overflow admission, slot recycling, non-silent output
python tests/e2e/online_serving/personaplex_realtime_duplex.py \
    --model /path/to/personaplex-7b-v1 \
    --input-wav /path/to/speech.wav \
    --output-dir /tmp/personaplex-realtime-duplex
```

#### Notes

- Realtime budget is 80 ms/frame. Verified eager per-tick latency at four
  concurrent sessions is ~70-74 ms on this hardware class, i.e. all four
  conversations stay realtime; single-session has comfortable headroom.
- Decoding is greedy only (temperature/top-k knobs are intentionally not
  exposed): greedy is what the parity gates pin against the reference
  implementation, and sampling amplifies sub-bit numeric drift into divergence.
- Session concurrency is governed by the deploy config
  (`vllm_omni/deploy/personaplex.yaml`); connections beyond the limit are
  rejected until a slot frees. A recycled slot is bit-exact with a fresh
  engine for the same inputs.
- Voice prompts: bundled `voices.tgz` names (`NATF*`/`NATM*` natural,
  `VARF*`/`VARM*` varied) or a path to your own `.pt`/`.wav`.
- Run the client near the server. The 80 ms cadence is sensitive to
  network jitter; over a high-latency link playback can stutter regardless of
  engine speed. On localhost it is smooth.
- Sliding window is 3000 frames (~4 min). The ring KV recycles beyond that;
  very long sessions keep running with a rolling context.
