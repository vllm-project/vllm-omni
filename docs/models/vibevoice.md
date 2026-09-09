# VibeVoice

vLLM-Omni supports the official `microsoft/VibeVoice-1.5B` text-to-speech
checkpoint without converting its weights. The integration targets inference
only; VibeVoice Realtime, ASR, training, and other model sizes are not included.

## Capabilities

- OpenAI-compatible `POST /v1/audio/speech`
- 24 kHz mono waveform output (`wav` or signed 16-bit `pcm`)
- Streaming PCM and structured SSE streaming
- One to four speakers, using four bundled default reference voices when
  `ref_audio` and `voice` are omitted
- Explicit reference audio per speaker and uploaded voices through
  `/v1/audio/voices`
- Classifier-free guidance with the checkpoint's positive and negative Qwen2
  branches
- Acoustic and semantic feedback after every generated audio token
- Single-GPU deployment by default; TP=2 remains experimental and unverified
  until a real multi-rank generation gate passes

Each generated audio token contains 3,200 samples, or approximately 133.3 ms at
24 kHz. `audio_eos_token_id` ends one audio segment; only the request EOS token
ends the request.

## Start the server

The default deploy configuration is `vllm_omni/deploy/vibevoice.yaml`. It uses
one GPU, tensor parallel size 1, `max_num_seqs=4`, and fixed 8 GiB positive and
negative KV-cache pools.

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer Qwen/Qwen2.5-1.5B
```

The tokenizer can be omitted when the checkpoint's
`preprocessor_config.json.language_model_pretrained_name` is available. In an
offline deployment, pre-cache that tokenizer or pass its local path explicitly.

To use reference audio from a server-local file, grant access only to the
containing directory:

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --allowed-local-media-path /srv/vibevoice-references
```

## Default reference voices

This development branch currently bundles four fallback reference voices. When
both `ref_audio` and `voice` are omitted, the adapter assigns the first N
defaults to the N speakers in first-appearance order. Plain text therefore uses
default voice 0, and a four-speaker script uses all four defaults.

!!! note
    The four files are byte-for-byte copies of Apache-2.0 reference/default
    assets already distributed for CosyVoice3, Step-Audio2, IndexTTS2, and
    Qwen3-TTS. See the
    [asset provenance manifest and audit](../design/vibevoice/ASSET_PROVENANCE.md)
    for immutable sources, hashes, licenses, and attribution. Slot numbers do
    not assert speaker identity.

Explicit references remain all-or-nothing: once `ref_audio` is provided, its
length must exactly match the number of speakers. A partial list is rejected
instead of silently mixing user and default voices.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output default-voice.wav \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Hello from the bundled default voice.",
    "response_format": "wav",
    "max_new_tokens": 1024
  }'
```

## Single-speaker synthesis

To select a custom voice, provide one reference audio as a data URL, an allowed
`file://` URI, or an uploaded voice. Omitting both selects bundled default voice
0.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output speech.wav \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Hello from VibeVoice.",
    "ref_audio": "file:///srv/vibevoice-references/speaker.wav",
    "response_format": "wav",
    "max_new_tokens": 1024
  }'
```

For a non-streaming response, `X-Finish-Reason` is `stop` when generation ended
naturally and `length` when `max_new_tokens` was reached.

## SSE streaming

Set both `stream=true` and `stream_format="sse"` to receive
`speech.audio.delta` events followed by one `speech.audio.done` event. Delta
audio is base64-encoded signed 16-bit mono PCM at 24 kHz.

```bash
curl -N http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "This response is streamed.",
    "ref_audio": "file:///srv/vibevoice-references/speaker.wav",
    "response_format": "pcm",
    "stream": true,
    "stream_format": "sse",
    "max_new_tokens": 1024
  }'
```

The terminal event includes `finish_reason`. Clients must treat
`speech.audio.error` or EOF before `speech.audio.done` as a failed request.
Applications that evaluate output quality should aggregate only samples with
`finish_reason="stop"`.

Raw PCM streaming is available with `stream_format="audio"`. Because raw audio
has no structured terminal event, clients that require `finish_reason` must use
SSE.

## Multiple speakers

Use `Speaker N:` lines and pass reference audios in first-appearance order.
VibeVoice supports at most four distinct speakers and exactly one reference per
speaker.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output conversation.wav \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Speaker 0: Welcome.\nSpeaker 1: Thank you.\nSpeaker 0: Let us begin.",
    "ref_audio": [
      "file:///srv/vibevoice-references/speaker-0.wav",
      "file:///srv/vibevoice-references/speaker-1.wav"
    ],
    "response_format": "wav",
    "max_new_tokens": 2048
  }'
```

## Uploaded voices

A voice uploaded through `POST /v1/audio/voices` can be selected by its `voice`
name instead of passing `ref_audio` on every request. `voice` and `ref_audio`
are mutually exclusive. Uploaded `voice` names are single-speaker only.
Multi-speaker requests can either omit both fields to use the bundled defaults,
or pass a complete reference list directly.

## Generation defaults

The bundled deployment matches the official generation behavior:

```text
temperature=0
top_p=1.0
top_k=-1
repetition_penalty=1.0
guidance_scale=1.3
num_diffusion_steps=10
```

The model constrains sampling to VibeVoice's audio BOS, audio token, audio EOS,
and request EOS tokens. The default checkpoint limit is 40,500 generated
tokens; applications should pass a smaller `max_new_tokens` when a bounded
request is required.

The 65,536-token context limit applies after the chat template, text, and every
reference-audio placeholder have been expanded. A request that asks for at
least one output token can therefore use at most 65,535 prompt tokens. To
guarantee `N` generated tokens, the expanded prompt must be no longer than
`65,536 - N`; with `max_new_tokens=40,500`, the prompt ceiling is 25,036
tokens.

Requests may override the two VibeVoice controls through `extra_params`:

```json
{
  "extra_params": {
    "guidance_scale": 1.3,
    "num_diffusion_steps": 10
  }
}
```

`guidance_scale` must be a JSON number in the inclusive range `[0.0, 20.0]`.
`num_diffusion_steps` must be a JSON integer in the inclusive range `[1, 50]`;
booleans, strings, and fractional values are rejected rather than coerced.
These are the only accepted VibeVoice `extra_params` keys, so misspellings fail
at request validation instead of being ignored.

The bounded CUDA Graph cache covers only the official controls
`guidance_scale=1.3`, `num_diffusion_steps=10`, and active batch sizes 1 through
4. Other valid controls and larger active batches use eager diffusion and do
not allocate additional graph entries.

Before the first request, the default deployment captures the official
diffusion graph keys for every active batch size from 1 through
`max_num_seqs`. Set `diffusion_graph_warmup_batch_sizes: []` under
`engine_extras.additional_config.vibevoice_runtime_config` to disable startup capture while
retaining lazy runtime capture, or provide a positive-integer list to capture
only those sizes. Duplicate values are removed and the list is sorted. Values
above `max_num_seqs`, booleans, numeric strings, and non-integers fail startup
validation. `enforce_eager: true` always skips startup graph capture.

## Memory tuning

VibeVoice's KV pools are fixed budgets because KV-pressure preemption is
unsafe for the stateful negative branch. Both the positive and negative pools
must stay equal (CFG trajectories are the same length), and neither may be
sized below the residency floor.

```text
KV cost/token = 28 layers x 2 KV heads x 128 head_dim x 2 (K+V) x 2 B (bf16)
              = 28,672 B = 28 KiB
block = 16 tokens = 448 KiB (vLLM allocation granularity)
audio rate = 7.5 tokens/s (3,200 samples @ 24 kHz)
```

```text
floor per pool = max_num_seqs x max_model_len x 28 KiB
```

| Profile | max_num_seqs | max_model_len | Pool floor | Suggested per pool | Both pools | Est. total VRAM | Max audio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 4 | 65,536 | 7.0 GiB | 8 GiB | 16 GiB | ~24 GiB | ~90 min |
| 2 concurrent | 2 | 65,536 | 3.5 GiB | 4 GiB | 8 GiB | ~16 GiB | ~90 min |
| 1 concurrent | 1 | 65,536 | 1.75 GiB | 2 GiB | 4 GiB | ~12 GiB | ~90 min |
| 1 concurrent, capped | 1 | 32,768 | 896 MiB | 1 GiB | 2 GiB | ~10 GiB | ~72 min |

Total VRAM = weights (~5.4 GiB) + both pools + graph pools/context (~2.5 GiB).
The default profile's ~24 GiB estimate matches the measured peak of 24,177 MiB.

Three disciplines for safe tuning:

1. **Keep both pools equal.** Any asymmetric shrink lets the negative branch
   exhaust first.
2. **Never go below the residency floor.** The startup guard rejects it; do
   not bypass the guard.
3. **Adjust linked settings together.** Lowering `max_num_seqs` also scales
   the diffusion graph warmup sizes; lowering `max_model_len` must still fit
   `prompt_tokens + expected_audio_tokens`.

The deploy YAML's `kv_cache_memory_bytes` comment block carries the compact
profile table for reference during local edits.

## Experimental TP=2 deployment

TP=1 is the only topology currently covered by an official-weight generation
gate. The overlay below is retained for development experiments; successful
startup does not establish TP=2 correctness. In particular, rank-consistent
diffusion RNG, replicated side modules, waveform parity, quality, and per-rank
memory still require validation. Create a local overlay such as
`/etc/vllm-omni/vibevoice-tp2.yaml`:

```yaml
base_config: /path/to/vllm_omni/deploy/vibevoice.yaml

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    kv_cache_memory_bytes: 6442450944  # 6 GiB per rank
    engine_extras:
      additional_config:
        vibevoice_runtime_config:
          negative_kv_cache_memory_bytes: 4294967296  # 4 GiB per rank
```

Unknown `vibevoice_runtime_config` keys and non-mapping values fail at startup;
they are not ignored or replaced with defaults.

Then launch with:

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --deploy-config /etc/vllm-omni/vibevoice-tp2.yaml
```

Do not use this overlay as a production support claim until the TP=2 acceptance
gate is recorded in the development runbook.

## Tests

CPU and contract tests do not require model weights:

```bash
pytest -q -s tests/model_executor/models/vibevoice \
  tests/worker/test_gpu_ar_model_runner.py \
  -m 'core_model and cpu' \
  --run-level core_model
```

Official-weight offline and OpenAI Speech tests require an H100 and an explicit
advanced run level:

```bash
VIBEVOICE_TEST_MODEL=microsoft/VibeVoice-1.5B \
VIBEVOICE_TEST_TOKENIZER=Qwen/Qwen2.5-1.5B \
pytest -q -s --run-level advanced_model \
  tests/e2e/offline_inference/test_vibevoice_tts.py \
  tests/e2e/online_serving/test_vibevoice_tts.py
```

The portable DFX configuration is
`tests/dfx/stability/tests/test_vibevoice.json`. Its report counts both `stop`
and valid `length` responses as `successful_requests`, while reporting them
separately as `natural_stop_requests` and `truncated_requests`.
`request_failures` is reserved for transport, SSE, PCM, engine, or terminal
metadata failures. Performance, experimental TP=2, and multi-scenario
long-duration tests are intentionally maintained outside the regular merge
gate.

## Limitations

- Only `microsoft/VibeVoice-1.5B` TTS is supported.
- Realtime/duplex inference, ASR, and training are not supported.
- Audio output is fixed to 24 kHz mono.
- A maximum of four speakers is accepted.
- Reference text and model-specific fields from other TTS APIs are rejected.
- TP=1 is the only verified target; TP=2 is experimental pending rank-consistency
  and official-weight generation evidence.
