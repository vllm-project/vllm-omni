# Nemotron official continuous-stream alignment (2026-09-07)

## Result

Nemotron passes the scoped H200 continuous-stream/KV-append acceptance after
aligning response lifetime with NVIDIA's official implementation. The earlier
requirement that each response must end with an unmodified model EOS was not
the official NIM stream-completion contract and is superseded by this report.
PersonaPlex assets are now available through a verified ModelScope mirror;
see the [PersonaPlex follow-up](personaplex_modelscope_20260907.md) for its
H200 smoke result and remaining strict-close boundary.

## Official sources examined

- NVIDIA model card and [deployment instructions](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/voicechat_realtime_instructions/deploy.md):
  file input should include about 20 seconds of trailing silence; the model
  continues generating while audio input continues.
- [Official API reference](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/voicechat_realtime_instructions/api-reference.md):
  audio append, graceful session close, response completion and session end.
- The actual `/s2s/audio_server.py` from
  `nvcr.io/nim/nvidia/nemotron-labs-voicechat:1.0.0`, not just its API prose.
  The verified source layer is
  `sha256:3cd5bf72eede965f674c9e7e7d2fb051bc8fcfc9e7efe661c000937d5bc7ffb2`.
  The public OCI layer was read without starting another container or pulling
  the large runtime layers. The extracted source SHA256 is
  `db188cac479571928eb6d0e6b9b31de5b49af94d8e769960473438db497a7a20`.

The NIM server emits transcript completion on `</s>`, but does not end the
audio response there. `_emit_response_done` is called by the send loop's
shutdown path. Graceful `session.close` drains accepted inference work,
finalizes the sequence, stops the sender, and emits `session.end`.

The public NeMo branch additionally contains optional forced turn-taking and
missing-EOS safeguards. They must not be confused with naturally sampled EOS,
and are not prerequisites for NIM's stream-level `response.done`. This change
does not invent or inject EOS, nor infer it from a waveform-silence threshold.

## Architecture changes

The public capability `response_lifecycle` separates retained KV ownership
from frontend response lifetime:

| Mode | Opt-in | Completion |
| --- | --- | --- |
| `model_turn` | Default; MiniCPM unchanged | Existing model-turn policy |
| `continuous_stream` | Nemotron | Explicit close after all accepted frames are delivered |

Continuous-stream adapters must provide accepted-input accounting, output
delivery accounting and a drain status. Missing hooks fail at session open.
Nemotron counts the engine's acknowledged sequence monotonically, so a retried
receipt cannot count an input frame twice. Delivery is marked only after the
output drain has finished publishing that output batch.

- One continuous response carries all requested audio, including silence.
- Model EOS completes an utterance transcript, not the response or retained KV.
- Server-side synthetic silence continuation is disabled for this lifecycle.
  The file client supplies the official trailing-silence input instead.
- Graceful close waits for accepted appends, flushes a partial final frame,
  verifies both text/audio delivery watermarks and emits `completed` with
  reason `stream_drained`. Resource cleanup follows that delivery barrier.
- Drain timeout or missing output fails explicitly; it is not converted into
  a successful close. Explicit cancel retains its cancellation semantics.
- Transcript events retain epoch fencing. Completion does not create a later
  cancellation notification for the already completed stream.

This aligns lifecycle semantics; it does not claim byte-identical NIM wire
output, NIM's RNNT/tool policy, or its 24kHz client resampling. Omni's tested
native audio format remains 16kHz float input and 22.05kHz PCM16 output.

## Verification

Final runtime Python fingerprint, excluding the unrelated profiler edit:

```text
44c4e7a14154f3a85cfdc40c8580b85c649f2e48364cb0df940c5c27a5390aa4
```

The final GPU launch records have this same fingerprint before and after the
runs. Existing `vllm-minghui` and the project's vLLM 0.28.0 environment were
reused; no alternate dependency environment or container was created.

| Check | Result |
| --- | --- |
| Broad CPU regression | 1048 passed, 1 CUDA-only skip, 1 deselected |
| Close boundary tests, expanded to whole/partial frames | 7 passed; includes timeout, cancel and no synthetic input |
| Nemotron H200, two sequential reopened sessions | Passed; not a concurrent multi-session claim |
| Per Nemotron session | 190 source frames + 250 silence frames; 440 append receipts; one physical KV request; all 440 audio frames received |
| Retained context | 58 to 496 tokens, within the 8192-token profile limit |
| Completion | One `response.done` with `completed/stream_drained`, one `session.end`; no cancellation substituted for success |
| MiniCPM H200 regression | C1/C2, three two-turn trials each; 18 protocol-completed responses with strict KV/receipt checks |
| Unpaced 440-frame stress input | Explicit `input_backpressure` rejection; not a successful burst-stream claim |

Nemotron produced 1,552,320 PCM16 bytes per session (35.2 seconds), with audible
output RMS above the existing 0.001 gate. These are delivery/correctness
results, not a new throughput comparison or an answer-quality score. The
previous MiniCPM termination-semantic caveat remains separate. No H20/H100/NPU
hardware sign-off or live KV migration is claimed.

## PersonaPlex authorization boundary

The historical missing-assets blocker below is superseded by the
[verified ModelScope acquisition](personaplex_modelscope_20260907.md).
The original HF credentials were not changed.

The [official PersonaPlex instructions](https://github.com/NVIDIA/personaplex/blob/3428dfd95309a7f3c84fd93259ded0f810d1ff91/README.md#accept-model-license)
require accepting the model license and authenticating to Hugging Face.
No usable weights/voice assets were found in scoped H200 caches. The existing
credential returned HTTP 401 from both `whoami-v2` and the gated model file;
this is not just a dependency or download-path issue. No connected browser
was available to check an already-authorized login either. Updated credentials
or an authorized local asset directory are required; access was not bypassed.

## Reproduction

Prerequisites: Linux, matching vLLM 0.28.0/torch/Omni and pytest dependencies.
Run from the repository root; CPU checks require no model weights:

```bash
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
cd tests
python -m pytest -sv entrypoints/openai/test_continuous_duplex.py
python -m pytest -q entrypoints/openai/test_continuous_duplex.py \
  entrypoints/openai_api/test_duplex_handler.py \
  e2e/features/fullduplex/nemotron_voicechat \
  -m 'core_model and cpu' --run-level=core_model

# Existing H100 nightly route; requires the native checkpoint, tokenizer,
# and turn_taking.wav. Set MODEL_PREFIX and the tokenizer override first.
python -m pytest -sv e2e/online_serving/test_nemotron_voicechat_duplex.py \
  -m 'full_model and H100 and omni and cards_1' --run-level=full_model
```

The existing CPU and H100 nightly sweeps collect these paths; no new CI job
was needed. On the shared H200 host the runs used the PID/create-time-scoped
launcher under `/root/duplex-official-alignment.2uKOQy`, not global fixture
cleanup. Raw events, WAVs, validated results, source metadata and failure logs
are retained in `duplex_official_alignment_20260907` alongside the worktree.
