# Native duplex hardware acceptance follow-up (2026-09-07)

## Historical status before official-source audit

The raw-EOS acceptance interpretation below is superseded by the
[official alignment report](nemotron_official_alignment_20260907.md).
NVIDIA NIM completes the continuous response after graceful stream drain,
not at every model utterance EOS. The earlier measurements are retained as
history; they are not the current Nemotron completion contract.
The historical PersonaPlex asset blocker is also superseded by the
[ModelScope/H200 follow-up](personaplex_modelscope_20260907.md).

The shared lifecycle fixes below are implemented and tested. The complete
three-model acceptance matrix is **not passed**: MiniCPM passes the scoped
H200 protocol/KV smoke, Nemotron still fails strict model-EOS acceptance, and
PersonaPlex lacks authorized weights/voice assets in the test environment.
There is no new throughput or answer-quality claim.

## Implemented fixes

- Bound data-plane reads by the frame period and remaining idle deadline,
  instead of borrowing the native control RPC's 60-second timeout. Include
  the read itself in the idle clock.
- Allow an unfinished auto-response to advance after quiet output polls,
  through the existing input sequencer and request/epoch cancellation fences.
- Handle exact-frame EOF even when no partial PCM tail remains and the model
  has not created a visible response. Arm the existing bounded decision owner;
  do not fabricate an empty response to drive further input.
- Arming wakes the owned output drain without sleeping in the wire mailbox.
  It retains the engine-selected response-stage ID across idle drain turnover.
  The existing queued-close regression still requires no extra append.
- Define continuation budgets in media time: 64 seconds for auto-response,
  8 seconds for explicit-response decisions. An 80ms model unit must not
  accidentally receive only 5.12 seconds from a 64-unit assumption.
- Mark auto-response budget exhaustion as `incomplete`, including the
  nonterminal listen/PAD path. A resource safety limit is not model EOS.
- Apply Nemotron speech/EOS boundaries at their acoustic frame. Split
  coalesced codec output into 80ms packets and suppress post-EOS silence until
  the next model speech boundary. This does not infer EOS from amplitude.
- Strengthen the Nemotron driver: all created responses must settle
  successfully, including after drain; preserve event logs on failure too.

## Validation

| Check | Result |
| --- | --- |
| Final CPU sweep | 1047 passed, 1 skipped, 1 deselected; 79.08s |
| Focused handler/Nemotron/PersonaPlex CPU | 300 passed |
| MiniCPM final H200 smoke | C1/C2, three two-turn trials each, 18 protocol-completed responses; strict receipts, retained KV identity/context and isolation checks passed |
| Ruff / diff | Nine task-owned Python files pass check/format; diff whitespace check passes |
| Nemotron strict H200 gate | Failed; no last-answer model EOS, not a successful sign-off |
| PersonaPlex H200 | Not run; authorized model/voice assets unavailable |

The skipped CPU module requires CUDA (PersonaPlex elastic). The MiniCPM
result does not resolve its previously documented chunk/turn termination
semantic risk. There is no H20/H100/NPU hardware sign-off in this follow-up.

Runtime fingerprint (sorted relative Python paths and their SHA256 digests,
excluding an unrelated concurrent profiler edit):

```text
90fedc9b0d02736c77d6a70b9c143466e484d0f02a3194b53d6bb6de9d4a8717
```

Final MiniCPM launch records use this fingerprint before and after the run.
The existing H200 container and vLLM 0.28.0 environment were reused. No shared
environment or unrelated GPU process was removed. The launcher checks GPU
occupancy and reclaims only PID/create-time-matched child processes.

## Nemotron: what the trace establishes

A diagnostic-only forwarding wrapper recorded real text EOS token 2 at
frames 10, 76 and 178; each was joined with the corresponding codec output.
The final answer's last text token appeared at frame 225. Hundreds of later
frames emitted PAD token 12, with no EOS and no pending EOS. The audio and
text frame clocks continued advancing, so this was not a stalled append or
an EOS stuck behind undelivered audio. The TTS checkpoint also explicitly
disables its separate EOS predictor (`disable_eos_prediction: true`).

The final answer reached the 800-frame/64-second media safety limit and was
correctly labeled `incomplete`. Later output could still create an active
response, which the strengthened driver rejected. Both realtime and burst
acceptance remain unsigned. A preliminary run that passed the old driver
was deliberately rejected after inspecting its incomplete/cancelled tail.

The observed final audio tail has peak amplitude 1/32768 and RMS about
2.51e-5. A model-listen plus decoded-silence ending policy could address this
case, but it would be a different, explicitly labeled adapter policy, not a
raw EOS. That policy was proposed to the user and **not enabled** without
confirmation. Neither sampling nor the test was changed to invent EOS.

## PersonaPlex access boundary

No usable PersonaPlex model/voice bundle was found in scoped H200 caches.
The official gated model revision examined was
`nvidia/personaplex-7b-v1@fdaf4090a61cb315c138a1faee287ffd6c716309`.
After resolving the server's network access through a temporary loopback
forward, both anonymous and existing-credential requests returned HTTP 401.
The temporary forward was closed. An authorized asset path or refreshed
server credentials after model-license approval is required; gated access
was not bypassed.

## Reproduction

Prerequisites: Linux, the matching vLLM 0.28.0/torch/Omni installation and
pytest development dependencies. L1 needs no weights. Run from the repo root:

```bash
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
cd tests

# Focused local tests
python -m pytest -sv e2e/features/fullduplex/nemotron_voicechat/test_data_plane.py

# CI-like L1
python -m pytest -q entrypoints/openai_api/test_duplex_handler.py \
  e2e/features/fullduplex/nemotron_voicechat \
  e2e/features/fullduplex/personaplex \
  -m 'core_model and cpu' --run-level=core_model

# Existing H100 nightly gate; this remains expected to expose the open boundary
python -m pytest -sv e2e/online_serving/test_nemotron_voicechat_duplex.py \
  -m 'full_model and H100 and omni and cards_1' --run-level=full_model
```

The last command requires the native Nemotron checkpoint, tokenizer and
`turn_taking.wav`. Set `VLLM_TEST_NEMOTRON_VOICECHAT_LLM_PATH` and `MODEL_PREFIX`
for cached assets. On shared machines use ownership-contained service
lifecycle management, not global fixture cleanup. Existing CPU and H100
nightly sweeps already collect these test paths; no new CI job was added.
