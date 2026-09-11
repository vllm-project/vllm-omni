# Gander validation record

## Snapshot and scope

Validated on 2026-09-10 using the materialized Gander Unit8/50 release
`Gander-Omni/Gander@24fc4cc8543f95daf99be53b6199403a7732688f`.
The integration is based on transactional-duplex
[PR #7294](https://github.com/vllm-project/vllm-omni/pull/7294) at
`36142ba106cdf458dfc77439cc8f769a68d1892d`. Later dependency revisions are
intentionally not included in this draft. The original running service was
not rebased or restarted during PR preparation.

The original single-session functional runs used one NVIDIA H200 (143 GB), one admitted session,
vLLM 0.28.0, PyTorch 2.13.0, Transformers 5.15.1, and the three-stage
`gander.yaml` deployment. Thinker and Talker run eager with one sequence per
stage; the Stage0 context limit is 40,960 and the replay journal limit is
256 MiB per session. Code2Wav retains the existing generation path.
This is correctness evidence, not a throughput, concurrency, or latency benchmark.

## Final real-weight checks

The final deployed-service run completed **10 passed in 460.63 seconds**:

| Check | Acceptance |
| --- | --- |
| Streaming speech | Real recorded audio input, model-generated text, non-silent PCM output. |
| Native model interruption | Actual model interrupt action, old-response cancellation, correct follow-up answer, and no old audio after the response terminal event. |
| Tool result and task slate | Model-generated `task_start`, deterministic local result, progress/slate updates, and spoken answer containing the result. |
| Slate reference | A follow-up question uses the latest task state after protected-prefix/KV replacement. |
| History editing and small-window rollover | Invalid edits preserve old context; move/delete/pin operations and repeated 8/4-window reconstruction allow subsequent inference. |
| Historical external event insertion | Validated observation is inserted in the selected historical order and inference resumes after reconstruction. |
| Default history window | Stream 140 seconds after an edit, cross the 128-unit threshold, retain history, and obtain another spoken answer. |
| Existing protocol smoke | Reuse the MiniCPM driver against Gander weights. |
| Two committed audio turns | Reuse the MiniCPM driver against Gander weights, preserving its assertions. |
| Two audio/video turns | Reuse the MiniCPM driver against Gander weights, preserving its assertions. |

The last three checks are protocol/driver reuse on **Gander**, not a fresh
base-MiniCPM checkpoint quality comparison. The deployed-service run used
wrappers calling the checked-in tests against an already-running engine.
The original Gander file contained seven tests (now eight with concurrency coverage) and normally starts its own
server through the shared fixture.

The default-window run recorded logical unit `u1-146`, a physical resource
generation increase from 1 to 2, and 114 retained/current units after rollover
and subsequent input. Its context had 2,302 tokens and replay metadata occupied
98,198,683 bytes, below the 268,435,456-byte limit. This demonstrates physical
reconstruction rather than just changing a Python prompt string.

A completed short follow-up can contain one audio delta. The strict Gander
interrupt check requires non-empty audio for every reply, multiple deltas in
at least one reply of the long/short fixture, a native interrupt, cancellation,
and the correct new answer. The original MiniCPM non-cancelling contract is
preserved. Packet counts do not establish speech fidelity.

## Speech omissions: reproduced causes and fixes

1. Resumable native units reset their output cursor within one model turn.
   Treating that cursor reset as a turn boundary restarted Talker KV for every
   unit. The bridge now uses model turn identity and the first handoff sequence.
2. A final Gander lexical unit replaced its own Talker context. It now retains
   the same-turn context and drains to EOS; base MiniCPM behavior is preserved.
3. The first Code2Wav block contained 28 zero placeholders instead of three
   silence codes (4218) plus 25 generated speech codes. Explicit full snapshots
   now retain `codes.audio`; prefill refresh also synchronizes the backing
   token list, clears prior outputs, and resets output placeholders.
4. Direct interrupt output bypassed the ordinary handoff that advanced the
   engine model turn. Valid follow-up speech could consequently be rejected as
   stale. The model declares `ends_model_turn`; the generic engine advances the
   boundary immediately. Codec transfer drops pending older-turn codes and
   resets the vocoder cache when an interrupted new turn arrives.

The first-block problem reproduced through both facade and batched vocoder
replay of the same corrupted codes. A reference Talker replay using recorded
conditioning could synthesize the full sentence. This narrows the issue to
handoff/codec state and does not establish complete official-runtime parity.
New deterministic regressions for full snapshots and token backing storage
failed against their old logic and passed with the fixes.

The native duplex path constructs embeddings directly; it does not use the
ordinary Chat multimodal-string assembly or the fallback TTS template. These
experiments do **not** attribute the Gander failures to PR #5524 or #6346, and
no historical attribution for the separate extra-newline issue is claimed.

## Browser checks

The actual browser, local WebSocket proxy, H200 engine, and AudioWorklet were
exercised using recorded speech as a Chromium microphone input. The model
reply was not injected or mocked. The previously incomplete sentence was:

> 平平淡淡也挺好的呀。要不咱们聊点别的，比如你平时喜欢看什么电影，或者有什么爱好？

After the fixes, independent Whisper-small ASR included the beginning and the
previously omitted connecting words. The final production-browser run received
and rendered 382,080 identical PCM16 bytes at 24 kHz (7.96 seconds), with zero
measured internal underrun. The common SHA-256 was
`5dbc767df7b492ebbf1dc54151f93546256e12e7ebafc4070fee62486654d3a3`.
Two additional greeting/comfort sentences had complete ASR transcriptions
apart from punctuation/homophone variation, with 134,400 and 407,040 identical
received/rendered bytes respectively.

In a real browser interruption run, 472,320 bytes of old-reply audio arrived;
playback was cancelled after 190,208 bytes. The cancellation sample count and
final rendered count were identical: no old samples rendered after cancellation.
The new reply received/rendered 453,120 identical bytes with zero internal
underrun and a complete ASR transcription of the follow-up answer.

These checks reach AudioWorklet output, not the OS mixer or physical headphones.
They resolve the recorded reproductions; they do not guarantee zero omissions
for arbitrary input. The frontend has no VAD playback gate, and a `listen`
display transition does not cancel audio. The optional app and deterministic
Node.js playback regression are in [Gander Live](https://github.com/vllm-project/vllm-omni/blob/main/apps/gander_live/README.md).

## CPU tests and local checks

Before PR packaging, expanded CPU regression completed **1,545 passed,
3 skipped, 26 deselected in 133.23 seconds**. After the final backing-token
regression was added, the complete connector file completed **158 passed**.
The broad run initially hit two Hub-connection failures in configuration tests;
rerunning with the existing Qwen3-0.6B configuration cache and
`HF_HUB_OFFLINE=1` passed. This cache does not require downloading model weights.

The packaged snapshot was then retested on H200: **1,916 passed, 3 skipped,
26 deselected in 187.51 seconds**, adding the duplex serving/handler and
streaming-contract directories to the broad CPU command above.

During packaging, repository formatting/import conventions and type annotations
were cleaned up without updating the pinned dependency. Changed-file mypy
reports 44 errors also present on the clean dependency snapshot: 31 in the
existing duplex client/AsyncOmni and 13 in the existing soft-interrupt driver.
A comparison of file, error message, and error code found no added diagnostics.
This is **not a clean mypy pass**. The draft preserves that baseline rather than
changing unrelated runtime logic. Hosted CI results are separate from these
local checks.

L1 regression files are discovered by the existing CPU shards in
`.buildkite/cuda/test-ready.yml`. A Gander L3 job in
`.buildkite/cuda/test-merge.yml` composes the pinned release and executes the
entire Gander file. Hosted Buildkite was not used for the H200 results above.

## Reproduce

Run from the repository root with a matching vLLM/Torch/pytest environment.
The single-file CPU check needs neither a GPU nor model weights:

```bash
python -m pytest tests/distributed/omni_connectors/test_chunk_transfer_adapter.py -q
```

For the broader CPU suite, provide the cached Qwen3-0.6B configuration or permit
Hub access (omit `HF_HUB_OFFLINE=1` if downloading it):

```bash
HF_HUB_OFFLINE=1 python -m pytest \
  tests/engine \
  tests/model_executor/models/minicpmo_4_5 \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_omni.py \
  tests/model_executor/stage_input_processors/test_minicpmo_4_5_async_chunk.py \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  -q -m 'core_model and cpu' --run-level core_model
```

For the eight real-weight tests, compose the release as described in the
[serving guide](../serving/gander.md), select a free large-memory CUDA GPU,
and let the fixture start its own service. Do not run a competing server on
that GPU. H200 is the hardware actually validated here; the CI job targets H100.

```bash
GANDER_MODEL=/path/to/new/gander-model \
python -m pytest -sv tests/e2e/online_serving/test_gander.py \
  -m 'advanced_model and cuda' --run-level advanced_model
node apps/gander_live/test_playback.cjs
```

## Exploratory quality evaluation and limits

Earlier development snapshots also ran 12 VoiceBench inputs and 8/100
Full-Duplex-Bench v3 recordings. All produced output, but factual errors and
speech distortion were observed. No GPT-4o judge, official Brain/Gateway,
trusted ASR binding, or benchmark tool execution was included. Those runs
predate the final speech fixes and have not been repeated as a full benchmark;
they are not acceptance scores for this snapshot.

A small earlier Daily-Omni comparison used the same 10 questions under AV/V/A
conditions: Gander scored 5/10, 6/10, 7/10; base MiniCPM-o 4.5 scored 7/10,
5/10, 7/10. The questions came from six videos, were not a representative random
sample, and used an earlier adapter snapshot. These figures cannot establish
that either model is generally better, or be compared with the paper's full
WorldSense/Daily-Omni results.

The paper's `Interrupt` timing metric concerns speaking before the user has
finished, whereas the native `interrupt` action here cancels the assistant's
current response. These are distinct measurements. No paper-equivalent score,
quality parity, millisecond interruption SLA, production-scale concurrency,
or non-CUDA hardware support is claimed. Gateway, Brain/GPT, persistent task
revision/cancellation/retry semantics, and business-level stale-result filtering
remain application responsibilities outside this inference integration.

## Review fixes and four-session validation (2026-09-10)

The deployment now admits four sessions, with four sequences per stage and
an active stream window of four. Async scheduling remains disabled. This
uses the same pinned #7294 dependency; later dependency commits are excluded.

Review reproductions confirmed missing tool-turn advancement, the lost resume
context epoch, cancellation of committed appends before their receipts during
context validation, configuration commits overwriting newly registered calls,
and token replay deduplication across distinct native units. These paths now
advance the tool turn, restore the resume gate, wait for append receipts before
validation, reconcile concurrent call registrations, and include native unit
identity in the handoff cursor. The async closed-unit guard also includes the
Gander interrupt terminator.

Five focused reproductions fail against the previous implementation. The
expanded CPU suite passes **1,956 tests**, with **3 skipped**, **26 deselected**,
and **21 warnings** (195.92 seconds). It also includes
`tests/worker/test_native_duplex_input_safety.py`; the async lookahead subset
passes all four cases.

On one H200 with the repaired source, four synchronized sessions each completed
two audio turns: **32, 26, 24, and 19 audio deltas**, with isolated identities
and native turn boundaries. A fifth session was rejected at capacity and a
replacement session was admitted after release. A separate simultaneous run
passed two independent model-generated tool queries (different result values),
history replacement/rollover, and native interruption. These are bounded
functional concurrency checks, not sustained throughput or latency benchmarks.

The Read the Docs log identified two strict-build warnings for relative links
to the browser app outside the docs tree. Those links now target GitHub; a new
hosted documentation build has not yet been verified. Complete input-arrival,
model-decision, audio-delivery, and rendered-playback timing instrumentation
remains unimplemented.

Two headless Chromium pages through the Mac WebSocket proxy each completed one
model-generated local tool query and received 12 audio deltas, with no browser
or protocol errors. Both transcripts contained the returned value. This checks
the concurrent browser/tool path with synthetic microphone input; it does not
measure acoustic playback fidelity or physical speaker output.
