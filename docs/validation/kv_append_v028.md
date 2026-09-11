# Native duplex KV append on vLLM 0.28.0

## Scope and acceptance status

This runbook covers the shared OpenAI native-duplex Stage0 append path on
vLLM 0.28.0. MiniCPM-o 4.5, PersonaPlex, and Nemotron VoiceChat now use the
same admission and transactional append interfaces. The
[feature design](../design/feature/duplex_kv_append.md) defines the current
contract and the active non-KV paths intentionally retained.

The interface cleanup is implemented; its acceptance requires results from
the exact candidate source snapshot. Historical H200 runs below do not sign
off this migration. This document does not report a new performance gain or
a completed three-model E2E matrix.

The [official Nemotron alignment](nemotron_official_alignment_20260907.md)
records the current continuous-stream contract and passing H200 delivery/KV
checks. It supersedes the raw-EOS interpretation in the earlier
[hardware acceptance follow-up](duplex_hardware_acceptance_20260907.md).
PersonaPlex assets are now available: the [ModelScope/H200 follow-up](personaplex_modelscope_20260907.md)
records the asset provenance. The [generated-audio drain follow-up](personaplex_stream_drain_20260907.md)
replaces cancelled close with checked delivery and explicit model-delay accounting.

### Cleanup validation on 2026-09-07

The cleanup's H200/vLLM 0.28.0 validation produced 1723 CPU passes with two
skips, plus 109 supplemental consumer-test passes. The skipped CPU cases need
CUDA (PersonaPlex elastic) or an HF config (Qwen3-Omni); the supplemental marker
sweep does not cover the already-skipped PD entrypoint module. Ruff and the
targeted documentation checks passed; full-branch pre-commit is not claimed.

MiniCPM C1/C2 each passed three two-turn trials on the final runtime source
(one warmup, two measured): 18 protocol-completed responses, with strict append
receipt, retained-request continuity, context and identity-isolation checks.
This is a functional smoke result, not an answer-completeness or speedup claim.
The existing MiniCPM chunk/turn termination-policy risk remains open.

Nemotron real-time input produced 195 fixed 80ms audio packets and three
completed responses, but its fourth response stayed open after final input
commit and was cancelled during timeout cleanup. The actual pre-cleanup
worktree reproduced the same failure (196 packets, three completed and one
cancelled response). Neither run emitted protocol error events. This exposes
an existing end-of-input/drain acceptance gap; it does not pass Nemotron E2E.
PersonaPlex real weights/voice assets were unavailable, so its GPU acceptance
also remains open. No model EOS policy was changed to bypass either boundary.

The final runtime Python fingerprint, excluding an unrelated concurrent
profiler edit, is:

```text
61e7b768877011defd3f2c02a88014b6ea4430eb06fac250649f6a0f047390b0
```

It hashes sorted relative filenames with per-file SHA256 digests. The local
and frozen H200 runtime sources matched. The broad CPU run preceded only a
StagePool comment edit; supplemental CPU and final GPU used that comment edit.

## One project-owned CUDA environment

Use Python 3.12 and a Linux CUDA 13.0-compatible vLLM 0.28.0 build. Historical
CUDA runs used an H200 inside the existing container. The MiniCPM default profile keeps
Stage0 async scheduling and prefix caching disabled; the async-on follow-up
is a separate opt-in validation. macOS is not the CUDA inference environment.

Create `.venv` only when it does not already contain an in-use environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements/fullduplex-test-cu130.txt
# cosmos-guardrail pulls GUI OpenCV, while vLLM requires headless OpenCV.
# Both distributions own cv2. Keep the actual imported implementation headless.
uv pip install --python .venv/bin/python --reinstall-package opencv-python-headless opencv-python-headless==5.0.0.93
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
```

Use a package mirror with the same pinned versions if PyPI downloads are slow;
do not solve download failures by changing the vLLM version. A file-only test
snapshot without `.git` needs `VLLM_OMNI_VERSION_OVERRIDE=0.28.0.dev0+kvappend`
for the editable Omni build. This labels the local Omni development snapshot;
upstream **vLLM remains 0.28.0**.

`tools/run_fullduplex_028.sh` is the project entry point. It checks the Python
prefix, vLLM/transformers versions, module origins, headless OpenCV, and that
the project's `ninja` is on PATH. Pointing only at `.venv/bin/python` is not
enough for compilation subprocesses that invoke `ninja` by name.

The entry point also pins `VLLM_USE_V2_MODEL_RUNNER=0` before constructing any
engine configuration, and rejects an explicit V2 request. Omni's workers use
V1 hooks; allowing the upstream scheduler to independently select V2 omits
the token history that V1 needs for async batch readmission. Other launchers
must set this environment variable themselves until runner selection is
represented consistently in the shared upstream configuration.

For controlled Stage0 async experiments, pass
`--stage-overrides '{"0":{"async_scheduling":true}}'` to the server. The legacy
CLI overlay now reselects the matching built-in AR scheduler; custom or
explicit scheduler classes remain the caller's responsibility. Verify that
the **Stage0** startup log names `OmniARAsyncScheduler` (the Stage1 log alone
is not evidence that Stage0 async is enabled).

Never remove a shared environment merely because its directory name looks
old: check the executable, installed version, symlink destination and active
processes first. Retire only identified project environments, preserving
other projects' environments and model caches.

## Contracts covered by this change

- `engine.kv_append` is a read-only 0.28 capability probe. The retired
  `_streaming_prompt_compat` installer and experimental forwarding module are
  removed. Importing Omni no longer adds append methods to upstream
  `EngineCore`/`AsyncMPClient` or invents a streaming-status enum alias.
- Initial admission uses the Omni client's `admit_duplex_request_async`, backed
  by upstream `add_request_async` with `request.resumable=True`. Its acknowledgement waits
  for `num_computed_tokens >= num_prompt_tokens`; no held last token or
  external finalize RPC remains.
- Subsequent units use the explicitly declared
  `StageEngineCoreProc.append_streaming_prompt_unit` utility and the native
  `WAITING_FOR_STREAMING_REQ` state. `_commit_native_append` is internal,
  synchronous receipt bookkeeping, not another RPC. The scheduler still
  delegates to upstream's private `_update_request_as_session` through its
  Omni override; a vLLM upgrade must recheck this boundary.
- Sampling snapshots traverse the real serialized utility path, including
  `StageEngineCoreProc`, and update scheduler `max_tokens` and worker sampling.
- Logical-epoch operation tombstones survive physical KV rollover/recovery;
  expired retries are rejected before submission.
- Actual audio/vision input embeddings are retained in a bounded CPU history
  for local KV recomputation. Output-token gaps retain ordinary embeddings.
- A final append no longer reserves a nonexistent extra 12-token audio unit.
  The scheduler-owned `kv_append_start` verifies the prepared unit's exact span.
- Discarded partial-prefill samples cannot change MiniCPM policy state or RNG.
- A worker-side, physical-request/unit fence prevents async lookahead after
  a unit stop from overwriting the terminator needed by the next KV append.
  `turn_eos` still allows its required following `chunk_eos` sample.
- Input failures mask KV write slots, suppress downstream payload, and terminate
  only the affected request. The original error survives request cleanup so a
  racing append receives `model_input_error`, not an ambiguous `not_found`.
  Deferred async outputs use the invalid-row mask instead of writing into
  the not-yet-materialized sampled-token list.
- Readiness queries and internal receipt-commit bookkeeping preserve the
  bounded model-input error tombstone after cleanup. There is no external
  finalize utility to race request teardown.

### Replay is an independent model contract

All three shared models declare scheduler-native append when the 0.28 probe
passes, subject to engine-side replica/context checks. Only MiniCPM declares
`supports_prompt_replay=True`; PersonaPlex and Nemotron declare `False`.
Atomic append does not prove that prompt replay can reconstruct model-owned
codec or recurrent state. PersonaPlex/Nemotron replica loss must remain a
typed safe failure, including `native_kv_replica_lost`, rather than entering
automatic replay or context rollover. MiniCPM's bounded replay/rebuild is a
separate acceptance item and does not provide live KV migration.

The generic `StagePool.submit_update` remains in use by public streaming
input, downstream stages, PD decode, and diffusion. PersonaPlex's independent
browser backend also retains its own active frame-stepper lifecycle. Neither
is an unused shim; this cleanup only removes the former shared native Stage0
fallback to the generic update path.

### Historical Stage0 async follow-up (2026-09-06)

The following records refer to earlier source snapshots. They are retained
as experiment history, not results for the admission/append cleanup.

Aligned async-on now completes the H200 four-session/two-turn gate. The
permanent `four-session-stage0-async-on` parameter is collected by the existing
L3 merge job, with V1 explicitly selected in the test server's environment.
This is separate from the default deploy setting, which remains **off**.

The earlier eighteen Seed-TTS A/B cases explicitly selected
`native_duplex=False`: they exercised text/chat fallback through Realtime,
not native KV append. Their 42 sessions and 168 audio turns remain valid TTS
measurements, but their throughput percentages are **not native KV-append
performance evidence**. Inferences about native performance from those numbers
are withdrawn. Use `run_native_duplex_benchmark.py` with
`test_minicpmo_4_5_native_duplex.json`; it requires real append receipts, stable
physical KV request identity, bounded context and model EOS before reporting
throughput. Keep async-on opt-in until the actual native workload is validated.

The subsequent native performance/correctness audit is recorded in
[KV append native performance, 2026-09-06](kv_append_native_performance_20260906.md).
It includes actual async-on measurements, rejected optimization candidates,
correlated submission deadlines, mixed-batch sampling, and transport backlog bounds.

This does not provide live KV migration, lossless arbitrary-context compaction,
or proof of support for every hardware/backend/concurrency configuration.
NPU failure hooks are wired and CPU source-contract checked; CUDA results are
not NPU hardware validation. MiniCPM's bounded replica replay retains its
narrower append/metrics contract, not bitwise original-history equivalence.

## Local regression commands

These L1 tests need the matching installed vLLM/torch build but no model weights:

```bash
tools/run_fullduplex_028.sh -m pytest -sv tests/worker/test_native_duplex_input_safety.py -m 'core_model and cpu'
HF_HUB_OFFLINE=1 tools/run_fullduplex_028.sh -m pytest -q \
  tests/config/test_stage_async_scheduling.py tests/tools/test_fullduplex_runtime_wrapper.py \
  tests/config/test_config_factory.py -m 'core_model and cpu'
tools/run_fullduplex_028.sh -m pytest -q \
  tests/core/sched tests/engine/duplex \
  tests/engine/test_kv_append.py tests/engine/test_stage_engine_core_proc.py \
  tests/engine/test_orchestrator.py tests/engine/test_stage_pool_collective_rpc.py \
  tests/engine/test_orchestrator_stage_input_bridge.py tests/engine/test_orchestrator_error_handling.py \
  tests/entrypoints/openai/test_duplex_protocol.py \
  tests/model_executor/models/personaplex/duplex/test_unified_runtime.py \
  tests/model_executor/models/nemotron_voicechat/duplex/test_runtime_contract.py \
  tests/worker/test_native_duplex_input_safety.py tests/worker/test_native_duplex_hooks.py \
  tests/helpers/tests/test_assertions.py -m 'core_model and cpu' --run-level core_model
```

## CUDA integration commands

Prerequisites: an idle H200/H100-class device with sufficient memory for the
declared single-GPU three-stage profile, complete MiniCPM-o 4.5 weights and
`assets/HT_ref_audio.wav`, and the checked-in WAV fixtures. `MODEL_PREFIX` must
contain `openbmb/MiniCPM-o-4_5` (a symlink to the actual cached checkpoint is
fine). Inspect GPU use first; the example GPU index is not a reservation.

```bash
export CUDA_VISIBLE_DEVICES=3
export MODEL_PREFIX=/path/to/model-parent
export HF_HUB_OFFLINE=1

# L3: real speech, two turns, independent sessions, resume/takeover.
tools/run_fullduplex_028.sh -m pytest -sv \
  tests/e2e/online_serving/test_minicpmo_4_5_duplex.py \
  -k 'single_session_response_required or two_sessions_resume_and_takeover' \
  -m 'advanced_model and cuda' --run-level advanced_model

# L3: configured four-session profile and two-slot contention; revalidate
# this shape on the exact candidate before making an acceptance claim.
tools/run_fullduplex_028.sh -m pytest -sv \
  tests/e2e/online_serving/test_minicpmo_4_5_duplex.py \
  -k four_sessions_rotate -m 'advanced_model and cuda' --run-level advanced_model

# Weekly reliability: real process loss/replay, rollover, lost reply, pending
# cancel/close, forced KV preemption and failed-input/concurrent-peer isolation.
tools/run_fullduplex_028.sh -m pytest -sv \
  tests/dfx/reliability/test_reliability_minicpmo_4_5_duplex.py \
  -m 'slow and H100 and cards_1' --run-level full_model
```

These commands cover MiniCPM hardware integration only. PersonaPlex and
Nemotron need their own cached weights, model-specific media, deployment
profiles, and real-audio runs after migration. Their CPU capability and
state-loss tests do not substitute for those runs. Answer-completeness checks
must distinguish a normal chunk EOS from a model turn EOS; the earlier
MiniCPM end-strategy risk is not resolved by removing compatibility APIs.

New CPU cases are collected by the existing ready/merge CPU sweeps. Reliability
cases extend the existing MiniCPM duplex weekly job. Ready/merge dependencies
also include the shared output carrier/template and AR runner files.

Use a fresh, task-specific `--basetemp` and `--junitxml` when recording evidence;
default pytest retention may delete an earlier run's media/event artifacts.
Fault-injection logs intentionally contain errors. A passing failure-isolation
test requires the injected failure on one session and real audio on its peer,
not an error-free log.

## Recording the cleanup decision

Record source SHA/worktree fingerprint, dependency versions, model/config and
input identities, commands, JUnit files, event logs, and output audio together.
Report CPU and each model/hardware E2E status separately. A completed protocol
response is insufficient when the answer is truncated or the audio workload
differs from the baseline. No earlier throughput percentage should be reused
as the speedup of this cleanup.
